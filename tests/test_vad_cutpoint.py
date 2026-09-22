"""AI-01 절단점 판정.

<h2>모델도 오디오도 없이 돈다</h2>
검증 대상은 "발화 확률이 이렇게 생겼을 때 어디서 자르는가"이지 silero 가 확률을 잘 내는가가
아니다. 후자는 우리 코드가 아니고, 그걸 함께 돌리면 테스트에 ffmpeg·S3·모델 파일이 필요해져
아무도 안 돌리게 된다.

<h2>가장 비싼 실패는 "잘못된 자리에서 자르는 것"이다</h2>
자른 자리가 말 중간이면 그 단어는 앞뒤 블록 어디서도 온전히 인식되지 않고, 손실이 정본에
그대로 남아 뒤 계층 전부가 그 문장을 못 본다. 조용히 일어나므로 여기서 잡아야 한다.
"""

from app.vad.cutpoint import (
    CUT_FALLBACK_OVERLAP,
    CUT_VAD_SILENCE,
    choose_cutpoint,
    find_silence_runs,
)

FRAME_MS = 32  # silero 16kHz 고정 창(512 샘플)


def probs(pattern: str) -> list[float]:
    """'s' = 무음, 'v' = 발화. 프레임 배열을 읽기 쉽게 쓴다."""
    return [0.05 if ch == "s" else 0.9 for ch in pattern]


class TestSilenceRuns:
    def test_연속된_무음을_한_구간으로_묶는다(self):
        runs = find_silence_runs(probs("vvsssvv"), FRAME_MS, window_start_ms=0)

        assert len(runs) == 1
        assert runs[0].start_ms == 2 * FRAME_MS
        assert runs[0].duration_ms == 3 * FRAME_MS

    def test_창_끝까지_이어진_무음도_닫는다(self):
        # 루프 안에서 닫히지 않는 자리다. 빠뜨리면 경계 직전의 긴 침묵 —
        # 회의가 잦아드는 자리라 가장 좋은 절단점인 경우 — 을 통째로 놓친다.
        runs = find_silence_runs(probs("vvssss"), FRAME_MS, window_start_ms=0)

        assert len(runs) == 1
        assert runs[0].duration_ms == 4 * FRAME_MS

    def test_창_시작_오프셋이_결과에_더해진다(self):
        runs = find_silence_runs(probs("ssv"), FRAME_MS, window_start_ms=580_000)

        assert runs[0].start_ms == 580_000


class TestChooseCutpoint:
    def test_무음_한가운데에서_자른다(self):
        """시작이나 끝이 아니라 가운데다 — 양쪽 여백을 최대로 남기는 지점이다.

        경계에서 자르면 디코딩 오차 몇십 ms 만으로도 앞뒤 블록 중 한쪽이 첫 음절을 먹는다.
        """
        # 발화 10프레임 → 무음 30프레임(960ms) → 발화 10프레임
        cut = choose_cutpoint(
            probs("v" * 10 + "s" * 30 + "v" * 10),
            frame_ms=FRAME_MS,
            window_start_ms=0,
            target_offset_ms=800,
        )

        assert cut.cut_reason == CUT_VAD_SILENCE
        assert cut.cut_offset_ms == (10 * FRAME_MS + 40 * FRAME_MS) // 2
        assert cut.silence_ms == 30 * FRAME_MS

    def test_700ms_미만_무음은_후보가_아니다(self):
        # 20프레임 = 640ms < 700ms. 짧은 숨은 절단점이 아니다 — 거기서 자르면
        # 사실상 말 중간에서 자르는 것과 같다.
        cut = choose_cutpoint(
            probs("v" * 5 + "s" * 20 + "v" * 5),
            frame_ms=FRAME_MS,
            window_start_ms=0,
            target_offset_ms=500,
        )

        assert cut.cut_reason == CUT_FALLBACK_OVERLAP
        assert cut.cut_offset_ms == 500
        # 0 을 채우면 "0ms 무음을 찾았다"로 읽혀 못 찾은 것과 구분되지 않는다.
        assert cut.silence_ms is None

    def test_가장_긴_무음을_고른다(self):
        """설계 문서가 정한 규칙이다 — "뒤쪽 ±20초에서 가장 긴 무음을 찾아 거기서 블록을 끊고".

        긴 침묵일수록 말이 실제로 끊긴 자리이고, 절단 오차가 앞뒤 발화를 건드릴 여지도 적다.
        """
        # 앞쪽에 조건을 넘기는 짧은 무음(25프레임), 뒤쪽에 더 긴 무음(60프레임).
        # 목표는 앞쪽에 붙여 둔다 — "가까운 것"을 고르면 앞쪽이 뽑힌다.
        pattern = "s" * 25 + "v" * 10 + "s" * 60
        target = 25 * FRAME_MS // 2

        cut = choose_cutpoint(probs(pattern), frame_ms=FRAME_MS, window_start_ms=0, target_offset_ms=target)

        assert cut.cut_reason == CUT_VAD_SILENCE
        assert cut.silence_ms == 60 * FRAME_MS

    def test_길이가_같으면_목표에_가까운_쪽(self):
        # 순서에 맡기면 창의 앞쪽이 늘 이겨, 블록이 목표보다 짧아지는 편향이 생긴다.
        pattern = "s" * 25 + "v" * 10 + "s" * 25
        target = (35 + 12) * FRAME_MS  # 뒤쪽 무음 한가운데 근처

        cut = choose_cutpoint(probs(pattern), frame_ms=FRAME_MS, window_start_ms=0, target_offset_ms=target)

        assert cut.cut_offset_ms > 30 * FRAME_MS

    def test_무음이_전혀_없으면_목표에서_자른다(self):
        # 말이 끊이지 않는 회의에서는 정상이다. 여기서 예외를 던지면 블록 조립이 멈춰
        # 그 회의가 STT 를 아예 못 받는다.
        cut = choose_cutpoint(probs("v" * 50), frame_ms=FRAME_MS, window_start_ms=0, target_offset_ms=600_000)

        assert cut.cut_reason == CUT_FALLBACK_OVERLAP
        assert cut.cut_offset_ms == 600_000

    def test_빈_확률_배열도_목표에서_자른다(self):
        cut = choose_cutpoint([], frame_ms=FRAME_MS, window_start_ms=0, target_offset_ms=1_000)

        assert cut.cut_reason == CUT_FALLBACK_OVERLAP

    def test_임계값을_요청으로_낮출_수_있다(self):
        # 700ms 는 명세의 초기값이지 불변값이 아니다. 서버 상수로 박으면 조정할 때마다 배포해야 한다.
        pattern = probs("v" * 5 + "s" * 20 + "v" * 5)

        assert choose_cutpoint(pattern, FRAME_MS, 0, 500).cut_reason == CUT_FALLBACK_OVERLAP
        assert choose_cutpoint(pattern, FRAME_MS, 0, 500, min_silence_ms=300).cut_reason == CUT_VAD_SILENCE
