"""절단점 판정 — 무음 구간에서 블록 경계를 고른다.

<h2>이 파일에 모델도 S3 도 없다</h2>
입력은 **발화 확률 배열**이고 출력은 절단점이다. 모델을 부르는 것과 무음을 고르는 것을
나눈 이유는, 판정 규칙이 이 파이프라인에서 가장 자주 바뀔 자리이기 때문이다 — 700ms 도
±20초도 명세가 정한 초기값이지 불변값이 아니다. 순수 함수로 두면 그 규칙을 오디오 없이
테스트할 수 있다.

<h2>왜 절단이 필요한가</h2>
녹음은 10분 단위로 잘라 STT 에 넣는데, 아무 데나 자르면 **말하는 중간이 잘린다.** 잘린
자리의 단어는 앞 블록에서도 뒤 블록에서도 온전히 인식되지 않고, 그 손실은 정본에 그대로
남아 뒤 계층 전부가 그 문장을 못 본다. 그래서 경계 근처에서 사람이 말을 쉬는 지점을 찾는다.
"""

from __future__ import annotations

from dataclasses import dataclass

# 명세 「청크 업로드 완료 통보」의 절단 규칙에서 온 값이다. 요청으로 덮을 수 있다 —
# 튜닝 대상이라 서버 상수로 박으면 값을 바꿀 때마다 배포해야 한다.
#
# 탐색 창(±20초)은 여기 없다. 창을 자르는 것은 Spring 이고(설계 문서 「VAD 입력만 wav」),
# 이 서버가 받는 wav 가 이미 그 창이다.
DEFAULT_MIN_SILENCE_MS = 700

# 무음 판정 임계값. silero 가 주는 발화 확률이 이 아래면 무음으로 본다.
# 0.5 는 silero 의 관용 기본값이고, 낮추면 숨소리를 발화로 세어 무음을 못 찾는다.
SPEECH_THRESHOLD = 0.5

CUT_VAD_SILENCE = "VAD_SILENCE"
CUT_FALLBACK_OVERLAP = "FALLBACK_OVERLAP"


@dataclass(frozen=True)
class Cutpoint:
    """판정 결과.

    silence_ms 는 실제로 찾은 무음 길이다. FALLBACK 이면 None — 0 으로 채우면 "0ms 무음을
    찾았다"로 읽혀서, 못 찾은 것과 찾았는데 짧은 것이 구분되지 않는다.
    """

    cut_offset_ms: int
    cut_reason: str
    silence_ms: int | None


@dataclass(frozen=True)
class SilenceRun:
    start_ms: int
    end_ms: int

    @property
    def duration_ms(self) -> int:
        return self.end_ms - self.start_ms

    @property
    def middle_ms(self) -> int:
        """무음 한가운데. **양쪽 여백을 최대로 남기는 지점이다.**

        시작이나 끝에서 자르면 무음 경계가 실제 발화와 맞닿아, 디코딩 오차 몇십 ms 만으로도
        앞뒤 블록 중 한쪽이 단어 첫 음절을 먹는다. 가운데면 그 오차를 양쪽이 나눠 흡수한다.
        """
        return (self.start_ms + self.end_ms) // 2


def find_silence_runs(
    speech_probs: list[float],
    frame_ms: int,
    window_start_ms: int,
    threshold: float = SPEECH_THRESHOLD,
) -> list[SilenceRun]:
    """발화 확률 배열에서 연속된 무음 구간을 모은다.

    프레임 하나하나가 아니라 **연속 구간**으로 묶는 이유는 판정 기준이 길이(700ms)이기
    때문이다. 프레임 단위로 보면 어느 프레임이 무음인지만 알고 얼마나 이어졌는지는 모른다.
    """
    runs: list[SilenceRun] = []
    run_start: int | None = None

    for index, prob in enumerate(speech_probs):
        silent = prob < threshold
        if silent and run_start is None:
            run_start = index
        elif not silent and run_start is not None:
            runs.append(_run_of(run_start, index, frame_ms, window_start_ms))
            run_start = None

    # 창 끝까지 무음이면 루프 안에서 닫히지 않는다. 빠뜨리면 경계 직전의 긴 침묵 —
    # 회의가 잦아드는 자리라 가장 좋은 절단점인 경우 — 을 통째로 놓친다.
    if run_start is not None:
        runs.append(_run_of(run_start, len(speech_probs), frame_ms, window_start_ms))

    return runs


def _run_of(start_index: int, end_index: int, frame_ms: int, window_start_ms: int) -> SilenceRun:
    return SilenceRun(
        start_ms=window_start_ms + start_index * frame_ms,
        end_ms=window_start_ms + end_index * frame_ms,
    )


def choose_cutpoint(
    speech_probs: list[float],
    frame_ms: int,
    window_start_ms: int,
    target_offset_ms: int,
    min_silence_ms: int = DEFAULT_MIN_SILENCE_MS,
    threshold: float = SPEECH_THRESHOLD,
) -> Cutpoint:
    """창 안에서 절단점을 고른다. 못 찾으면 목표 지점에서 그대로 자른다.

    <h2>가장 긴 무음을 고른다</h2>
    설계 문서가 정한 규칙이다 — *"VAD 가 뒤쪽 ±20초에서 **가장 긴 무음**을 찾아 거기서 블록을
    끊고"*. 긴 침묵일수록 말이 실제로 끊긴 자리이고, 절단 오차가 앞뒤 발화를 건드릴 여지도 그만큼
    적다.

    길이가 같으면 목표에 가까운 쪽을 쓴다. 순서에 맡기면 같은 입력에서 창의 앞쪽이 늘 이겨,
    블록이 목표보다 짧아지는 편향이 생긴다.

    <h2>못 찾으면 실패가 아니다</h2>
    말이 끊이지 않는 회의에서는 700ms 무음이 없는 것이 정상이다. 그때는 목표 지점에서 자르고
    이유를 FALLBACK_OVERLAP 으로 남긴다 — 오버랩으로 손실을 덮는다는 뜻이고, 그 사실이
    stt_block.cut_reason 에 남아 나중에 인식 품질을 조사할 때 근거가 된다.

    여기서 예외를 던지면 블록 조립 자체가 멈춘다. 절단점을 못 고르는 것보다 회의가 통째로
    STT 를 못 받는 쪽이 훨씬 나쁘다.
    """
    candidates = [
        run
        for run in find_silence_runs(speech_probs, frame_ms, window_start_ms, threshold)
        if run.duration_ms >= min_silence_ms
    ]

    if not candidates:
        return Cutpoint(target_offset_ms, CUT_FALLBACK_OVERLAP, None)

    best = max(candidates, key=lambda run: (run.duration_ms, -abs(run.middle_ms - target_offset_ms)))
    return Cutpoint(best.middle_ms, CUT_VAD_SILENCE, best.duration_ms)
