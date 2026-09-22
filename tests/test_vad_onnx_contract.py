"""silero ONNX 계약 — 모델 파일이 코드가 기대하는 모양인가.

<h2>왜 이 테스트가 있나</h2>
`app/vad/silero.py` 는 텐서 **이름과 모양**을 직접 안다(`input`·`state`·`sr` → 확률·다음 state).
그 가정이 틀리면 첫 호출에서 깨지는데, 나머지 테스트는 전부 순수 함수 위에서 돌기 때문에
아무도 잡지 못한다.

특히 위험한 순간은 **모델을 새 버전으로 갈아끼울 때**다. v4 계열은 state 가 h·c 두 개로
나뉘어 있었고, 그런 변화는 파일만 바꿔도 조용히 들어온다. 그때 여기서 빨간불이 떠야 한다.

<h2>모델이 없으면 건너뛴다</h2>
파일은 이미지에 함께 넣는 것이라 개발 환경에 없을 수 있다. 없는 것을 실패로 만들면 사람들이
`-k` 로 빼기 시작하고, 그러면 있을 때도 안 돌게 된다.
"""

from pathlib import Path

import numpy as np
import pytest

from app.vad import silero
from app.vad.audio import FRAME_SAMPLES, SAMPLE_RATE
from app.vad.silero import _CONTEXT_SAMPLES, speech_probs

MODEL = Path(__file__).resolve().parent.parent / "models" / "silero_vad.onnx"

# ⚠ 모듈 전체에 걸지 않는다. "모델이 없으면 거절한다"는 테스트는 **모델이 없을 때 돌아야 할
# 테스트**인데, 모듈 스킵이면 정확히 그 상황에서 건너뛴다 — 검증하려던 조건이 스킵 조건과
# 같아지는 자리다(CodeRabbit PR #15). 모델을 실제로 여는 테스트에만 붙인다.
needs_model = pytest.mark.skipif(not MODEL.is_file(), reason=f"모델 파일이 없다: {MODEL}")


@needs_model
def test_모델이_코드가_아는_이름과_모양을_요구한다():
    import onnxruntime

    session = onnxruntime.InferenceSession(str(MODEL), providers=["CPUExecutionProvider"])
    inputs = {i.name: i for i in session.get_inputs()}

    # silero.py 가 이 이름들로 넘긴다. 하나라도 다르면 첫 호출에서 깨진다.
    assert set(inputs) == {"input", "state", "sr"}
    # state 는 [2, batch, 128]. 가운데가 동적이라 우리가 1 로 쓴다.
    assert inputs["state"].shape[0] == 2
    assert inputs["state"].shape[2] == 128
    # 출력 순서에 기대고 있다 — [0] 확률, [1] 다음 state.
    assert len(session.get_outputs()) == 2


@needs_model
def test_프레임_수만큼_확률이_나온다():
    """플러밍 검증이다 — state 를 이어 넘기는 것까지 포함해 끝까지 도는지 본다.

    ⚠ **판정 품질은 검증하지 않는다.** 합성 신호는 사람 목소리가 아니라 silero 가 무음으로
    보는 것이 정상이고, 여기서 확률의 절대값을 못박으면 모델을 올릴 때마다 의미 없이 깨진다.
    실제 음성으로 확인하는 것은 gold set 회의를 돌려 보는 자리다.
    """
    frames = 30
    pcm = (np.zeros(FRAME_SAMPLES * frames) * 32767).astype(np.int16).tobytes()

    probs = speech_probs(pcm, str(MODEL))

    assert len(probs) == frames
    assert all(0.0 <= p <= 1.0 for p in probs)


class _입력을_기록하는_세션:
    """실제 추론 대신 넘어온 입력만 모은다.

    모델을 진짜로 돌리지 않는 이유는, 검증하려는 것이 **모델의 판정**이 아니라 우리가
    모델에 넣는 입력의 모양이기 때문이다. 판정을 합성 신호로 검증하려 들면 값을 못박게
    되고 그건 모델을 올릴 때마다 의미 없이 깨진다(아래 플러밍 테스트 주석과 같은 이유).
    """

    def __init__(self) -> None:
        self.windows: list[np.ndarray] = []

    def run(self, _outputs, feeds):
        self.windows.append(feeds["input"].copy())
        return [np.array([[0.0]], dtype=np.float32), np.zeros((2, 1, 128), dtype=np.float32)]


def test_프레임_앞에_직전_64샘플을_붙여_넣는다(monkeypatch):
    """v5 는 16kHz 에서 576 샘플(64 컨텍스트 + 512)을 받는다.

    512 만 넣으면 **모든 프레임이 무음으로 나온다.** 입력 shape 가 [None, None] 이라
    예외는 안 나므로, 이 회귀는 오직 여기서만 잡힌다. 실제로 그 상태로 배포돼 있었고,
    전 프레임 무음 → 창 전체가 무음 런 하나 → VAD_SILENCE 로 목표 지점 절단이 되어
    "VAD 를 안 쓴 것과 같은데 성공으로 기록되는" 실패였다.
    """
    session = _입력을_기록하는_세션()
    monkeypatch.setattr(silero, "_load", lambda _path: session)

    # 프레임 세 개. 값이 서로 달라야 꼬리가 제대로 넘어갔는지 볼 수 있다.
    samples = np.arange(FRAME_SAMPLES * 3, dtype=np.int16) * 3
    speech_probs(samples.tobytes(), "쓰이지 않는 경로")

    assert len(session.windows) == 3
    assert all(w.shape == (1, FRAME_SAMPLES + _CONTEXT_SAMPLES) for w in session.windows)

    audio = samples.astype(np.float32) / 32768.0

    # 첫 프레임 앞에는 우리가 받은 오디오가 없다 — 0 으로 시작한다.
    assert np.array_equal(
        session.windows[0][0, :_CONTEXT_SAMPLES], np.zeros(_CONTEXT_SAMPLES, dtype=np.float32)
    )

    # 두 번째부터는 **직전 프레임의 꼬리**가 앞에 온다. 여기가 어긋나면 경계마다
    # 컨텍스트가 틀려 판정이 조용히 나빠진다.
    assert np.allclose(
        session.windows[1][0, :_CONTEXT_SAMPLES],
        audio[FRAME_SAMPLES - _CONTEXT_SAMPLES : FRAME_SAMPLES],
    )
    assert np.allclose(session.windows[1][0, _CONTEXT_SAMPLES:], audio[FRAME_SAMPLES : FRAME_SAMPLES * 2])


def test_호출마다_컨텍스트를_새로_시작한다(monkeypatch):
    """앞 요청의 꼬리를 이어 쓰면 창 하나의 판정이 '직전에 어떤 요청이 왔는지'에 따라
    달라진다. 같은 wav 를 두 번 보내면 같은 절단점이 나와야 한다."""
    session = _입력을_기록하는_세션()
    monkeypatch.setattr(silero, "_load", lambda _path: session)

    pcm = (np.arange(FRAME_SAMPLES * 2, dtype=np.int16) * 3).tobytes()
    speech_probs(pcm, "쓰이지 않는 경로")
    speech_probs(pcm, "쓰이지 않는 경로")

    assert np.array_equal(session.windows[0], session.windows[2])
    assert np.array_equal(session.windows[1], session.windows[3])


@needs_model
def test_프레임을_못_채우면_빈_결과다():
    # 0 으로 채워 넣으면 그 패딩이 무음으로 잡혀 없는 무음이 하나 생긴다.
    pcm = (np.zeros(FRAME_SAMPLES - 1)).astype(np.int16).tobytes()

    assert speech_probs(pcm, str(MODEL)) == []


def test_모델이_없으면_명확히_거절한다():
    from app.errors import LayerError, LayerErrorKind

    with pytest.raises(LayerError) as caught:
        speech_probs(b"\x00" * (FRAME_SAMPLES * 2), str(MODEL.parent / "없는파일.onnx"))

    # 조용히 통과시키면 무음을 하나도 못 찾아 모든 블록이 FALLBACK_OVERLAP 이 되고,
    # 그게 "절단이 잘 안 되는 회의"로 위장된다.
    assert caught.value.code == "VAD_MODEL_MISSING"
    assert caught.value.kind is LayerErrorKind.PERMANENT


def test_샘플레이트가_16k_로_고정돼_있다():
    # 8k 를 쓰면 자음 구분이 나빠져 무음 판정이 흔들린다. 값이 바뀌면 프레임 길이(32ms)도
    # 함께 어긋나 절단점이 통째로 밀린다.
    assert SAMPLE_RATE == 16_000
    assert FRAME_SAMPLES == 512
