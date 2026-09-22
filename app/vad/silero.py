"""silero-vad(ONNX) 실행기 — PCM 을 발화 확률로 바꾼다.

<h2>onnxruntime 을 직접 쓴다</h2>
래퍼 패키지를 끼우지 않는다. 이 계층이 모델에서 쓰는 것은 입력 셋(`input`·`state`·`sr`)과
출력 둘(확률·다음 state)뿐이라, 래퍼가 주는 편의보다 그 API 가 버전마다 바뀌는 위험이 크다.
torch 를 끌어오지 않는 것도 이유다 — 추론만 하는 서버에 학습 프레임워크가 들어가면 이미지가
GB 단위로 커진다.

<h2>모델 파일은 이미지에 함께 넣는다</h2>
런타임에 받아오지 않는다. 모델이 바뀌면 절단점이 바뀌는데, 그게 배포와 무관하게 조용히
일어나면 "어제와 오늘의 블록 경계가 다른" 이유를 아무도 못 찾는다. 경로는 설정으로 받되
없으면 명확히 실패한다.

<h2>state 를 이어서 넘긴다</h2>
silero v5 는 RNN 이라 프레임 사이에 상태가 흐른다. 매 프레임 0 으로 초기화하면 모델이
직전 맥락을 잃고, 말 중간의 짧은 숨을 전부 무음으로 본다 — 그러면 700ms 조건을 만족하는
가짜 무음이 창마다 생긴다.

<h2>프레임 앞에 직전 64 샘플을 붙인다</h2>
v5 는 16kHz 에서 **576 샘플**(직전 프레임 꼬리 64 + 새 512)을 받는다. 앞의 64 는 내부
STFT 가 프레임 경계에서 쓰는 컨텍스트라, 이것 없이 512 만 넣으면 모델이 **모든 프레임을
무음으로 본다.** 입력 shape 가 `[None, None]` 이라 512 를 넣어도 예외 없이 돌아가므로
그 오작동이 조용하다 — 실제로 그 상태였고, 전 프레임이 무음이면 창 전체가 무음 런 하나가
되어 `VAD_SILENCE` 로 목표 지점에서 자른다. VAD 를 안 쓴 것과 결과가 같은데 성공으로
기록되는, 가장 나쁜 모양의 실패였다(41분 실오디오에서 발화 프레임 0/1875 로 확인).

공식 래퍼(torch)가 해 주던 일이다. 래퍼를 안 쓰는 판단은 유지하되, 래퍼가 하던 일 중
빠뜨리면 안 되는 것이 이것이다.
"""

from __future__ import annotations

import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from app.errors import LayerError, LayerErrorKind
from app.vad.audio import FRAME_SAMPLES, SAMPLE_RATE

# silero v5 의 은닉 상태 모양. 모델이 요구하는 값이라 바꿀 수 없다.
_STATE_SHAPE = (2, 1, 128)

# 프레임 앞에 붙이는 직전 샘플 수(16kHz 기준). 모델이 요구하는 값이라 바꿀 수 없다 —
# 512 + 64 = 576 이 v5 의 입력 길이다. 8kHz 로 내리면 32 이지만 이 서버는 16k 고정이다.
_CONTEXT_SAMPLES = 64

_lock = threading.Lock()
_session = None
_session_path: str | None = None

_executor_lock = threading.Lock()
_executor: ThreadPoolExecutor | None = None
_executor_size: int | None = None


def _executor_of(max_workers: int) -> ThreadPoolExecutor:
    """추론 전용 스레드 풀. **크기를 우리가 정한다.**

    기본 실행기(asyncio.to_thread)를 쓰면 크기가 `min(32, cpu+4)` 라 동시 요청이 몰릴 때
    CPU 코어보다 많은 추론이 서로 코어를 뺏는다. t3.medium(2 vCPU)에 Qdrant 까지 같이 떠
    있어 전부가 느려지고, 그러면 헬스체크까지 늦어져 컨테이너가 재시작된다.

    onnxruntime 세션도 스레드를 1 로 묶어 두었다(_load) — 병렬은 여기 한 곳에서만 정한다.
    """
    global _executor, _executor_size

    with _executor_lock:
        if _executor is None or _executor_size != max_workers:
            if _executor is not None:
                _executor.shutdown(wait=False)
            _executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="vad")
            _executor_size = max_workers
        return _executor


def _load(model_path: str):
    """세션을 한 번만 만든다.

    요청마다 만들면 20초 창 하나 처리하는 데 모델 로딩이 매번 붙는다. onnxruntime 세션은
    스레드 안전하므로 공유해도 된다 — 상태는 세션이 아니라 우리가 넘기는 `state` 에 있다.
    """
    global _session, _session_path

    with _lock:
        if _session is not None and _session_path == model_path:
            return _session

        if not Path(model_path).is_file():
            raise LayerError(
                LayerErrorKind.PERMANENT,
                "VAD_MODEL_MISSING",
                f"silero-vad 모델을 찾을 수 없습니다: {model_path}",
            )

        try:
            import onnxruntime
        except ImportError as exc:  # pragma: no cover - 의존성 누락은 배포 문제다
            raise LayerError(
                LayerErrorKind.PERMANENT,
                "VAD_RUNTIME_MISSING",
                "onnxruntime 이 설치되어 있지 않습니다.",
            ) from exc

        options = onnxruntime.SessionOptions()
        # CPU 한 장으로 20초를 도는 작업이다. 스레드를 늘리면 동시 요청끼리 코어를 뺏는다.
        options.inter_op_num_threads = 1
        options.intra_op_num_threads = 1

        _session = onnxruntime.InferenceSession(
            model_path, sess_options=options, providers=["CPUExecutionProvider"]
        )
        _session_path = model_path
        return _session


def speech_probs(pcm: bytes, model_path: str) -> list[float]:
    """s16le PCM → 프레임별 발화 확률.

    프레임을 못 채우는 꼬리는 버린다. 0 으로 채워 넣으면 그 패딩이 무음으로 잡혀서,
    창 끝에 실제로 없는 무음이 항상 하나 생긴다.
    """
    session = _load(model_path)

    samples = np.frombuffer(pcm, dtype=np.int16)
    usable = (len(samples) // FRAME_SAMPLES) * FRAME_SAMPLES
    if usable == 0:
        return []

    # int16 → float32 [-1, 1]. 모델이 정규화된 파형을 전제한다.
    audio = samples[:usable].astype(np.float32) / 32768.0
    frames = audio.reshape(-1, FRAME_SAMPLES)

    state = np.zeros(_STATE_SHAPE, dtype=np.float32)
    sample_rate = np.array(SAMPLE_RATE, dtype=np.int64)

    # 첫 프레임의 컨텍스트는 0 이다 — 창의 시작 앞에는 우리가 받은 오디오가 없다.
    # 호출마다 새로 시작한다(state 와 같은 이유): 앞 요청의 꼬리를 이어 쓰면 창 하나의
    # 판정이 직전에 어떤 요청이 왔는지에 따라 달라진다.
    context = np.zeros((1, _CONTEXT_SAMPLES), dtype=np.float32)

    probs: list[float] = []
    for frame in frames:
        window = np.concatenate([context, frame.reshape(1, -1)], axis=1)
        outputs = session.run(
            None,
            {"input": window, "state": state, "sr": sample_rate},
        )
        probs.append(float(outputs[0][0][0]))
        state = outputs[1]
        context = frame[-_CONTEXT_SAMPLES:].reshape(1, -1)

    return probs


async def speech_probs_async(pcm: bytes, model_path: str, max_workers: int) -> list[float]:
    """추론을 이벤트 루프 밖에서 돌린다.

    40초 창이면 프레임이 1250 개고 그만큼 session.run 이 돈다. 이걸 루프에서 그대로 부르면
    그 동안 **다른 요청도 /health 도 전부 멈춘다** — 워커가 하나뿐이라 헬스체크가 늦어지면
    컨테이너가 재시작되고, 돌던 요청까지 함께 죽는다.
    """
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(_executor_of(max_workers), speech_probs, pcm, model_path)
