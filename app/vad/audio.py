"""VAD 입력 wav 읽기.

<h2>디코딩하지 않는다</h2>
받는 것은 이미 **Spring 이 잘라 만든 ±20초 wav** 다(설계 문서 「전송 포맷 — VAD 입력만 wav」).
브라우저가 올리는 청크는 opus 지만, 그걸 푸는 것도 붙이는 것도 Spring 의 ffmpeg 이 한다 —
**오디오 원본을 다루는 쪽이 한 곳이어야 한다**(README 「이 서버가 하지 않는 것」).

그래서 여기에 ffmpeg 이 없다. 표준 라이브러리로 wav 헤더만 읽고 PCM 을 그대로 넘긴다.
런타임 이미지에 시스템 바이너리를 넣지 않아도 되고, 같은 오디오 처리가 두 곳에 생기지도 않는다.

<h2>형식을 고쳐주지 않는다</h2>
16kHz mono 16-bit 이 아니면 거절한다. 리샘플링을 여기서 하면 그 순간 이 서버가 오디오를
가공하기 시작하고, 위의 경계가 무너진다. 무엇을 보내야 하는지는 계약으로 못박는 편이 낫다.
"""

from __future__ import annotations

import io
import wave

from app.errors import LayerError, LayerErrorKind

# silero 가 지원하는 것은 8k·16k 뿐이다. 16k 로 고정한다 — 8k 는 자음 구분이 나빠져
# 무음 판정이 흔들린다.
SAMPLE_RATE = 16_000

# silero 가 16kHz 에서 요구하는 고정 창 크기(샘플). 이 값이 곧 프레임 하나의 길이다.
FRAME_SAMPLES = 512
FRAME_MS = FRAME_SAMPLES * 1000 // SAMPLE_RATE  # 32ms

_BYTES_PER_SAMPLE = 2

# 받아들일 최대 길이. 설계의 탐색 창이 ±20초라 40초가 상한이고, 회의 시작·끝에서는 그보다 짧다.
#
# 상한을 두는 이유는 **이 서버가 무엇을 받을지 고를 수 없기 때문**이다. s3Key 는 요청이 정하고,
# 실수든 아니든 10분짜리 원본을 가리키면 그 전부가 메모리에 올라온 뒤 프레임 수만큼 추론이 돈다.
# t3.medium 한 대에 Qdrant 까지 같이 떠 있어 그게 곧 인스턴스 전체의 정지다.
MAX_WINDOW_MS = 40_000
MAX_FRAMES = SAMPLE_RATE * MAX_WINDOW_MS // 1000

# wav 헤더 여유를 더한 바이트 상한. S3 에서 받을 때 이만큼만 읽는다.
MAX_WAV_BYTES = MAX_FRAMES * _BYTES_PER_SAMPLE + 4096


def read_pcm(wav_bytes: bytes) -> bytes:
    """VAD 입력 wav → s16le PCM.

    형식이 다르면 PERMANENT 다. 같은 파일을 다시 보내도 같은 결과이고, 고쳐야 할 곳은
    이 서버가 아니라 wav 를 만든 쪽이다.
    """
    try:
        with wave.open(io.BytesIO(wav_bytes), "rb") as source:
            channels = source.getnchannels()
            width = source.getsampwidth()
            rate = source.getframerate()
            total = source.getnframes()

            # readframes **앞에서** 막는다. 읽고 나서 재면 이미 메모리에 올라온 뒤다.
            if total > MAX_FRAMES:
                raise LayerError(
                    LayerErrorKind.PERMANENT,
                    "AUDIO_TOO_LONG",
                    f"VAD 입력은 {MAX_WINDOW_MS // 1000}초를 넘을 수 없습니다"
                    f"(받은 값: {total * 1000 // rate if rate else '?'}ms).",
                )

            frames = source.readframes(total)
    except (wave.Error, EOFError) as exc:
        raise LayerError(
            LayerErrorKind.PERMANENT,
            "AUDIO_NOT_WAV",
            f"VAD 입력이 wav 가 아닙니다: {exc}",
        ) from exc

    if (channels, width, rate) != (1, _BYTES_PER_SAMPLE, SAMPLE_RATE):
        # 리샘플링해 주지 않는다 — 그 순간 이 서버가 오디오를 가공하기 시작한다.
        raise LayerError(
            LayerErrorKind.PERMANENT,
            "AUDIO_FORMAT_UNSUPPORTED",
            f"VAD 입력은 {SAMPLE_RATE}Hz mono 16-bit wav 여야 합니다"
            f"(받은 값: {rate}Hz {channels}ch {width * 8}-bit).",
        )

    return frames


def frame_count(pcm: bytes) -> int:
    """PCM 이 몇 프레임인가. 남는 꼬리는 버린다 — silero 가 고정 길이만 받는다."""
    return len(pcm) // (FRAME_SAMPLES * _BYTES_PER_SAMPLE)
