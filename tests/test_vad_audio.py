"""VAD 입력 wav 읽기.

<h2>고쳐주지 않는 것을 검증한다</h2>
이 서버는 오디오를 가공하지 않는다 — 자르고 변환하는 것은 Spring 의 ffmpeg 이 한다(설계 문서
「전송 포맷 — VAD 입력만 wav」). 그래서 형식이 다르면 조용히 리샘플링하는 대신 거절해야 하고,
그게 이 파일이 보는 것이다. 여기서 관대해지면 오디오 처리가 두 곳에 생긴다.
"""

import io
import wave

import pytest

from app.errors import LayerError, LayerErrorKind
from app.vad.audio import FRAME_SAMPLES, MAX_FRAMES, SAMPLE_RATE, frame_count, read_pcm


def wav_bytes(samples: int, *, rate: int = SAMPLE_RATE, channels: int = 1, width: int = 2) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as sink:
        sink.setnchannels(channels)
        sink.setsampwidth(width)
        sink.setframerate(rate)
        sink.writeframes(b"\x00" * samples * width * channels)
    return buffer.getvalue()


class TestReadPcm:
    def test_16k_mono_16bit_은_그대로_읽는다(self):
        pcm = read_pcm(wav_bytes(FRAME_SAMPLES * 3))

        assert len(pcm) == FRAME_SAMPLES * 3 * 2

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"rate": 8_000},  # silero 가 받긴 하지만 자음 구분이 나빠 무음 판정이 흔들린다
            {"channels": 2},  # 스테레오를 섞어 주면 그 순간 이 서버가 오디오를 가공한다
            {"width": 1},  # 8-bit
        ],
    )
    def test_형식이_다르면_거절한다(self, kwargs):
        with pytest.raises(LayerError) as caught:
            read_pcm(wav_bytes(FRAME_SAMPLES, **kwargs))

        # 같은 파일을 다시 보내도 같은 결과다. 고쳐야 할 곳은 이 서버가 아니라 wav 를 만든 쪽이다.
        assert caught.value.kind is LayerErrorKind.PERMANENT
        assert caught.value.code == "AUDIO_FORMAT_UNSUPPORTED"
        assert not caught.value.retryable

    def test_40초까지는_받는다(self):
        # 탐색 창이 ±20초라 40초가 상한이다. 경계값을 거절하면 정상 입력이 막힌다.
        pcm = read_pcm(wav_bytes(MAX_FRAMES))

        assert len(pcm) == MAX_FRAMES * 2

    def test_40초를_넘으면_거절한다(self):
        """이 서버는 무엇을 받을지 고를 수 없다 — s3Key 는 요청이 정한다.

        실수든 아니든 10분짜리 원본을 가리키면 그 전부가 메모리에 올라온 뒤 프레임 수만큼
        추론이 돈다. t3.medium 한 대에 Qdrant 까지 같이 떠 있어 그건 곧 인스턴스 전체의 정지다.
        """
        with pytest.raises(LayerError) as caught:
            read_pcm(wav_bytes(MAX_FRAMES + 1))

        assert caught.value.code == "AUDIO_TOO_LONG"
        assert caught.value.kind is LayerErrorKind.PERMANENT

    def test_wav_가_아니면_거절한다(self):
        with pytest.raises(LayerError) as caught:
            read_pcm(b"this is not a wav file")

        assert caught.value.code == "AUDIO_NOT_WAV"
        assert caught.value.kind is LayerErrorKind.PERMANENT


class TestFrameCount:
    def test_프레임을_못_채우는_꼬리는_버린다(self):
        # 0 으로 채워 넣으면 그 패딩이 무음으로 잡혀 창 끝에 없는 무음이 항상 하나 생긴다.
        pcm = b"\x00" * ((FRAME_SAMPLES * 2 + 100) * 2)

        assert frame_count(pcm) == 2
