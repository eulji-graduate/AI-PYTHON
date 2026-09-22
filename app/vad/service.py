"""AI-01 조립 — S3 → wav 읽기 → 모델 → 판정.

네 단계를 여기서만 잇는다. 각 단계가 자기 파일에 있고 이 파일에는 순서와 실패 처리만 있다 —
그래야 판정 규칙(cutpoint.py)을 오디오 없이, wav 읽기(audio.py)를 모델 없이 테스트할 수 있다.
"""

from __future__ import annotations

import logging

from app.clients import s3
from app.config import Settings
from app.schemas.vad import CutpointRequest, CutpointResponse
from app.vad import silero
from app.vad.audio import FRAME_MS, MAX_WAV_BYTES, read_pcm
from app.vad.cutpoint import CUT_FALLBACK_OVERLAP, choose_cutpoint

log = logging.getLogger(__name__)


async def find_cutpoint(request: CutpointRequest, settings: Settings) -> CutpointResponse:
    """절단점을 찾는다. 못 찾는 것은 실패가 아니다."""
    # 상한을 걸어 받는다 — s3Key 는 요청이 정하고, 원본을 가리키면 그 전부가 메모리에 올라온다.
    wav = await s3.fetch(request.bucket, request.s3_key, settings.s3_region, MAX_WAV_BYTES)
    pcm = read_pcm(wav)

    probs = await silero.speech_probs_async(pcm, settings.vad_model_path, settings.vad_max_workers)
    if not probs:
        # 프레임 하나를 못 채울 만큼 짧다. 목표에서 자르고 이유를 남긴다 — 여기서 예외를
        # 던지면 블록 조립이 멈춰 그 회의가 STT 를 아예 못 받는다.
        log.warning(
            "VAD 입력이 너무 짧다 — meetingId=%s key=%s bytes=%s",
            request.meeting_id,
            request.s3_key,
            len(pcm),
        )
        return CutpointResponse(cut_offset_ms=request.target_offset_ms, cut_reason=CUT_FALLBACK_OVERLAP)

    cut = choose_cutpoint(
        probs,
        frame_ms=FRAME_MS,
        # wav 안의 위치를 회의 기준으로 되돌린다. 이게 없으면 Spring 이 stt_block 에 넣을
        # start_ms·end_ms 를 만들 수 없다.
        window_start_ms=request.window_start_offset_ms,
        target_offset_ms=request.target_offset_ms,
        min_silence_ms=request.min_silence_ms,
    )

    log.info(
        "VAD 절단점 — meetingId=%s target=%sms cut=%sms reason=%s silence=%s",
        request.meeting_id,
        request.target_offset_ms,
        cut.cut_offset_ms,
        cut.cut_reason,
        cut.silence_ms,
    )
    return CutpointResponse(
        cut_offset_ms=cut.cut_offset_ms, cut_reason=cut.cut_reason, silence_ms=cut.silence_ms
    )
