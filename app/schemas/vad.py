"""AI-01 VAD 절단점 계산의 요청·응답.

<h2>입력은 청크가 아니라 Spring 이 잘라 만든 wav 다</h2>
설계 문서 「전송 포맷」이 정한 것이다 — 브라우저가 올리는 청크는 opus 지만 **VAD 입력만
wav** 이고, 자르고 변환하는 것은 Spring 의 ffmpeg 이 한다. 그래서 `s3Key` 가 가리키는 것은
원본 청크가 아니라 **이미 ±20초로 잘린 16kHz mono wav** 다.

그 결과 이 서버에는 창을 자를 이유가 없고(`windowMs` 가 없는 이유), 대신 그 wav 가 회의의
어디서 시작하는지를 알아야 절단점을 회의 기준으로 답할 수 있다(`windowStartOffsetMs`).

<h2>계약은 제안 상태다</h2>
명세에 AI-01 은 한 줄짜리 표 항목("S3 키로 ±20초만 전달. onnxruntime 버전 silero-vad")과
동작 규칙뿐이고 요청·응답 예시가 없다. 소비처인 Spring 블록 조립 경로가 아직 없다.
바뀌면 이 파일과 라우터만 고치면 된다 — 판정 로직(app/vad/cutpoint.py)은 그대로다.
"""

from pydantic import Field

from app.schemas.common import CamelModel
from app.vad.cutpoint import DEFAULT_MIN_SILENCE_MS


class CutpointRequest(CamelModel):
    """절단점을 찾을 wav 와 목표 지점.

    오프셋은 전부 **회의 시작 기준 경과 ms** 다. 브라우저마다 시계가 달라 절대 시각을 쓰지
    않는다는 설계 결정과 같은 좌표계이고, 응답도 여기 맞춰야 Spring 이 stt_block 의
    start_ms·end_ms 에 그대로 넣을 수 있다.
    """

    meeting_id: int
    bucket: str
    # Spring 이 만든 ±20초 wav. 원본 청크가 아니다.
    s3_key: str

    # 그 wav 의 첫 샘플이 회의의 어느 지점인가. 없으면 절단점을 회의 기준으로 되돌릴 수 없다.
    window_start_offset_ms: int = Field(ge=0)

    # 여기서 자르고 싶다(= 10분 경계). 무음을 못 찾았을 때의 절단점이기도 하다.
    target_offset_ms: int = Field(ge=0)

    # 명세가 정한 초기값이지 불변값이 아니다. 서버 상수로 박으면 조정할 때마다 배포해야 한다.
    min_silence_ms: int = Field(default=DEFAULT_MIN_SILENCE_MS, ge=100, le=10_000)


class CutpointResponse(CamelModel):
    """절단점과 그 근거.

    <h2>cutReason 이 응답의 절반이다</h2>
    stt_block.cut_reason 에 그대로 저장돼, 나중에 인식 품질이 나쁜 블록을 조사할 때 "말 중간을
    잘라서인가"를 가르는 유일한 단서가 된다. 값을 지어내면 그 조사가 성립하지 않는다.

    silenceMs 는 VAD_SILENCE 일 때만 값이 있다. FALLBACK 에서 0 을 채우면 "0ms 무음을 찾았다"로
    읽혀 못 찾은 것과 구분되지 않는다.
    """

    cut_offset_ms: int
    cut_reason: str
    silence_ms: int | None = None
