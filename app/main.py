"""AI EC2 계층 서버 진입점.

Spring 이 이 서버를 부르는 방향만 존재한다(단방향). 이 서버는 업무 DB에 접속하지 않고,
오디오는 항상 S3 키로 주고받는다 — 두 인스턴스 사이에 공유 볼륨이 없다.
"""

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from starlette.requests import Request

from app.errors import LayerError
from app.routers import internal

app = FastAPI(
    title="잇다 · AI 계층 서버",
    version="0.1.0",
    description="회의 분석 계층(L1.5~L5) · VAD · 벡터. 내부 호출 전용.",
)

app.include_router(internal.router)


@app.exception_handler(LayerError)
async def layer_error_handler(_: Request, exc: LayerError) -> JSONResponse:
    """실패를 Spring 이 그대로 기록할 수 있는 모양으로 내려준다.

    `retryable` 을 응답에 넣는 이유: 재시도 판단을 Spring 이 메시지 문자열로
    추측하게 두면 영구 실패를 세 번 재시도해 토큰만 태운다. 판정은 실패를 만든
    쪽이 한다.
    """
    # 영구 실패는 422 — 같은 입력으로 다시 보내지 말라는 뜻이다.
    # 일시적 실패는 503 — 큐에 남겨 두고 백오프 후 재시도하라는 뜻이다.
    http_status = 422 if not exc.retryable else 503
    return JSONResponse(
        status_code=http_status,
        content={
            "code": exc.code,
            "kind": exc.kind.value,
            "retryable": exc.retryable,
            "retryAfterSec": exc.retry_after_sec,
            "message": exc.message,
        },
    )


@app.get("/health")
async def health() -> dict:
    """인프라 liveness — 무인증. ALB·컨테이너 헬스체크가 토큰을 들고 있을 수 없다.
    계층 수용 가능 여부는 `/internal/health`(AI-10)가 따로 답한다."""
    return {"status": "UP"}
