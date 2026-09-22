"""계층 호출 실패 분류.

명세 「레이어 호출 실패 정책」을 코드로 옮긴 것이다. 셋을 구분하는 이유는
재시도 여부가 완전히 다르기 때문이다.

    TRANSIENT   타임아웃 · 5xx · 순단        → 지수 백오프 3회 (2s · 8s · 30s ±20%)
    RATE_LIMIT  429                         → Retry-After 존중
    PERMANENT   스키마 위반 · 컨텍스트 초과   → 즉시 실패. 재시도해도 토큰만 태운다

PERMANENT 를 재시도하면 같은 입력으로 같은 실패를 세 번 하면서 과금만 3배가 된다.
"""

from enum import Enum


class LayerErrorKind(str, Enum):
    TRANSIENT = "TRANSIENT"
    RATE_LIMIT = "RATE_LIMIT"
    PERMANENT = "PERMANENT"


class LayerError(Exception):
    """계층 실행 실패. Spring 이 analysis_layer.error_code 에 그대로 적는다."""

    def __init__(
        self,
        kind: LayerErrorKind,
        code: str,
        message: str,
        retry_after_sec: float | None = None,
    ) -> None:
        super().__init__(message)
        self.kind = kind
        self.code = code
        self.message = message
        self.retry_after_sec = retry_after_sec

    @property
    def retryable(self) -> bool:
        return self.kind is not LayerErrorKind.PERMANENT


def classify_provider_error(exc: Exception) -> LayerError:
    """Gemini SDK 예외를 위 셋으로 나눈다.

    SDK 내부 예외 계층에 기대지 않고 상태코드와 메시지로 판정한다 — SDK 버전이
    올라가며 예외 클래스가 바뀌어도 분류가 조용히 무너지지 않게.
    """
    code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
    text = str(exc).lower()

    if code == 429 or "resource_exhausted" in text or "rate limit" in text:
        return LayerError(
            LayerErrorKind.RATE_LIMIT,
            "RATE_LIMITED",
            str(exc),
            retry_after_sec=_parse_retry_after(exc),
        )

    # 컨텍스트 초과·잘못된 스키마는 400 계열로 오고, 같은 입력으로 다시 보내도 같은 결과다.
    if isinstance(code, int) and 400 <= code < 500 and code not in (408, 429):
        return LayerError(LayerErrorKind.PERMANENT, "PROVIDER_REJECTED", str(exc))

    if "context" in text and ("exceed" in text or "too long" in text or "token" in text):
        return LayerError(LayerErrorKind.PERMANENT, "CONTEXT_EXCEEDED", str(exc))

    return LayerError(LayerErrorKind.TRANSIENT, "PROVIDER_UNAVAILABLE", str(exc))


def _parse_retry_after(exc: Exception) -> float | None:
    value = getattr(exc, "retry_after", None)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
