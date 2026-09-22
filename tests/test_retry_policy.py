"""재시도 대기시간 정책.

계층 호출 실패는 셋으로 갈리고 대기시간 계산이 다르다. 여기가 틀리면 레이트리밋에
걸린 상태로 계속 때려 쿼터만 태우거나, 반대로 바로 재개할 수 있는데 30초를 쉰다.
"""

from app.clients.gemini import GeminiClient
from app.config import Settings
from app.errors import LayerError, LayerErrorKind

SETTINGS = Settings(internal_token="t", retry_delays_sec=(2.0, 8.0, 30.0), retry_jitter_ratio=0.2)
CLIENT = GeminiClient(SETTINGS)


def _rate_limit(retry_after):
    return LayerError(LayerErrorKind.RATE_LIMIT, "RATE_LIMITED", "429", retry_after_sec=retry_after)


def test_Retry_After_0은_그대로_0이다():
    # 서버가 "바로 다시 보내도 된다"고 답한 것이다. 참/거짓으로 보면 0 이 falsy 라
    # 우리 백오프로 떨어져 30초를 기다리게 된다.
    assert CLIENT._delay_for(2, _rate_limit(0)) == 0.0


def test_Retry_After가_있으면_그_값을_쓴다():
    assert CLIENT._delay_for(0, _rate_limit(45)) == 45.0


def test_음수_Retry_After는_0으로_깎는다():
    assert CLIENT._delay_for(0, _rate_limit(-5)) == 0.0


def test_Retry_After가_없으면_지수_백오프로_떨어진다():
    delay = CLIENT._delay_for(2, _rate_limit(None))

    # 30초 ±20%
    assert 24.0 <= delay <= 36.0


def test_일시적_실패는_시도_순서대로_커진다():
    transient = LayerError(LayerErrorKind.TRANSIENT, "PROVIDER_UNAVAILABLE", "timeout")

    first = CLIENT._delay_for(0, transient)
    third = CLIENT._delay_for(2, transient)

    assert 1.6 <= first <= 2.4
    assert 24.0 <= third <= 36.0
