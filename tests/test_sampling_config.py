"""샘플링 설정 — 같은 입력에 같은 출력이 나와야 한다.

이 파일이 지키는 것은 **측정 가능성**이다. 흔들리는 계층 위에서는 프롬프트를 고쳐도
나아졌는지 말할 수 없고, 사용자에게는 같은 회의를 재분석할 때 판정이 바뀌는 것으로 보인다.

2026-08-14 실측 — L3.5 게이트에 같은 프롬프트를 반복 호출했다. `temperature=0` 만일 때 2회 중
항목 12건에서 2건, `seed` 를 더한 뒤 3회 중 항목 14건에서 매회 3건의 판정이 뒤집혔다.

⚠ **그래서 이 테스트는 "흔들리지 않는다"를 지키지 않는다.** 그건 우리 코드로 지킬 수 없다는
것이 위 실측의 결론이다. 여기서 지키는 것은 **줄 수 있는 것을 주고 있는지** 하나다 —
두 값 중 하나가 조용히 빠지면 흔들림이 더 커지는데, 잡음 바닥 안에 묻혀서 아무도 모른다.

실제 흔들림은 반복 실행으로 재야 한다. 1회 측정으로 개선을 판단하지 않는다.
"""

import asyncio
import json
import types as pytypes

from app.clients.gemini import GeminiClient
from app.config import Settings


class _FakeResponse:
    text = json.dumps({"ok": True})
    usage_metadata = pytypes.SimpleNamespace(prompt_token_count=11, candidates_token_count=7)


def _capture_config(settings: Settings) -> dict:
    """generate_json 이 SDK 에 실제로 넘기는 config 를 잡아낸다.

    _ensure_client 를 갈아끼운다 — 실물 SDK 클라이언트를 만들면 API 키를 요구하고,
    그러면 이 테스트가 자격증명 유무에 따라 켜졌다 꺼졌다 한다.
    """
    seen: dict = {}

    async def fake_generate_content(*, model, contents, config):
        seen["model"] = model
        seen["config"] = config
        return _FakeResponse()

    fake_client = pytypes.SimpleNamespace(
        aio=pytypes.SimpleNamespace(models=pytypes.SimpleNamespace(generate_content=fake_generate_content))
    )

    client = GeminiClient(settings)
    client._ensure_client = lambda: fake_client  # type: ignore[method-assign]

    asyncio.run(client.generate_json(prompt="p", response_schema={"type": "OBJECT"}))
    return seen


def test_temperature와_seed를_함께_넘긴다():
    settings = Settings(internal_token="t", gemini_seed=20260814)

    config = _capture_config(settings)["config"]

    # 온도만 0 이어도 1·2위 확률이 근접한 자리에서 갈린다. 둘이 짝이다.
    assert config.temperature == 0
    assert config.seed == 20260814


def test_seed는_설정값을_그대로_쓴다():
    # 값 자체에 의미가 없다 — 바뀌지 않는 것이 요구사항이고, 그래서 설정에서 온다.
    # 상수를 클라이언트에 박아두면 측정 때 시드를 바꿔 잡음 바닥을 재는 것이 불가능해진다.
    config = _capture_config(Settings(internal_token="t", gemini_seed=99))["config"]

    assert config.seed == 99


def test_구조화_출력_강제는_그대로다():
    # seed 를 더하면서 응답 스키마 강제가 빠지지 않았는지 함께 본다 — 그게 빠지면
    # 뒤 계층 전체가 "대체로 JSON" 위에 서게 된다(모듈 주석).
    config = _capture_config(Settings(internal_token="t"))["config"]

    assert config.response_mime_type == "application/json"
    assert config.response_schema is not None
