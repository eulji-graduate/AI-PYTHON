"""제공자 동시 호출 상한.

Spring 이 주제 단위 계층(L3·L3.5·L4)을 **주제별로 동시에** 부르기 시작하면서 필요해졌다.
회의 하나에 주제가 7개면 그만큼이 한꺼번에 나가고 분석이 두 건 겹치면 그 두 배다 —
제공자 레이트리밋을 우리가 스스로 당기는 모양이 된다.

여기서 지키는 것은 둘이다. **상한이 실제로 걸리는가**, 그리고 **백오프 대기가 남의 자리를
막지 않는가**. 후자를 놓치면 한 호출이 30초를 쉬는 동안 그 자리가 비어 있는데도 다른 호출이
줄을 선다 — 제공자는 놀고 우리만 느려진다.
"""

import asyncio

import pytest

from app.clients.gemini import GeminiClient
from app.config import Settings


class _FakeModels:
    """동시 실행 수를 세는 가짜 제공자. 호출을 붙잡아 겹침을 만든다."""

    def __init__(self, hold: asyncio.Event) -> None:
        self._hold = hold
        self.in_flight = 0
        self.peak = 0
        self.started = asyncio.Event()

    async def generate_content(self, *, model, contents, config):
        self.in_flight += 1
        self.peak = max(self.peak, self.in_flight)
        self.started.set()
        try:
            await self._hold.wait()
            return _FakeResponse()
        finally:
            self.in_flight -= 1


class _FakeResponse:
    text = '{"ok": true}'
    usage_metadata = None


def _client(limit: int, models) -> GeminiClient:
    client = GeminiClient(Settings(internal_token="t", gemini_max_concurrency=limit))
    fake = type("Fake", (), {"aio": type("Aio", (), {"models": models})()})()
    client._ensure_client = lambda: fake  # type: ignore[method-assign]
    return client


@pytest.mark.asyncio
async def test_상한을_넘겨_동시에_부르지_않는다():
    hold = asyncio.Event()
    models = _FakeModels(hold)
    client = _client(2, models)

    calls = [
        asyncio.create_task(client.generate_json(prompt="p", response_schema={"type": "OBJECT"}))
        for _ in range(5)
    ]
    await models.started.wait()
    # 붙잡힌 동안 나머지가 밀려 들어올 틈을 준다.
    await asyncio.sleep(0.05)

    peak_while_held = models.peak
    hold.set()
    await asyncio.gather(*calls)

    # 5개를 한꺼번에 던졌지만 제공자에게는 2개까지만 나갔다.
    assert peak_while_held == 2
    assert models.peak == 2


@pytest.mark.asyncio
async def test_상한이_크면_전부_동시에_나간다():
    """상한이 병목이 아닐 때는 붙잡지 않는다 — 위 테스트가 우연히 통과한 것이 아님을 본다."""
    hold = asyncio.Event()
    models = _FakeModels(hold)
    client = _client(5, models)

    calls = [
        asyncio.create_task(client.generate_json(prompt="p", response_schema={"type": "OBJECT"}))
        for _ in range(5)
    ]
    await models.started.wait()
    await asyncio.sleep(0.05)

    peak_while_held = models.peak
    hold.set()
    await asyncio.gather(*calls)

    assert peak_while_held == 5


@pytest.mark.asyncio
async def test_백오프로_쉬는_동안에는_자리를_비운다():
    """⚠ 상한이 재시도 대기까지 감싸면 기다리는 동안 남의 자리를 막는다.

    429 는 Retry-After 를 존중해 최대 30초를 쉰다. 그 시간을 잡고 있으면 제공자는 노는데
    우리만 줄을 서고, 상한이 처리량을 깎는 장치가 된다.
    """
    settings = Settings(
        internal_token="t",
        gemini_max_concurrency=1,
        retry_delays_sec=(0.05,),
    )
    client = GeminiClient(settings)

    attempts = {"n": 0}
    other_ran = asyncio.Event()

    class _FlakyModels:
        async def generate_content(self, *, model, contents, config):
            attempts["n"] += 1
            if attempts["n"] == 1:
                # 첫 시도는 일시적 실패 — 호출자가 백오프로 쉰다.
                raise RuntimeError("503 Service Unavailable")
            return _FakeResponse()

    fake = type("Fake", (), {"aio": type("Aio", (), {"models": _FlakyModels()})()})()
    client._ensure_client = lambda: fake  # type: ignore[method-assign]

    async def other() -> None:
        # 상한이 1 이므로, 백오프가 자리를 잡고 있으면 이 호출은 그 시간만큼 못 들어간다.
        await client.generate_json(prompt="other", response_schema={"type": "OBJECT"})
        other_ran.set()

    first = asyncio.create_task(client.generate_json(prompt="p", response_schema={"type": "OBJECT"}))
    second = asyncio.create_task(other())

    # 백오프(0.05초)보다 짧게 기다려도 두 번째가 들어갈 수 있어야 한다.
    await asyncio.wait_for(other_ran.wait(), timeout=1.0)

    await asyncio.gather(first, second)
