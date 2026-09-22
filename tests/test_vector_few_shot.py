"""AI-08 벡터 저장 · AI-09 유사 발화 조회.

검증의 축이 둘이다.

1. **세 필터가 실제로 좁히는가** — company_id 가 빠지면 타사 발화가 프롬프트에 들어간다.
   정확도 문제가 아니라 유출이라, 「다른 회사 것이 안 나온다」가 이 파일의 중심 테스트다.
2. **재시도가 중복을 만들지 않는가** — 포인트 id 를 원본 행 id 에서 결정적으로 유도하는
   이유가 그것이다. 무작위 id 면 재시도 워커가 돌 때마다 같은 예시가 복제되고, 그 복제본이
   검색 상위를 채워 few-shot 이 같은 문장만 다섯 개 보게 된다.

⚠ Qdrant 로컬 모드(":memory:")로 **실물 클라이언트**를 돌린다. 가짜 저장소를 두면 필터가
실제로 걸리는지를 검증하지 못한다 — 가짜는 우리가 맞다고 믿는 대로 동작한다.

임베딩은 DRY_RUN 스텁이다. 값에 의미가 없으므로 **순위를 주장하지 않는다** — 무엇이 걸러지고
무엇이 남는지만 본다. 유사도 품질은 이 테스트의 축이 아니다.
"""

import pytest
from fastapi.testclient import TestClient

from app.config import get_settings
from app.layers import few_shot
from app.layers.few_shot import _STORES
from app.main import app

TOKEN = "test-internal-token"
COMPANY = 7
OTHER_COMPANY = 8


@pytest.fixture(autouse=True)
def _settings(monkeypatch):
    monkeypatch.setenv("INTERNAL_TOKEN", TOKEN)
    monkeypatch.setenv("GEMINI_API_KEY", "")
    monkeypatch.setenv("DRY_RUN", "1")
    # 로컬 모드. 같은 프로세스 안의 임시 DB 라 테스트끼리 섞이지 않게 매번 비운다.
    monkeypatch.setenv("QDRANT_URL", ":memory:")
    get_settings.cache_clear()
    _STORES.clear()
    yield
    get_settings.cache_clear()
    _STORES.clear()


@pytest.fixture
def client():
    return TestClient(app)


def _auth():
    return {"X-Internal-Token": TOKEN}


def _item(vector_id: int, text: str, *, company=COMPANY, layer="L4", provenance="HUMAN_VERIFIED"):
    return {
        "vectorId": vector_id,
        "companyId": company,
        "layer": layer,
        "inputText": text,
        "payload": {"title": f"할 일 {vector_id}", "assigneeMemberId": 42},
        "provenance": provenance,
    }


def _upsert(client, *items):
    return client.post("/internal/vector/upsert", headers=_auth(), json={"items": list(items)})


def _similar(client, **overrides):
    body = {"companyId": COMPANY, "layer": "L4", "queryText": "서준님이 정리해주세요."}
    body.update(overrides)
    return client.post("/internal/similar", headers=_auth(), json=body)


def test_저장하면_조회된다(client):
    _upsert(client, _item(1, "서준님이 정리해주세요."))

    response = _similar(client)

    assert response.status_code == 200
    examples = response.json()["examples"]
    assert len(examples) == 1
    assert examples[0]["inputText"] == "서준님이 정리해주세요."
    # payload 는 확정 tuple 이다. 그대로 실려 나와야 프롬프트에 예시로 붙는다.
    assert examples[0]["payload"]["assigneeMemberId"] == 42


def test_다른_회사_예시는_나오지_않는다(client):
    """이 파일에서 가장 중요한 테스트다 — 걸러지지 않으면 타사 회의 발화가 프롬프트에 주입된다."""
    _upsert(client, _item(1, "타사 발화입니다.", company=OTHER_COMPANY))

    response = _similar(client)

    assert response.json()["examples"] == []


def test_다른_계층_예시는_나오지_않는다(client):
    # L1.5 예시는 지시어-선행사 쌍이라 모양이 다르다. L4 에 섞이면 정확도가 떨어진다.
    _upsert(client, _item(1, "그분이요.", layer="L1.5"))

    response = _similar(client, layer="L4")

    assert response.json()["examples"] == []


def test_AUTO_예시는_나오지_않는다(client):
    # 모델이 자기 출력을 예시로 다시 학습하는 루프를 막는다.
    _upsert(client, _item(1, "AI 가 만든 것입니다.", provenance="AUTO"))

    response = _similar(client)

    assert response.json()["examples"] == []


def test_같은_행을_다시_보내도_중복되지_않는다(client):
    """재시도 워커가 같은 행을 다시 넘기는 것이 정상 경로다(vector_synced=false)."""
    first = _upsert(client, _item(1, "서준님이 정리해주세요."))
    second = _upsert(client, _item(1, "서준님이 정리해주세요."))

    # 포인트 id 가 원본 행 id 에서 유도되므로 같아야 한다 — 다르면 복제본이 쌓인다.
    assert first.json()["upserted"][0]["pointId"] == second.json()["upserted"][0]["pointId"]
    assert len(_similar(client).json()["examples"]) == 1


def test_행마다_결과를_돌려준다(client):
    """배치 전체를 하나의 성공/실패로 답하면 어느 행을 다시 보낼지 알 수 없다."""
    response = _upsert(client, _item(1, "첫째"), _item(2, "둘째"))

    upserted = response.json()["upserted"]
    assert [u["vectorId"] for u in upserted] == [1, 2]
    assert all(u["pointId"] for u in upserted)


def test_빈_배치는_그냥_통과한다(client):
    response = _upsert(client)

    assert response.status_code == 200
    assert response.json()["upserted"] == []


def test_health가_구현_목록에_AI_08_09를_포함한다(client):
    """목록과 라우팅이 어긋나면 워커가 미구현 계층을 부르거나, 구현된 걸 안 부른다."""
    response = client.get("/internal/health", headers=_auth())

    implemented = response.json()["implemented"]
    assert "AI-08" in implemented
    assert "AI-09" in implemented


@pytest.mark.asyncio
async def test_조회가_실패해도_계층을_세우지_않는다(monkeypatch):
    """few-shot 은 정확도를 올리는 재료이지 계층의 입력이 아니다.

    Qdrant 가 내려갔다고 여섯 계층이 전부 실패하면 파이프라인이 인덱스 하나에 인질로 잡힌다.
    """

    async def boom(*args, **kwargs):
        raise RuntimeError("qdrant down")

    monkeypatch.setattr(few_shot.EmbeddingClient, "embed", boom)

    examples = await few_shot.lookup(tenant_id=COMPANY, layer="L4", query_text="아무 발화")

    assert examples == []


@pytest.mark.asyncio
async def test_질의문이_비면_조회하지_않는다():
    # 빈 질의로 검색하면 아무 예시나 상위로 올라온다 — 관련 없는 예시를 붙이는 것은
    # 예시를 안 붙이는 것보다 나쁘다.
    assert await few_shot.lookup(tenant_id=COMPANY, layer="L4", query_text=None) == []
    assert await few_shot.lookup(tenant_id=COMPANY, layer="L4", query_text="") == []
