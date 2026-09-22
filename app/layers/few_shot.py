"""few-shot 조회 — AI-09 `/internal/similar` 의 내부 사용.

Spring 이 따로 조회하지 않고 여기서 부른다. 인스턴스가 분리돼 있어 벡터·예시를
네트워크로 왕복시킬 이유가 없기 때문이다. 대신 무엇을 썼는지 `usedFewShot` 으로
돌려줘야 review_log.input_context 가 온전해진다 — 그게 없으면 나중에 "이 라벨은
어떤 예시를 보고 나온 판단인지"를 재현할 수 없다.

<h2>조회가 실패해도 계층을 세우지 않는다</h2>
few-shot 은 **정확도를 올리는 재료**이지 계층의 입력이 아니다. Qdrant 가 잠깐 내려갔다고
여섯 계층이 전부 실패하면, 분석 파이프라인이 인덱스 하나에 인질로 잡힌다. 그래서 여기서
잡고 빈 목록으로 계속한다.

⚠ 대신 **로그로 크게 남긴다.** 빈 목록은 "예시가 없다"와 "조회가 실패했다" 둘 다로 읽히고,
응답의 usedFewShot 만 보면 구분되지 않는다. 응답에 필드를 더해 구분하려면 Spring DTO 까지
같이 고쳐야 해서, 지금은 관측을 로그에 둔다.
"""

from __future__ import annotations

import logging

from app.clients.embedding import TASK_QUERY, EmbeddingClient
from app.clients.qdrant import VectorStore
from app.config import Settings, get_settings
from app.schemas.common import FewShotExample

logger = logging.getLogger(__name__)

# 세 필터는 전부 필수다. tenant 가 빠지면 다른 회사 발화가 프롬프트에 들어간다 —
# 정확도 문제가 아니라 유출이다.
PROVENANCE_HUMAN_VERIFIED = "HUMAN_VERIFIED"


#: 접속 설정이 같으면 **같은 저장소를 쓴다.** 요청마다 새로 만들면 세 가지가 함께 나빠진다 —
#: 연결이 요청 수만큼 열리고, 컬렉션 존재 확인이 매번 왕복하고, 로컬 모드(":memory:")에서는
#: 아예 다른 DB 가 되어 방금 넣은 것을 못 찾는다(테스트가 그 경로를 탄다).
_STORES: dict[tuple[str, str, int], VectorStore] = {}


def store_of(settings: Settings) -> VectorStore:
    key = (settings.qdrant_url, settings.qdrant_collection, settings.embed_dim)
    store = _STORES.get(key)
    if store is None:
        store = VectorStore(
            url=settings.qdrant_url,
            collection=settings.qdrant_collection,
            dim=settings.embed_dim,
            api_key=settings.qdrant_api_key,
        )
        _STORES[key] = store
    return store


async def lookup(
    *,
    tenant_id: int,
    layer: str,
    query_text: str | None,
    dept_id: int | None = None,
    top_k: int = 5,
    provenance: str = PROVENANCE_HUMAN_VERIFIED,
    settings: Settings | None = None,
) -> list[FewShotExample]:
    if not query_text:
        return []

    settings = settings or get_settings()
    if not settings.few_shot_enabled:
        return []

    try:
        # 임베딩 대상은 근거 발화 텍스트다. 저장 쪽(AI-08)과 같은 모델·차원을 쓰되
        # task_type 만 질의용으로 바꾼다 — 그래야 찾는 쪽과 찾아지는 쪽이 맞물린다.
        embedder = EmbeddingClient(settings)
        vectors = await embedder.embed([query_text], task_type=TASK_QUERY)
        if not vectors:
            return []

        hits = await store_of(settings).search(
            vector=vectors[0],
            company_id=tenant_id,
            layer=layer,
            provenance=provenance,
            dept_id=dept_id,
            top_k=top_k,
        )
    except Exception:
        # 계층을 세우지 않는다(위 주석). 스택까지 남겨야 원인이 임베딩인지 인덱스인지 갈린다.
        logger.warning(
            "few-shot 조회 실패 — 예시 없이 계층을 계속한다. tenantId=%s layer=%s",
            tenant_id,
            layer,
            exc_info=True,
        )
        return []

    return [FewShotExample(input_text=hit.input_text, payload=hit.payload, score=hit.score) for hit in hits]
