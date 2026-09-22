"""Qdrant — few-shot 예시의 인덱스.

**MySQL 이 원본이고 여기는 인덱스다**(V5.10 주석). Spring 이 라벨을 먼저 커밋한 뒤 AI-08 로
넘기므로, 이 저장소가 잠깐 비어도 잃는 것은 검색 품질뿐이고 라벨 자체는 안전하다. 반대로
여기를 원본처럼 쓰면 벡터는 검색에 걸리는데 꺼낼 내용이 없는 상태가 생긴다.

<h2>포인트 id 를 만들어 쓰지 않고 받아서 유도한다</h2>
Qdrant 의 포인트 id 는 부호 없는 정수나 UUID 만 받는다. 그래서 Spring 의 원본 행 id
(meeting_tuple_vector.id)로부터 **결정적으로** UUID 를 만든다 — 같은 행을 두 번 보내면 같은
포인트에 덮어써진다. 무작위 id 를 쓰면 재시도 워커가 돌 때마다 같은 예시가 복제되고,
그 복제본들이 검색 상위를 독차지해 few-shot 이 같은 문장만 다섯 개 보게 된다.

<h2>세 필터는 선택이 아니다</h2>
company_id · layer · provenance 는 검색에서 항상 걸린다. 각각 빠졌을 때의 결과가 다르다 —
company_id 는 타사 발화 유출, layer 는 다른 계층 예시 오염, provenance 는 모델이 자기 출력을
다시 학습하는 루프다(V5.10 주석). 그래서 조회 함수의 인자에서 기본값을 주지 않는다.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass

# 포인트 id 를 원본 행 id 에서 유도할 때 쓰는 이름공간. 값 자체에 의미는 없고,
# **바뀌면 안 된다** — 바뀌는 순간 같은 행이 다른 포인트가 되어 전부 중복된다.
POINT_NAMESPACE = uuid.UUID("6f1d2c5a-8b34-4f7e-9c21-0a5e3d8b7c46")

FIELD_COMPANY_ID = "companyId"
FIELD_LAYER = "layer"
FIELD_PROVENANCE = "provenance"
FIELD_DEPT_ID = "deptId"
FIELD_INPUT_TEXT = "inputText"
FIELD_PAYLOAD = "payload"
FIELD_VECTOR_ID = "vectorId"


def point_id_of(vector_id: int) -> str:
    """원본 행 id → 포인트 id. 결정적이라 재시도가 중복을 만들지 않는다."""
    return str(uuid.uuid5(POINT_NAMESPACE, f"meeting_tuple_vector:{vector_id}"))


@dataclass(frozen=True)
class SearchHit:
    input_text: str
    payload: dict
    score: float


class VectorStore:
    """Qdrant 접근을 한 곳에 모은다. 계층 코드가 클라이언트를 직접 만들면 컬렉션 이름과
    차원이 여러 군데로 흩어지고, 한 곳만 고쳐졌을 때 조용히 다른 컬렉션을 보게 된다."""

    def __init__(self, *, url: str, collection: str, dim: int, api_key: str | None = None) -> None:
        self._url = url
        self._collection = collection
        self._dim = dim
        self._api_key = api_key or None
        self._client = None
        self._ensured = False

    def _ensure_client(self):
        if self._client is None:
            from qdrant_client import AsyncQdrantClient

            # ":memory:" 는 테스트용 로컬 모드다. 같은 코드 경로로 돌려야 테스트가
            # 실제 동작을 검증한다 — 가짜 저장소를 따로 두면 우리가 맞다고 믿는 대로 돈다.
            if self._url == ":memory:":
                self._client = AsyncQdrantClient(":memory:")
            else:
                self._client = AsyncQdrantClient(url=self._url, api_key=self._api_key)
        return self._client

    async def ensure_collection(self) -> None:
        """없으면 만든다. **차원은 여기서 박힌다** — 임베딩 차원을 바꾸면 이 컬렉션은
        더 이상 맞지 않으므로 새 이름으로 만들고 재색인해야 한다."""
        if self._ensured:
            return

        from qdrant_client import models

        client = self._ensure_client()
        if not await client.collection_exists(self._collection):
            await client.create_collection(
                collection_name=self._collection,
                vectors_config=models.VectorParams(
                    size=self._dim,
                    # 코사인. 임베딩이 정규화돼 나오므로 내적과 사실상 같지만, 정규화가
                    # 깨진 벡터가 섞여도 길이에 휘둘리지 않는 쪽을 고른다.
                    distance=models.Distance.COSINE,
                ),
            )
            # 필터 대상에 인덱스를 건다. 없으면 컬렉션이 커질수록 필터가 전수 검사가 된다.
            #
            # 로컬 모드(":memory:")는 페이로드 인덱스를 지원하지 않는다 — 걸어도 효과가 없고
            # 경고만 남는다. **필터 동작 자체는 인덱스 없이도 같으므로** 테스트가 검증하려는
            # 것(무엇이 걸러지는가)은 그대로 확인된다. 여기서 건너뛰는 것은 성능 최적화뿐이다.
            for field in () if self._url == ":memory:" else (FIELD_COMPANY_ID, FIELD_DEPT_ID):
                await client.create_payload_index(
                    collection_name=self._collection,
                    field_name=field,
                    field_schema=models.PayloadSchemaType.INTEGER,
                )
            for field in () if self._url == ":memory:" else (FIELD_LAYER, FIELD_PROVENANCE):
                await client.create_payload_index(
                    collection_name=self._collection,
                    field_name=field,
                    field_schema=models.PayloadSchemaType.KEYWORD,
                )
        self._ensured = True

    async def upsert(
        self,
        *,
        vector_id: int,
        vector: list[float],
        company_id: int,
        layer: str,
        provenance: str,
        input_text: str,
        payload: dict,
        dept_id: int | None,
    ) -> str:
        from qdrant_client import models

        await self.ensure_collection()
        client = self._ensure_client()
        point_id = point_id_of(vector_id)

        await client.upsert(
            collection_name=self._collection,
            points=[
                models.PointStruct(
                    id=point_id,
                    vector=vector,
                    payload={
                        FIELD_VECTOR_ID: vector_id,
                        FIELD_COMPANY_ID: company_id,
                        FIELD_LAYER: layer,
                        FIELD_PROVENANCE: provenance,
                        FIELD_DEPT_ID: dept_id,
                        # 원문을 함께 둔다. 검색 결과에 예시 텍스트가 필요한데, 이걸 빼면
                        # 조회할 때마다 MySQL 을 한 번 더 왕복해야 한다.
                        FIELD_INPUT_TEXT: input_text,
                        FIELD_PAYLOAD: payload,
                    },
                )
            ],
        )
        return point_id

    async def search(
        self,
        *,
        vector: list[float],
        company_id: int,
        layer: str,
        provenance: str,
        dept_id: int | None,
        top_k: int,
    ) -> list[SearchHit]:
        from qdrant_client import models

        await self.ensure_collection()
        client = self._ensure_client()

        must = [
            models.FieldCondition(key=FIELD_COMPANY_ID, match=models.MatchValue(value=company_id)),
            models.FieldCondition(key=FIELD_LAYER, match=models.MatchValue(value=layer)),
            models.FieldCondition(key=FIELD_PROVENANCE, match=models.MatchValue(value=provenance)),
        ]
        if dept_id is not None:
            # 부서를 준 경우에만 좁힌다. 안 주면 회사 전체가 대상이다 — 팀이 작을 때
            # 강제로 좁히면 예시가 0건이 되어 few-shot 이 아예 안 붙는다.
            must.append(models.FieldCondition(key=FIELD_DEPT_ID, match=models.MatchValue(value=dept_id)))

        found = await client.query_points(
            collection_name=self._collection,
            query=vector,
            query_filter=models.Filter(must=must),
            limit=top_k,
            with_payload=True,
        )

        hits: list[SearchHit] = []
        for point in found.points:
            data = point.payload or {}
            text = data.get(FIELD_INPUT_TEXT)
            payload = data.get(FIELD_PAYLOAD)
            if not text or payload is None:
                # 원문이나 payload 가 없는 포인트는 예시로 쓸 수 없다. 형식만 맞는 빈
                # 예시를 프롬프트에 넣으면 모델이 그 빈 모양을 따라 한다.
                continue
            hits.append(SearchHit(input_text=text, payload=payload, score=point.score))
        return hits
