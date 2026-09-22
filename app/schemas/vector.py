"""AI-08 벡터 저장 · AI-09 유사 발화 조회의 요청·응답.

<h2>임베딩 대상은 근거 발화 원문이다 — 확정 tuple 이 아니다</h2>
검색 시점에 손에 있는 것은 tuple 이 아니라 **새 발화**다. tuple 을 임베딩하면 쿼리와 키가
다른 공간에 놓여 유사도가 망가진다(V5.10 주석). 그래서 `inputText` 가 벡터가 되고 확정
tuple 은 `payload` 로 따라붙는다.

    저장  근거 발화 → 벡터(key) + payload = 확정 tuple
    검색  새 발화   → 벡터(query) → 가장 가까운 key → payload 를 few-shot 으로
"""

from pydantic import Field

from app.schemas.common import CamelModel, FewShotExample

PROVENANCE_HUMAN_VERIFIED = "HUMAN_VERIFIED"
PROVENANCE_AUTO = "AUTO"


class VectorUpsertItem(CamelModel):
    """저장할 예시 하나.

    `vectorId` 는 Spring 의 원본 행 id(meeting_tuple_vector.id)다. **포인트 id 를 여기서
    유도하므로 필수다** — 없으면 무작위 id 를 쓰게 되고, 재시도 워커가 돌 때마다 같은 예시가
    복제되어 검색 상위를 자기 복제본으로 채운다.
    """

    vector_id: int
    company_id: int
    layer: str
    input_text: str
    payload: dict
    dept_id: int | None = None
    provenance: str = PROVENANCE_HUMAN_VERIFIED


class VectorUpsertRequest(CamelModel):
    """배치로 받는다. 재시도 워커가 밀린 행을 한 번에 넘기는 것이 정상 경로다."""

    items: list[VectorUpsertItem] = Field(default_factory=list)


class VectorUpsertResult(CamelModel):
    """행마다 결과를 돌려준다. **Spring 이 이 값으로 vector_synced 와 qdrant_point_id 를
    적는다** — 배치 전체를 하나의 성공/실패로 답하면, 일부만 들어간 배치에서 어느 행을
    다시 보내야 하는지 알 수 없어 전부 재시도하게 된다."""

    vector_id: int
    point_id: str


class VectorUpsertResponse(CamelModel):
    upserted: list[VectorUpsertResult] = Field(default_factory=list)
    model: str


class SimilarRequest(CamelModel):
    """세 필터(companyId · layer · provenance)가 전부 필수다.

    각각 빠졌을 때의 결과가 다르다 — companyId 는 **타사 발화 유출**, layer 는 다른 계층
    예시 오염, provenance 는 모델이 자기 출력을 다시 학습하는 루프다. 기본값을 주지 않는
    이유가 그것이다(provenance 만 사람 검증본으로 고정한다).
    """

    company_id: int
    layer: str
    query_text: str
    dept_id: int | None = None
    top_k: int = 5
    provenance: str = PROVENANCE_HUMAN_VERIFIED


class SimilarResponse(CamelModel):
    examples: list[FewShotExample] = Field(default_factory=list)
    model: str
