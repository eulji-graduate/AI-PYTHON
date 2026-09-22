"""AI-06 · L4 — assignment tuple 추출 DTO.

L4 를 첫 계층으로 짜는 이유는 여기서 잡은 틀(요청 DTO → 프롬프트 → 응답 스키마 →
후처리)을 L1.5 · L2 · L3 · L3.5 · L5 가 그대로 복제하기 때문이다. 계층마다 다른 건
프롬프트 파일과 응답 모델뿐이다.
"""

from datetime import date
from typing import Literal

from pydantic import Field

from app.schemas.common import (
    UNKNOWN,
    UNKNOWN_PERSON,
    CamelModel,
    FewShotExample,
    Participant,
    Usage,
    Utterance,
)

# assigneeSource 를 nullable 로 두는 대신 enum 안에 넣는다 — 구조화 출력에서
# "nullable + enum" 조합은 제공자마다 처리가 갈려서, 값으로 받고 후처리에서 None 으로 바꾼다.
SOURCE_UNKNOWN = UNKNOWN

__all__ = [
    "SOURCE_UNKNOWN",
    "UNKNOWN_PERSON",  # 재수출 — 기존 임포트 경로(app.schemas.l4)를 깨지 않는다
    "AssignmentTuple",
    "ExtractTuplesRequest",
    "ExtractTuplesResponse",
    "TopicItem",
]


class TopicItem(CamelModel):
    """L3.5 게이트를 통과한 정리 항목.

    `gate_status` 를 필수 리터럴로 둔다. 주석으로 "CONFIRMED 만 보낸다"고 적어두면
    지켜지지 않을 여지가 남고, DISCUSSED 항목이 섞여 들어오면 아직 결정되지도 않은
    논의가 배정으로 확정된다 — L3.5 게이트를 지나온 의미가 사라진다.
    필드로 요구하면 잘못 보낸 호출은 422 로 거절된다.
    """

    item_type: Literal["DECISION", "DISCUSSION", "BLOCKER"]
    gate_status: Literal["CONFIRMED"]
    content: str

    # 이 항목의 근거 발화. 채워서 보내면 tuple 의 근거도 이 안에서만 고를 수 있게 좁힌다
    # (build_response_schema 참조). 비워 보내면 주제의 발화 전체가 근거 후보가 된다.
    evidence_utterance_ids: list[int] = Field(default_factory=list)


class ExtractTuplesRequest(CamelModel):
    """명세의 요청에 `utterances` 를 더했다.

    명세 예시에는 `items` 만 있는데, 그것만으로는 `evidenceUtteranceId` 를 닫힌
    목록으로 강제할 수 없다. 후보 발화 목록을 함께 받아 그 id 들만 enum 으로 넣으면
    없는 발화를 근거로 붙이는 경로가 구조적으로 사라진다.
    """

    tenant_id: int
    meeting_id: int
    topic: str
    items: list[TopicItem]
    utterances: list[Utterance]
    participants: list[Participant]
    query_text: str | None = None

    # 회의 날짜(YYYY-MM-DD). "다음 주까지" 같은 상대 표현을 절대 날짜로 바꾸는 기준점이다.
    # 없으면 상대 표현은 계산하지 않고 dueDate 를 null 로 둔다 — 기준일을 모르는 채
    # 계산하면 그럴듯하게 틀린 마감이 보드에 꽂힌다.
    meeting_date: date | None = None

    # EXTRACT        주제 전체를 보고 뽑기 (기본)
    # EXTRACT_NARROW 근거 발화 앞뒤 3발화만 보고 다시 뽑기 — L5 관점 다변화의 한쪽
    view: Literal["EXTRACT", "EXTRACT_NARROW"] = "EXTRACT"


class AssignmentTuple(CamelModel):
    title: str
    assignee_candidate_person_id: int | None = None
    assignee_source: Literal["EXPLICIT_CALL", "FIRST_PERSON"] | None = None
    due_date: str | None = None
    evidence_utterance_id: int


class ExtractTuplesResponse(CamelModel):
    tuples: list[AssignmentTuple] = Field(default_factory=list)
    used_few_shot: list[FewShotExample] = Field(default_factory=list)
    usage: Usage = Field(default_factory=Usage)
    model: str
    prompt_version: str
