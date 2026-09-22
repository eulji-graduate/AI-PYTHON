"""AI-05 · L3.5 — 확정/논의 게이트 DTO.

L3 가 정리한 항목 각각에 대해 "이게 정말 **확정**인가"만 판정한다.
`CONFIRMED` 만 L4(tuple 추출)로 넘어가므로, 여기서 잘못 올린 항목은
아직 합의도 안 된 일이 담당자에게 배정되는 결과가 된다.

**precision 우선이다 — 애매하면 버린다**(명세). 여기서 '버린다'는 항목을 지우는 것이
아니라 `DISCUSSED` 로 남기는 것이다. "이 안건은 담당자가 안 정해졌다"고 짚어주는 것
자체가 결과물이고, 나중에 게이트를 조일지 풀지 판단할 근거가 된다(V5.8 주석).
"""

from typing import Literal

from pydantic import Field

from app.schemas.common import CamelModel, FewShotExample, Participant, Usage, Utterance

GateStatus = Literal["CONFIRMED", "DISCUSSED"]
ItemType = Literal["DECISION", "DISCUSSION", "BLOCKER"]

REASON_MAX = 1000


class GateCandidate(CamelModel):
    """판정 대상 항목 하나. L3 산출에 Spring 이 키를 붙여 보낸다."""

    # 응답을 요청 항목에 되짚는 키. Spring 이 meeting_decision.id 를 문자열로 넣어도 되고
    # 저장 전이면 임시 순번을 넣어도 된다 — 이쪽은 되돌려주기만 한다.
    item_key: str

    item_type: ItemType
    content: str
    evidence_utterance_id: int


class GateRequest(CamelModel):
    tenant_id: int
    meeting_id: int
    topic: str
    items: list[GateCandidate]

    # 판정 근거를 다시 읽기 위한 발화. 항목의 content 만으로는 "정말 합의됐는지"를
    # 볼 수 없다 — 합의는 발화의 어투와 뒤따르는 반응에 있다.
    utterances: list[Utterance]
    participants: list[Participant]
    query_text: str | None = None


class GateVerdict(CamelModel):
    item_key: str
    gate_status: GateStatus
    reason: str


class GateResponse(CamelModel):
    verdicts: list[GateVerdict] = Field(default_factory=list)
    used_few_shot: list[FewShotExample] = Field(default_factory=list)
    usage: Usage = Field(default_factory=Usage)
    model: str
    prompt_version: str
