"""AI-04 · L3 — 주제별 정리 DTO. 주제마다 한 번씩 호출된다(명세 「주제별 N회」).

산출은 BACKEND 의 `meeting_decision`(V5.8) 한 행에 그대로 대응한다 —
`item_type` · `content` · `reason` · `evidence_transcript_id`.

**`reason` 을 필수로 받는다.** L3 가 왜 이 항목을 결정으로 분류했는지 남기지 않으면,
나중에 오분류를 발견해도 프롬프트의 어느 부분이 문제였는지 되짚을 수 없다.
`gate_status` 는 여기서 정하지 않는다 — 그건 L3.5 의 일이다(한 계층 한 목표).
"""

from typing import Literal

from pydantic import Field

from app.schemas.common import CamelModel, FewShotExample, Participant, Usage, Utterance

ItemType = Literal["DECISION", "DISCUSSION", "BLOCKER"]

CONTENT_MAX = 1000
REASON_MAX = 1000
SUMMARY_MAX = 2000


class SummarizeTopicRequest(CamelModel):
    tenant_id: int
    meeting_id: int

    # L2 가 매긴 순번·주제명을 그대로 되돌려 받는다. Spring 이 meeting_decision.topic_seq 로
    # 저장하므로 여기서 새로 매기면 L2 의 분할과 어긋난다.
    topic_seq: int
    topic: str

    # 이 주제의 발화(L2 의 utteranceIds — 오버랩 포함).
    utterances: list[Utterance]
    participants: list[Participant]
    query_text: str | None = None


class TopicItemDraft(CamelModel):
    """게이트를 통과하기 전의 항목. `gateStatus` 가 없는 것이 L4 의 `TopicItem` 과 다른 점이다."""

    item_type: ItemType
    content: str
    reason: str
    evidence_utterance_id: int


class SummarizeTopicResponse(CamelModel):
    topic_seq: int
    topic: str

    # 주제 한 문단 요약. 회의 전체 요약(meeting_summary.overview)은 Spring 이 조립한다 —
    # 이 계층은 주제 하나만 본다.
    summary: str = ""

    items: list[TopicItemDraft] = Field(default_factory=list)
    used_few_shot: list[FewShotExample] = Field(default_factory=list)
    usage: Usage = Field(default_factory=Usage)
    model: str
    prompt_version: str
