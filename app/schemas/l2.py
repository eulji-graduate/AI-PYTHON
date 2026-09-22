"""AI-03 · L2 — 주제 분할 DTO.

회의 발화를 주제 단위로 나눈다. 뒤 계층(L3)이 주제별로 N회 돌기 때문에, 여기서
경계를 잘못 그으면 하나의 결정이 두 주제로 쪼개지거나(중복 액션) 서로 다른 안건이
한 주제에 뭉쳐(요약이 뭉개짐) 뒤 계층이 전부 그 위에서 돈다.

**모델은 경계의 '시작'만 정한다.** 끝은 다음 주제의 시작 직전으로 코드가 계산한다 —
시작·끝을 둘 다 모델이 정하면 겹침과 구멍이 생기고, 그걸 후처리로 메우는 규칙이
또 판단을 하게 된다. 시작만 받으면 겹침도 구멍도 구조적으로 불가능하다.
"""

from pydantic import Field

from app.schemas.common import CamelModel, FewShotExample, Usage, Utterance

# 인접 주제 경계에서 겹쳐 넘기는 발화 수(명세 「AI-03 · 오버랩 3발화」).
# 경계 발화가 어느 주제에 속하는지는 원래 애매하다. 겹쳐 두면 L3 가 양쪽에서 보게 되고,
# 중복은 뒤에서 병합할 수 있지만 누락은 되살릴 수 없다.
OVERLAP_UTTERANCES = 3

TOPIC_MAX = 300


class SegmentTopicsRequest(CamelModel):
    tenant_id: int
    meeting_id: int
    utterances: list[Utterance]
    query_text: str | None = None


class TopicSegment(CamelModel):
    topic_seq: int
    topic: str

    # 이 주제의 고유 구간. 경계는 겹치지 않는다.
    start_utterance_id: int
    end_utterance_id: int

    # L3 에 넘길 발화 = 고유 구간 + 앞 주제 끝 오버랩. 이걸 따로 두는 이유는
    # "이 주제의 범위"와 "이 주제를 읽을 때 볼 발화"가 다르기 때문이다 —
    # 같은 값으로 두면 오버랩이 다음 주제의 구간으로 기록돼 결정이 두 번 저장된다.
    utterance_ids: list[int] = Field(default_factory=list)


class SegmentTopicsResponse(CamelModel):
    topics: list[TopicSegment] = Field(default_factory=list)
    used_few_shot: list[FewShotExample] = Field(default_factory=list)
    usage: Usage = Field(default_factory=Usage)
    model: str
    prompt_version: str
