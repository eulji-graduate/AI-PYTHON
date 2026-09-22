"""AI-02 · L1.5 — 지시어 해소 DTO.

"그거 제가 할게요"의 **그거**가 무엇인지, "그분한테 넘기죠"의 **그분**이 누구인지를
앞 발화에 잇는다. L4 가 담당자를 정하기 전에 여기서 대명사가 풀려야 하고,
풀리지 않으면 L4 는 담당자를 추측하게 된다 — 검토 사유 `WRONG_ASSIGNEE` 가
L1.5 로 귀속되는 이유다(명세 「검토 사유 코드」).

**확신도 자기보고를 받지 않는다.** 모델이 스스로 말한 confidence 는 85~95 에 몰리고
실제 정확도와 맞지 않는다. 대신 해소되지 않으면 `UNRESOLVED` 로 기권한다.
"""

from typing import Literal

from pydantic import Field

from app.schemas.common import CamelModel, FewShotExample, Participant, Usage, Utterance

# 지시어가 가리키는 대상의 종류. 종류를 돌려주는 이유는 소비자가 다르기 때문이다 —
# PERSON 은 L4 의 담당자 판정에, TOPIC·ARTIFACT 는 L3 의 항목 본문에 쓰인다.
ReferenceType = Literal["PERSON", "TOPIC", "ARTIFACT", "TIME", "UNRESOLVED"]


class ResolveReferenceRequest(CamelModel):
    tenant_id: int
    meeting_id: int

    # 문맥 전체. 지시어는 선행사가 앞 발화에 있으므로 대상 발화만 보내면 풀 수 없다.
    utterances: list[Utterance]

    # 해소 대상 발화. 비워 보내면 전체 발화를 대상으로 본다 — Spring 이 아직
    # 지시어 후보를 추리지 않는 단계에서도 계층이 돌아가야 한다.
    target_utterance_ids: list[int] = Field(default_factory=list)

    participants: list[Participant]
    query_text: str | None = None


class ResolvedReference(CamelModel):
    """지시어 하나의 해소 결과."""

    utterance_id: int

    # 발화 안에 실제로 나타난 지시 표현 원문("그거"·"아까 그건"). 후처리가 발화 텍스트에
    # 이 문자열이 있는지 확인한다 — 없으면 모델이 만들어낸 것이므로 버린다.
    surface: str

    reference_type: ReferenceType

    # PERSON 일 때의 대상. 명단 밖이면 None 이다 — "사람을 가리키지만 누군지 모른다"는
    # 그 자체로 쓸모 있는 답이다(L4 가 담당자로 쓰면 안 된다는 뜻).
    resolved_person_id: int | None = None

    # TOPIC·ARTIFACT 일 때 가리키는 대상의 표현. PERSON 이면 비어 있을 수 있다.
    resolved_text: str | None = None

    # 선행사가 있는 발화. 근거 강제 — 이 값이 없으면 항목 자체를 반환하지 않는다.
    evidence_utterance_id: int


class ResolveReferenceResponse(CamelModel):
    references: list[ResolvedReference] = Field(default_factory=list)
    used_few_shot: list[FewShotExample] = Field(default_factory=list)
    usage: Usage = Field(default_factory=Usage)
    model: str
    prompt_version: str
