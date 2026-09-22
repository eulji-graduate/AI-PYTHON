"""AI-07 · L5 — 관점 다변화 검증 DTO.

**다수결을 반환하지 않는다.** 같은 모델에 같은 프롬프트를 여러 번 돌리면 오류가 서로
상관되어, 틀릴 때도 똑같이 틀린다. 그 100% 일치가 오답에 최고 신뢰도를 부여한다.

대신 **관점을 바꿔서 묻는다** — `EXTRACT_NARROW`(앞뒤 3발화만 보고 다시 뽑기) vs
`VERIFY`("이 tuple 이 맞나?"). 생성과 검증은 다른 작업이라 오류가 덜 겹친다.
불일치는 다수결로 덮지 않고 `agree=false` 로 넘긴다 — 앙상블의 가치는 정답 선택이
아니라 **불확실성 탐지**다.
"""

from datetime import date
from typing import Literal

from pydantic import Field

from app.schemas.common import CamelModel, Participant, Usage, Utterance
from app.schemas.l4 import AssignmentTuple, TopicItem

View = Literal["EXTRACT_NARROW", "VERIFY"]
Verdict = Literal["ACCEPT", "REJECT"]

REASON_MAX = 500

# 두 관점이 비교하는 필드. dueDate 를 넣는 이유는 틀린 마감이 그대로 보드에 꽂히기 때문이다.
COMPARED_FIELDS = ("title", "assigneeCandidatePersonId", "dueDate")

# 그중 **검토로 보낼지를 정하는** 필드. title 이 빠져 있다.
#
# 왜 title 이 빠지나 — 좁은 시야에서 같은 일을 조금 다르게 부르는 것은 흔하다.
# 실호출에서 바로 나온 예: "제품 로드맵 초안 작성" vs "제품 로드맵 초안 작성 및 공유"
# (담당자·기한·근거 발화가 전부 같았고 VERIFY 관점도 ACCEPT 였다).
# 그런 것을 검토로 보내면 목록이 부풀어 **진짜 오배정이 표현 차이들 사이에 묻힌다.**
# 앙상블의 가치는 불확실성 탐지인데, 탐지된 것이 전부 노이즈면 신호가 사라진다.
#
# 이 판단은 _match 가 title 로 짝을 맞추지 않는 이유와 같다 — 그쪽에서 피한 문제가
# 필드 비교에서 그대로 돌아오고 있었다(BACKEND 실호출 검증에서 발견 · python#11).
#
# **title 을 버리는 것이 아니다.** disagreementFields 에는 그대로 실린다 —
# 제목이 자주 갈리는지는 프롬프트를 조일 근거이고, 그 신호까지 없애면 볼 방법이 없다.
# 다만 사람을 부르지는 않는다. 사람이 고칠 것은 담당자와 기한이지 제목 표현이 아니다.
#
# tuple 의 정체성은 (근거 발화 · 담당자 · 기한)이다. 같은 근거 발화에서 정말로 다른 일이
# 나온 경우는 대개 근거가 갈리거나 재현 자체가 안 돼 notReproduced 로 잡힌다.
BLOCKING_FIELDS = ("assigneeCandidatePersonId", "dueDate")


class VerifyRequest(CamelModel):
    """검증 대상 tuple 하나 + 그것을 다시 뽑는 데 필요한 문맥.

    문맥을 통째로 받는 이유는 Python 이 **내부에서 L4 를 한 번 더 돌리기** 때문이다.
    Spring 이 두 관점을 각각 호출해 결과를 모아 넘기면 인스턴스 간 왕복이 두 번이 되고,
    관점 조합 규칙(한쪽 실패 시 안전한 쪽)이 Spring 과 Python 두 곳에 생긴다.
    """

    tenant_id: int
    meeting_id: int
    topic: str

    # 검증 대상 — L4(view=EXTRACT)가 뽑은 기준 tuple.
    tuple: AssignmentTuple

    items: list[TopicItem] = Field(default_factory=list)
    utterances: list[Utterance]
    participants: list[Participant]
    meeting_date: date | None = None
    query_text: str | None = None


class ViewResult(CamelModel):
    """관점 하나의 결과. 관점마다 채워지는 필드가 다르다(명세 응답 예시와 같은 모양)."""

    view: View

    # EXTRACT_NARROW — 좁은 시야로 다시 뽑은 tuple. 재현되지 않았으면 None 이다.
    tuple: AssignmentTuple | None = None

    # VERIFY — 기준 tuple 이 근거 발화로 확인되는가.
    verdict: Verdict | None = None
    reason: str | None = None

    # 이 관점이 실패했다. 실패를 숨기지 않는 이유는 agree=false 의 원인이 '불일치'인지
    # '한쪽이 죽음'인지가 다르기 때문이다 — 후자가 계속 쌓이면 고칠 것은 코드다.
    error: str | None = None

    # 관점별 토큰. 최상단 usage 는 이 둘의 합이다. 나눠서도 두는 이유는 두 관점의
    # 비용이 크게 다르기 때문이다 — VERIFY 는 발화를 좁혀 보내므로 훨씬 싸고,
    # 합계만 있으면 어느 쪽을 줄여야 하는지 판단할 수 없다.
    usage: Usage = Field(default_factory=Usage)


class VerifyResponse(CamelModel):
    # 검토로 보내지 않아도 되는가. **"맞다"가 아니라 "확신할 수 있다"** 이다.
    #
    # ⚠ disagreementFields 가 비어 있다는 뜻은 아니다. title 만 갈린 경우 agree=true 인 채로
    #   그 필드가 실려 나간다(BLOCKING_FIELDS 주석). 표현 차이는 기록할 값이지 사람을 부를
    #   이유가 아니다.
    agree: bool

    # 두 관점이 갈린 필드 **전부**. 검토 여부를 정하는 것은 이 중 BLOCKING_FIELDS 뿐이다.
    #
    disagreement_fields: list[str] = Field(default_factory=list)

    # 첫 회차가 갈렸는데 **재실행 두 번이 모두 동의**했는가 — 즉 그 갈림이 잡음이었을 가능성.
    #
    # ⚠ **이 값이 true 여도 agree 는 바뀌지 않는다.** 판정은 첫 회차 그대로이고 여기서는 세기만
    #   한다. 모델이 같은 입력에 같은 출력을 주지 않는다는 것은 알지만(2026-08-15 실측),
    #   그건 L3.5 에서 잰 값이고 **L5 에서 잰 적이 없다.** 미검증 전제로 판정을 뒤집으면
    #   진짜 오배정이 자동확정으로 나갈 수 있고, 이 계층이 막으려던 것이 정확히 그것이다.
    #
    # 이 값을 세면 곧 **L5 의 잡음 비율**이다("첫 회차 갈림 중 몇 %가 재실행에서 뒤집혔나").
    # 그 수치가 다음 결정의 근거가 된다 —
    #   높으면  다수결로 회수할 값이 있다 → 판정까지 뒤집는다
    #   낮으면  갈림이 대개 진짜다 → 다수결은 물론 이 재실행 자체를 걷어낸다
    tie_broken: bool = False

    # 돌린 관점 전부. 재실행이 있었으면 회차마다 둘씩 쌓여 여섯 개다.
    results: list[ViewResult] = Field(default_factory=list)
    usage: Usage = Field(default_factory=Usage)
    model: str
    prompt_version: str
