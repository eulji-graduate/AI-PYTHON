"""OVERVIEW · 회의 개요 DTO. 회의당 한 번 호출된다.

BACKEND 의 `meeting_summary.overview` 한 칸에 대응한다. 지금 그 칸에는 Spring 이 L3 의
주제별 요약을 이어 붙인 값이 들어 있고("· 주제이름\\n요약" × 주제 수), 이 계층이 성공하면
그 값을 짧은 개요로 덮는다. 실패하면 이어 붙인 값이 그대로 남는다 — 그래서 이 계층은
파이프라인의 맨 끝이고, 실패가 회의를 실패시키지 않는다.

**발화를 받지 않는다.** 이것이 다른 계층과 갈리는 지점이다.

  ① 전사를 다시 읽히면 주제 요약과 **다른 말을 하는 개요**가 나온다. 사용자는 개요와
     주제 요약 중 어느 쪽을 믿을지 알 수 없고, 그 불일치를 되짚을 방법도 없다.
  ② Spring 이 이 계층의 출력으로 개요 칸을 덮는다. 그 칸을 다시 입력으로 주면 재실행·
     재개에서 **자기 출력을 다시 압축**하고, 돌릴수록 내용이 사라진다.

그래서 주제 이름과 **확정 항목**만 받는다. 둘 다 `meeting_decision` 에 있어 몇 번 돌려도
같은 값이다. 확정만 오는 이유는 Spring 이 거른다 — 논의 중인 항목을 개요에 넣으면 합의되지
않은 것이 합의된 것처럼 읽힌다(L4 가 CONFIRMED 만 받는 것과 같은 규칙).
"""

from typing import Literal

from pydantic import Field

from app.schemas.common import CamelModel, Participant, Usage

# L3 의 ItemType 과 같은 집합이다 — 같은 표(meeting_decision)에서 온 값이라 갈리면 안 된다.
ItemType = Literal["DECISION", "DISCUSSION", "BLOCKER"]

CONTENT_MAX = 1000
OVERVIEW_MAX = 2000


class DigestItem(CamelModel):
    """개요가 참고할 확정 항목 하나.

    근거 발화 id 가 없다. 개요는 회의 전체를 몇 문장으로 줄이는 것이고 특정 발화를 가리키지
    않는다 — 근거가 필요한 화면은 주제별 항목(ANLZ-03)을 그대로 쓴다.
    """

    item_type: ItemType
    content: str


class MeetingTopicDigest(CamelModel):
    """주제 하나와 그 안에서 확정된 것들.

    `topicSeq` 는 L2 가 매긴 순번을 그대로 받는다. 개요는 순번을 쓰지 않지만, 프롬프트가
    주제를 회의 진행 순서로 읽어야 "먼저 A 를 정하고 그다음 B" 같은 흐름이 나온다.
    """

    topic_seq: int
    topic: str
    items: list[DigestItem] = Field(default_factory=list)


class SummarizeMeetingRequest(CamelModel):
    tenant_id: int
    meeting_id: int

    # 확정 항목이 하나도 없는 주제는 Spring 이 넘기지 않는다. 이름만 오면 모델이 내용을
    # 지어낼 여지가 생긴다.
    topics: list[MeetingTopicDigest]
    participants: list[Participant]
    query_text: str | None = None


class SummarizeMeetingResponse(CamelModel):
    # 빈 문자열이 정상 응답이다 — 줄일 것이 없으면 비워 보낸다. Spring 이 빈 값으로는
    # 덮지 않으므로(이어 붙인 값보다 나쁘다) 여기서 억지로 채우지 않는다.
    overview: str = ""

    usage: Usage = Field(default_factory=Usage)
    model: str
    prompt_version: str
