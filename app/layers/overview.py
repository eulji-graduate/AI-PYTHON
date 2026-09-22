"""OVERVIEW · 회의 개요.

L3(`l3.py`)를 복제해 세 곳을 바꾼 것이다 — 입력(발화 → 주제 묶음) · SPEC · 후처리.

<h2>few-shot 을 쓰지 않는다</h2>
few-shot 예시는 사람이 검수한 라벨(review_log)에서 온다. 그런데 개요는 **검토 대상이
아니다** — RVW-02 가 고칠 수 있는 값은 담당자·기한·제목·내용이고 개요는 거기 없다. 라벨이
영원히 쌓이지 않으므로 조회를 걸어도 항상 빈 목록이고, 그러면 호출만 늘어난다.

<h2>주제 순서를 그대로 읽힌다</h2>
topicSeq 순으로 프롬프트에 싣는다. 순서를 섞으면 "먼저 A 를 정하고 그다음 B" 같은 흐름이
나오지 않고, 개요가 항목을 다시 나열한 목록이 된다 — 그건 지금 이어 붙인 값과 같아진다.
"""

from __future__ import annotations

from app.layers import formatting as fmt
from app.layers.runner import LayerRunner, LayerSpec
from app.schemas.overview import (
    OVERVIEW_MAX,
    MeetingTopicDigest,
    SummarizeMeetingRequest,
    SummarizeMeetingResponse,
)

SPEC = LayerSpec(
    layer="OVERVIEW",
    prompt_file="overview_summarize_meeting.v1.txt",
    prompt_version="v1",
    dry_run_payload={"overview": ""},
)


def build_response_schema() -> dict:
    """항목 enum 이 없다 — 개요는 특정 발화나 항목을 가리키지 않는 자유 문장이다."""
    return {
        "type": "OBJECT",
        "properties": {"overview": {"type": "STRING"}},
        "required": ["overview"],
    }


async def summarize_meeting(
    request: SummarizeMeetingRequest, runner: LayerRunner
) -> SummarizeMeetingResponse:
    if not request.topics:
        # 줄일 것이 없다. 부르면 빈 입력에 돈만 쓰고 빈 개요가 돌아온다.
        # Spring 도 같은 검사를 하지만(digests.isEmpty) 여기서도 막는다 — 이 엔드포인트를
        # 직접 부르는 경로(수동 재현·테스트)가 남아 있다.
        return SummarizeMeetingResponse(
            model=runner.model_name,
            prompt_version=SPEC.prompt_version,
        )

    raw, usage = await runner.run(
        SPEC,
        variables=build_prompt_variables(request),
        response_schema=build_response_schema(),
    )

    return SummarizeMeetingResponse(
        overview=fmt.clip(fmt.as_text(raw.get("overview")), OVERVIEW_MAX),
        usage=usage,
        model=runner.model_name,
        prompt_version=SPEC.prompt_version,
    )


def build_prompt_variables(request: SummarizeMeetingRequest) -> dict[str, str]:
    return {
        "PARTICIPANTS": fmt.format_participants(request.participants),
        "TOPICS": format_topics(request.topics),
    }


def format_topics(topics: list[MeetingTopicDigest]) -> str:
    """주제와 확정 항목을 진행 순서대로 적는다.

    항목 종류를 함께 적는 이유 — 결정과 블로커를 같은 무게로 읽으면 "정해졌다"와 "막혀 있다"가
    한 문장에 섞인다. 개요에서 그 둘이 뭉치면 읽는 사람이 회의 결과를 잘못 판단한다.
    """
    lines: list[str] = []
    for topic in sorted(topics, key=lambda t: t.topic_seq):
        lines.append(f"### {topic.topic}")
        for item in topic.items:
            lines.append(f"- [{item.item_type}] {item.content}")
        lines.append("")
    return "\n".join(lines).strip()
