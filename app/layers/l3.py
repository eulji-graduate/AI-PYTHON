"""L3 · 주제별 정리 (AI-04).

L4(`l4.py`)를 복제해 세 곳만 바꾼 것이다 — SPEC · build_response_schema · 후처리.

`gate_status` 를 여기서 정하지 않는 것이 설계의 핵심이다. 정리와 게이트를 한
프롬프트에 섞으면 "무엇이 오갔나"와 "그게 확정인가"가 한 번의 생성으로 결정되고,
그러면 L3.5 를 따로 둔 이유(precision 우선의 별도 판단)가 사라진다.
"""

from __future__ import annotations

from app.layers import few_shot
from app.layers import formatting as fmt
from app.layers.runner import LayerRunner, LayerSpec
from app.schemas.common import FewShotExample, Utterance
from app.schemas.l3 import (
    CONTENT_MAX,
    REASON_MAX,
    SUMMARY_MAX,
    SummarizeTopicRequest,
    SummarizeTopicResponse,
    TopicItemDraft,
)

SPEC = LayerSpec(
    layer="L3",
    prompt_file="l3_summarize_topic.v1.txt",
    prompt_version="v1",
    dry_run_payload={"summary": "", "items": []},
)

ITEM_TYPES = ["DECISION", "DISCUSSION", "BLOCKER"]
FALLBACK_TYPE = "DISCUSSION"


def build_response_schema(utterances: list[Utterance]) -> dict:
    return {
        "type": "OBJECT",
        "properties": {
            "summary": {"type": "STRING"},
            "items": {
                "type": "ARRAY",
                "items": {
                    "type": "OBJECT",
                    "properties": {
                        "itemType": {"type": "STRING", "enum": ITEM_TYPES},
                        "content": {"type": "STRING"},
                        "reason": {"type": "STRING"},
                        "evidenceUtteranceId": {
                            "type": "STRING",
                            "enum": fmt.utterance_enum(utterances),
                        },
                    },
                    "required": ["itemType", "content", "reason", "evidenceUtteranceId"],
                    "propertyOrdering": ["itemType", "content", "reason", "evidenceUtteranceId"],
                },
            },
        },
        "required": ["summary", "items"],
    }


async def summarize_topic(request: SummarizeTopicRequest, runner: LayerRunner) -> SummarizeTopicResponse:
    examples = await few_shot.lookup(
        tenant_id=request.tenant_id,
        layer=SPEC.layer,
        query_text=request.query_text,
    )

    if not request.utterances:
        return SummarizeTopicResponse(
            topic_seq=request.topic_seq,
            topic=request.topic,
            used_few_shot=examples,
            model=runner.model_name,
            prompt_version=SPEC.prompt_version,
        )

    schema = build_response_schema(request.utterances)
    variables = build_prompt_variables(request, examples)

    raw, usage = await runner.run(SPEC, variables=variables, response_schema=schema)

    return SummarizeTopicResponse(
        topic_seq=request.topic_seq,
        topic=request.topic,
        summary=fmt.clip(fmt.as_text(raw.get("summary")), SUMMARY_MAX),
        items=parse_items(raw, request.utterances),
        used_few_shot=examples,
        usage=usage,
        model=runner.model_name,
        prompt_version=SPEC.prompt_version,
    )


def build_prompt_variables(request: SummarizeTopicRequest, examples: list[FewShotExample]) -> dict[str, str]:
    return {
        "TOPIC": request.topic,
        "PARTICIPANTS": fmt.format_participants(request.participants),
        "UTTERANCES": fmt.format_utterances(request.utterances),
        "FEW_SHOT": fmt.format_few_shot(examples),
    }


def parse_items(raw: dict, utterances: list[Utterance]) -> list[TopicItemDraft]:
    allowed_utterances = fmt.allowed_utterance_ids(utterances)

    results: list[TopicItemDraft] = []
    seen: set[tuple[str, int]] = set()

    for item in raw.get("items") or []:
        if not isinstance(item, dict):
            continue

        content = fmt.clip(fmt.as_text(item.get("content")), CONTENT_MAX)
        if not content:
            continue

        evidence_id = fmt.as_int(item.get("evidenceUtteranceId"))
        # 근거 강제 — 목록에 없는 id 는 그 항목을 버린다.
        if evidence_id is None or evidence_id not in allowed_utterances:
            continue

        # 분류가 깨져 돌아오면 버리지 않고 DISCUSSION 으로 내린다. 버리면 오간 내용이
        # 통째로 사라지지만, 내리면 최악이 '사람이 한 번 더 보는 것'이다.
        item_type = item.get("itemType")
        if item_type not in ITEM_TYPES:
            item_type = FALLBACK_TYPE

        reason = fmt.clip(fmt.as_text(item.get("reason")), REASON_MAX)
        if not reason:
            # 근거 없는 분류는 오분류 조사의 출발점을 잃는다. 그래도 항목 자체는
            # 살리되, 비어 있음을 감추지 않는다.
            reason = "(모델이 분류 근거를 반환하지 않음)"

        key = (content, evidence_id)
        if key in seen:
            continue
        seen.add(key)

        results.append(
            TopicItemDraft(
                item_type=item_type,
                content=content,
                reason=reason,
                evidence_utterance_id=evidence_id,
            )
        )

    return results
