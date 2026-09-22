"""L3.5 · 확정/논의 게이트 (AI-05).

L4(`l4.py`)를 복제해 세 곳만 바꾼 것이다 — SPEC · build_response_schema · 후처리.

**precision 우선을 프롬프트가 아니라 후처리로 강제한다.** 프롬프트의 "애매하면
DISCUSSED" 는 지켜지지 않을 확률이 남지만, 여기서는 그 확률이 남는 자리를 없앤다:

    모델이 판정을 돌려주지 않은 항목 → DISCUSSED

즉 **누락이 통과가 되지 않는다.** 반대로 두면(누락 = CONFIRMED) 응답이 잘리거나
항목 하나를 건너뛴 것만으로 합의되지 않은 일이 배정으로 나간다. 게이트에서 가장
비싼 실패는 그것이다.
"""

from __future__ import annotations

from app.layers import few_shot
from app.layers import formatting as fmt
from app.layers.runner import LayerRunner, LayerSpec
from app.schemas.common import FewShotExample
from app.schemas.l3_5 import (
    REASON_MAX,
    GateCandidate,
    GateRequest,
    GateResponse,
    GateVerdict,
)

SPEC = LayerSpec(
    layer="L3.5",
    prompt_file="l3_5_gate.v1.txt",
    prompt_version="v1",
    dry_run_payload={"verdicts": []},
)

CONFIRMED = "CONFIRMED"
DISCUSSED = "DISCUSSED"
GATE_STATUSES = [CONFIRMED, DISCUSSED]

# 모델이 답하지 않은 항목에 붙는 사유. 사람이 화면에서 이 문구를 보면 "게이트가
# 판정을 못 했다"와 "게이트가 논의로 판정했다"를 구분할 수 있어야 한다.
MISSING_VERDICT_REASON = "게이트가 이 항목을 판정하지 않았다 — 확정으로 올리지 않는다(precision 우선)."


def build_response_schema(items: list[GateCandidate]) -> dict:
    """itemKey 를 요청 항목의 enum 으로 박는다 — 없는 항목에 대한 판정이 나올 수 없게."""
    return {
        "type": "OBJECT",
        "properties": {
            "verdicts": {
                "type": "ARRAY",
                "items": {
                    "type": "OBJECT",
                    "properties": {
                        "itemKey": {"type": "STRING", "enum": [item.item_key for item in items]},
                        "gateStatus": {"type": "STRING", "enum": GATE_STATUSES},
                        "reason": {"type": "STRING"},
                    },
                    "required": ["itemKey", "gateStatus", "reason"],
                    "propertyOrdering": ["itemKey", "gateStatus", "reason"],
                },
            }
        },
        "required": ["verdicts"],
    }


async def gate(request: GateRequest, runner: LayerRunner) -> GateResponse:
    examples = await few_shot.lookup(
        tenant_id=request.tenant_id,
        layer=SPEC.layer,
        query_text=request.query_text,
    )

    if not request.items:
        return GateResponse(
            verdicts=[],
            used_few_shot=examples,
            model=runner.model_name,
            prompt_version=SPEC.prompt_version,
        )

    schema = build_response_schema(request.items)
    variables = build_prompt_variables(request, examples)

    raw, usage = await runner.run(SPEC, variables=variables, response_schema=schema)

    return GateResponse(
        verdicts=parse_verdicts(raw, request.items),
        used_few_shot=examples,
        usage=usage,
        model=runner.model_name,
        prompt_version=SPEC.prompt_version,
    )


def build_prompt_variables(request: GateRequest, examples: list[FewShotExample]) -> dict[str, str]:
    return {
        "TOPIC": request.topic,
        "PARTICIPANTS": fmt.format_participants(request.participants),
        "ITEMS": _format_items(request.items),
        "UTTERANCES": fmt.format_utterances(request.utterances),
        "FEW_SHOT": fmt.format_few_shot(examples),
    }


def parse_verdicts(raw: dict, items: list[GateCandidate]) -> list[GateVerdict]:
    """**요청 항목 수만큼 정확히 반환한다.** 판정이 없는 항목은 DISCUSSED 로 채운다.

    요청 순서를 유지하는 것도 계약이다 — Spring 이 순서로 맞추지는 않지만, 순서가
    흔들리면 응답을 눈으로 대조할 때 사람이 항목을 놓친다.
    """
    by_key: dict[str, GateVerdict] = {}

    for entry in raw.get("verdicts") or []:
        if not isinstance(entry, dict):
            continue
        key = fmt.as_text(entry.get("itemKey"))
        status = entry.get("gateStatus")
        # 목록 밖 키, 알 수 없는 판정값은 버린다 — 아래에서 DISCUSSED 로 채워진다.
        if not key or status not in GATE_STATUSES:
            continue
        if key in by_key:  # 같은 항목을 두 번 판정했다. 먼저 온 것만 쓴다
            continue
        by_key[key] = GateVerdict(
            item_key=key,
            gate_status=status,
            reason=fmt.clip(fmt.as_text(entry.get("reason")), REASON_MAX)
            or "(모델이 판정 근거를 반환하지 않음)",
        )

    return [
        by_key.get(
            item.item_key,
            GateVerdict(item_key=item.item_key, gate_status=DISCUSSED, reason=MISSING_VERDICT_REASON),
        )
        for item in items
    ]


def _format_items(items: list[GateCandidate]) -> str:
    lines = [
        f"- itemKey={item.item_key} · [{item.item_type}] {item.content}"
        f" (근거 발화 id={item.evidence_utterance_id})"
        for item in items
    ]
    return "\n".join(lines) or fmt.NONE_MARK
