"""L1.5 · 지시어 해소 (AI-02).

L4(`l4.py`)를 복제해 세 곳만 바꾼 것이다 — SPEC · build_response_schema · 후처리.

이 계층에만 있는 검증이 하나 있다: **surface 가 그 발화 안에 실제로 있는가.**
지시 표현은 발화에서 그대로 인용할 수 있는 값이라, 없는 표현이 돌아오면 그건
모델이 만들어낸 것이다. 근거 강제를 문자열 수준까지 밀어붙일 수 있는 드문 자리다.
"""

from __future__ import annotations

from app.layers import few_shot
from app.layers import formatting as fmt
from app.layers.runner import LayerRunner, LayerSpec
from app.schemas.common import UNKNOWN_PERSON, FewShotExample, Participant, Utterance
from app.schemas.l1_5 import (
    ResolvedReference,
    ResolveReferenceRequest,
    ResolveReferenceResponse,
)

SPEC = LayerSpec(
    layer="L1.5",
    prompt_file="l1_5_resolve_reference.v1.txt",
    prompt_version="v1",
    dry_run_payload={"references": []},
)

REFERENCE_TYPES = ["PERSON", "TOPIC", "ARTIFACT", "TIME", "UNRESOLVED"]
UNRESOLVED = "UNRESOLVED"

SURFACE_MAX = 100
RESOLVED_TEXT_MAX = 300


def targets(request: ResolveReferenceRequest) -> list[Utterance]:
    """지시어를 찾을 발화. 비워 보내면 전체가 대상이다.

    L4 의 `evidence_pool` 과 같은 정책이다 — 좁힐 근거를 주지 않은 호출에서
    전부 버리면 조용히 빈 결과가 되고, 미구현이 '지시어 0건'으로 위장된다.
    """
    if not request.target_utterance_ids:
        return request.utterances

    approved = set(request.target_utterance_ids)
    narrowed = [u for u in request.utterances if u.utterance_id in approved]
    return narrowed or request.utterances


def build_response_schema(
    participants: list[Participant], target_utterances: list[Utterance], context_utterances: list[Utterance]
) -> dict:
    """대상 발화와 **문맥 전체**를 각각 다른 enum 으로 박는다.

    지시어가 나타나는 발화(utteranceId)와 선행사가 있는 발화(evidenceUtteranceId)는
    다른 집합이다 — 선행사는 대상 밖 앞 발화에 있는 것이 정상이므로, 근거 enum 을
    대상으로 좁히면 정답을 고를 수 없게 된다.
    """
    return {
        "type": "OBJECT",
        "properties": {
            "references": {
                "type": "ARRAY",
                "items": {
                    "type": "OBJECT",
                    "properties": {
                        "utteranceId": {
                            "type": "STRING",
                            "enum": fmt.utterance_enum(target_utterances),
                        },
                        "surface": {"type": "STRING"},
                        "referenceType": {"type": "STRING", "enum": REFERENCE_TYPES},
                        "resolvedPersonId": {
                            "type": "STRING",
                            "enum": fmt.person_enum(participants),
                            "nullable": True,
                        },
                        "resolvedText": {"type": "STRING", "nullable": True},
                        "evidenceUtteranceId": {
                            "type": "STRING",
                            "enum": fmt.utterance_enum(context_utterances),
                        },
                    },
                    "required": ["utteranceId", "surface", "referenceType", "evidenceUtteranceId"],
                    "propertyOrdering": [
                        "utteranceId",
                        "surface",
                        "referenceType",
                        "resolvedPersonId",
                        "resolvedText",
                        "evidenceUtteranceId",
                    ],
                },
            }
        },
        "required": ["references"],
    }


async def resolve_reference(
    request: ResolveReferenceRequest, runner: LayerRunner
) -> ResolveReferenceResponse:
    examples = await few_shot.lookup(
        tenant_id=request.tenant_id,
        layer=SPEC.layer,
        query_text=request.query_text,
    )

    target_utterances = targets(request)

    # 대상도 문맥도 없으면 부를 것이 없다. enum 이 빈 배열인 스키마는 만들 수 없고,
    # 불러도 결과가 전부 버려지므로 토큰만 태운다.
    if not target_utterances or not request.utterances:
        return ResolveReferenceResponse(
            references=[],
            used_few_shot=examples,
            model=runner.model_name,
            prompt_version=SPEC.prompt_version,
        )

    schema = build_response_schema(request.participants, target_utterances, request.utterances)
    variables = build_prompt_variables(request, examples)

    raw, usage = await runner.run(SPEC, variables=variables, response_schema=schema)

    return ResolveReferenceResponse(
        references=parse_references(raw, request, target_utterances),
        used_few_shot=examples,
        usage=usage,
        model=runner.model_name,
        prompt_version=SPEC.prompt_version,
    )


def build_prompt_variables(
    request: ResolveReferenceRequest, examples: list[FewShotExample]
) -> dict[str, str]:
    target_ids = fmt.allowed_utterance_ids(targets(request))
    return {
        "PARTICIPANTS": fmt.format_participants(request.participants),
        "UTTERANCES": fmt.format_utterances(request.utterances, target_ids, fmt.TARGET_MARK),
        "FEW_SHOT": fmt.format_few_shot(examples),
    }


def parse_references(
    raw: dict, request: ResolveReferenceRequest, target_utterances: list[Utterance]
) -> list[ResolvedReference]:
    """스키마를 걸었어도 한 번 더 검증한다 — 특히 surface 의 실재 여부를."""
    allowed_targets = {u.utterance_id: u.text for u in target_utterances}
    allowed_context = fmt.allowed_utterance_ids(request.utterances)
    allowed_persons = fmt.allowed_person_ids(request.participants)

    results: list[ResolvedReference] = []
    seen: set[tuple[int, str]] = set()

    for item in raw.get("references") or []:
        if not isinstance(item, dict):
            continue

        utterance_id = fmt.as_int(item.get("utteranceId"))
        if utterance_id is None or utterance_id not in allowed_targets:
            continue

        surface = fmt.clip(fmt.as_text(item.get("surface")), SURFACE_MAX)
        # 지시 표현은 발화에서 그대로 인용할 수 있다. 발화에 없는 표현이 돌아왔다면
        # 모델이 만들어낸 것이고, 그걸 통과시키면 존재하지 않는 지시어에 대한 해소가
        # 뒤 계층으로 흘러간다.
        if not surface or surface not in allowed_targets[utterance_id]:
            continue

        evidence_id = fmt.as_int(item.get("evidenceUtteranceId"))
        if evidence_id is None or evidence_id not in allowed_context:
            continue

        reference_type = item.get("referenceType")
        if reference_type not in REFERENCE_TYPES:
            reference_type = UNRESOLVED

        person_id = (
            fmt.resolve_person(item.get("resolvedPersonId"), allowed_persons)
            if reference_type == "PERSON"
            else None  # 사람이 아닌 지시어에 붙은 personId 는 의미가 없다
        )

        key = (utterance_id, surface)
        if key in seen:
            continue
        seen.add(key)

        results.append(
            ResolvedReference(
                utterance_id=utterance_id,
                surface=surface,
                reference_type=reference_type,
                resolved_person_id=person_id,
                resolved_text=_resolved_text(item, reference_type, person_id),
                evidence_utterance_id=evidence_id,
            )
        )

    return results


def _resolved_text(item: dict, reference_type: str, person_id: int | None) -> str | None:
    """대상 표현. **한 항목이 서로 다른 두 대상을 가리키지 않게** 정리한다.

    <표> 어떤 경우에 비우나
      UNRESOLVED            해소가 안 됐는데 대상 표현이 남으면 뒤 계층이 그걸 답으로 읽는다
      PERSON + personId 있음  personId 가 답이다. 자유 텍스트가 함께 오면 둘 중 무엇이 대상인지
                            소비자가 판단해야 하고, 어긋나면 조용히 갈린다(CodeRabbit PR #4 지적)
      `unknown_person` 문자열  탈출구 문자열이 대상 표현으로 새는 것

    <표> 남기는 경우
      PERSON + personId 없음  명단 밖 사람이다. "그분 = 김민섭 팀장(명단 밖)" 에서 그 이름이
                            유일한 단서고, 비우면 사람이 검토할 때 아무 정보가 없다.
                            L4 는 어차피 personId 로만 담당자를 정하므로 오배정 경로가 아니다.
    """
    text = fmt.clip(fmt.as_text(item.get("resolvedText")), RESOLVED_TEXT_MAX)
    if not text or text == UNKNOWN_PERSON or reference_type == UNRESOLVED:
        return None
    if reference_type == "PERSON" and person_id is not None:
        return None
    return text
