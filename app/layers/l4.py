"""L4 · assignment tuple 추출 (AI-06).

이 파일이 나머지 계층의 원본이다. 새 계층을 추가할 때 복제해서 바꾸는 것은
① SPEC(프롬프트 파일·버전) ② build_response_schema ③ 후처리 세 곳뿐이다.
참석자·발화 포맷과 값 되돌리기는 `formatting.py` 가 공통으로 갖는다.

핵심은 정확도 4원칙을 **프롬프트가 아니라 응답 스키마로** 거는 것이다.
프롬프트의 "목록에서만 고르세요"는 지켜지지 않을 확률이 남지만,
enum 으로 박으면 목록 밖 값이 나올 방법 자체가 없다.
"""

from __future__ import annotations

from app.layers import few_shot
from app.layers import formatting as fmt
from app.layers.runner import LayerRunner, LayerSpec
from app.schemas.common import FewShotExample, Participant, Utterance
from app.schemas.l4 import (
    SOURCE_UNKNOWN,
    AssignmentTuple,
    ExtractTuplesRequest,
    ExtractTuplesResponse,
    TopicItem,
)

SPEC = LayerSpec(
    layer="L4",
    prompt_file="l4_extract_tuples.v1.txt",
    prompt_version="v1",
    dry_run_payload={"tuples": []},
)

TITLE_MAX = 300

_VIEW_INSTRUCTION = {
    # 기본 관점 — 주제 전체를 보고 뽑는다.
    "EXTRACT": "주제 전체를 읽고 배정을 모두 뽑는다.",
    # L5 관점 다변화의 한쪽. 같은 프롬프트를 두 번 돌리는 게 아니라 시야를 좁힌다 —
    # 넓은 문맥에서 생긴 추론(앞에서 언급된 사람을 뒤 발화의 담당자로 잇는 것 등)이
    # 근거 발화만으로 재현되는지 보는 것이 목적이다.
    "EXTRACT_NARROW": (
        "근거 발화와 그 앞뒤 3발화만 근거로 삼는다. "
        "그 범위에서 담당자가 확인되지 않으면 unknown_person 으로 둔다."
    ),
}


def evidence_pool(items: list[TopicItem], utterances: list[Utterance]) -> list[Utterance]:
    """근거로 지정할 수 있는 발화만 골라낸다.

    항목(items)이 자기 근거 발화 id 를 들고 왔으면 **그 발화들로 좁힌다.** L4 의 일은
    L3.5 가 확정한 항목을 배정으로 바꾸는 것이므로, 근거도 그 항목이 선 근거여야
    추적이 이어진다. 좁히지 않으면 주제 안의 아무 발화나 근거로 붙을 수 있고,
    그러면 "확정된 항목"과 "뽑힌 배정"의 연결이 끊긴다.

    비워 보내면(아직 Spring 이 채우지 않는 단계) 주제의 발화 전체를 후보로 둔다 —
    좁힐 근거가 없을 때 전부 버리면 배정을 하나도 못 뽑는다.
    """
    approved = {uid for item in items for uid in item.evidence_utterance_ids}
    if not approved:
        return utterances

    narrowed = [u for u in utterances if u.utterance_id in approved]
    # 항목이 준 id 가 발화 목록에 하나도 없으면 둘이 어긋난 것이다. 전부 버리면
    # 조용히 빈 결과가 되므로, 문맥 전체를 후보로 두고 후처리 검증에 맡긴다.
    return narrowed or utterances


def build_response_schema(participants: list[Participant], evidence_utterances: list[Utterance]) -> dict:
    """참석자와 **근거 가능 발화**를 그대로 enum 으로 박은 응답 스키마.

    회의마다 참석자가 다르므로 스키마는 요청 시점에 만들어진다. 고정 스키마로 두면
    personId 를 검증할 방법이 없어져 4원칙의 '닫힌 목록'이 프롬프트 부탁으로 내려앉는다.
    """
    return {
        "type": "OBJECT",
        "properties": {
            "tuples": {
                "type": "ARRAY",
                "items": {
                    "type": "OBJECT",
                    "properties": {
                        "title": {"type": "STRING"},
                        "assigneeCandidatePersonId": {
                            "type": "STRING",
                            "enum": fmt.person_enum(participants),
                        },
                        "assigneeSource": {
                            "type": "STRING",
                            "enum": ["EXPLICIT_CALL", "FIRST_PERSON", SOURCE_UNKNOWN],
                        },
                        "dueDate": {"type": "STRING", "nullable": True},
                        "evidenceUtteranceId": {
                            "type": "STRING",
                            "enum": fmt.utterance_enum(evidence_utterances),
                        },
                    },
                    "required": [
                        "title",
                        "assigneeCandidatePersonId",
                        "assigneeSource",
                        "evidenceUtteranceId",
                    ],
                    "propertyOrdering": [
                        "title",
                        "assigneeCandidatePersonId",
                        "assigneeSource",
                        "dueDate",
                        "evidenceUtteranceId",
                    ],
                },
            }
        },
        "required": ["tuples"],
    }


async def extract_tuples(request: ExtractTuplesRequest, runner: LayerRunner) -> ExtractTuplesResponse:
    examples = await few_shot.lookup(
        tenant_id=request.tenant_id,
        layer=SPEC.layer,
        query_text=request.query_text,
    )

    pool = evidence_pool(request.items, request.utterances)

    # 근거 후보가 없으면 반환할 수 있는 항목이 없다. 모델을 부르지 않고 끝낸다 —
    # enum 이 빈 배열인 스키마는 만들 수도 없고, 불러도 결과가 전부 버려지므로
    # 토큰만 태운다.
    if not pool:
        return ExtractTuplesResponse(
            tuples=[],
            used_few_shot=examples,
            model=runner.model_name,
            prompt_version=SPEC.prompt_version,
        )

    schema = build_response_schema(request.participants, pool)
    variables = build_prompt_variables(request, examples)

    raw, usage = await runner.run(SPEC, variables=variables, response_schema=schema)

    return ExtractTuplesResponse(
        tuples=parse_tuples(raw, request.participants, pool),
        used_few_shot=examples,
        usage=usage,
        model=runner.model_name,
        prompt_version=SPEC.prompt_version,
    )


def build_prompt_variables(request: ExtractTuplesRequest, examples: list[FewShotExample]) -> dict[str, str]:
    """프롬프트 자리표시자 → 값. 프롬프트 파일과 이 함수가 어긋나면
    `render_prompt` 가 PROMPT_UNRESOLVED 로 즉시 실패한다(조용히 넘어가지 않는다)."""
    eligible_ids = fmt.allowed_utterance_ids(evidence_pool(request.items, request.utterances))
    return {
        "TOPIC": request.topic,
        "MEETING_DATE": fmt.format_meeting_date(request.meeting_date),
        "PARTICIPANTS": fmt.format_participants(request.participants),
        "UTTERANCES": fmt.format_utterances(request.utterances, eligible_ids),
        "ITEMS": _format_items(request.items),
        "FEW_SHOT": fmt.format_few_shot(examples),
        "VIEW_INSTRUCTION": _VIEW_INSTRUCTION[request.view],
    }


def parse_tuples(
    raw: dict, participants: list[Participant], evidence_utterances: list[Utterance]
) -> list[AssignmentTuple]:
    """스키마를 걸었어도 한 번 더 검증한다.

    구조화 출력은 제공자 구현에 의존하고, 그 구현이 흔들리는 순간 조용히 오염된
    담당자·근거가 액션 테이블까지 흘러간다. 여기서 걸러내면 최악이 '항목 누락'이다.

    `evidence_utterances` 는 `evidence_pool()` 이 좁힌 목록이어야 한다 — 스키마에
    박은 enum 과 같은 집합으로 검증해야 둘이 어긋나지 않는다.
    """
    allowed_persons = fmt.allowed_person_ids(participants)
    allowed_utterances = fmt.allowed_utterance_ids(evidence_utterances)

    results: list[AssignmentTuple] = []
    seen: set[tuple[str, int | None, int]] = set()

    for item in raw.get("tuples") or []:
        if not isinstance(item, dict):
            continue

        title = fmt.clip(fmt.as_text(item.get("title")), TITLE_MAX)
        if not title:
            continue

        evidence_id = fmt.as_int(item.get("evidenceUtteranceId"))
        # 근거 강제 — 목록에 없는 id 는 그 항목을 버린다(§규칙 2).
        if evidence_id is None or evidence_id not in allowed_utterances:
            continue

        assignee = fmt.resolve_person(item.get("assigneeCandidatePersonId"), allowed_persons)

        source = item.get("assigneeSource")
        if source not in ("EXPLICIT_CALL", "FIRST_PERSON"):
            source = None

        key = (title, assignee, evidence_id)
        if key in seen:  # 세그먼트별 호출의 중복 병합
            continue
        seen.add(key)

        results.append(
            AssignmentTuple(
                title=title,
                assignee_candidate_person_id=assignee,
                assignee_source=source,
                due_date=fmt.as_iso_date(item.get("dueDate")),
                evidence_utterance_id=evidence_id,
            )
        )

    return results


def _format_items(items: list[TopicItem]) -> str:
    lines = [f"- [{item.item_type}] {item.content}" for item in items]
    return "\n".join(lines) or fmt.NONE_MARK
