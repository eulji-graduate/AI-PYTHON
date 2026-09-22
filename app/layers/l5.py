"""L5 · 관점 다변화 검증 (AI-07).

다른 계층과 달리 이 계층은 **모델을 두 번, 서로 다른 관점으로 부른다.**

    EXTRACT_NARROW   l4 를 view=EXTRACT_NARROW 로 재실행 — 근거 발화 앞뒤만 보고 다시 뽑기
    VERIFY           "이 tuple 이 맞나?" — 생성이 아니라 검사

두 관점을 **Python 안에서 조합한다.** Spring 이 각각 호출해 결과를 모으면 인스턴스 간
왕복이 두 번이 되고, 더 나쁘게는 조합 규칙("한쪽만 실패하면 안전한 쪽으로")이 Spring 과
Python 두 곳에 생겨 한쪽만 고쳐지는 상태가 만들어진다.

두 호출은 병렬로 돈다. 순차로 돌리면 사용자가 기다리는 시간이 그대로 두 배가 되는데,
두 관점은 서로의 결과를 보지 않으므로(그게 관점 다변화의 전제다) 순서가 의미 없다.

**모든 불일치가 사람을 부를 이유는 아니다.** 검토 여부는 `BLOCKING_FIELDS`(담당자·기한)와
재현 실패로만 정하고, `title` 표현 차이는 기록만 한다 — 그것까지 검토로 보내면 목록이
부풀어 진짜 오배정이 표현 차이들 사이에 묻힌다(python#11).
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass

from app.errors import LayerError, LayerErrorKind
from app.layers import formatting as fmt
from app.layers import l4
from app.layers.runner import LayerRunner, LayerSpec
from app.schemas.common import Participant, Usage, Utterance
from app.schemas.l4 import AssignmentTuple, ExtractTuplesRequest
from app.schemas.l5 import (
    BLOCKING_FIELDS,
    COMPARED_FIELDS,
    REASON_MAX,
    VerifyRequest,
    VerifyResponse,
    ViewResult,
)

SPEC = LayerSpec(
    layer="L5",
    prompt_file="l5_verify.v1.txt",
    prompt_version="v1",
    dry_run_payload={"verdict": "ACCEPT", "reason": "(DRY_RUN)"},
)

VERDICTS = ["ACCEPT", "REJECT"]

# 한 관점이 재현하지 못했을 때 붙는 불일치 표시. 필드 하나가 갈린 것과 "아예 안 나왔다"는
# 다른 신호라 따로 둔다 — 후자가 잦으면 좁은 시야에서 근거가 부족하다는 뜻이다.
NOT_REPRODUCED = "notReproduced"


def build_response_schema() -> dict:
    """VERIFY 관점의 응답. 참석자·발화 enum 이 없는 유일한 계층이다 —
    새 값을 뽑는 것이 아니라 주어진 것을 판정하기만 하므로 닫을 목록이 없다."""
    return {
        "type": "OBJECT",
        "properties": {
            "verdict": {"type": "STRING", "enum": VERDICTS},
            "reason": {"type": "STRING"},
        },
        "required": ["verdict", "reason"],
        "propertyOrdering": ["verdict", "reason"],
    }


async def verify(request: VerifyRequest, runner: LayerRunner) -> VerifyResponse:
    """두 관점을 돌리고, **갈렸을 때만** 두 번 더 돌려 그 갈림이 잡음이었는지 기록한다.

    <h2>⚠ 판정은 바꾸지 않는다 — 세는 것까지다</h2>
    첫 회차가 갈리면 `agree=False` 다. 재실행에서 두 번 다 동의로 나와도 그대로다.
    바뀌는 것은 `tie_broken` 이 켜지는 것뿐이다.

    다수결로 덮고 싶은 유혹이 있다 — 모델이 같은 입력에 같은 출력을 주지 않으므로
    (2026-08-15 실측: L3.5 가 같은 프롬프트 3회에 항목 14건 중 3건씩 뒤집혔다) 잡음 한 번에
    억울하게 검토로 떨어지는 배정이 있을 것이다. 그런데 **그 "있을 것"이 아직 추정이다.**
    L3.5 에서 잰 값이고 L5 에서 잰 적이 없다.

    그리고 이 계층이 지키는 것은 정답 선택이 아니라 **불확실성 탐지**다
    (`test_불일치는_다수결로_덮지_않는다` 가 그 결정을 이름으로 못박고 있다). 뒤집는 쪽이
    틀렸을 때의 대가가 비대칭이다 —

        맞았을 때  검토 목록에서 한두 건이 빠진다(사람 클릭 몇 번)
        틀렸을 때  진짜 오배정이 자동확정으로 나가 보드에 꽂힌다. 회수 경로가 없다

    이 저장소가 반복해서 지켜 온 "틀린 값보다 빈 값"의 방향을 근거 없이 뒤집지 않는다.

    <h2>그래서 지금은 데이터를 모은다</h2>
    `tie_broken` 을 세면 곧 **L5 의 잡음 비율**이다("첫 회차 갈림 중 몇 %가 재실행에서
    동의로 뒤집혔나"). 그 값이 나오면 판정까지 뒤집을지 정할 수 있고, 그 전환은 아래
    `_respond(...)` 에 `agree=True` 를 넘기는 한 줄이다.

        잡음이 높다  → 다수결로 회수할 값이 있다
        잡음이 낮다  → 갈림이 대개 진짜다. 다수결은 물론 이 재실행 자체도 걷어낸다

    <h2>왜 갈렸을 때만 돌리나</h2>
    **행복 경로에 비용을 물리지 않는다.** 첫 회차가 동의하면 거기서 끝이다. Gemini 쿼터가
    병목인 상황에서 이 차이가 곧 "같은 한도로 회의를 몇 건 더 돌리는가"다.
    """
    first = await _run_once(request, runner)
    if first.agree:
        return _respond(first, runner, tie_broken=False)

    # 갈렸다. **판정을 바꾸려는 것이 아니라** 이 갈림이 잡음이었는지 세려고 두 번 더 묻는다.
    others = await asyncio.gather(
        _run_once(request, runner),
        _run_once(request, runner),
        return_exceptions=False,
    )

    rounds = [first, *others]
    # 재실행이 둘 다 동의했다 = 첫 회차가 잡음이었을 가능성이 크다. 그 사실만 남긴다.
    tie_broken = all(round_.agree for round_ in others)
    return _respond(first, runner, tie_broken=tie_broken, extra=rounds[1:])


@dataclass(frozen=True)
class _Round:
    """한 회차의 두 관점과 그 판정. 회차끼리 비교하려면 판정을 값으로 들고 있어야 한다."""

    narrow: ViewResult
    verify: ViewResult
    agree: bool
    disagreements: list[str]


async def _run_once(request: VerifyRequest, runner: LayerRunner) -> _Round:
    """두 관점을 한 번 돌려 판정까지 낸다. 예전 verify() 본문 그대로다."""
    narrow, verify_result = await asyncio.gather(
        _run_narrow(request, runner),
        _run_verify(request, runner),
        return_exceptions=False,
    )

    # 둘 다 실패했으면 검증이 아예 수행되지 않은 것이다. 그걸 agree=false 로 돌려주면
    # "관점이 갈렸다"로 기록돼 검증이 돈 것처럼 보인다 — 계층 실패로 던져야
    # Spring 이 analysis_layer 를 FAILED 로 남기고 재시도한다.
    if narrow.error and verify_result.error:
        raise _both_views_failed(narrow, verify_result)

    disagreements = compare(request.tuple, narrow)
    # 한 관점만 실패해도 안전한 쪽으로 — agree=false 로 검토에 보낸다(명세).
    # 실패한 관점을 "동의"로 세면 검증이 반쪽만 돌았는데 자동 확정으로 나간다.
    #
    # 검토 여부는 **갈린 필드 전부가 아니라 BLOCKING_FIELDS 로만** 정한다. title 표현 차이로
    # 사람을 부르면 검토 목록이 부풀어 진짜 오배정이 그 사이에 묻힌다(BLOCKING_FIELDS 주석).
    agree = not blocking(disagreements) and not narrow.error and verify_result.verdict == "ACCEPT"
    return _Round(narrow=narrow, verify=verify_result, agree=agree, disagreements=disagreements)


def _respond(
    first: _Round,
    runner: LayerRunner,
    *,
    tie_broken: bool,
    extra: list[_Round] | None = None,
) -> VerifyResponse:
    """**첫 회차의 판정이 곧 응답이다.** 재실행은 세기만 하고 결과를 바꾸지 않는다.

    판정까지 뒤집으려면 여기에 `agree` 를 받는 인자를 더하면 된다 — 그 전환은 `tie_broken`
    수치를 보고 정한다(verify 주석). 지금 그 인자를 미리 만들어 두지 않는 이유는, 안 쓰는
    경로가 있으면 "이미 그렇게 도는 줄" 알게 되기 때문이다.

    results 에는 돌린 관점을 전부 싣고 usage 도 전부 더한다. 재실행 비용을 응답에서 숨기면
    QLTY-03 비용 집계가 실제보다 적게 잡히고, **"이 측정이 얼마나 비싼가"** 를 나중에 판단할
    수 없다 — 그 값이 재실행을 걷어낼지 정하는 다른 한 축이다.
    """
    rounds = [first, *(extra or [])]
    views = [view for round_ in rounds for view in (round_.narrow, round_.verify)]
    return VerifyResponse(
        agree=first.agree,
        disagreement_fields=first.disagreements,
        tie_broken=tie_broken,
        results=views,
        usage=_sum_usage(*views),
        model=runner.model_name,
        prompt_version=SPEC.prompt_version,
    )


def compare(baseline: AssignmentTuple, narrow: ViewResult) -> list[str]:
    """기준 tuple 과 좁은 시야의 재추출을 필드 단위로 맞춰본다.

    좁은 시야에서 **재현되지 않은 것 자체가 신호다.** 넓은 문맥에서만 성립하는 배정은
    앞에서 언급된 사람을 뒤 발화의 담당자로 이어 붙인 추론일 때가 많고, 그게 담당자
    오배정(`WRONG_ASSIGNEE`)의 주된 경로다.

    관점이 실패한 경우는 여기서 불일치로 세지 않는다 — 실패는 `error` 로 이미 드러나고,
    "갈렸다"와 "못 물어봤다"를 섞으면 disagreementFields 가 원인 조사에 쓸모없어진다.
    """
    if narrow.error:
        return []
    if narrow.tuple is None:
        return [NOT_REPRODUCED]

    differences = []
    if narrow.tuple.title != baseline.title:
        differences.append("title")
    if narrow.tuple.assignee_candidate_person_id != baseline.assignee_candidate_person_id:
        differences.append("assigneeCandidatePersonId")
    if narrow.tuple.due_date != baseline.due_date:
        differences.append("dueDate")

    # COMPARED_FIELDS 는 명세가 돌려주기로 한 필드명 목록이다. 위 비교와 어긋나면
    # 응답에 없는 필드명이 나가므로 순서를 그것에 맞춘다.
    return [field for field in COMPARED_FIELDS if field in differences]


def blocking(disagreements: list[str]) -> list[str]:
    """갈린 필드 중 **검토로 보내야 하는** 것만 남긴다.

    `notReproduced` 는 목록에 없어도 막는다 — 좁은 시야에서 아예 안 나온 것은 필드 하나가
    다른 것과 성질이 다르고, 넓은 문맥에서만 성립하는 배정이 담당자 오배정의 주된 경로다.

    title 만 갈린 경우 이 함수가 빈 목록을 주고, 그래서 agree 는 true 가 된다. 그때도
    disagreementFields 에는 title 이 남아 있다 — 값을 버리지 않는다.
    """
    return [field for field in disagreements if field in BLOCKING_FIELDS or field == NOT_REPRODUCED]


async def _run_narrow(request: VerifyRequest, runner: LayerRunner) -> ViewResult:
    """L4 를 좁은 시야로 재실행한다. 같은 프롬프트를 두 번 돌리는 것이 아니라
    `VIEW_INSTRUCTION` 이 바뀐다 — 그래서 오류가 서로 상관되지 않는다."""
    try:
        narrow_request = ExtractTuplesRequest(
            tenant_id=request.tenant_id,
            meeting_id=request.meeting_id,
            topic=request.topic,
            items=request.items,
            utterances=_evidence_window(request),
            participants=request.participants,
            query_text=request.query_text,
            meeting_date=request.meeting_date,
            view="EXTRACT_NARROW",
        )
        response = await l4.extract_tuples(narrow_request, runner)
    except LayerError as exc:  # 한쪽 실패는 전체 실패가 아니다 — 안전한 쪽으로 접는다
        return ViewResult(view="EXTRACT_NARROW", error=_describe(exc))

    return ViewResult(
        view="EXTRACT_NARROW",
        tuple=_match(response.tuples, request.tuple),
        usage=response.usage,
    )


async def _run_verify(request: VerifyRequest, runner: LayerRunner) -> ViewResult:
    try:
        raw, usage = await runner.run(
            SPEC,
            variables=_verify_variables(request),
            response_schema=build_response_schema(),
        )
    except LayerError as exc:
        return ViewResult(view="VERIFY", error=_describe(exc))

    verdict = raw.get("verdict")
    # 판정값이 깨져 돌아오면 ACCEPT 로 넘기지 않는다. 애매한 것은 검토로 보낸다.
    if verdict not in VERDICTS:
        verdict = "REJECT"

    return ViewResult(
        view="VERIFY",
        verdict=verdict,
        reason=fmt.clip(fmt.as_text(raw.get("reason")), REASON_MAX) or "(모델이 판정 근거를 반환하지 않음)",
        usage=usage,
    )


def _evidence_window(request: VerifyRequest) -> list[Utterance]:
    """근거 발화와 그 앞뒤 3발화만 남긴다.

    프롬프트에 "앞뒤 3발화만 보라"고 적어도 나머지 발화가 함께 들어가 있으면 모델은
    그것을 읽는다. 관점을 좁히는 유일하게 확실한 방법은 **넘기지 않는 것**이다.
    """
    window = 3
    ids = [u.utterance_id for u in request.utterances]
    if request.tuple.evidence_utterance_id not in ids:
        # 근거 발화가 문맥에 없다 — 좁힐 기준이 없으므로 전부 넘긴다. 이 경우
        # 재현이 안 되면 compare 가 notReproduced 로 잡는다.
        return request.utterances

    center = ids.index(request.tuple.evidence_utterance_id)
    return request.utterances[max(0, center - window) : center + window + 1]


def _match(candidates: list[AssignmentTuple], baseline: AssignmentTuple) -> AssignmentTuple | None:
    """같은 근거 발화에서 나온 tuple 을 짝으로 본다.

    title 로 맞추지 않는 이유: 좁은 시야에서 표현이 조금 달라지는 것은 흔하고, 그걸
    '재현 실패'로 세면 거의 모든 항목이 검토로 넘어가 게이트가 무의미해진다.
    근거 발화는 두 관점이 같아야 하는 유일한 값이다.
    """
    for candidate in candidates:
        if candidate.evidence_utterance_id == baseline.evidence_utterance_id:
            return candidate
    return None


def _verify_variables(request: VerifyRequest) -> dict[str, str]:
    """VERIFY 도 좁은 시야로 본다 — 예전에는 여기만 전체 발화를 실었다.

    프롬프트(l5_verify.v1.txt)가 이렇게 못박고 있다 —

        그 발화(와 바로 앞뒤 문맥)만으로 판단한다.
        회의 전체를 뒤져 근거를 새로 찾아주지 않는다 — 그렇게 하면 검증이 아니라
        두 번째 추출이 된다.

    전체를 넘기면 그 규칙이 부탁이 된다. `_evidence_window` 주석이 이미 답을 적어 두었다 —
    **관점을 좁히는 유일하게 확실한 방법은 넘기지 않는 것**이다. NARROW 는 그렇게 하고
    있었고 VERIFY 만 아니었다.

    ⚠ 두 관점이 같은 발화를 보게 되는 것 아닌가 — 아니다. 좁히는 것은 **문맥의 폭**이고,
    두 관점을 가르는 것은 프롬프트다(NARROW 는 재추출, VERIFY 는 검사). 같은 재료로 다른
    질문을 하는 것이 이 계층의 설계이고, 서로 다른 재료를 보게 하는 것이 아니다.

    덤으로 입력 토큰이 줄어든다. L5 는 실측에서 **회의 입력 토큰의 26%** 였고 그 대부분이
    같은 발화를 관점마다 통째로 다시 싣는 데서 나왔다. 쿼터가 병목일 때 이 차이가 곧
    "같은 한도로 회의를 몇 건 더 돌리는가"가 된다.
    """
    return {
        "TOPIC": request.topic,
        "MEETING_DATE": fmt.format_meeting_date(request.meeting_date),
        "PARTICIPANTS": fmt.format_participants(request.participants),
        "TARGET_TUPLE": _format_target(request.tuple, request.participants),
        "UTTERANCES": fmt.format_utterances(_evidence_window(request)),
    }


def _format_target(target: AssignmentTuple, participants: list[Participant]) -> str:
    names = {p.person_id: p.name for p in participants if p.person_id is not None}
    assignee = (
        f"personId={target.assignee_candidate_person_id}"
        f" ({names.get(target.assignee_candidate_person_id, '이름 미상')})"
        if target.assignee_candidate_person_id is not None
        else "미정"
    )
    return (
        f"- 할 일   : {target.title}\n"
        f"- 담당자  : {assignee}\n"
        f"- 판정근거: {target.assignee_source or 'UNKNOWN'}\n"
        f"- 기한    : {target.due_date or '없음'}\n"
        f"- 근거 발화 id: {target.evidence_utterance_id}"
    )


def _sum_usage(*results: ViewResult) -> Usage:
    """두 관점의 토큰을 합산한다. 한쪽만 세면 L5 가 실제보다 싸 보이고,
    특화 모델 전환의 손익분기점(QLTY-03)이 그만큼 틀어진다."""
    return Usage(
        tokens_in=sum(r.usage.tokens_in for r in results),
        tokens_out=sum(r.usage.tokens_out for r in results),
    )


def _describe(exc: LayerError) -> str:
    """실패 사유는 **오류 코드까지만** 싣는다.

    원문 메시지를 그대로 실으면 제공자 응답 본문이 우리 API 응답에 섞이고, 거기 무엇이
    들어 있을지는 보증할 수 없다(`ProviderAvailability` 가 사유에 상태코드만 담는 것과 같은 이유).
    본문이 필요한 진단은 서버 로그를 본다.
    """
    return exc.code


def _both_views_failed(*results: ViewResult) -> LayerError:
    detail = json.dumps([r.error for r in results], ensure_ascii=False)
    return LayerError(
        LayerErrorKind.TRANSIENT,
        "ALL_VIEWS_FAILED",
        f"두 관점이 모두 실패했습니다 — 검증이 수행되지 않았습니다: {detail}",
    )
