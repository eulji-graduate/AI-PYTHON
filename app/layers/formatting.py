"""계층 공통 — 프롬프트에 넣을 문자열 만들기 · 모델이 돌려준 값 되돌리기.

`runner.py` 가 호출·재시도·토큰 집계를 한 곳에 둔 것과 같은 이유다. 계층이 6개인데
참석자 목록 포맷이나 `unknown_person` 되돌리기를 계층마다 복제하면 같은 버그를
여섯 번 고치게 되고, 실제로는 여섯 번째를 빠뜨린다.

계층이 각자 갖는 것은 여전히 셋뿐이다 — 프롬프트 파일 · 응답 스키마 · 후처리.
"""

from __future__ import annotations

import json
from datetime import date, timedelta

from app.schemas.common import UNKNOWN_PERSON, FewShotExample, Participant, Utterance

NONE_MARK = "- (없음)"


def person_enum(participants: list[Participant]) -> list[str]:
    """참석자 personId + 탈출구. 구조화 출력의 enum 은 **문자열만** 지원하므로 문자열이다."""
    values = [str(p.person_id) for p in participants if p.person_id is not None]
    values.append(UNKNOWN_PERSON)
    return values


def utterance_enum(utterances: list[Utterance]) -> list[str]:
    return [str(u.utterance_id) for u in utterances]


def as_int(value: object) -> int | None:
    """`unknown_person` 같은 탈출구 문자열은 None 이 된다."""
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None


def as_iso_date(value: object) -> str | None:
    """형식이 어긋나면 None. 기권 우선 — 틀린 기한은 잘못된 마감으로 보드에 꽂힌다."""
    if not value:
        return None
    text = str(value).strip()
    if not text or text.lower() in ("null", "none", "unknown"):
        return None
    try:
        return date.fromisoformat(text).isoformat()
    except ValueError:
        return None


def as_text(value: object) -> str:
    return str(value or "").strip()


def clip(text: str, limit: int) -> str:
    """자유 텍스트 필드의 상한. 모델이 문단을 통째로 넣으면 DB 컬럼(VARCHAR)에서 잘리고,
    잘린 위치가 어디인지 아무도 모른 채 저장된다 — 자를 거면 우리가 자른다."""
    text = text.strip()
    return text if len(text) <= limit else text[:limit].rstrip()


def allowed_person_ids(participants: list[Participant]) -> set[int]:
    return {p.person_id for p in participants if p.person_id is not None}


def allowed_utterance_ids(utterances: list[Utterance]) -> set[int]:
    return {u.utterance_id for u in utterances}


def resolve_person(value: object, allowed: set[int]) -> int | None:
    """명단 밖 값은 `unknown_person` 과 같게 취급한다 — 스키마 enum 을 걸었어도 한 번 더 본다.

    구조화 출력은 제공자 구현에 의존하고, 그 구현이 흔들리는 순간 조용히 오염된
    담당자가 액션 테이블까지 흘러간다. 여기서 걸러내면 최악이 '항목 누락'이다.
    """
    person_id = as_int(value)
    return person_id if person_id in allowed else None


def format_participants(participants: list[Participant]) -> str:
    lines = []
    for person in participants:
        pid = UNKNOWN_PERSON if person.person_id is None else str(person.person_id)
        lines.append(f"- personId={pid} · {person.name}")
    return "\n".join(lines) or NONE_MARK


WEEKDAYS = "월화수목금토일"


def format_meeting_date(meeting_date: date | None) -> str:
    """기준일 + 이번 주·다음 주 달력.

    <h2>요일 산수를 모델에게 시키지 않는다</h2>
    ISO 날짜만 주면 모델이 ① 그 날의 요일을 알아내고 ② "다음 주 화요일"이 며칠인지 세야 한다.
    둘 다 자주 틀린다 — 실측에서 2026-08-14(목) 기준 "다음 주 화요일까지"를 **2026-08-28(금)**
    으로 냈다. 열흘이 밀렸고 요일도 안 맞았다.

    달력을 코드가 만들어 넘기면 그 계산이 **찾아보기**가 된다. 날짜 산수는 파이썬이 틀릴 수
    없는 일이고, 모델이 틀릴 수 있는 일이다 — 그러면 파이썬이 한다(같은 판단: L2 가 구간을
    코드로 계산하는 것, l2.py).

    ⚠ 프롬프트 파일은 그대로지만 **모델이 보는 값이 바뀐다.** 기한 정확도를 이 변경 전후로
    비교할 때 prompt_version 만 보면 같은 버전으로 보이므로, 지표를 되짚을 때 이 함수의
    변경 이력을 함께 봐야 한다.
    """
    if meeting_date is None:
        # 프롬프트의 "기준일이 없으면 상대 표현을 계산하지 말라"와 짝이다.
        return "(제공되지 않음 — 상대적 기한 표현은 계산하지 말고 dueDate 를 null 로 둔다)"

    monday = meeting_date - timedelta(days=meeting_date.weekday())
    lines = [f"{meeting_date.isoformat()} ({WEEKDAYS[meeting_date.weekday()]})", ""]
    for offset, label in ((0, "이번 주"), (7, "다음 주")):
        days = [monday + timedelta(days=offset + i) for i in range(7)]
        formatted = " · ".join(f"{WEEKDAYS[d.weekday()]} {d.isoformat()}" for d in days)
        lines.append(f"{label}: {formatted}")
    return "\n".join(lines)


EVIDENCE_MARK = "   ← 근거 지정 가능"
TARGET_MARK = "   ← 해소 대상"


def format_utterances(
    utterances: list[Utterance],
    eligible_ids: set[int] | None = None,
    marker: str = EVIDENCE_MARK,
) -> str:
    """문맥은 전부 보여주고 표시로만 대상을 좁힌다.

    후보만 보여주면 앞뒤 맥락이 사라져 "그거"·"아까 그건" 같은 지시어를 해석할 수 없다.
    `eligible_ids` 가 None 이면 표시를 붙이지 않는다(전부 대상인 계층).

    표시 문구는 계층마다 다르다 — L4 는 "근거로 쓸 수 있는 발화"를, L1.5 는
    "지시어를 찾을 발화"를 좁힌다. 프롬프트 문구와 여기 값이 어긋나면 모델이 표시를
    무시하게 되므로 상수로 두고 프롬프트와 같은 말을 쓴다.
    """
    lines = []
    for utterance in utterances:
        # 화자 미정을 숨기지 않는다. 1인칭 발화의 화자가 미정이면 담당자도 미정이어야 한다.
        speaker = "미정" if utterance.speaker_id is None else f"personId={utterance.speaker_id}"
        line = f"- id={utterance.utterance_id} · 화자 {speaker} · {utterance.text}"
        if eligible_ids is not None and utterance.utterance_id in eligible_ids:
            line += marker
        lines.append(line)
    return "\n".join(lines) or NONE_MARK


def format_few_shot(examples: list[FewShotExample]) -> str:
    if not examples:
        return "- (없음 — 축적 전이다. 예시 없이 규칙만으로 판단한다)"
    lines = []
    for example in examples:
        payload = json.dumps(example.payload, ensure_ascii=False)
        lines.append(f'- 발화: "{example.input_text}"\n  확정 결과: {payload}')
    return "\n".join(lines)
