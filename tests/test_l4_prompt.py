"""L4 프롬프트 렌더링.

프롬프트에 `{{NEW_VAR}}` 를 추가하고 배선을 잊는 사고를 여기서 잡는다. 그대로 나가면
모델에게 "{{NEW_VAR}}" 라는 문자열을 근거로 답하라고 시키는 셈이 되고, 결과는
그럴듯하게 나와서 사람이 눈치채지 못한다.
"""

import pytest

from app.errors import LayerError
from app.layers.l4 import SPEC, build_prompt_variables
from app.layers.runner import load_prompt, render_prompt
from app.schemas.common import FewShotExample
from app.schemas.l4 import ExtractTuplesRequest

REQUEST = ExtractTuplesRequest.model_validate(
    {
        "tenantId": 7,
        "meetingId": 500,
        "topic": "제품 로드맵",
        "items": [
            {
                "itemType": "DECISION",
                "gateStatus": "CONFIRMED",
                "content": "온보딩 플로우를 3단계로 축소",
            }
        ],
        "utterances": [
            {"utteranceId": 8812, "speakerId": 7, "startMs": 1284000, "text": "서준님이 정리해주세요"},
            {"utteranceId": 8813, "speakerId": None, "startMs": 1286000, "text": "그럼 그렇게 가시죠"},
        ],
        "participants": [{"personId": 7, "name": "김서준"}, {"personId": None, "name": "명단 외"}],
        "queryText": "서준님이 정리해주세요",
        "meetingDate": "2026-08-05",
        "view": "EXTRACT",
    }
)


def _render(request=REQUEST, examples=None):
    return render_prompt(load_prompt(SPEC.prompt_file), build_prompt_variables(request, examples or []))


def test_자리표시자가_전부_치환된다():
    rendered = _render()

    assert "{{" not in rendered


def test_참석자와_발화가_프롬프트에_실린다():
    rendered = _render()

    assert "personId=7 · 김서준" in rendered
    assert "personId=unknown_person · 명단 외" in rendered
    assert "id=8812" in rendered
    # 화자 미정을 숨기지 않는다 — 1인칭 발화의 화자가 미정이면 담당자도 미정이어야 한다.
    assert "화자 미정" in rendered


def test_예시가_없으면_없다고_명시한다():
    rendered = _render(examples=[])

    # 빈칸으로 두면 모델이 앞 문맥을 예시로 오독한다.
    assert "(없음" in rendered


def test_예시가_있으면_발화와_확정결과가_함께_들어간다():
    example = FewShotExample(
        input_text="이 부분은 제가 금요일까지 하겠습니다",
        payload={"title": "리팩터링", "assigneeCandidatePersonId": 42},
    )

    rendered = _render(examples=[example])

    assert "이 부분은 제가 금요일까지 하겠습니다" in rendered
    assert "리팩터링" in rendered


def test_NARROW_관점은_시야를_좁히라고_지시한다():
    narrow = REQUEST.model_copy(update={"view": "EXTRACT_NARROW"})

    rendered = _render(narrow)

    assert "앞뒤 3발화" in rendered


def test_치환되지_않은_자리표시자는_즉시_실패한다():
    with pytest.raises(LayerError) as exc:
        render_prompt("주제는 {{TOPIC}} 이고 참석자는 {{PARTICIPANTS}}", {"TOPIC": "x"})

    assert exc.value.code == "PROMPT_UNRESOLVED"
    assert not exc.value.retryable


def test_프롬프트가_안_쓰는_변수도_실패로_잡는다():
    # 프롬프트에서 자리표시자를 지웠는데 값은 계속 만들고 있는 경우.
    # 조용히 두면 그 정보가 모델에 안 가는데도 코드만 보면 가는 것처럼 읽힌다.
    with pytest.raises(LayerError) as exc:
        render_prompt("주제는 {{TOPIC}}", {"TOPIC": "x", "MEETING_DATE": "2026-08-05"})

    assert exc.value.code == "PROMPT_UNUSED_VARIABLE"


def test_회의_날짜가_프롬프트에_실린다():
    rendered = _render()

    assert "2026-08-05" in rendered


def test_회의_날짜가_없으면_계산하지_말라고_지시한다():
    no_date = REQUEST.model_copy(update={"meeting_date": None})

    rendered = _render(no_date)

    # 기준일 없이 "다음 주까지"를 계산하면 그럴듯하게 틀린 마감이 보드에 꽂힌다.
    assert "제공되지 않음" in rendered
    assert "null 로 둔다" in rendered


def test_회의_내용에_중괄호가_있어도_실패하지_않는다():
    # 참석자가 "{{UTTERANCES}} 라고 적어주세요" 라고 말하는 회의도 있을 수 있다.
    tricky = REQUEST.model_copy(update={"topic": "템플릿 {{UTTERANCES}} 정리"})

    rendered = _render(tricky)

    # 값으로 들어온 토큰은 다시 훑지 않으므로 그대로 남는다(오탐 없음).
    assert "템플릿 {{UTTERANCES}} 정리" in rendered


def test_값_안의_자리표시자가_다시_치환되지_않는다():
    rendered = render_prompt(
        "주제 {{TOPIC}} · 발화 {{UTTERANCES}}",
        {"TOPIC": "{{UTTERANCES}}", "UTTERANCES": "실제 발화"},
    )

    # TOPIC 에 들어온 "{{UTTERANCES}}" 가 실제 발화로 바뀌면 회의 내용이 프롬프트
    # 구조를 바꾸는 주입 경로가 된다.
    assert rendered == "주제 {{UTTERANCES}} · 발화 실제 발화"
