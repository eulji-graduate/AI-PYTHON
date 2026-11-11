"""L3 프롬프트 계약 검증.

여기서 보는 것은 프롬프트의 **품질이 아니라 배선**이다. 품질은 `evals/`로 재고(실제 모델 호출),
이 테스트는 모델 없이도 깨지는 것 — 파일 이름·자리표시자·렌더링·롤백 경로·샘플 누설 — 만 본다.
"""

import json
from pathlib import Path

from app.layers.l3 import SPEC, build_prompt_variables
from app.layers.runner import PROMPT_DIR, load_prompt, render_prompt
from app.schemas.common import Participant, Utterance
from app.schemas.l3 import SummarizeTopicRequest

FIXTURES = Path(__file__).resolve().parent.parent / "evals" / "fixtures"


def sample_request() -> SummarizeTopicRequest:
    return SummarizeTopicRequest(
        tenant_id=1,
        meeting_id=1,
        topic_seq=1,
        topic="일정 논의",
        utterances=[Utterance(utterance_id=1, speaker_id=2, text="안녕하세요 오늘 회의를 시작합니다")],
        participants=[Participant(person_id=2, name="참석자")],
    )


class TestPromptWiring:
    def test_프롬프트_파일_이름이_버전과_일치하고_존재한다(self):
        assert SPEC.prompt_file == f"l3_summarize_topic.{SPEC.prompt_version}.txt"
        assert (PROMPT_DIR / SPEC.prompt_file).is_file()

    def test_필수_자리표시자가_각각_한_번씩만_있다(self):
        template = load_prompt(SPEC.prompt_file)

        for placeholder in ("{{TOPIC}}", "{{PARTICIPANTS}}", "{{UTTERANCES}}", "{{FEW_SHOT}}"):
            assert template.count(placeholder) == 1, placeholder

    def test_렌더링하면_자리표시자가_남지_않고_발화가_들어간다(self):
        # render_prompt 는 미치환·미사용 변수를 예외로 막는다 — 통과 자체가 배선 검증이다.
        prompt = render_prompt(load_prompt(SPEC.prompt_file), build_prompt_variables(sample_request(), []))

        assert "{{" not in prompt
        assert "안녕하세요 오늘 회의를 시작합니다" in prompt

    def test_v2로_전환돼_있다(self):
        assert SPEC.prompt_version == "v2"

    def test_이전_버전_프롬프트는_되돌릴_수_있게_남아_있다(self):
        # 전환이 나빴을 때 SPEC 두 줄만 되돌리면 되도록 v1 파일을 지우지 않는다.
        assert (PROMPT_DIR / "l3_summarize_topic.v1.txt").is_file()


class TestPromptDoesNotLeakSamples:
    """프롬프트를 평가 샘플에 맞추면 점수만 오르고 일반 성능은 오르지 않는다.

    샘플 회의에만 나오는 낱말·문장이 프롬프트에 들어가면 그건 설명이 아니라 정답 누설이다.
    """

    def test_정답_항목_문장이_프롬프트에_없다(self):
        template = load_prompt(SPEC.prompt_file)
        gold = json.loads((FIXTURES / "demo_meeting.gold.json").read_text(encoding="utf-8"))

        leaked = [item["content"] for item in gold["items"] if item["content"] in template]

        assert leaked == []

    def test_샘플_발화에만_나오는_구절이_프롬프트에_없다(self):
        template = load_prompt(SPEC.prompt_file)
        phrases = [
            "일정표",
            "상품 API",
            "인증 API",
            "소셜 로그인",
            "카카오",
            "테스트 서버",
            "푸시 알림",
            "강서연",
            "와이어프레임",
            "후기 영상",
            "면담 시간",
            "다면평가",
        ]

        assert [p for p in phrases if p in template] == []

    def test_샘플_회의의_발화_원문이_프롬프트에_없다(self):
        template = load_prompt(SPEC.prompt_file)
        sentences = []
        for name in ("demo_meeting", "heldout_marketing", "heldout_hr"):
            meeting = json.loads((FIXTURES / f"{name}.json").read_text(encoding="utf-8"))
            sentences += [u["text"] for u in meeting["utterances"]]

        assert [s for s in sentences if s in template] == []
