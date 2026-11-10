"""L2 프롬프트 계약 검증.

여기서 보는 것은 프롬프트의 **품질이 아니라 배선**이다. 품질은 `evals/` 로 재고(실제 모델 호출),
이 테스트는 모델 없이도 깨지는 것 — 파일 이름·자리표시자·렌더링·롤백 경로 — 만 본다.
"""

import json
from pathlib import Path

from app.layers.l2 import SPEC, build_prompt_variables
from app.layers.runner import PROMPT_DIR, load_prompt, render_prompt
from app.schemas.common import Utterance
from app.schemas.l2 import SegmentTopicsRequest

FIXTURES = Path(__file__).resolve().parent.parent / "evals" / "fixtures"


def sample_request() -> SegmentTopicsRequest:
    return SegmentTopicsRequest(
        tenant_id=1,
        meeting_id=1,
        utterances=[Utterance(utterance_id=1, speaker_id=2, text="안녕하세요 오늘 회의를 시작합니다")],
    )


class TestPromptWiring:
    def test_프롬프트_파일_이름이_버전과_일치하고_존재한다(self):
        assert SPEC.prompt_file == f"l2_segment_topics.{SPEC.prompt_version}.txt"
        assert (PROMPT_DIR / SPEC.prompt_file).is_file()

    def test_필수_자리표시자가_각각_한_번씩만_있다(self):
        template = load_prompt(SPEC.prompt_file)

        assert template.count("{{UTTERANCES}}") == 1
        assert template.count("{{FEW_SHOT}}") == 1

    def test_렌더링하면_자리표시자가_남지_않고_발화가_들어간다(self):
        # render_prompt 는 미치환·미사용 변수를 예외로 막는다 — 통과 자체가 배선 검증이다.
        prompt = render_prompt(load_prompt(SPEC.prompt_file), build_prompt_variables(sample_request(), []))

        assert "{{" not in prompt
        assert "안녕하세요 오늘 회의를 시작합니다" in prompt

    def test_이전_버전_프롬프트는_되돌릴_수_있게_남아_있다(self):
        # 전환이 나빴을 때 SPEC 두 줄만 되돌리면 되도록 v1 파일을 지우지 않는다.
        assert (PROMPT_DIR / "l2_segment_topics.v1.txt").is_file()


class TestPromptDoesNotLeakSamples:
    """프롬프트를 평가 샘플에 맞추면 점수만 오르고 일반 성능은 오르지 않는다.

    샘플 회의에만 나오는 낱말이 프롬프트에 들어가면 그건 설명이 아니라 정답 누설이다.
    """

    def test_평가_샘플의_고유_낱말이_프롬프트에_없다(self):
        template = load_prompt(SPEC.prompt_file)
        leaked = []
        for gold_file in sorted(FIXTURES.glob("*.gold.json")):
            for topic in json.loads(gold_file.read_text(encoding="utf-8"))["topics"]:
                # 정답 주제명 중 일반 안건명("미결 논의")은 프롬프트가 일부러 쓰는 말이라 제외한다.
                name = topic["topic"]
                if name != "미결 논의" and name in template:
                    leaked.append(name)

        assert leaked == []

    def test_샘플_발화에만_나오는_구절이_프롬프트에_없다(self):
        template = load_prompt(SPEC.prompt_file)
        phrases = ["소셜 로그인", "카카오", "후기 영상", "면담 시간", "다면평가", "테스트 서버", "푸시 알림"]

        assert [p for p in phrases if p in template] == []
