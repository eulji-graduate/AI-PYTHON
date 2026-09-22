"""OVERVIEW 회의 개요 — 입력 조립과 빈 입력 처리.

이 계층의 위험은 산출물이 틀리는 것이 아니라 **입력이 잘못 들어오는 것**이다.
발화를 받으면 주제 요약과 다른 말을 하는 개요가 나오고, 자기 출력을 되먹이면 돌릴수록
내용이 사라진다. 그래서 여기서 보는 것은 "무엇을 프롬프트에 싣는가"다.
"""

import pytest

from app.layers.overview import build_prompt_variables, build_response_schema, format_topics
from app.schemas.common import Participant
from app.schemas.overview import DigestItem, MeetingTopicDigest, SummarizeMeetingRequest

TOPICS = [
    MeetingTopicDigest(
        topic_seq=2,
        topic="인증 리팩터링",
        items=[DigestItem(item_type="BLOCKER", content="보안팀 검토 대기")],
    ),
    MeetingTopicDigest(
        topic_seq=1,
        topic="배포 일정",
        items=[
            DigestItem(item_type="DECISION", content="배포를 다음 스프린트로 미룸"),
            DigestItem(item_type="DISCUSSION", content="롤백 절차는 더 논의"),
        ],
    ),
]


class TestFormatTopics:
    def test_진행_순서대로_적는다(self):
        """topicSeq 순으로 싣는다. 섞으면 "먼저 A 그다음 B" 같은 흐름이 나오지 않는다."""
        text = format_topics(TOPICS)

        assert text.index("배포 일정") < text.index("인증 리팩터링")

    def test_항목_종류를_함께_적는다(self):
        """결정과 블로커가 같은 무게로 읽히면 "정해졌다"와 "막혀 있다"가 한 문장에 섞인다."""
        text = format_topics(TOPICS)

        assert "[DECISION] 배포를 다음 스프린트로 미룸" in text
        assert "[BLOCKER] 보안팀 검토 대기" in text
        assert "[DISCUSSION] 롤백 절차는 더 논의" in text


class TestPromptVariables:
    def test_발화를_싣지_않는다(self):
        """이 계층의 계약이다 — 전사를 다시 읽히면 주제 요약과 모순되는 개요가 나온다."""
        request = SummarizeMeetingRequest(
            tenant_id=1,
            meeting_id=2,
            topics=TOPICS,
            participants=[Participant(person_id=7, name="김서준")],
        )

        variables = build_prompt_variables(request)

        assert set(variables) == {"PARTICIPANTS", "TOPICS"}


class TestResponseSchema:
    def test_개요_문장_하나만_요구한다(self):
        """항목 enum 이 없다 — 개요는 특정 발화나 항목을 가리키지 않는다."""
        schema = build_response_schema()

        assert schema["required"] == ["overview"]
        assert set(schema["properties"]) == {"overview"}


class TestEmptyInput:
    @pytest.mark.asyncio
    async def test_주제가_없으면_모델을_부르지_않는다(self):
        """확정된 것이 없는 회의다. 부르면 빈 입력에 돈만 쓰고 빈 개요가 돌아온다."""

        class NeverCalledRunner:
            model_name = "gemini-flash"

            async def run(self, *args, **kwargs):  # pragma: no cover - 불리면 테스트 실패다
                raise AssertionError("주제가 없으면 모델을 부르지 않아야 한다")

        from app.layers.overview import summarize_meeting

        request = SummarizeMeetingRequest(tenant_id=1, meeting_id=2, topics=[], participants=[])

        response = await summarize_meeting(request, NeverCalledRunner())

        # 빈 문자열이 정상 응답이다. Spring 이 빈 값으로는 개요를 덮지 않는다.
        assert response.overview == ""
        assert response.model == "gemini-flash"
