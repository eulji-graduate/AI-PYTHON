"""L2 주제 분할 후처리 검증.

여기서 검증하는 것은 프롬프트 품질이 아니라 **구조적 불변식**이다. 모델이 무엇을
답하든 구간은 겹치지 않고 구멍이 없어야 한다 — 그게 시작점만 받기로 한 이유다.
겹치면 결정이 두 번 저장되고, 구멍이 생기면 그 안의 결정은 영영 뽑히지 않는다.
"""

from app.layers.l2 import build_response_schema, parse_topics
from app.schemas.common import Utterance
from app.schemas.l2 import OVERLAP_UTTERANCES

UTTERANCES = [Utterance(utterance_id=100 + i, speaker_id=7, text=f"발화 {i}") for i in range(10)]


def ids_of(segment) -> list[int]:
    return segment.utterance_ids


class TestResponseSchema:
    def test_시작점만_받고_끝은_받지_않는다(self):
        # 끝까지 모델이 정하면 겹침·구멍을 후처리가 판단으로 메워야 한다.
        properties = build_response_schema(UTTERANCES)["properties"]["topics"]["items"]["properties"]

        assert set(properties) == {"topic", "startUtteranceId"}

    def test_시작점은_전달된_발화만_허용한다(self):
        properties = build_response_schema(UTTERANCES)["properties"]["topics"]["items"]["properties"]

        assert properties["startUtteranceId"]["enum"] == [str(u.utterance_id) for u in UTTERANCES]


class TestParseTopics:
    def test_구간이_겹치지도_비지도_않는다(self):
        raw = {
            "topics": [
                {"topic": "인사", "startUtteranceId": "100"},
                {"topic": "로드맵", "startUtteranceId": "104"},
                {"topic": "배포 일정", "startUtteranceId": "107"},
            ]
        }

        topics = parse_topics(raw, UTTERANCES)

        assert [(t.start_utterance_id, t.end_utterance_id) for t in topics] == [
            (100, 103),
            (104, 106),
            (107, 109),
        ]
        assert [t.topic_seq for t in topics] == [1, 2, 3]

    def test_모델이_순서를_섞어도_정렬한다(self):
        raw = {
            "topics": [
                {"topic": "배포 일정", "startUtteranceId": "107"},
                {"topic": "인사", "startUtteranceId": "100"},
                {"topic": "로드맵", "startUtteranceId": "104"},
            ]
        }

        topics = parse_topics(raw, UTTERANCES)

        assert [t.topic for t in topics] == ["인사", "로드맵", "배포 일정"]

    def test_첫_주제를_첫_발화로_당긴다(self):
        # 모델이 인사·잡담을 건너뛰고 시작점을 잡는 일이 흔한데, 그 구간을 버리면
        # 거기서 나온 결정이 어떤 주제에도 속하지 않게 된다.
        raw = {"topics": [{"topic": "로드맵", "startUtteranceId": "104"}]}

        [topic] = parse_topics(raw, UTTERANCES)

        assert topic.start_utterance_id == 100
        assert topic.end_utterance_id == 109

    def test_같은_발화에서_시작하는_주제는_하나로_합친다(self):
        # 길이 0 인 구간이 생기면 L3 가 발화 없는 주제를 요약하게 된다.
        raw = {
            "topics": [
                {"topic": "로드맵", "startUtteranceId": "104"},
                {"topic": "로드맵(중복)", "startUtteranceId": "104"},
            ]
        }

        topics = parse_topics(raw, UTTERANCES)

        assert len(topics) == 1
        assert topics[0].topic == "로드맵"

    def test_목록_밖_시작점은_버린다(self):
        raw = {
            "topics": [
                {"topic": "로드맵", "startUtteranceId": "104"},
                {"topic": "환각", "startUtteranceId": "9999"},
            ]
        }

        assert [t.topic for t in parse_topics(raw, UTTERANCES)] == ["로드맵"]

    def test_주제명이_비면_버린다(self):
        raw = {"topics": [{"topic": "   ", "startUtteranceId": "104"}]}

        assert parse_topics(raw, UTTERANCES) == []

    def test_아무것도_못_받으면_빈_결과다(self):
        # 여기서 임의로 "전체를 한 주제"로 만들면 분할이 실패한 사실이 감춰진다.
        assert parse_topics({"topics": []}, UTTERANCES) == []

    def test_오버랩_3발화를_앞_주제에서_가져온다(self):
        raw = {
            "topics": [
                {"topic": "인사", "startUtteranceId": "100"},
                {"topic": "로드맵", "startUtteranceId": "104"},
            ]
        }

        _, second = parse_topics(raw, UTTERANCES)

        # 고유 구간은 104~109 지만, L3 가 읽을 발화는 101 부터다.
        assert second.start_utterance_id == 104
        assert ids_of(second)[:OVERLAP_UTTERANCES] == [101, 102, 103]
        assert ids_of(second)[-1] == 109

    def test_첫_주제는_앞이_없으므로_오버랩이_없다(self):
        raw = {
            "topics": [
                {"topic": "인사", "startUtteranceId": "100"},
                {"topic": "로드맵", "startUtteranceId": "104"},
            ]
        }

        first, _ = parse_topics(raw, UTTERANCES)

        assert ids_of(first) == [100, 101, 102, 103]

    def test_모든_발화가_어떤_주제에든_속한다(self):
        raw = {
            "topics": [
                {"topic": "로드맵", "startUtteranceId": "103"},
                {"topic": "배포", "startUtteranceId": "108"},
            ]
        }

        covered = set()
        for topic in parse_topics(raw, UTTERANCES):
            covered.update(range(topic.start_utterance_id, topic.end_utterance_id + 1))

        assert covered == {u.utterance_id for u in UTTERANCES}
