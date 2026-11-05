"""L2 주제 분할 후처리 검증.

여기서 검증하는 것은 프롬프트 품질이 아니라 **구조적 불변식**이다. 모델이 무엇을
답하든 구간은 겹치지 않고 구멍이 없어야 한다 — 그게 시작점만 받기로 한 이유다.
겹치면 결정이 두 번 저장되고, 구멍이 생기면 그 안의 결정은 영영 뽑히지 않는다.
"""

from itertools import pairwise

from app.layers.l2 import build_response_schema, parse_topics
from app.schemas.common import Utterance
from app.schemas.l2 import MIN_TOPIC_UTTERANCES, OVERLAP_UTTERANCES, TOPIC_MAX

UTTERANCES = [Utterance(utterance_id=100 + i, speaker_id=7, text=f"발화 {i}") for i in range(10)]


def utterances_of(count: int, first_id: int = 100) -> list[Utterance]:
    return [Utterance(utterance_id=first_id + i, speaker_id=7, text=f"발화 {i}") for i in range(count)]


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


class TestShortTopicMerge:
    """짧은 구간 병합 — 모델이 두세 발화짜리 곁가지를 새 주제로 자르는 경향을 코드로 막는다.

    ⚠️ 임계값(`MIN_TOPIC_UTTERANCES`)은 샘플 회의 1건으로 정했다. 여기서 고정하는 것은
    그 값이 옳다는 주장이 아니라 **병합 규칙의 동작**이다.
    """

    def test_임계값은_3발화다(self):
        # 값이 바뀌면 아래 경계 테스트의 전제가 달라진다 — 조용히 통과하지 않게 못박는다.
        assert MIN_TOPIC_UTTERANCES == 3

    def test_3발화_미만인_마지막_구간은_앞_주제에_합친다(self):
        raw = {
            "topics": [
                {"topic": "인사", "startUtteranceId": "100"},
                {"topic": "로드맵", "startUtteranceId": "104"},
                {"topic": "배포", "startUtteranceId": "108"},
            ]
        }

        topics = parse_topics(raw, UTTERANCES)

        assert [(t.start_utterance_id, t.end_utterance_id) for t in topics] == [(100, 103), (104, 109)]
        assert [t.topic_seq for t in topics] == [1, 2]
        # 구간이 넓어졌으니 제목도 처음 안건만 말하지 않는다.
        assert [t.topic for t in topics] == ["인사", "로드맵 외 1건"]

    def test_정확히_3발화인_구간은_합치지_않는다(self):
        raw = {
            "topics": [
                {"topic": "인사", "startUtteranceId": "100"},
                {"topic": "로드맵", "startUtteranceId": "104"},
                {"topic": "배포", "startUtteranceId": "107"},
            ]
        }

        topics = parse_topics(raw, UTTERANCES)

        assert [(t.start_utterance_id, t.end_utterance_id) for t in topics] == [
            (100, 103),
            (104, 106),
            (107, 109),
        ]
        assert [t.topic for t in topics] == ["인사", "로드맵", "배포"]

    def test_연속된_짧은_구간은_모두_앞_주제로_합친다(self):
        raw = {
            "topics": [
                {"topic": "인사", "startUtteranceId": "100"},
                {"topic": "가", "startUtteranceId": "106"},
                {"topic": "나", "startUtteranceId": "108"},
                {"topic": "다", "startUtteranceId": "110"},
            ]
        }

        [topic] = parse_topics(raw, utterances_of(12))

        assert (topic.start_utterance_id, topic.end_utterance_id) == (100, 111)
        assert topic.topic == "인사 외 3건"

    def test_합친_뒤에도_구간이_겹치지도_비지도_않는다(self):
        raw = {
            "topics": [
                {"topic": "인사", "startUtteranceId": "100"},
                {"topic": "로드맵", "startUtteranceId": "104"},
                {"topic": "가", "startUtteranceId": "108"},
                {"topic": "나", "startUtteranceId": "110"},
            ]
        }

        topics = parse_topics(raw, utterances_of(12))

        assert topics[0].start_utterance_id == 100
        assert topics[-1].end_utterance_id == 111
        for previous, following in pairwise(topics):
            assert following.start_utterance_id == previous.end_utterance_id + 1

    def test_첫_구간이_짧아도_뒤_주제를_지우지_않는다(self):
        # 첫 주제는 앞이 없어 합칠 곳이 없다. 뒤 주제를 당겨 오면 멀쩡한 주제가 사라진다.
        raw = {
            "topics": [
                {"topic": "인사", "startUtteranceId": "100"},
                {"topic": "로드맵", "startUtteranceId": "102"},
            ]
        }

        topics = parse_topics(raw, UTTERANCES)

        assert [(t.start_utterance_id, t.end_utterance_id) for t in topics] == [(100, 101), (102, 109)]
        assert [t.topic for t in topics] == ["인사", "로드맵"]

    def test_합친_제목은_상한을_넘지_않고_접미사를_잃지_않는다(self):
        raw = {
            "topics": [
                {"topic": "가" * TOPIC_MAX, "startUtteranceId": "100"},
                {"topic": "배포", "startUtteranceId": "108"},
            ]
        }

        [topic] = parse_topics(raw, UTTERANCES)

        assert len(topic.topic) <= TOPIC_MAX
        assert topic.topic.endswith(" 외 1건")

    def test_합친_뒤_오버랩은_합쳐진_경계를_따른다(self):
        raw = {
            "topics": [
                {"topic": "인사", "startUtteranceId": "100"},
                {"topic": "로드맵", "startUtteranceId": "104"},
                {"topic": "배포", "startUtteranceId": "108"},
            ]
        }

        _, second = parse_topics(raw, UTTERANCES)

        assert ids_of(second) == [101, 102, 103, 104, 105, 106, 107, 108, 109]

    def test_샘플_회의에서_관측한_과분할을_정답_구조로_정리한다(self):
        # 2026-11-05 baseline: 정답 4주제(시작 1·9·15·21)를 모델이 24·26 에서 더 잘라 6주제로 답했다.
        # 24~25, 26~27 은 2발화씩이라 앞 주제(21~)에 합쳐진다.
        raw = {
            "topics": [
                {"topic": "앱 개발 일정 및 1차 범위", "startUtteranceId": "1"},
                {"topic": "온보딩 플로우 검토 및 디자인 시안 요청", "startUtteranceId": "9"},
                {"topic": "API 명세 문서화", "startUtteranceId": "15"},
                {"topic": "소셜 로그인 도입 검토", "startUtteranceId": "21"},
                {"topic": "테스트 서버 세팅 담당", "startUtteranceId": "24"},
                {"topic": "푸시 알림 일정", "startUtteranceId": "26"},
            ]
        }

        topics = parse_topics(raw, utterances_of(27, first_id=1))

        assert [(t.start_utterance_id, t.end_utterance_id) for t in topics] == [
            (1, 8),
            (9, 14),
            (15, 20),
            (21, 27),
        ]
        assert topics[-1].topic == "소셜 로그인 도입 검토 외 2건"

    def test_시작점을_빠뜨린_모양은_고치지_못한다(self):
        # 같은 baseline 의 다른 실행: 모델이 21 을 빠뜨리고 [1, 9, 15, 24] 로 답했다.
        # 이 규칙은 "너무 많이 자른 것"만 합친다 — 빠뜨린 시작점은 되살릴 수 없다.
        raw = {
            "topics": [
                {"topic": "앱 개발 일정", "startUtteranceId": "1"},
                {"topic": "온보딩 플로우", "startUtteranceId": "9"},
                {"topic": "API 명세 및 인증 논의", "startUtteranceId": "15"},
                {"topic": "테스트 서버 세팅 및 푸시 알림 논의", "startUtteranceId": "24"},
            ]
        }

        topics = parse_topics(raw, utterances_of(27, first_id=1))

        assert [t.start_utterance_id for t in topics] == [1, 9, 15, 24]
