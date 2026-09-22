"""L1.5 응답 스키마 강제와 후처리 검증.

여기가 뚫리면 존재하지 않는 지시어에 대한 해소가 뒤 계층으로 흘러가고, L4 가 그걸
근거로 담당자를 정한다 — 검토 사유 `WRONG_ASSIGNEE` 가 L1.5 로 귀속되는 경로다.
"""

from app.layers.l1_5 import build_response_schema, parse_references, targets
from app.schemas.common import UNKNOWN_PERSON, Participant, Utterance
from app.schemas.l1_5 import ResolveReferenceRequest

PARTICIPANTS = [
    Participant(person_id=7, name="김서준"),
    Participant(person_id=42, name="이태연"),
    Participant(person_id=None, name="명단 외"),
]
UTTERANCES = [
    Utterance(utterance_id=100, speaker_id=7, text="A/B 테스트 도구 후보를 세 개 추렸어요"),
    Utterance(utterance_id=101, speaker_id=42, text="그거 제가 정리해서 공유할게요"),
    Utterance(utterance_id=102, speaker_id=None, text="네 좋습니다"),
]


def request(**overrides) -> ResolveReferenceRequest:
    body = {
        "tenant_id": 7,
        "meeting_id": 500,
        "utterances": UTTERANCES,
        "participants": PARTICIPANTS,
    }
    body.update(overrides)
    return ResolveReferenceRequest(**body)


class TestTargets:
    def test_대상을_지정하면_그것만_본다(self):
        assert [u.utterance_id for u in targets(request(target_utterance_ids=[101]))] == [101]

    def test_비워_보내면_전체가_대상이다(self):
        # 좁힐 근거를 안 준 호출에서 전부 버리면 조용히 빈 결과가 되고,
        # 미구현이 '지시어 0건'으로 위장된다.
        assert len(targets(request())) == 3

    def test_지정한_id가_발화에_하나도_없으면_전체로_되돌린다(self):
        assert len(targets(request(target_utterance_ids=[999]))) == 3


class TestResponseSchema:
    def test_지시어_발화와_선행사_발화의_목록이_다르다(self):
        # 선행사는 대상 밖 앞 발화에 있는 것이 정상이다. 근거 enum 을 대상으로 좁히면
        # 정답을 고를 방법이 없어진다.
        schema = build_response_schema(PARTICIPANTS, [UTTERANCES[1]], UTTERANCES)
        properties = schema["properties"]["references"]["items"]["properties"]

        assert properties["utteranceId"]["enum"] == ["101"]
        assert properties["evidenceUtteranceId"]["enum"] == ["100", "101", "102"]

    def test_대상_인물은_참석자와_탈출구만_허용한다(self):
        schema = build_response_schema(PARTICIPANTS, UTTERANCES, UTTERANCES)
        enum = schema["properties"]["references"]["items"]["properties"]["resolvedPersonId"]["enum"]

        assert enum == ["7", "42", UNKNOWN_PERSON]

    def test_확신도_필드가_없다(self):
        # 자기보고 신뢰도는 85~95 에 몰리고 실제 정확도와 맞지 않는다(명세).
        schema = build_response_schema(PARTICIPANTS, UTTERANCES, UTTERANCES)
        properties = schema["properties"]["references"]["items"]["properties"]

        assert "confidence" not in properties


class TestParseReferences:
    def test_정상_항목을_통과시킨다(self):
        raw = {
            "references": [
                {
                    "utteranceId": "101",
                    "surface": "그거",
                    "referenceType": "ARTIFACT",
                    "resolvedText": "A/B 테스트 도구 후보",
                    "evidenceUtteranceId": "100",
                }
            ]
        }

        [reference] = parse_references(raw, request(), UTTERANCES)

        assert reference.utterance_id == 101
        assert reference.surface == "그거"
        assert reference.reference_type == "ARTIFACT"
        assert reference.resolved_text == "A/B 테스트 도구 후보"
        assert reference.evidence_utterance_id == 100

    def test_발화에_없는_지시_표현은_버린다(self):
        # 지시 표현은 발화에서 그대로 인용할 수 있는 값이다. 없는 표현이 돌아왔다면
        # 모델이 만들어낸 것이고, 근거 강제를 문자열 수준까지 밀어붙일 수 있는 자리다.
        raw = {
            "references": [
                {
                    "utteranceId": "101",
                    "surface": "저 문서",  # 101 번 발화에 없는 말
                    "referenceType": "ARTIFACT",
                    "evidenceUtteranceId": "100",
                }
            ]
        }

        assert parse_references(raw, request(), UTTERANCES) == []

    def test_대상_밖_발화의_해소는_버린다(self):
        raw = {
            "references": [
                {
                    "utteranceId": "100",
                    "surface": "세 개",
                    "referenceType": "ARTIFACT",
                    "evidenceUtteranceId": "100",
                }
            ]
        }

        assert parse_references(raw, request(target_utterance_ids=[101]), [UTTERANCES[1]]) == []

    def test_목록_밖_선행사는_버린다(self):
        raw = {
            "references": [
                {
                    "utteranceId": "101",
                    "surface": "그거",
                    "referenceType": "ARTIFACT",
                    "evidenceUtteranceId": "9999",
                }
            ]
        }

        assert parse_references(raw, request(), UTTERANCES) == []

    def test_알_수_없는_종류는_UNRESOLVED로_내린다(self):
        raw = {
            "references": [
                {
                    "utteranceId": "101",
                    "surface": "그거",
                    "referenceType": "MYSTERY",
                    "resolvedText": "무언가",
                    "evidenceUtteranceId": "100",
                }
            ]
        }

        [reference] = parse_references(raw, request(), UTTERANCES)

        assert reference.reference_type == "UNRESOLVED"
        # 해소되지 않았는데 대상 표현이 남아 있으면 뒤 계층이 그걸 답으로 읽는다.
        assert reference.resolved_text is None

    def test_담당자를_찾았으면_대상_표현을_비운다(self):
        # personId 가 답인데 자유 텍스트가 함께 오면 한 항목이 두 대상을 가리키게 되고,
        # 둘이 어긋나도 소비자는 알 수 없다(CodeRabbit PR #4 지적).
        raw = {
            "references": [
                {
                    "utteranceId": "101",
                    "surface": "제가",
                    "referenceType": "PERSON",
                    "resolvedPersonId": "42",
                    "resolvedText": "김서준",  # personId=42 는 이태연이다 — 어긋난 값
                    "evidenceUtteranceId": "101",
                }
            ]
        }

        [reference] = parse_references(raw, request(), UTTERANCES)

        assert reference.resolved_person_id == 42
        assert reference.resolved_text is None

    def test_명단_밖_사람이면_이름을_남긴다(self):
        # "그분 = 김민섭 팀장(명단 밖)" 에서 그 이름이 유일한 단서다. 비우면 사람이
        # 검토할 때 아무 정보가 없다. L4 는 personId 로만 담당자를 정하므로 오배정 경로가 아니다.
        raw = {
            "references": [
                {
                    "utteranceId": "101",
                    "surface": "그거",
                    "referenceType": "PERSON",
                    "resolvedPersonId": UNKNOWN_PERSON,
                    "resolvedText": "김민섭 팀장",
                    "evidenceUtteranceId": "100",
                }
            ]
        }

        [reference] = parse_references(raw, request(), UTTERANCES)

        assert reference.resolved_person_id is None
        assert reference.resolved_text == "김민섭 팀장"

    def test_명단_밖_인물은_None이지만_PERSON은_유지한다(self):
        # "사람을 가리키지만 누군지 모른다"는 그 자체로 쓸모 있는 답이다 —
        # L4 가 이걸 담당자로 쓰면 안 된다는 뜻이 된다.
        raw = {
            "references": [
                {
                    "utteranceId": "101",
                    "surface": "제가",
                    "referenceType": "PERSON",
                    "resolvedPersonId": UNKNOWN_PERSON,
                    "evidenceUtteranceId": "101",
                }
            ]
        }

        [reference] = parse_references(raw, request(), UTTERANCES)

        assert reference.reference_type == "PERSON"
        assert reference.resolved_person_id is None

    def test_사람이_아닌_지시어에_붙은_personId는_떨어뜨린다(self):
        raw = {
            "references": [
                {
                    "utteranceId": "101",
                    "surface": "그거",
                    "referenceType": "ARTIFACT",
                    "resolvedPersonId": "7",
                    "evidenceUtteranceId": "100",
                }
            ]
        }

        [reference] = parse_references(raw, request(), UTTERANCES)

        assert reference.resolved_person_id is None

    def test_같은_발화의_같은_표현은_한_번만_남긴다(self):
        entry = {
            "utteranceId": "101",
            "surface": "그거",
            "referenceType": "ARTIFACT",
            "evidenceUtteranceId": "100",
        }

        assert len(parse_references({"references": [entry, dict(entry)]}, request(), UTTERANCES)) == 1

    def test_탈출구_문자열이_대상_표현으로_새지_않는다(self):
        raw = {
            "references": [
                {
                    "utteranceId": "101",
                    "surface": "그거",
                    "referenceType": "PERSON",
                    "resolvedText": UNKNOWN_PERSON,
                    "evidenceUtteranceId": "100",
                }
            ]
        }

        [reference] = parse_references(raw, request(), UTTERANCES)

        assert reference.resolved_text is None
