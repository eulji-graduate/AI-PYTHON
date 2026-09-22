"""L3 주제별 정리 후처리 검증.

L3 는 "무엇이 오갔나"까지만 답한다. "그게 확정인가"는 L3.5 의 일이고, 둘을 한
프롬프트에 섞지 않았는지도 여기서 본다 — 섞이는 순간 게이트를 따로 둔 이유가 사라진다.
"""

from app.layers.l3 import build_response_schema as l3_schema
from app.layers.l3 import parse_items
from app.schemas.common import Utterance

UTTERANCES = [
    Utterance(utterance_id=200, speaker_id=7, text="배포는 다음 스프린트로 미루죠"),
    Utterance(utterance_id=201, speaker_id=42, text="네 그렇게 가시죠"),
    Utterance(utterance_id=202, speaker_id=7, text="로그 수집은 어떻게 할까요?"),
]


class TestL3ParseItems:
    def test_정상_항목을_통과시킨다(self):
        raw = {
            "items": [
                {
                    "itemType": "DECISION",
                    "content": "배포를 다음 스프린트로 미룸",
                    "reason": "발화 201에서 '그렇게 가시죠'로 합의됨",
                    "evidenceUtteranceId": "200",
                }
            ]
        }

        [item] = parse_items(raw, UTTERANCES)

        assert item.item_type == "DECISION"
        assert item.evidence_utterance_id == 200

    def test_목록_밖_근거는_버린다(self):
        raw = {
            "items": [{"itemType": "DECISION", "content": "x", "reason": "y", "evidenceUtteranceId": "9999"}]
        }

        assert parse_items(raw, UTTERANCES) == []

    def test_알_수_없는_분류는_버리지_않고_DISCUSSION으로_내린다(self):
        # 버리면 오간 내용이 통째로 사라지지만, 내리면 최악이 '사람이 한 번 더 보는 것'이다.
        raw = {
            "items": [
                {
                    "itemType": "MYSTERY",
                    "content": "배포 미룸",
                    "reason": "근거",
                    "evidenceUtteranceId": "200",
                }
            ]
        }

        [item] = parse_items(raw, UTTERANCES)

        assert item.item_type == "DISCUSSION"

    def test_분류_근거가_비면_비어_있음을_감추지_않는다(self):
        raw = {
            "items": [
                {"itemType": "DECISION", "content": "배포 미룸", "reason": "", "evidenceUtteranceId": "200"}
            ]
        }

        [item] = parse_items(raw, UTTERANCES)

        assert "반환하지 않음" in item.reason

    def test_같은_내용과_근거는_한_번만_남긴다(self):
        entry = {
            "itemType": "DECISION",
            "content": "배포 미룸",
            "reason": "근거",
            "evidenceUtteranceId": "200",
        }

        assert len(parse_items({"items": [entry, dict(entry)]}, UTTERANCES)) == 1

    def test_게이트_판정을_이_계층이_하지_않는다(self):
        # 정리와 게이트를 한 프롬프트에 섞으면 L3.5 를 따로 둔 이유가 사라진다.
        properties = l3_schema(UTTERANCES)["properties"]["items"]["items"]["properties"]

        assert "gateStatus" not in properties
