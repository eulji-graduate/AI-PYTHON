"""L3.5 게이트 후처리 검증.

게이트에서 가장 비싼 실패는 **합의되지 않은 일이 확정으로 올라가는 것**이고,
그 경로는 오판정만이 아니라 **누락**으로도 열린다. 그래서 이 파일의 절반은
"모델이 답하지 않았을 때 무엇이 되는가"를 본다.
"""

from app.layers.l3_5 import DISCUSSED, MISSING_VERDICT_REASON, parse_verdicts
from app.layers.l3_5 import build_response_schema as gate_schema
from app.schemas.l3_5 import GateCandidate

ITEMS = [
    GateCandidate(
        item_key="1", item_type="DECISION", content="배포를 다음 스프린트로 미룸", evidence_utterance_id=200
    ),
    GateCandidate(item_key="2", item_type="DISCUSSION", content="로그 수집 방식", evidence_utterance_id=202),
]


class TestGateSchema:
    def test_판정_대상은_요청_항목으로_닫힌다(self):
        properties = gate_schema(ITEMS)["properties"]["verdicts"]["items"]["properties"]

        assert properties["itemKey"]["enum"] == ["1", "2"]
        assert properties["gateStatus"]["enum"] == ["CONFIRMED", "DISCUSSED"]

    def test_확신도_필드가_없다(self):
        properties = gate_schema(ITEMS)["properties"]["verdicts"]["items"]["properties"]

        assert "confidence" not in properties


class TestGateParseVerdicts:
    def test_요청_항목_수만큼_순서대로_반환한다(self):
        raw = {
            "verdicts": [
                {"itemKey": "2", "gateStatus": "DISCUSSED", "reason": "누가 할지 미정"},
                {"itemKey": "1", "gateStatus": "CONFIRMED", "reason": "발화 201에서 합의"},
            ]
        }

        verdicts = parse_verdicts(raw, ITEMS)

        assert [v.item_key for v in verdicts] == ["1", "2"]

    def test_판정이_없는_항목은_DISCUSSED다(self):
        # precision 우선의 실체. 누락이 통과가 되면 응답이 잘린 것만으로
        # 합의되지 않은 일이 배정으로 나간다.
        raw = {"verdicts": [{"itemKey": "1", "gateStatus": "CONFIRMED", "reason": "합의됨"}]}

        _, second = parse_verdicts(raw, ITEMS)

        assert second.item_key == "2"
        assert second.gate_status == DISCUSSED
        assert second.reason == MISSING_VERDICT_REASON

    def test_응답이_통째로_비어도_확정이_생기지_않는다(self):
        verdicts = parse_verdicts({"verdicts": []}, ITEMS)

        assert len(verdicts) == 2
        assert all(v.gate_status == DISCUSSED for v in verdicts)

    def test_알_수_없는_판정값은_DISCUSSED로_떨어진다(self):
        raw = {"verdicts": [{"itemKey": "1", "gateStatus": "PROBABLY", "reason": "음"}]}

        first, _ = parse_verdicts(raw, ITEMS)

        assert first.gate_status == DISCUSSED

    def test_목록_밖_항목에_대한_판정은_버린다(self):
        raw = {
            "verdicts": [
                {"itemKey": "999", "gateStatus": "CONFIRMED", "reason": "없는 항목"},
                {"itemKey": "1", "gateStatus": "CONFIRMED", "reason": "합의됨"},
            ]
        }

        verdicts = parse_verdicts(raw, ITEMS)

        assert [v.item_key for v in verdicts] == ["1", "2"]

    def test_같은_항목을_두_번_판정하면_먼저_온_것을_쓴다(self):
        raw = {
            "verdicts": [
                {"itemKey": "1", "gateStatus": "DISCUSSED", "reason": "먼저"},
                {"itemKey": "1", "gateStatus": "CONFIRMED", "reason": "나중"},
            ]
        }

        first, _ = parse_verdicts(raw, ITEMS)

        assert first.gate_status == DISCUSSED
