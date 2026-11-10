"""반복 측정 틀(`evals/`) 테스트 — 채점 함수, 픽스처 정합성, 요청 스키마, 전체 흐름.

⚠️ 요청 payload를 **실제 요청 스키마**(`app/schemas`)에 통과시켜 본다. 측정 스크립트는 서버를 띄워야만
   돌아서, 스키마가 바뀌어도 아무도 모른 채 스크립트만 조용히 422를 받게 된다 — 그걸 여기서 먼저 잡는다.
"""

import json
from datetime import date
from itertools import pairwise

import pytest

from app.schemas.l1_5 import ResolveReferenceRequest
from app.schemas.l2 import OVERLAP_UTTERANCES, SegmentTopicsRequest
from app.schemas.l3 import SummarizeTopicRequest
from app.schemas.l3_5 import GateRequest
from app.schemas.l4 import ExtractTuplesRequest
from app.schemas.l5 import VerifyRequest
from app.schemas.overview import SummarizeMeetingRequest
from evals import run_eval, scoring

FIXTURES = run_eval.FIXTURES


@pytest.fixture(scope="module")
def meeting() -> dict:
    return json.loads((FIXTURES / "demo_meeting.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def gold() -> dict:
    return json.loads((FIXTURES / "demo_meeting.gold.json").read_text(encoding="utf-8"))


# ── 픽스처 정합성 ────────────────────────────────────────────────────────────


def test_회의일은_월요일이고_발화_번호는_1부터_이어진다(meeting):
    assert date.fromisoformat(meeting["meetingDate"]).weekday() == 0
    ids = [u["utteranceId"] for u in meeting["utterances"]]
    assert ids == list(range(1, len(ids) + 1))


def test_주제_구간은_빈틈과_겹침_없이_발화_전체를_덮는다(meeting, gold):
    total = len(meeting["utterances"])
    topics = sorted(gold["topics"], key=lambda t: t["startUtteranceId"])
    assert topics[0]["startUtteranceId"] == 1
    assert topics[-1]["endUtteranceId"] == total
    for previous, current in pairwise(topics):
        assert current["startUtteranceId"] == previous["endUtteranceId"] + 1


def test_정답_배정은_확정_항목의_근거_발화와_같다(gold):
    confirmed = {i["evidenceUtteranceId"] for i in gold["items"] if i["gateStatus"] == "CONFIRMED"}
    discussed = {i["evidenceUtteranceId"] for i in gold["items"] if i["gateStatus"] == "DISCUSSED"}
    for target in gold["tuples"]:
        assert target["evidenceUtteranceId"] in confirmed
        assert target["evidenceUtteranceId"] not in discussed


def test_정답_배정의_근거는_그_주제_안에_있고_담당자는_참석자다(meeting, gold):
    people = {p["personId"] for p in meeting["participants"]}
    for target in gold["tuples"]:
        topic = run_eval.topic_of(gold, target["topicSeq"])
        assert topic["startUtteranceId"] <= target["evidenceUtteranceId"] <= topic["endUtteranceId"]
        assert target["assigneePersonId"] in people


def test_정답_마감일은_회의일_뒤의_날짜다(meeting, gold):
    meeting_date = date.fromisoformat(meeting["meetingDate"])
    for target in gold["tuples"]:
        assert date.fromisoformat(target["dueDate"]) > meeting_date


# ── 요청이 실제 스키마를 통과하는가 ─────────────────────────────────────────────


def test_모든_계층의_요청이_실제_요청_스키마를_통과한다(meeting, gold):
    ResolveReferenceRequest.model_validate(run_eval.l1_5_request(meeting))
    SegmentTopicsRequest.model_validate(run_eval.l2_request(meeting))
    for payload in run_eval.l3_requests(meeting, gold):
        SummarizeTopicRequest.model_validate(payload)
    for payload in run_eval.l3_5_requests(meeting, gold):
        GateRequest.model_validate(payload)
    for payload in run_eval.l4_requests(meeting, gold):
        ExtractTuplesRequest.model_validate(payload)
    for payload, _ in run_eval.l5_requests(meeting, gold):
        VerifyRequest.model_validate(payload)
    SummarizeMeetingRequest.model_validate(run_eval.overview_request(meeting, gold))


def test_게이트_요청은_논의_항목까지_담고_L4_요청은_확정_항목만_담는다(meeting, gold):
    gate_keys = {item["itemKey"] for r in run_eval.l3_5_requests(meeting, gold) for item in r["items"]}
    assert gate_keys == {item["itemKey"] for item in gold["items"]}
    for payload in run_eval.l4_requests(meeting, gold):
        assert {item["gateStatus"] for item in payload["items"]} == {"CONFIRMED"}


def test_L5는_배정마다_맞는_요청과_틀린_요청을_하나씩_만든다(meeting, gold):
    requests = run_eval.l5_requests(meeting, gold)
    assert len(requests) == 2 * len(gold["tuples"])
    for (right, right_is_wrong), (wrong, wrong_is_wrong) in zip(requests[::2], requests[1::2], strict=True):
        assert (right_is_wrong, wrong_is_wrong) == (False, True)
        assert right["tuple"]["assigneeCandidatePersonId"] != wrong["tuple"]["assigneeCandidatePersonId"]


def test_L3_요청은_운영과_같이_앞_주제_끝_발화를_얹는다(meeting, gold):
    requests = run_eval.l3_requests(meeting, gold)
    assert [r["topicSeq"] for r in requests] == [t["topicSeq"] for t in gold["topics"]]

    for request, topic in zip(requests, gold["topics"], strict=True):
        ids = [u["utteranceId"] for u in request["utterances"]]
        assert ids[-1] == topic["endUtteranceId"]
        own_start = ids.index(topic["startUtteranceId"])
        # 첫 주제는 앞이 없어 얹을 게 없고, 나머지는 앞 주제의 끝 OVERLAP_UTTERANCES 발화가 앞에 붙는다.
        assert own_start == (0 if topic["startUtteranceId"] == 1 else OVERLAP_UTTERANCES)
        assert ids == list(range(ids[0], ids[-1] + 1))


def test_L3_요청은_오버랩_수를_바꿀_수_있고_0이면_고유_구간만_보낸다(meeting, gold):
    for request, topic in zip(run_eval.l3_requests(meeting, gold, overlap=0), gold["topics"], strict=True):
        ids = [u["utteranceId"] for u in request["utterances"]]
        assert (ids[0], ids[-1]) == (topic["startUtteranceId"], topic["endUtteranceId"])


# ── 채점 ────────────────────────────────────────────────────────────────────────


def _l3_result(
    gold: dict, *, drop: set[str] = frozenset(), retype: dict[str, str] | None = None
) -> list[dict]:
    """정답 항목을 그대로 돌려주는 L3 응답(주제별). 일부를 빼거나 유형을 바꿔 오류를 만든다."""
    retype = retype or {}
    results = []
    for topic in gold["topics"]:
        items = [
            {
                "itemType": retype.get(i["itemKey"], i["itemType"]),
                "content": i["content"],
                "evidenceUtteranceId": i["evidenceUtteranceId"],
            }
            for i in gold["items"]
            if i["topicSeq"] == topic["topicSeq"] and i["itemKey"] not in drop
        ]
        results.append({"topicSeq": topic["topicSeq"], "items": items})
    return results


def test_항목_채점_완벽한_입력(gold):
    score = scoring.score_items(_l3_result(gold), gold["items"], gold["topics"])
    assert score == {
        "item_recall": 1.0,
        "item_type_accuracy": 1.0,
        "item_count": len(gold["items"]),
        "overlap_item_count": 0,
        "overlap_item_rate": 0.0,
    }


def test_항목_채점_놓친_항목은_재현율을_깎고_유형_정확도는_짝이_맞은_것만_본다(gold):
    score = scoring.score_items(_l3_result(gold, drop={"i2"}), gold["items"], gold["topics"])
    assert score["item_recall"] == pytest.approx((len(gold["items"]) - 1) / len(gold["items"]))
    assert score["item_type_accuracy"] == 1.0


def test_항목_채점_유형만_틀리면_재현율은_유지되고_유형_정확도만_떨어진다(gold):
    score = scoring.score_items(_l3_result(gold, retype={"i8": "BLOCKER"}), gold["items"], gold["topics"])
    assert score["item_recall"] == 1.0
    assert score["item_type_accuracy"] == pytest.approx((len(gold["items"]) - 1) / len(gold["items"]))


def test_항목_채점_오버랩_발화에서_뽑힌_항목은_따로_세고_재현율을_올리지_않는다(gold):
    result = _l3_result(gold, drop={"i2"})
    # 주제 1의 정답(근거 7)을 주제 2의 오버랩에서 다시 뽑은 상황 — 다른 주제에서 맞힌 건 맞힌 게 아니다.
    topic_two = next(r for r in result if r["topicSeq"] == 2)
    topic_two["items"].append({"itemType": "DECISION", "content": "일정표 공유", "evidenceUtteranceId": 7})

    score = scoring.score_items(result, gold["items"], gold["topics"])

    assert score["overlap_item_count"] == 1
    assert score["overlap_item_rate"] == pytest.approx(1 / score["item_count"])
    assert score["item_recall"] == pytest.approx((len(gold["items"]) - 1) / len(gold["items"]))


def test_항목_채점_아무것도_못_뽑으면_재현율_0이고_나머지는_측정_불가다(gold):
    score = scoring.score_items([], gold["items"], gold["topics"])
    assert score["item_recall"] == 0.0
    assert score["item_type_accuracy"] is None
    assert score["overlap_item_rate"] is None
    assert score["item_count"] == 0


def test_배정_채점_완벽한_입력(gold):
    perfect = [_as_predicted(t) for t in gold["tuples"]]
    score = scoring.score_tuples(perfect, gold["tuples"])
    assert score["f1"] == 1.0
    assert score["source_accuracy"] == 1.0
    assert score["due_accuracy"] == 1.0
    assert score["extra"] == 0


def test_배정_채점_마감일만_틀리면_짝은_유지되고_마감_정확도만_떨어진다(gold):
    predicted = [_as_predicted(t) for t in gold["tuples"]]
    predicted[0]["dueDate"] = "2099-01-01"
    score = scoring.score_tuples(predicted, gold["tuples"])
    assert score["f1"] == 1.0
    assert score["due_accuracy"] == pytest.approx(0.8)


def test_배정_채점_뽑히면_안_되는_배정은_정밀도를_깎는다(gold):
    predicted = [_as_predicted(t) for t in gold["tuples"]]
    predicted.append({"assigneeCandidatePersonId": 4, "evidenceUtteranceId": 25})
    score = scoring.score_tuples(predicted, gold["tuples"])
    assert score["recall"] == 1.0
    assert score["precision"] == pytest.approx(5 / 6)
    assert score["extra"] == 1


def test_배정_채점_아무것도_못_뽑으면_재현율_0이고_정밀도는_측정_불가다(gold):
    score = scoring.score_tuples([], gold["tuples"])
    assert score["recall"] == 0.0
    assert score["precision"] is None
    assert score["f1"] is None


def test_게이트_채점은_논의를_확정으로_올린_오탐을_따로_센다(gold):
    verdicts = [{"itemKey": i["itemKey"], "gateStatus": i["gateStatus"]} for i in gold["items"]]
    verdicts[-1]["gateStatus"] = "CONFIRMED"  # 정답은 DISCUSSED
    score = scoring.score_gate(verdicts, gold["items"])
    assert score["false_confirmed"] == 1
    assert score["missed_confirmed"] == 0
    assert score["accuracy"] == pytest.approx(7 / 8)


def test_게이트_채점은_판정이_없는_확정_항목을_놓친_것으로_센다(gold):
    score = scoring.score_gate([], gold["items"])
    assert score["missed_confirmed"] == 6
    assert score["false_confirmed"] == 0


def test_주제_채점은_시작_발화_일치만_본다(gold):
    predicted = [{"startUtteranceId": 1}, {"startUtteranceId": 9}, {"startUtteranceId": 12}]
    score = scoring.score_topics(predicted, gold["topics"])
    assert score["start_precision"] == pytest.approx(2 / 3)
    assert score["start_recall"] == pytest.approx(2 / 4)
    assert score["topic_count_diff"] == -1


def test_지시어_채점은_해소_문장에_핵심어가_모두_있어야_맞다고_본다(gold):
    predicted = [
        {"utteranceId": 19, "resolvedText": "인증 API 문서"},
        {"utteranceId": 20, "resolvedText": "다른 일"},
    ]
    score = scoring.score_references(predicted, gold["references"])
    assert score["found_rate"] == pytest.approx(2 / 3)
    assert score["resolved_rate"] == pytest.approx(1 / 3)


def test_검증_채점은_틀린_담당자가_거절된_비율을_센다():
    score = scoring.score_verify([True, True, False], [False, False, True])
    assert score["positive_agree_rate"] == pytest.approx(2 / 3)
    assert score["negative_reject_rate"] == pytest.approx(2 / 3)


def test_요약은_값이_없으면_전부_None이고_None은_건너뛴다():
    assert scoring.summarize([None, None])["mean"] is None
    stat = scoring.summarize([0.8, None, 0.6, 1.0])
    assert stat["n"] == 3
    assert stat["spread"] == pytest.approx(0.4)


def test_흔들림은_실행마다_갈린_항목만_센다():
    info = scoring.instability([{"a": "X", "b": "Y"}, {"a": "X", "b": "Z"}])
    assert info["unstable_keys"] == ["b"]
    assert info["unstable_rate"] == pytest.approx(0.5)


# ── 서버 없이 전체 흐름 ─────────────────────────────────────────────────────────


def _as_predicted(target: dict) -> dict:
    return {
        "assigneeCandidatePersonId": target["assigneePersonId"],
        "assigneeSource": target["assigneeSource"],
        "dueDate": target["dueDate"],
        "evidenceUtteranceId": target["evidenceUtteranceId"],
    }


def _perfect_server(gold: dict):
    """정답을 그대로 돌려주는 가짜 서버 — 측정 흐름이 점수를 제대로 모으는지만 본다."""

    def post(_base_url, path, _token, payload, _timeout):
        usage = {"tokensIn": 10, "tokensOut": 5}
        if path == run_eval.PATHS["l1_5"]:
            refs = [
                {"utteranceId": r["utteranceId"], "resolvedText": " ".join(r["keywords"])}
                for r in gold["references"]
            ]
            return {"references": refs, "usage": usage}, 0.01
        if path == run_eval.PATHS["l2"]:
            return {"topics": gold["topics"], "usage": usage}, 0.01
        if path == run_eval.PATHS["l3"]:
            items = [
                {
                    "itemType": i["itemType"],
                    "content": i["content"],
                    "reason": "",
                    "evidenceUtteranceId": i["evidenceUtteranceId"],
                }
                for i in gold["items"]
                if i["topicSeq"] == payload["topicSeq"]
            ]
            return {"topicSeq": payload["topicSeq"], "items": items, "usage": usage}, 0.01
        if path == run_eval.PATHS["l3_5"]:
            status = {i["itemKey"]: i["gateStatus"] for i in gold["items"]}
            verdicts = [
                {"itemKey": i["itemKey"], "gateStatus": status[i["itemKey"]]} for i in payload["items"]
            ]
            return {"verdicts": verdicts, "usage": usage}, 0.01
        if path == run_eval.PATHS["l4"]:
            evidence = {i["evidenceUtteranceIds"][0] for i in payload["items"]}
            tuples = [_as_predicted(t) for t in gold["tuples"] if t["evidenceUtteranceId"] in evidence]
            return {"tuples": tuples, "usage": usage}, 0.01
        if path == run_eval.PATHS["l5"]:
            sent = payload["tuple"]
            truth = next(t for t in gold["tuples"] if t["evidenceUtteranceId"] == sent["evidenceUtteranceId"])
            return {
                "agree": sent["assigneeCandidatePersonId"] == truth["assigneePersonId"],
                "usage": usage,
            }, 0.01
        return {"overview": "회의 요약", "usage": usage}, 0.01

    return post


def test_완벽한_서버로_전체를_돌리면_모든_계층이_만점이고_흔들림이_없다(monkeypatch, meeting, gold):
    from types import SimpleNamespace

    monkeypatch.setattr(run_eval, "post", _perfect_server(gold))
    args = SimpleNamespace(base_url="http://x", timeout=1.0)

    runs = []
    for _ in range(2):
        run = run_eval.Run(args, "t", meeting, gold)
        for layer in run_eval.LAYERS:
            getattr(run, f"run_{layer}")()
        runs.append(
            {"metrics": run.metrics, "outcomes": run.outcomes, "usage": run.usage, "failures": run.failures}
        )
    summary = run_eval.build_summary(runs)

    assert summary["failures"] == 0
    assert summary["metrics"]["l4"]["f1"]["mean"] == 1.0
    assert summary["metrics"]["l3_5"]["accuracy"]["mean"] == 1.0
    assert summary["metrics"]["l3_5"]["false_confirmed"]["mean"] == 0
    assert summary["metrics"]["l5"]["positive_agree_rate"]["mean"] == 1.0
    assert summary["metrics"]["l5"]["negative_reject_rate"]["mean"] == 1.0
    assert summary["metrics"]["l2"]["start_recall"]["mean"] == 1.0
    assert summary["metrics"]["l3"]["item_recall"]["mean"] == 1.0
    assert summary["metrics"]["l3"]["overlap_item_count"]["mean"] == 0
    assert summary["instability"]["l3"]["unstable"] == 0
    assert summary["usage"]["l3"]["calls"]["mean"] == len(gold["topics"])
    assert summary["instability"]["l4"]["unstable"] == 0
    assert summary["usage"]["l5"]["calls"]["mean"] == 2 * len(gold["tuples"])


def test_호출이_실패하면_0점이_아니라_실패로_센다(monkeypatch, meeting, gold):
    from types import SimpleNamespace

    def failing(*_args):
        raise run_eval.CallFailed("HTTP 503 /x")

    monkeypatch.setattr(run_eval, "post", failing)
    run = run_eval.Run(SimpleNamespace(base_url="http://x", timeout=1.0), "t", meeting, gold)
    run.run_l2()
    assert run.failures == ["l2: HTTP 503 /x"]
    assert "l2" not in run.metrics
