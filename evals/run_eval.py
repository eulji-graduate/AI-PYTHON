"""AI 계층 반복 측정 — 시연용 샘플 회의를 같은 서버에 N번 흘려 점수와 흔들림을 잰다.

사용 예 (서버는 따로 띄운다 — DRY_RUN=1이면 배선만, 키가 있으면 실제 품질을 잰다):

    INTERNAL_TOKEN=... uv run python -m evals.run_eval --runs 5 --out evals/out/baseline.json

⚠️ 토큰은 환경변수로만 받는다. 인자·파일·출력 어디에도 찍지 않는다.
⚠️ **1회 측정으로 개선을 판단하지 않는다.** 이 스크립트가 같은 입력을 여러 번 보내는 이유다 —
   요약의 `spread`(최댓값-최솟값)를 넘는 차이만 효과로 읽는다(`app/clients/gemini.py` 주석의 실측).
⚠️ 계층별 격리 모드다. 입력을 앞 계층 출력이 아니라 **정답 라벨**에서 만든다 — 앞 계층이 틀려도
   뒤 계층 점수가 같이 무너지지 않아야 어느 계층을 고쳐야 하는지 보인다.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from app.schemas.l2 import OVERLAP_UTTERANCES
from evals import scoring

HERE = Path(__file__).resolve().parent
FIXTURES = HERE / "fixtures"
LAYERS = ("l1_5", "l2", "l3", "l3_5", "l4", "l5", "overview")

PATHS = {
    "l1_5": "/internal/layers/l1-5/resolve-reference",
    "l2": "/internal/layers/l2/segment-topics",
    "l3": "/internal/layers/l3/summarize-topic",
    "l3_5": "/internal/layers/l3-5/gate",
    "l4": "/internal/layers/l4/extract-tuples",
    "l5": "/internal/layers/l5/verify",
    "overview": "/internal/layers/overview/summarize-meeting",
}


class CallFailed(Exception):
    """호출이 실패했다 — 점수를 0으로 두지 않고 실패로 따로 센다."""


# ── 요청 만들기 (순수 함수 — 네트워크 없이 테스트가 실제 요청 스키마로 검증한다) ──────────


def base(meeting: dict) -> dict:
    return {"tenantId": meeting["tenantId"], "meetingId": meeting["meetingId"]}


def utterances_of(meeting: dict, topic: dict | None = None) -> list[dict]:
    items = meeting["utterances"]
    if topic is None:
        return items
    return [u for u in items if topic["startUtteranceId"] <= u["utteranceId"] <= topic["endUtteranceId"]]


def confirmed_items_of(gold: dict, topic_seq: int) -> list[dict]:
    """L4·L5가 받는 항목 — 게이트를 통과한(CONFIRMED) 것만이다."""
    return [
        {
            "itemType": item["itemType"],
            "gateStatus": "CONFIRMED",
            "content": item["content"],
            "evidenceUtteranceIds": [item["evidenceUtteranceId"]],
        }
        for item in gold["items"]
        if item["topicSeq"] == topic_seq and item["gateStatus"] == "CONFIRMED"
    ]


def topic_of(gold: dict, topic_seq: int) -> dict:
    return next(topic for topic in gold["topics"] if topic["topicSeq"] == topic_seq)


def other_person(meeting: dict, person_id: int) -> int:
    """일부러 틀린 담당자 — 참석자 중 다른 사람 하나(순서상 다음)."""
    ids = [p["personId"] for p in meeting["participants"]]
    return ids[(ids.index(person_id) + 1) % len(ids)]


def l1_5_request(meeting: dict) -> dict:
    return {
        **base(meeting),
        "utterances": meeting["utterances"],
        "targetUtteranceIds": [],
        "participants": meeting["participants"],
    }


def l2_request(meeting: dict) -> dict:
    return {**base(meeting), "utterances": meeting["utterances"]}


def l3_requests(
    meeting: dict, gold: dict, overlap: int = OVERLAP_UTTERANCES, *, include_topic_boundary: bool = False
) -> list[dict]:
    """L3 입력 — 정답 주제 구간에 **운영과 같이 앞 주제 끝 `overlap`발화를 얹어** 보낸다.

    다른 계층은 고유 구간(`utterances_of`)만 보내지만 L3는 그러면 안 된다. 운영의 L2(`parse_topics`)가
    오버랩을 얹어 L3에 넘기고, 오버랩 구간에서 항목이 중복으로 뽑히는 것이 바로 이 계층에서 일어나는 일이다.
    """
    utterances = meeting["utterances"]
    requests = []
    for topic in gold["topics"]:
        start = next(i for i, u in enumerate(utterances) if u["utteranceId"] >= topic["startUtteranceId"])
        end = max(i for i, u in enumerate(utterances) if u["utteranceId"] <= topic["endUtteranceId"])
        payload = {
            **base(meeting),
            "topicSeq": topic["topicSeq"],
            "topic": topic["topic"],
            "utterances": utterances[max(0, start - overlap) : end + 1],
            "participants": meeting["participants"],
        }
        if include_topic_boundary:
            payload["topicStartUtteranceId"] = utterances[start]["utteranceId"]
        requests.append(payload)
    return requests


def l3_5_requests(meeting: dict, gold: dict) -> list[dict]:
    requests = []
    for topic in gold["topics"]:
        items = [
            {
                "itemKey": item["itemKey"],
                "itemType": item["itemType"],
                "content": item["content"],
                "evidenceUtteranceId": item["evidenceUtteranceId"],
            }
            for item in gold["items"]
            if item["topicSeq"] == topic["topicSeq"]
        ]
        if items:
            requests.append(
                {
                    **base(meeting),
                    "topic": topic["topic"],
                    "items": items,
                    "utterances": utterances_of(meeting, topic),
                    "participants": meeting["participants"],
                }
            )
    return requests


def l4_requests(meeting: dict, gold: dict) -> list[dict]:
    requests = []
    for topic in gold["topics"]:
        items = confirmed_items_of(gold, topic["topicSeq"])
        if items:
            requests.append(
                {
                    **base(meeting),
                    "topic": topic["topic"],
                    "items": items,
                    "utterances": utterances_of(meeting, topic),
                    "participants": meeting["participants"],
                    "meetingDate": meeting["meetingDate"],
                    "view": "EXTRACT",
                }
            )
    return requests


def _tuple_payload(target: dict, person_id: int) -> dict:
    return {
        "title": target["keywords"][0],
        "assigneeCandidatePersonId": person_id,
        "assigneeSource": target["assigneeSource"],
        "dueDate": target["dueDate"],
        "evidenceUtteranceId": target["evidenceUtteranceId"],
    }


def l5_requests(meeting: dict, gold: dict) -> list[tuple[dict, bool]]:
    """정답 배정마다 둘 — 맞는 담당자(합의가 나와야 함)와 일부러 틀린 담당자(불일치가 나와야 함).

    돌려주는 값은 `(요청, 일부러 틀렸는가)`이다.
    """
    requests = []
    for target in gold["tuples"]:
        topic = topic_of(gold, target["topicSeq"])
        common = {
            **base(meeting),
            "topic": topic["topic"],
            "items": confirmed_items_of(gold, topic["topicSeq"]),
            "utterances": utterances_of(meeting, topic),
            "participants": meeting["participants"],
            "meetingDate": meeting["meetingDate"],
        }
        right = {**common, "tuple": _tuple_payload(target, target["assigneePersonId"])}
        wrong_person = other_person(meeting, target["assigneePersonId"])
        wrong = {**common, "tuple": _tuple_payload(target, wrong_person)}
        requests.append((right, False))
        requests.append((wrong, True))
    return requests


def overview_request(meeting: dict, gold: dict) -> dict:
    topics = [
        {
            "topicSeq": topic["topicSeq"],
            "topic": topic["topic"],
            "items": [
                {"itemType": item["itemType"], "content": item["content"]}
                for item in gold["items"]
                if item["topicSeq"] == topic["topicSeq"] and item["gateStatus"] == "CONFIRMED"
            ],
        }
        for topic in gold["topics"]
    ]
    return {**base(meeting), "topics": topics, "participants": meeting["participants"]}


# ── 호출 ────────────────────────────────────────────────────────────────────────


def post(base_url: str, path: str, token: str, payload: dict, timeout: float) -> tuple[dict, float]:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}{path}",
        data=body,
        method="POST",
        headers={"Content-Type": "application/json", "X-Internal-Token": token},
    )
    started = time.monotonic()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        # ⚠️ 응답 본문을 싣지 않는다 — 제공자 응답이 섞여 있을 수 있다(README 실패 정책).
        raise CallFailed(f"HTTP {error.code} {path}") from error
    except (urllib.error.URLError, TimeoutError) as error:
        raise CallFailed(f"연결 실패 {path}: {error.__class__.__name__}") from error
    return data, time.monotonic() - started


class Run:
    """한 번의 전체 실행. 호출 수·토큰·지연을 계층별로 모은다."""

    def __init__(self, args: argparse.Namespace, token: str, meeting: dict, gold: dict) -> None:
        self.args, self.token, self.meeting, self.gold = args, token, meeting, gold
        self.metrics: dict[str, dict] = {}
        self.outcomes: dict[str, dict[str, str]] = {}
        self.usage: dict[str, dict[str, float]] = {}
        self.failures: list[str] = []
        self.overview_text = ""
        self.l3_results: list[dict] | None = None

    def call(self, layer: str, payload: dict) -> dict | None:
        try:
            data, elapsed = post(self.args.base_url, PATHS[layer], self.token, payload, self.args.timeout)
        except CallFailed as error:
            self.failures.append(f"{layer}: {error}")
            return None
        bucket = self.usage.setdefault(layer, {"calls": 0, "tokensIn": 0, "tokensOut": 0, "seconds": 0.0})
        usage = data.get("usage") or {}
        bucket["calls"] += 1
        bucket["tokensIn"] += usage.get("tokensIn", 0)
        bucket["tokensOut"] += usage.get("tokensOut", 0)
        bucket["seconds"] += elapsed
        return data

    def run_l1_5(self) -> None:
        data = self.call("l1_5", l1_5_request(self.meeting))
        if data is not None:
            self.metrics["l1_5"] = scoring.score_references(
                data.get("references", []), self.gold["references"]
            )

    def run_l2(self) -> None:
        data = self.call("l2", l2_request(self.meeting))
        if data is not None:
            self.metrics["l2"] = scoring.score_topics(data.get("topics", []), self.gold["topics"])

    def run_l3(self) -> None:
        results: list[dict] = []
        for payload in l3_requests(
            self.meeting, self.gold, include_topic_boundary=self.args.l3_topic_boundary
        ):
            data = self.call("l3", payload)
            if data is not None:
                results.append({"topicSeq": payload["topicSeq"], "items": data.get("items", [])})
        self.l3_results = results
        # ⚠️ 정답 항목별로 (주제, 근거 발화)가 뽑혔는지와 유형을 접어 둔다 — 실행 간 같은 항목이 갈렸는지 비교한다.
        got = {
            (result["topicSeq"], item["evidenceUtteranceId"]): item["itemType"]
            for result in results
            for item in result["items"]
        }
        self.outcomes["l3"] = {
            item["itemKey"]: got.get((item["topicSeq"], item["evidenceUtteranceId"]), "<없음>")
            for item in self.gold["items"]
        }
        self.metrics["l3"] = scoring.score_items(results, self.gold["items"], self.gold["topics"])

    def run_l3_5(self) -> None:
        verdicts: list[dict] = []
        for payload in l3_5_requests(self.meeting, self.gold):
            data = self.call("l3_5", payload)
            if data is not None:
                verdicts.extend(data.get("verdicts", []))
        got = {verdict["itemKey"]: verdict["gateStatus"] for verdict in verdicts}
        self.outcomes["l3_5"] = {
            item["itemKey"]: got.get(item["itemKey"], "<없음>") for item in self.gold["items"]
        }
        self.metrics["l3_5"] = scoring.score_gate(verdicts, self.gold["items"])

    def run_l4(self) -> None:
        predicted: list[dict] = []
        for payload in l4_requests(self.meeting, self.gold):
            data = self.call("l4", payload)
            if data is not None:
                predicted.extend(data.get("tuples", []))
        # ⚠️ 정답 키(담당자·근거)로 결과를 접어 둔다 — 반복 실행 간 같은 배정이 갈렸는지 비교한다.
        outcomes = {}
        for target in self.gold["tuples"]:
            hit = next(
                (
                    t
                    for t in predicted
                    if t.get("assigneeCandidatePersonId") == target["assigneePersonId"]
                    and t["evidenceUtteranceId"] == target["evidenceUtteranceId"]
                ),
                None,
            )
            outcomes[target["key"]] = (
                "<없음>" if hit is None else f"{hit.get('assigneeSource')}|{hit.get('dueDate')}"
            )
        self.outcomes["l4"] = outcomes
        self.metrics["l4"] = scoring.score_tuples(predicted, self.gold["tuples"])

    def run_l5(self) -> None:
        right_agree: list[bool] = []
        wrong_agree: list[bool] = []
        for payload, is_wrong in l5_requests(self.meeting, self.gold):
            data = self.call("l5", payload)
            if data is not None:
                (wrong_agree if is_wrong else right_agree).append(bool(data.get("agree")))
        self.metrics["l5"] = scoring.score_verify(right_agree, wrong_agree)

    def run_overview(self) -> None:
        data = self.call("overview", overview_request(self.meeting, self.gold))
        if data is None:
            return
        self.overview_text = data.get("overview", "")
        # ⚠️ 개요는 자유 문장이라 점수를 매기지 않는다 — 사람이 읽을 원문을 남기고 길이만 잰다.
        self.metrics["overview"] = {"chars": len(self.overview_text)}


# ── 요약·출력 ───────────────────────────────────────────────────────────────────


def metric_values(runs: list[dict], layer: str, name: str) -> list[float | None]:
    return [run["metrics"].get(layer, {}).get(name) for run in runs]


def build_summary(runs: list[dict]) -> dict:
    summary: dict[str, Any] = {"metrics": {}, "instability": {}, "usage": {}, "failures": 0}
    for layer in sorted({layer for run in runs for layer in run["metrics"]}):
        names = sorted({name for run in runs for name in run["metrics"].get(layer, {})})
        summary["metrics"][layer] = {
            name: scoring.summarize(metric_values(runs, layer, name)) for name in names
        }
    for layer in sorted({layer for run in runs for layer in run["outcomes"]}):
        per_run = [run["outcomes"][layer] for run in runs if layer in run["outcomes"]]
        summary["instability"][layer] = scoring.instability(per_run)
    for layer in sorted({layer for run in runs for layer in run["usage"]}):
        summary["usage"][layer] = {
            key: scoring.summarize([run["usage"].get(layer, {}).get(key) for run in runs])
            for key in ("calls", "tokensIn", "tokensOut", "seconds")
        }
    summary["failures"] = sum(len(run["failures"]) for run in runs)
    return summary


def fmt(value: float | None) -> str:
    return "-" if value is None else f"{value:.3f}"


def print_report(summary: dict, runs: int) -> None:
    print(f"\n# 반복 측정 결과 ({runs}회)\n")
    print("| 계층 | 지표 | 평균 | 최소 | 최대 | 폭(spread) |")
    print("|---|---|---|---|---|---|")
    for layer, metrics in summary["metrics"].items():
        for name, stat in metrics.items():
            cells = [fmt(stat[key]) for key in ("mean", "min", "max", "spread")]
            print(f"| {layer} | {name} | " + " | ".join(cells) + " |")
    print("\n## 실행마다 결과가 갈린 항목")
    for layer, info in summary["instability"].items():
        print(f"- {layer}: {info['unstable']}/{info['items']} 항목 {info['unstable_keys']}")
    print("\n## 토큰·지연 (실행당 평균)")
    for layer, usage in summary["usage"].items():
        print(
            f"- {layer}: 호출 {fmt(usage['calls']['mean'])}회, 입력 {fmt(usage['tokensIn']['mean'])}, "
            f"출력 {fmt(usage['tokensOut']['mean'])}, {fmt(usage['seconds']['mean'])}초"
        )
    print(f"\n실패한 호출: {summary['failures']}건")
    print("개선 판단: 프롬프트·모델을 바꾼 뒤의 평균 차이가 위 '폭'보다 작으면 효과로 읽지 않는다.")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument(
        "--layers", default=",".join(LAYERS), help=f"쉼표 구분. 기본 전부: {','.join(LAYERS)}"
    )
    parser.add_argument("--fixture", type=Path, default=FIXTURES / "demo_meeting.json")
    parser.add_argument("--gold", type=Path, default=FIXTURES / "demo_meeting.gold.json")
    parser.add_argument(
        "--l3-topic-boundary", action="store_true", help="L3 요청에 현재 주제의 시작 발화 ID를 포함"
    )
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    token = os.environ.get("INTERNAL_TOKEN", "")
    if not token:
        print("INTERNAL_TOKEN 환경변수가 필요합니다 (값은 출력하지 않습니다).", file=sys.stderr)
        return 2
    selected = [layer.strip() for layer in args.layers.split(",") if layer.strip()]
    unknown = [layer for layer in selected if layer not in LAYERS]
    if unknown:
        print(f"알 수 없는 계층: {unknown}", file=sys.stderr)
        return 2

    meeting = json.loads(args.fixture.read_text(encoding="utf-8"))
    gold = json.loads(args.gold.read_text(encoding="utf-8"))

    runs: list[dict] = []
    for index in range(args.runs):
        run = Run(args, token, meeting, gold)
        for layer in selected:
            getattr(run, f"run_{layer}")()
        result = {
            "run": index + 1,
            "metrics": run.metrics,
            "outcomes": run.outcomes,
            "usage": run.usage,
            "failures": run.failures,
            "overview": run.overview_text,
        }
        if args.out and run.l3_results is not None:
            result["l3_results"] = run.l3_results
            result["l3_topic_boundary"] = args.l3_topic_boundary
        runs.append(result)
        print(f"실행 {index + 1}/{args.runs} 완료 (실패 호출 {len(run.failures)}건)", file=sys.stderr)

    summary = build_summary(runs)
    print_report(summary, args.runs)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps({"summary": summary, "runs": runs}, ensure_ascii=False, indent=2)
        args.out.write_text(payload, encoding="utf-8")
        print(f"\n저장: {args.out}", file=sys.stderr)
    return 1 if summary["failures"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
