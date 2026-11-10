"""반복 측정용 채점 함수 — 순수 함수만 둔다(네트워크·파일 없음).

정확도는 한 번 재서 판단하지 않는다. 같은 입력도 호출마다 결과가 달라지므로(gemini.py 주석의
실측), 여기서는 한 번의 점수를 만들고 `summarize`가 여러 번의 분포(평균·폭)를 만든다.
"""

from __future__ import annotations

import statistics
from collections.abc import Iterable


def _ratio(numerator: int, denominator: int) -> float | None:
    """분모가 0이면 `None`이다 — 0점과 "잴 게 없다"를 같은 값으로 두지 않는다."""
    return None if denominator == 0 else numerator / denominator


def score_topics(predicted: list[dict], gold: list[dict]) -> dict:
    """주제 분할 — 시작 발화 id가 정답과 일치하는가. 끝은 코드가 계산하므로 보지 않는다."""
    predicted_starts = {topic["startUtteranceId"] for topic in predicted}
    gold_starts = {topic["startUtteranceId"] for topic in gold}
    hit = len(predicted_starts & gold_starts)
    return {
        "start_precision": _ratio(hit, len(predicted_starts)),
        "start_recall": _ratio(hit, len(gold_starts)),
        "topic_count_diff": len(predicted) - len(gold),
    }


def score_references(predicted: list[dict], gold: list[dict]) -> dict:
    """지시어 해소 — 정답 발화에서 해소 결과가 나왔고, 해소 문장에 핵심어가 있는가."""
    by_utterance: dict[int, list[dict]] = {}
    for reference in predicted:
        by_utterance.setdefault(reference["utteranceId"], []).append(reference)

    found = 0
    resolved_ok = 0
    for target in gold:
        candidates = by_utterance.get(target["utteranceId"], [])
        if candidates:
            found += 1
        if any(
            all(keyword in (candidate.get("resolvedText") or "") for keyword in target["keywords"])
            for candidate in candidates
        ):
            resolved_ok += 1

    return {
        "found_rate": _ratio(found, len(gold)),
        "resolved_rate": _ratio(resolved_ok, len(gold)),
        "extra_references": max(0, len(predicted) - found),
    }


def score_items(predicted: list[dict], gold_items: list[dict], topics: list[dict]) -> dict:
    """주제별 정리(L3) — 정답 항목을 그 주제에서 같은 근거 발화로 뽑았는가, 오버랩에서 뽑은 항목이 얼마나 되는가.

    `predicted`는 주제별 응답 `{"topicSeq", "items": [...]}`의 목록이고, `topics`는 정답 주제의 고유 구간
    (`startUtteranceId`~`endUtteranceId`)이다.

    ⚠️ 짝은 (주제, 근거 발화 id)다. 다른 주제에서 같은 발화를 근거로 뽑은 항목은 맞힌 것으로 세지 않는다 —
       그건 L2가 앞 주제 끝에 얹은 오버랩에서 중복으로 뽑힌 것이고 `overlap_item_*`로 따로 센다.
    ⚠️ 근거 발화가 정답과 정확히 같아야 맞다. 같은 약속을 다음 발화(맞장구)를 근거로 접으면 놓친 것으로 센다.
       엄격한 기준이지만, 그 접힘이 바로 뒤 계층(L3.5·L4)이 항목을 못 받는 경로라서 일부러 그렇게 둔다.
    """
    own = {t["topicSeq"]: (t["startUtteranceId"], t["endUtteranceId"]) for t in topics}
    types_by_key: dict[tuple[int, int], set[str]] = {}
    total = 0
    overlap = 0
    for result in predicted:
        seq = result["topicSeq"]
        for item in result.get("items", []):
            total += 1
            evidence = item["evidenceUtteranceId"]
            types_by_key.setdefault((seq, evidence), set()).add(item["itemType"])
            if seq in own and not own[seq][0] <= evidence <= own[seq][1]:
                overlap += 1

    hit = 0
    type_hit = 0
    for target in gold_items:
        types = types_by_key.get((target["topicSeq"], target["evidenceUtteranceId"]))
        if types is None:
            continue
        hit += 1
        if target["itemType"] in types:
            type_hit += 1

    return {
        "item_recall": _ratio(hit, len(gold_items)),
        "item_type_accuracy": _ratio(type_hit, hit),
        "item_count": total,
        "overlap_item_count": overlap,
        "overlap_item_rate": _ratio(overlap, total),
    }


def score_gate(verdicts: list[dict], gold_items: list[dict]) -> dict:
    """확정/논의 게이트 — 정확도와, 논의인데 확정으로 올린 오탐 수.

    ⚠️ 이 계층의 약속은 precision 우선이다. 논의 항목을 확정으로 올리면 결정되지도 않은 일이
       배정으로 번진다 — 정확도보다 `false_confirmed`가 더 중요한 숫자다.
    """
    predicted = {verdict["itemKey"]: verdict["gateStatus"] for verdict in verdicts}
    correct = 0
    false_confirmed = 0
    missed_confirmed = 0
    for item in gold_items:
        got = predicted.get(item["itemKey"])
        if got == item["gateStatus"]:
            correct += 1
        elif got == "CONFIRMED" and item["gateStatus"] == "DISCUSSED":
            false_confirmed += 1
        elif got != "CONFIRMED" and item["gateStatus"] == "CONFIRMED":
            missed_confirmed += 1
    return {
        "accuracy": _ratio(correct, len(gold_items)),
        "false_confirmed": false_confirmed,
        "missed_confirmed": missed_confirmed,
    }


def _tuple_key(person_id: int | None, evidence_id: int) -> tuple[int | None, int]:
    return person_id, evidence_id


def score_tuples(predicted: list[dict], gold: list[dict]) -> dict:
    """배정 추출 — (담당자, 근거 발화)로 짝을 짓고, 짝이 된 것의 필드별 정확도를 본다.

    짝 기준에 마감일·제목을 넣지 않는다. 마감일이 틀린 배정을 "없는 배정"으로 세면 담당자는
    맞고 날짜만 틀린 경우와 아예 못 뽑은 경우가 같은 점수가 된다.
    """
    gold_by_key = {_tuple_key(g["assigneePersonId"], g["evidenceUtteranceId"]): g for g in gold}
    matched = 0
    source_ok = 0
    due_ok = 0
    seen: set[tuple[int | None, int]] = set()
    for tuple_ in predicted:
        key = _tuple_key(tuple_.get("assigneeCandidatePersonId"), tuple_["evidenceUtteranceId"])
        target = gold_by_key.get(key)
        if target is None or key in seen:
            continue
        seen.add(key)
        matched += 1
        source_ok += int(tuple_.get("assigneeSource") == target["assigneeSource"])
        due_ok += int(tuple_.get("dueDate") == target["dueDate"])

    precision = _ratio(matched, len(predicted))
    recall = _ratio(matched, len(gold))
    f1 = None
    if precision is not None and recall is not None and (precision + recall) > 0:
        f1 = 2 * precision * recall / (precision + recall)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "source_accuracy": _ratio(source_ok, matched),
        "due_accuracy": _ratio(due_ok, matched),
        "extra": len(predicted) - matched,
    }


def score_verify(positive_agree: list[bool], negative_agree: list[bool]) -> dict:
    """관점 다변화 검증 — 맞는 배정은 합의, 일부러 틀린 배정은 불일치가 나와야 한다."""
    return {
        "positive_agree_rate": _ratio(sum(positive_agree), len(positive_agree)),
        "negative_reject_rate": _ratio(len(negative_agree) - sum(negative_agree), len(negative_agree)),
    }


def summarize(runs: Iterable[float | None]) -> dict:
    """한 지표의 반복 결과 — 평균·표준편차·최솟값·최댓값.

    ⚠️ **개선 판단은 이 폭(`max - min`)을 넘는 차이만 효과로 읽는다.** 표본이 작으면 표준편차는
       믿기 어려우므로 폭을 같이 본다. 값이 하나도 없으면 전부 `None`이다.
    """
    values = [value for value in runs if value is not None]
    if not values:
        return {"n": 0, "mean": None, "stdev": None, "min": None, "max": None, "spread": None}
    return {
        "n": len(values),
        "mean": statistics.fmean(values),
        "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
        "spread": max(values) - min(values),
    }


def instability(outcomes_per_run: list[dict[str, str]]) -> dict:
    """같은 항목이 실행마다 다른 결과를 냈는가 — 항목별로 결과를 모아 갈린 비율을 센다."""
    keys = {key for outcomes in outcomes_per_run for key in outcomes}
    unstable = sorted(
        key for key in keys if len({outcomes.get(key, "<없음>") for outcomes in outcomes_per_run}) > 1
    )
    return {
        "items": len(keys),
        "unstable": len(unstable),
        "unstable_rate": _ratio(len(unstable), len(keys)),
        "unstable_keys": unstable,
    }
