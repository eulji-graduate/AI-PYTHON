"""L2 주제 분할 프로브 — 같은 회의를 N번 보내 **시작점 집합별 빈도**와 틀린 사례를 보여 준다.

`run_eval` 은 지표 평균(precision·recall)만 남겨서, 틀렸을 때 모델이 *어떤 시작점*을 냈는지는
알 수 없다. 이 스크립트는 그걸 센다. 같은 실수가 매번 나오는지(체계적), 가끔 나오는지(흔들림)가
갈리면 고칠 방법이 달라진다.

    INTERNAL_TOKEN=... uv run python -m evals.probe_l2 --runs 30 \\
        --fixture evals/fixtures/heldout_hr.json --gold evals/fixtures/heldout_hr.gold.json

서버는 따로 띄운다(`evals/README.md`). 토큰은 환경변수로만 받는다(출력하지 않는다).
"""

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

FIXTURES = Path(__file__).resolve().parent / "fixtures"
PATH = "/internal/layers/l2/segment-topics"


def post(base_url: str, token: str, payload: dict) -> dict:
    request = urllib.request.Request(
        base_url + PATH,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        method="POST",
        headers={"Content-Type": "application/json", "X-Internal-Token": token},
    )
    last: Exception | None = None
    for attempt in range(3):  # 일시 오류만 재시도한다. 4xx 응답은 그대로 실패시킨다.
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError:
            raise
        except (urllib.error.URLError, TimeoutError) as error:
            last = error
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"호출 실패: {last}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--runs", type=int, default=30)
    parser.add_argument("--workers", type=int, default=3, help="동시 호출 수(기본 3)")
    parser.add_argument("--fixture", type=Path, default=FIXTURES / "demo_meeting.json")
    parser.add_argument("--gold", type=Path, default=FIXTURES / "demo_meeting.gold.json")
    parser.add_argument("--out", type=Path, default=None, help="실행별 원본(시작점·제목)을 JSON 으로 저장")
    args = parser.parse_args()

    token = os.environ.get("INTERNAL_TOKEN", "")
    if not token:
        print("INTERNAL_TOKEN 환경변수가 필요합니다 (값은 출력하지 않습니다).", file=sys.stderr)
        return 2

    meeting = json.loads(args.fixture.read_text(encoding="utf-8"))
    gold = json.loads(args.gold.read_text(encoding="utf-8"))
    gold_starts = tuple(sorted(topic["startUtteranceId"] for topic in gold["topics"]))
    payload = {
        "tenantId": meeting["tenantId"],
        "meetingId": meeting["meetingId"],
        "utterances": meeting["utterances"],
    }

    def one(_: int) -> dict:
        data = post(args.base_url, token, payload)
        topics = data.get("topics", [])
        return {
            "starts": [topic["startUtteranceId"] for topic in topics],
            "titles": [topic["topic"] for topic in topics],
            "promptVersion": data.get("promptVersion"),
        }

    started = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        runs = list(pool.map(one, range(args.runs)))
    elapsed = time.time() - started

    exact = sum(1 for run in runs if tuple(sorted(run["starts"])) == gold_starts)
    print(
        f"runs={len(runs)} 정답 시작점={list(gold_starts)} 프롬프트={runs[0]['promptVersion']} ({elapsed:.0f}초)"
    )
    print(f"정확 일치: {exact}/{len(runs)} = {exact / len(runs):.1%}")
    print("시작점 집합별 빈도:")
    for pattern, count in Counter(tuple(run["starts"]) for run in runs).most_common():
        mark = "  <- 정답" if tuple(sorted(pattern)) == gold_starts else ""
        print(f"  {list(pattern)}: {count}회{mark}")

    shown: set[tuple[int, ...]] = set()
    print("틀린 사례의 제목(집합별 1건):")
    for run in runs:
        key = tuple(run["starts"])
        if tuple(sorted(key)) != gold_starts and key not in shown:
            shown.add(key)
            print(f"  {list(key)}")
            for start, title in zip(run["starts"], run["titles"], strict=True):
                print(f"    {start:>3}: {title}")

    if args.out:
        record = {"goldStarts": list(gold_starts), "exact": exact, "n": len(runs), "runs": runs}
        args.out.write_text(json.dumps(record, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
