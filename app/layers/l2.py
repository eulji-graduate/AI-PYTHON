"""L2 · 주제 분할 (AI-03).

L4(`l4.py`)를 복제해 세 곳만 바꾼 것이다 — SPEC · build_response_schema · 후처리.

이 계층의 후처리가 하는 일이 조금 특별하다. **모델에게는 시작점만 받고 구간은 코드가
만든다.** 시작·끝을 둘 다 모델이 정하면 구간이 겹치거나 사이에 구멍이 생기고, 겹친
구간의 결정은 두 번 저장되며 구멍 속 결정은 영영 뽑히지 않는다. 시작점만 받아
정렬·중복 제거하면 그 두 실패가 **일어날 방법 자체가 없어진다** — 정확도 4원칙을
스키마로 거는 것과 같은 발상이다.

오버랩 3발화(명세)도 프롬프트로 부탁하지 않고 여기서 붙인다.
"""

from __future__ import annotations

from app.layers import few_shot
from app.layers import formatting as fmt
from app.layers.runner import LayerRunner, LayerSpec
from app.schemas.common import FewShotExample, Utterance
from app.schemas.l2 import (
    MIN_TOPIC_UTTERANCES,
    OVERLAP_UTTERANCES,
    TOPIC_MAX,
    SegmentTopicsRequest,
    SegmentTopicsResponse,
    TopicSegment,
)

SPEC = LayerSpec(
    layer="L2",
    prompt_file="l2_segment_topics.v2.txt",
    prompt_version="v2",
    dry_run_payload={"topics": []},
)


def build_response_schema(utterances: list[Utterance]) -> dict:
    """시작 발화 id 를 발화 목록의 enum 으로 박는다. 끝은 받지 않는다."""
    return {
        "type": "OBJECT",
        "properties": {
            "topics": {
                "type": "ARRAY",
                "items": {
                    "type": "OBJECT",
                    "properties": {
                        "topic": {"type": "STRING"},
                        "startUtteranceId": {
                            "type": "STRING",
                            "enum": fmt.utterance_enum(utterances),
                        },
                    },
                    "required": ["topic", "startUtteranceId"],
                    "propertyOrdering": ["topic", "startUtteranceId"],
                },
            }
        },
        "required": ["topics"],
    }


async def segment_topics(request: SegmentTopicsRequest, runner: LayerRunner) -> SegmentTopicsResponse:
    examples = await few_shot.lookup(
        tenant_id=request.tenant_id,
        layer=SPEC.layer,
        query_text=request.query_text,
    )

    if not request.utterances:
        return SegmentTopicsResponse(
            topics=[],
            used_few_shot=examples,
            model=runner.model_name,
            prompt_version=SPEC.prompt_version,
        )

    schema = build_response_schema(request.utterances)
    variables = build_prompt_variables(request, examples)

    raw, usage = await runner.run(SPEC, variables=variables, response_schema=schema)

    return SegmentTopicsResponse(
        topics=parse_topics(raw, request.utterances),
        used_few_shot=examples,
        usage=usage,
        model=runner.model_name,
        prompt_version=SPEC.prompt_version,
    )


def build_prompt_variables(request: SegmentTopicsRequest, examples: list[FewShotExample]) -> dict[str, str]:
    return {
        "UTTERANCES": fmt.format_utterances(request.utterances),
        "FEW_SHOT": fmt.format_few_shot(examples),
    }


def merge_short_topics(starts: list[tuple[int, str]], total: int) -> list[tuple[int, str]]:
    """`MIN_TOPIC_UTTERANCES` 발화보다 짧은 구간의 시작점을 버려 앞 주제에 합친다.

    입력은 발화 순번 순으로 정렬되고 첫 주제가 0 번에 붙은 `(발화 순번, 주제명)` 목록이다.
    합쳐진 주제의 이름은 `"<앞 주제명> 외 N건"` 이다 — 이름을 그대로 두면 구간이 넓어졌는데
    제목은 처음 안건만 말해, 화면이 실제보다 좁게 말하게 된다.

    첫 주제는 앞이 없어서 짧아도 합칠 곳이 없다. 뒤 주제를 끌어당기면 멀쩡한 주제를
    지우게 되므로 그대로 둔다.
    """
    kept: list[list] = [[index, topic, 0] for index, topic in starts]  # [시작 순번, 주제명, 합친 수]
    i = 1
    while i < len(kept):
        end_index = kept[i + 1][0] if i + 1 < len(kept) else total
        if end_index - kept[i][0] < MIN_TOPIC_UTTERANCES:
            kept[i - 1][2] += 1 + kept[i][2]
            del kept[i]  # 다음 구간이 i 자리로 올라오므로 i 는 올리지 않는다
        else:
            i += 1

    merged: list[tuple[int, str]] = []
    for index, topic, absorbed in kept:
        if absorbed:
            suffix = f" 외 {absorbed}건"
            topic = fmt.clip(topic, TOPIC_MAX - len(suffix)) + suffix
        merged.append((index, topic))
    return merged


def parse_topics(raw: dict, utterances: list[Utterance]) -> list[TopicSegment]:
    """시작점 목록 → 겹침도 구멍도 없는 구간.

    ① 목록에 없는 시작점은 버린다
    ② 발화 순서로 정렬하고 중복 시작점은 하나로 합친다
    ③ 첫 주제를 첫 발화까지 당긴다 — 앞 구간이 어떤 주제에도 안 속하면 그 안의 결정이 사라진다
    ④ `MIN_TOPIC_UTTERANCES` 발화보다 짧은 구간은 앞 주제에 합친다 — 주제명은 "X 외 N건"
    ⑤ 끝 = 다음 시작 직전. 마지막 주제는 마지막 발화까지
    ⑥ L3 에 넘길 발화에 앞 주제 끝 오버랩을 얹는다
    """
    order = {u.utterance_id: index for index, u in enumerate(utterances)}

    starts: list[tuple[int, str]] = []  # (발화 순번, 주제명)
    for item in raw.get("topics") or []:
        if not isinstance(item, dict):
            continue
        start_id = fmt.as_int(item.get("startUtteranceId"))
        topic = fmt.clip(fmt.as_text(item.get("topic")), TOPIC_MAX)
        if start_id is None or start_id not in order or not topic:
            continue
        starts.append((order[start_id], topic))

    if not starts:
        return []

    starts.sort(key=lambda pair: pair[0])

    # 같은 발화에서 시작하는 주제가 둘이면 구간 길이가 0인 주제가 생긴다. 먼저 온 것만 남긴다.
    deduped: list[tuple[int, str]] = []
    for index, topic in starts:
        if deduped and deduped[-1][0] == index:
            continue
        deduped.append((index, topic))

    # 첫 주제를 첫 발화로 당긴다. 모델이 인사·잡담 구간을 건너뛰고 시작점을 잡는 일이
    # 흔한데, 그 구간을 버리면 거기서 나온 결정이 어떤 주제에도 속하지 않게 된다.
    deduped[0] = (0, deduped[0][1])

    deduped = merge_short_topics(deduped, total=len(utterances))

    segments: list[TopicSegment] = []
    for seq, (start_index, topic) in enumerate(deduped, start=1):
        end_index = deduped[seq][0] - 1 if seq < len(deduped) else len(utterances) - 1
        overlap_start = max(0, start_index - OVERLAP_UTTERANCES)
        segments.append(
            TopicSegment(
                topic_seq=seq,
                topic=topic,
                start_utterance_id=utterances[start_index].utterance_id,
                end_utterance_id=utterances[end_index].utterance_id,
                utterance_ids=[u.utterance_id for u in utterances[overlap_start : end_index + 1]],
            )
        )
    return segments
