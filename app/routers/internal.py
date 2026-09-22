"""내부 API — Spring EC2 → AI EC2 (AI-01 ~ AI-11).

전부 `X-Internal-Token` 을 요구한다.

AI-01 이 붙으면서 미구현 스텁이 사라졌다. 다시 생기면 501 로 거절할 것 — 200 에 빈 결과를
돌려주면 Spring 오케스트레이션이 "계층이 정상 완료했고 산출물이 없다"로 기록해 버려서,
미구현이 품질 문제로 위장된다.
"""

from functools import lru_cache

from fastapi import APIRouter, Depends

from app.clients.embedding import TASK_DOCUMENT, EmbeddingClient
from app.clients.gemini import GeminiClient
from app.config import Settings, get_settings
from app.layers import few_shot, l1_5, l2, l3, l3_5, l4, l5, overview
from app.layers.runner import LayerRunner
from app.schemas.l1_5 import ResolveReferenceRequest, ResolveReferenceResponse
from app.schemas.l2 import SegmentTopicsRequest, SegmentTopicsResponse
from app.schemas.l3 import SummarizeTopicRequest, SummarizeTopicResponse
from app.schemas.l3_5 import GateRequest, GateResponse
from app.schemas.l4 import ExtractTuplesRequest, ExtractTuplesResponse
from app.schemas.l5 import VerifyRequest, VerifyResponse
from app.schemas.overview import SummarizeMeetingRequest, SummarizeMeetingResponse
from app.schemas.vad import CutpointRequest, CutpointResponse
from app.schemas.vector import (
    SimilarRequest,
    SimilarResponse,
    VectorUpsertRequest,
    VectorUpsertResponse,
    VectorUpsertResult,
)
from app.security import require_internal_token
from app.vad import service as vad_service

router = APIRouter(
    prefix="/internal",
    dependencies=[Depends(require_internal_token)],
)

# AI-10 이 돌려주는 목록. 라우팅과 따로 관리하면 하나를 붙이고 다른 하나를 잊는다 —
# 그러면 워커가 "구현됐다"를 보고 호출했다가 501 을 받는다.
IMPLEMENTED = [
    "AI-01",
    "AI-02",
    "AI-03",
    "AI-04",
    "AI-05",
    "AI-06",
    "AI-07",
    "AI-08",
    "AI-09",
    "AI-10",
    "AI-11",
]


@lru_cache(maxsize=1)
def _gemini_client() -> GeminiClient:
    """제공자 클라이언트를 **프로세스에 하나로** 유지한다.

    ⚠ 요청마다 만들면 두 가지가 깨진다 —

        동시 호출 상한   인스턴스마다 자기 몫의 세마포어를 갖게 되어 아무것도 제한하지 않는다
        연결 재사용      genai.Client 가 매 요청 새로 만들어져 커넥션 풀이 그때마다 버려진다

    설정은 get_settings 가 이미 캐시하므로(lru_cache) 여기서 다시 받아도 같은 객체다.
    """
    return GeminiClient(get_settings())


def get_runner(settings: Settings = Depends(get_settings)) -> LayerRunner:
    return LayerRunner(_gemini_client(), settings)


# ── 계층 (파이프라인 순서대로) ────────────────────────────────────────────────
# 전부 l4.py 의 틀을 복제한 것이다 — SPEC · 응답 스키마 · 후처리 세 곳만 다르다.


# AI-02 · L1.5 지시어 해소 — L4 의 담당자 판정이 여기 의존한다.
@router.post("/layers/l1-5/resolve-reference", response_model=ResolveReferenceResponse)
async def resolve_reference(
    request: ResolveReferenceRequest,
    runner: LayerRunner = Depends(get_runner),
) -> ResolveReferenceResponse:
    return await l1_5.resolve_reference(request, runner)


# AI-03 · L2 주제 분할 — 오버랩 3발화는 응답 후처리가 붙인다(프롬프트 부탁이 아니다).
@router.post("/layers/l2/segment-topics", response_model=SegmentTopicsResponse)
async def segment_topics(
    request: SegmentTopicsRequest,
    runner: LayerRunner = Depends(get_runner),
) -> SegmentTopicsResponse:
    return await l2.segment_topics(request, runner)


# AI-04 · L3 주제별 정리 — 주제마다 한 번씩 호출된다(명세 「주제별 N회」).
@router.post("/layers/l3/summarize-topic", response_model=SummarizeTopicResponse)
async def summarize_topic(
    request: SummarizeTopicRequest,
    runner: LayerRunner = Depends(get_runner),
) -> SummarizeTopicResponse:
    return await l3.summarize_topic(request, runner)


# AI-05 · L3.5 확정/논의 게이트 — 판정이 없는 항목은 DISCUSSED 로 채운다(precision 우선).
@router.post("/layers/l3-5/gate", response_model=GateResponse)
async def gate(
    request: GateRequest,
    runner: LayerRunner = Depends(get_runner),
) -> GateResponse:
    return await l3_5.gate(request, runner)


# AI-06 · L4 tuple 추출 — 다른 계층의 원본 틀.
@router.post("/layers/l4/extract-tuples", response_model=ExtractTuplesResponse)
async def extract_tuples(
    request: ExtractTuplesRequest,
    runner: LayerRunner = Depends(get_runner),
) -> ExtractTuplesResponse:
    return await l4.extract_tuples(request, runner)


# AI-07 · L5 관점 다변화 검증 — 두 관점을 이 안에서 조합한다(Spring 왕복 1회).
@router.post("/layers/l5/verify", response_model=VerifyResponse)
async def verify(
    request: VerifyRequest,
    runner: LayerRunner = Depends(get_runner),
) -> VerifyResponse:
    return await l5.verify(request, runner)


# AI-11 · OVERVIEW 회의 개요 — 파이프라인의 맨 끝. 회의당 한 번.
#
# 발화를 받지 않는 유일한 계층이다. 전사를 다시 읽히면 주제 요약과 다른 말을 하는 개요가
# 나오고, 개요 칸은 Spring 이 이 출력으로 덮는 자리라 그 값을 되먹이면 자기 출력을 다시
# 압축한다(schemas/overview.py 주석).
#
# 실패해도 Spring 은 회의를 완료로 둔다 — 개요 칸에 주제 요약을 이어 붙인 값이 남아 있다.
# 그래서 이 엔드포인트가 없던 동안에도 파이프라인은 끝까지 돌았다(404 → 계층만 FAILED).
@router.post("/layers/overview/summarize-meeting", response_model=SummarizeMeetingResponse)
async def summarize_meeting(
    request: SummarizeMeetingRequest,
    runner: LayerRunner = Depends(get_runner),
) -> SummarizeMeetingResponse:
    return await overview.summarize_meeting(request, runner)


# ── AI-10 · 헬스체크 — 운영 점검·구현 목록 노출용 ─────────────────────────────
# 인프라 liveness(/health, 무인증)와 구분한다. 이쪽은 "계층을 받을 준비가 됐는지"라
# 토큰을 요구하고 모델 설정 여부까지 본다 — status: UP 만으로는 **떠 있지만 일은 못 하는**
# 서버(키 없음·dryRun·계층 덜 배포됨)를 구분할 수 없다.
#
# ⚠ Spring 소비처 없음(의도) — 2026-08-16 전수 대조로 확인했다. 예전 주석은 '워커 백오프
# 판단용'이라 적었으나 그렇게 부르는 코드는 없었다. 지금 이 응답은 사람이 배포를 확인할 때
# 읽는다. 소비처를 지어내지 않으려고 그 문구를 걷어낸다.
#
# 응답 필드는 줄이지 않는다 — 언젠가 기계가 읽는다면 geminiConfigured·dryRun·implemented 가
# 판단의 핵심이 된다. 다만 그 자리는 매 주기 백오프가 아니라 **기동·배포 게이트**가 맞다고
# 본다(주기마다 왕복이 붙고, 체크와 호출 사이 틈에서 조용히 일을 건너뛰는 실패가 생긴다).
@router.get("/health")
async def internal_health(settings: Settings = Depends(get_settings)) -> dict:
    return {
        "status": "UP",
        "model": settings.gemini_model,
        "geminiConfigured": bool(settings.gemini_api_key),
        "dryRun": settings.dry_run,
        "implemented": IMPLEMENTED,
    }


# ── AI-08 · 벡터 저장 ─────────────────────────────────────────────────────────
# Spring 이 라벨을 MySQL 에 먼저 커밋한 뒤 이걸 부른다(V5.10 · meeting_tuple_vector).
# 원본이 저쪽에 있으므로 여기서 실패해도 라벨은 안전하고, vector_synced=false 로 남아
# 재시도 워커가 다시 넘긴다. 그래서 **부분 성공을 행 단위로 답한다** — 배치 전체를 하나의
# 성공/실패로 답하면 어느 행을 다시 보내야 하는지 몰라 전부 재시도하게 된다.
@router.post("/vector/upsert", response_model=VectorUpsertResponse)
async def vector_upsert(
    request: VectorUpsertRequest,
    settings: Settings = Depends(get_settings),
) -> VectorUpsertResponse:
    embedder = EmbeddingClient(settings)
    if not request.items:
        return VectorUpsertResponse(upserted=[], model=embedder.model_name)

    # 임베딩 대상은 **근거 발화 원문**이다. 확정 tuple 은 payload 로만 따라간다 —
    # tuple 을 임베딩하면 쿼리(새 발화)와 다른 공간에 놓여 유사도가 망가진다.
    vectors = await embedder.embed([item.input_text for item in request.items], task_type=TASK_DOCUMENT)

    store = few_shot.store_of(settings)
    results = []
    for item, vector in zip(request.items, vectors, strict=True):
        point_id = await store.upsert(
            vector_id=item.vector_id,
            vector=vector,
            company_id=item.company_id,
            layer=item.layer,
            provenance=item.provenance,
            input_text=item.input_text,
            payload=item.payload,
            dept_id=item.dept_id,
        )
        results.append(VectorUpsertResult(vector_id=item.vector_id, point_id=point_id))

    return VectorUpsertResponse(upserted=results, model=embedder.model_name)


# ── AI-09 · 유사 발화 조회 ────────────────────────────────────────────────────
# 계층은 이 엔드포인트를 거치지 않고 few_shot.lookup 을 직접 부른다(같은 프로세스 안이라
# 왕복시킬 이유가 없다). 이 엔드포인트는 **같은 조회를 밖에서 확인할 수 있게** 열어 둔다 —
# few-shot 이 이상할 때 계층 전체를 돌리지 않고 검색만 떼어 볼 수 있어야 한다.
# ⚠ Spring 소비처 없음(의도) — 2026-08-16 전수 대조로 확인했다. 고아로 보인다고 지우지 마라.
@router.post("/similar", response_model=SimilarResponse)
async def similar(
    request: SimilarRequest,
    settings: Settings = Depends(get_settings),
) -> SimilarResponse:
    examples = await few_shot.lookup(
        tenant_id=request.company_id,
        layer=request.layer,
        query_text=request.query_text,
        dept_id=request.dept_id,
        top_k=request.top_k,
        provenance=request.provenance,
        settings=settings,
    )
    return SimilarResponse(examples=examples, model=settings.gemini_embed_model)


# ── AI-01 · VAD 절단점 계산 ───────────────────────────────────────────────────
# 계층이 아니라 **블록 조립의 전처리**다. 10분 경계 근처에서 사람이 말을 쉬는 지점을 찾아
# 거기서 자른다 — 아무 데나 자르면 말 중간이 잘리고, 잘린 단어는 앞뒤 블록 어디서도 온전히
# 인식되지 않아 그 손실이 정본에 그대로 남는다.
#
# 못 찾는 것은 실패가 아니다. 말이 끊이지 않는 회의에서는 700ms 무음이 없는 것이 정상이고,
# 그때는 목표 지점에서 자르고 FALLBACK_OVERLAP 을 남긴다. 여기서 에러를 주면 블록 조립이
# 통째로 멈춰 그 회의가 STT 를 아예 못 받는다.
#
# ⚠ 요청·응답 계약은 명세에 없어 **제안 상태**다(schemas/vad.py). 소비처인 Spring 블록
# 조립 경로가 아직 없어 맞춰볼 상대가 없다.
@router.post("/vad/cutpoint", response_model=CutpointResponse)
async def vad_cutpoint(
    request: CutpointRequest,
    settings: Settings = Depends(get_settings),
) -> CutpointResponse:
    return await vad_service.find_cutpoint(request, settings)
