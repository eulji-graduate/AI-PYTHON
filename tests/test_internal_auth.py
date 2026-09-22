"""내부 API 인증 — 토큰 없이 통과하는 경로가 없어야 한다.

보안그룹으로 인바운드를 좁혀도 이 테스트를 둔다. 같은 VPC 안의 다른 인스턴스가
뚫렸을 때 방어선이 하나도 없는 상태가 되지 않게.
"""

import pytest
from fastapi.testclient import TestClient

from app.config import get_settings
from app.layers.few_shot import _STORES
from app.main import app

TOKEN = "test-internal-token"


@pytest.fixture(autouse=True)
def _settings(monkeypatch):
    monkeypatch.setenv("INTERNAL_TOKEN", TOKEN)
    monkeypatch.setenv("GEMINI_API_KEY", "")
    monkeypatch.setenv("DRY_RUN", "1")
    # AI-08·09 는 벡터 저장소를 탄다. 로컬 모드로 두면 실물 클라이언트를 그대로 쓰면서
    # 외부 의존 없이 라우팅을 확인할 수 있다.
    monkeypatch.setenv("QDRANT_URL", ":memory:")
    get_settings.cache_clear()
    _STORES.clear()
    yield
    get_settings.cache_clear()
    _STORES.clear()


@pytest.fixture
def client():
    return TestClient(app)


def test_토큰_없으면_401(client):
    response = client.get("/internal/health")

    assert response.status_code == 401


def test_틀린_토큰이면_401(client):
    response = client.get("/internal/health", headers={"X-Internal-Token": "wrong"})

    assert response.status_code == 401


def test_올바른_토큰이면_통과(client):
    response = client.get("/internal/health", headers={"X-Internal-Token": TOKEN})

    assert response.status_code == 200
    assert response.json()["status"] == "UP"


def test_인프라_liveness는_무인증(client):
    # ALB·컨테이너 헬스체크는 토큰을 들고 있을 수 없다.
    response = client.get("/health")

    assert response.status_code == 200


def test_AI_01_은_스텁이_아니라_실제_핸들러에_닿는다(client):
    """AI-01 은 스모크로 200 을 요구할 수 없다 — S3·ffmpeg·모델 파일이 전부 있어야 한다.

    그래도 **라우팅이 실제 핸들러에 닿는지**는 지켜야 한다. 스텁이 남아 있으면 501 이고,
    스키마가 요청을 거르면 FastAPI 의 검증 422(`detail`)라 핸들러를 지나지 못한다.
    둘 다 아니고 LayerError 의 모양(`code`)이 나오면 핸들러 안까지 들어간 것이다.

    이 환경에서는 boto3 가 없어 S3_RUNTIME_MISSING(PERMANENT → 422)이 나온다. 그 코드 자체를
    못박지는 않는다 — 의존성이 깔린 환경에서는 S3 단계에서 다른 코드가 나오는 것이 정상이다.
    """
    response = client.post(
        "/internal/vad/cutpoint",
        headers={"X-Internal-Token": TOKEN},
        json={
            "meetingId": 500,
            "bucket": "z-recordings",
            "s3Key": "org-1/vad/meeting-500/0040.wav",
            "windowStartOffsetMs": 580_000,
            "targetOffsetMs": 600_000,
        },
    )

    assert response.status_code != 501, "미구현 스텁이 남아 있다"
    body = response.json()

    # S3·모델·객체가 다 있는 환경에서는 200 이 정상이다. 그때 CutpointResponse 에는 code 가
    # 없으므로, code 만 요구하면 제대로 도는 환경에서 오히려 실패한다(CodeRabbit PR #15).
    if response.status_code == 200:
        assert "cutReason" in body
    else:
        assert "code" in body, f"핸들러에 닿지 못했다: {response.status_code} {response.text}"


PARTICIPANTS = [{"personId": 7, "name": "김서준"}]
UTTERANCES = [{"utteranceId": 8812, "speakerId": 7, "text": "제가 정리할게요"}]

# 각 계층을 **실제로 통과하는** 최소 요청. 빈 본문을 보내면 FastAPI 가 핸들러에 닿기 전에
# 422 를 돌려주므로, 라우팅이 미구현 스텁에 걸려 있어도 테스트가 통과해버린다.
# DRY_RUN=1 이라 Gemini 는 호출되지 않고 계층이 고정 스텁으로 200 을 만든다.
IMPLEMENTED_CALLS = {
    "AI-02": (
        "/internal/layers/l1-5/resolve-reference",
        {"tenantId": 7, "meetingId": 500, "utterances": UTTERANCES, "participants": PARTICIPANTS},
    ),
    "AI-03": (
        "/internal/layers/l2/segment-topics",
        {"tenantId": 7, "meetingId": 500, "utterances": UTTERANCES},
    ),
    # 발화가 없는 유일한 본문이다 — 개요는 확정 항목만 읽는다(schemas/overview.py 주석).
    "AI-11": (
        "/internal/layers/overview/summarize-meeting",
        {
            "tenantId": 7,
            "meetingId": 500,
            "topics": [
                {
                    "topicSeq": 1,
                    "topic": "제품 로드맵",
                    "items": [{"itemType": "DECISION", "content": "A/B 테스트 도구를 비교하기로 함"}],
                }
            ],
            "participants": PARTICIPANTS,
        },
    ),
    "AI-04": (
        "/internal/layers/l3/summarize-topic",
        {
            "tenantId": 7,
            "meetingId": 500,
            "topicSeq": 1,
            "topic": "제품 로드맵",
            "utterances": UTTERANCES,
            "participants": PARTICIPANTS,
        },
    ),
    "AI-05": (
        "/internal/layers/l3-5/gate",
        {
            "tenantId": 7,
            "meetingId": 500,
            "topic": "제품 로드맵",
            "items": [
                {
                    "itemKey": "1",
                    "itemType": "DECISION",
                    "content": "A/B 테스트 도구 비교",
                    "evidenceUtteranceId": 8812,
                }
            ],
            "utterances": UTTERANCES,
            "participants": PARTICIPANTS,
        },
    ),
    "AI-06": (
        "/internal/layers/l4/extract-tuples",
        {
            "tenantId": 7,
            "meetingId": 500,
            "topic": "제품 로드맵",
            "items": [],
            "utterances": UTTERANCES,
            "participants": PARTICIPANTS,
        },
    ),
    "AI-07": (
        "/internal/layers/l5/verify",
        {
            "tenantId": 7,
            "meetingId": 500,
            "topic": "제품 로드맵",
            "tuple": {
                "title": "A/B 테스트 도구 비교 정리",
                "assigneeCandidatePersonId": 7,
                "assigneeSource": "FIRST_PERSON",
                "evidenceUtteranceId": 8812,
            },
            "utterances": UTTERANCES,
            "participants": PARTICIPANTS,
        },
    ),
    # 계층이 아니라 벡터 API 다. 여기 함께 두는 이유는 implemented 목록이 계층만 담는 것이
    # 아니기 때문이다 — 목록에 있는데 라우팅이 없으면 워커가 501 을 맞는 것은 똑같다.
    "AI-08": (
        "/internal/vector/upsert",
        {
            "items": [
                {
                    "vectorId": 1,
                    "companyId": 7,
                    "layer": "L4",
                    "inputText": "제가 정리할게요",
                    "payload": {"title": "정리", "assigneeMemberId": 7},
                }
            ]
        },
    ),
    "AI-09": (
        "/internal/similar",
        {"companyId": 7, "layer": "L4", "queryText": "제가 정리할게요"},
    ),
}

# 본문 없이 GET 으로 확인되는 계층. 위 표와 합쳐 implemented 전체와 대조한다.
BODYLESS_IMPLEMENTED = {"AI-10"}

# 외부 자원이 있어야 200 이 되는 API. 스모크 호출로 200 을 요구할 수 없다 — AI-01 은
# S3·ffmpeg·모델 파일이 전부 있어야 한다. 라우팅이 실제 핸들러에 닿는지는 위의 전용
# 테스트가 따로 본다. 여기서 빼면 implemented 목록 대조가 그만큼 헐거워지므로 남겨 둔다.
EXTERNAL_DEPENDENCY_IMPLEMENTED = {"AI-01"}


def test_health의_구현목록이_실제_라우팅과_일치한다(client):
    """AI-10 이 답하는 `implemented` 와 실제로 도는 계층이 **양방향으로** 같아야 한다.

    워커가 이 목록을 보고 호출 여부를 정한다. 어긋나는 두 방향 모두 사고다.
      목록에만 있음 → 워커가 미구현 계층을 부른다(501 폭풍)
      라우팅에만 있음 → 구현해둔 계층을 워커가 건너뛴다

    각 계층을 **유효한 본문으로 실제 호출해서** 200 을 받는지 본다. 빈 본문으로
    `!= 501` 만 보면 FastAPI 가 핸들러 전에 422 를 내므로, 라우팅이 미구현 스텁에
    걸려 있어도 통과한다 — 아무것도 검증하지 못한다(CodeRabbit PR #4 지적).
    """
    implemented = client.get("/internal/health", headers={"X-Internal-Token": TOKEN}).json()["implemented"]

    assert set(implemented) == set(IMPLEMENTED_CALLS) | BODYLESS_IMPLEMENTED | EXTERNAL_DEPENDENCY_IMPLEMENTED

    for api_id, (path, body) in IMPLEMENTED_CALLS.items():
        response = client.post(path, headers={"X-Internal-Token": TOKEN}, json=body)
        assert response.status_code == 200, f"{api_id}({path}) → {response.status_code} {response.text}"


def test_L4는_발화가_없으면_모델을_부르지_않는다(client):
    body = {
        "tenantId": 7,
        "meetingId": 500,
        "topic": "제품 로드맵",
        "items": [],
        "utterances": [],
        "participants": [{"personId": 7, "name": "김서준"}],
        "view": "EXTRACT",
    }

    response = client.post(
        "/internal/layers/l4/extract-tuples",
        headers={"X-Internal-Token": TOKEN},
        json=body,
    )

    # GEMINI_API_KEY 가 비어 있어도 200 이어야 한다 = 호출 자체가 없었다는 뜻.
    assert response.status_code == 200
    assert response.json()["tuples"] == []
