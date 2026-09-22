# syntax=docker/dockerfile:1
#
# AI 계층 서버 이미지. ECR z-ai 로 커밋 SHA 태그만 올린다(latest 없음 · Immutable).
#
# ⚠️ 이 프로젝트는 pip + requirements.txt 가 아니라 **uv + uv.lock** 이다.
#    `pip install -r requirements.txt` 로 만들면 그런 파일이 없어서 실패한다.

# ── 빌드 스테이지 ─────────────────────────────────────────────────────────────
FROM python:3.12-slim AS builder

# uv 버전을 고정한다 — 로컬·CI·이미지가 같은 의존성 해석 결과를 갖게 한다.
# ghcr.io/astral-sh/uv 이미지에서 COPY 하는 방법도 있지만, 태그를 검증할 수 없는
# 외부 이미지를 빌드 경로에 넣지 않으려고 PyPI 에서 고정 버전으로 설치한다.
RUN pip install --no-cache-dir uv==0.11.28

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /app

# 의존성 레이어를 먼저 만든다 — 소스만 바뀌면 이 레이어는 캐시에서 재사용된다.
# .python-version 도 함께 넣는다(uv 가 인터프리터를 고를 때 본다. 베이스가 3.12 라
# 새로 내려받지 않는다 — UV_PYTHON_DOWNLOADS=never).
COPY pyproject.toml uv.lock .python-version ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-install-project

COPY app ./app
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev

# ── 실행 스테이지 ─────────────────────────────────────────────────────────────
FROM python:3.12-slim

# ⚠ ffmpeg 을 넣지 않는다. AI-01 이 받는 것은 원본 청크(opus)가 아니라 **Spring 이 잘라 만든
# ±20초 wav** 이고(설계 문서 「전송 포맷 — VAD 입력만 wav」), 자르고 변환하는 것은 Spring 의
# ffmpeg 이 한다 — 오디오 원본을 다루는 쪽이 한 곳이어야 한다.
#
# 여기에 넣으면 같은 오디오 처리가 두 곳에 생기고 이미지도 100MB 남짓 커진다.

# 비루트로 돈다. 이 서버는 인터넷에 노출되지 않지만(Spring SG 에서만 인바운드),
# 컨테이너가 뚫렸을 때 범위를 좁히는 비용이 거의 없다.
RUN groupadd --system --gid 1001 app \
    && useradd --system --uid 1001 --gid app --no-create-home --shell /usr/sbin/nologin app

WORKDIR /app
COPY --from=builder --chown=app:app /app /app

# silero-vad ONNX 모델. **이미지에 함께 넣는다** — 런타임에 받아오면 모델이 바뀔 때 절단점이
# 배포와 무관하게 조용히 달라지고, "어제와 오늘의 블록 경계가 다른" 이유를 아무도 못 찾는다.
# 파일이 없으면 AI-01 이 VAD_MODEL_MISSING(PERMANENT)으로 명확히 거절한다.
COPY --chown=app:app models/ /app/models/

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

USER app
EXPOSE 8000

# liveness 는 무인증 /health 를 쓴다. AI-10(/internal/health)은 X-Internal-Token 이
# 필요해서 헬스체크로 쓸 수 없다 — 토큰을 이미지·컨테이너 설정에 심어야 하기 때문이다.
# curl 을 설치하지 않으려고 파이썬 표준 라이브러리로 확인한다(이미지 표면 최소화).
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD ["python", "-c", "import sys,urllib.request; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3).status == 200 else 1)"]

# 워커 수는 컨테이너 밖에서 정한다(t3.medium 에 Qdrant 와 같이 뜨므로 기본 1개).
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
