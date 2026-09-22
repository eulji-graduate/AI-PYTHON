#!/usr/bin/env bash
# =============================================================================
# 잇다 AI 계층 서버(FastAPI) 배포 스크립트 — AI EC2 에서 실행된다
#
# 배치 위치: 이 파일을 AI EC2 의 /opt/z-ai-worker/deploy.sh 에 올려둔다
#           (프로비저닝 시 1회 배치 — ai-cd.yml 이 SSM 으로 이걸 호출만 한다)
#
# 호출 규약: sudo /opt/z-ai-worker/deploy.sh <커밋SHA> <전체 이미지 URI>
#           $1 = 커밋 SHA (배포 대상 커밋 · 로그와 검증용)
#           $2 = 전체 이미지 URI (<account>.dkr.ecr.<region>.amazonaws.com/z-ai:<sha>)
#
# ⚠️ **$2 가 이미지다. $1 을 이미지로 쓰면 안 된다.**
#    BACKEND 는 Docker Hub 라 `사용자/이름:SHA` 를 스크립트가 조립할 수 있지만, ECR 은
#    레지스트리 호스트(계정·리전)가 필요해서 CD 가 URI 를 통째로 넘긴다(ai-cd.yml 주석).
#    이 규약이 어긋나 있어서 초기 EC2 스크립트가 SHA 를 이미지 이름으로 읽었다.
#
# 전제:
#   - EC2 에 docker · docker compose · aws CLI 설치됨
#   - EC2 IAM Role 권한
#       ssm:GetParametersByPath  (아래 PARAM_PATH 대상)
#       ecr:GetAuthorizationToken · ecr:BatchGetImage · ecr:GetDownloadUrlForLayer
#     ⚠️ ECR 은 Docker Hub 와 달리 **익명 pull 이 안 된다.** 로그인 없이 compose pull 하면
#        "no basic auth credentials" 로 실패한다 — 아래 [2/5] 가 그 자리다.
# =============================================================================
set -euo pipefail
umask 077

# ------- 환경 -------
REGION="ap-northeast-2"
# AI 롤은 /z/ai/prod/* 만 읽는다(BACKEND application.yaml 머리말 — 같은 값을 양쪽에 둔다).
PARAM_PATH="/z/ai/prod/"
CONTAINER="z-ai"
APP_PORT="8000"
COMPOSE_FILE="/opt/z-ai-worker/docker-compose.yml"
ENV_FILE="/opt/z-ai-worker/.env.runtime"
# --------------------

# ── 인자 검증 ────────────────────────────────────────────────────────────────
if [ "$#" -lt 2 ] || [ -z "${1:-}" ] || [ -z "${2:-}" ]; then
  echo "사용법: $(basename "$0") <커밋SHA> <전체 이미지 URI>" >&2
  echo "  예: $(basename "$0") a1b2c3d 123456789012.dkr.ecr.ap-northeast-2.amazonaws.com/z-ai:a1b2c3d" >&2
  exit 1
fi
IMAGE_TAG="$1"
AI_IMAGE_URI="$2"

# 커밋 SHA 형식(7~40 hex)만 허용한다. latest·develop 같은 별칭을 넘기면 커밋과 무관한
# 이미지가 뜰 수 있다 — ECR 리포가 Immutable 이라 태그는 항상 SHA 다(ai-cd.yml).
if ! printf '%s' "${IMAGE_TAG}" | grep -Eq '^[0-9a-f]{7,40}$'; then
  echo "오류: 첫 번째 인자는 커밋 SHA(7~40자리 hex)여야 합니다: '${IMAGE_TAG}'" >&2
  exit 1
fi

# URI 가 그 SHA 를 가리키는지 확인한다. CD 가 둘을 따로 만들어 넘기므로 어긋날 수 있고,
# 어긋난 채로 배포하면 "배포했다는 커밋"과 "실제로 도는 커밋"이 달라진다.
if [ "${AI_IMAGE_URI##*:}" != "${IMAGE_TAG}" ]; then
  echo "오류: 이미지 태그가 커밋 SHA 와 다릅니다 — SHA=${IMAGE_TAG} URI=${AI_IMAGE_URI}" >&2
  exit 1
fi

# 레지스트리 호스트만 잘라낸다(로그인 대상). URI 의 첫 '/' 앞이 그것이다.
ECR_REGISTRY="${AI_IMAGE_URI%%/*}"
if [ "${ECR_REGISTRY}" = "${AI_IMAGE_URI}" ]; then
  echo "오류: 이미지 URI 에 레지스트리 호스트가 없습니다: '${AI_IMAGE_URI}'" >&2
  exit 1
fi

export AI_IMAGE="${AI_IMAGE_URI}"
export RUNTIME_ENV_FILE="${ENV_FILE}"

echo "=== [1/5] SSM Parameter Store 에서 설정 로드: ${PARAM_PATH} ==="
# --output text 는 컬럼을 TAB 으로 구분 → IFS=TAB 으로 값 안의 공백 보존
tmp_env="$(mktemp "${ENV_FILE}.tmp.XXXXXX")"
trap 'rm -f -- "${tmp_env}"' EXIT
aws ssm get-parameters-by-path \
  --path "${PARAM_PATH}" \
  --recursive \
  --with-decryption \
  --region "${REGION}" \
  --query "Parameters[].[Name,Value]" \
  --output text \
| while IFS=$'\t' read -r name value; do
    key="$(basename "${name}")"          # /z/ai/prod/GEMINI_API_KEY -> GEMINI_API_KEY
    echo "${key}=${value}"
  done > "${tmp_env}"

if [[ ! -s "${tmp_env}" ]]; then
  echo "오류: Parameter Store 에서 환경변수를 가져오지 못했습니다(${PARAM_PATH})." >&2
  exit 1
fi

chmod 600 "${tmp_env}"
mv -f "${tmp_env}" "${ENV_FILE}"
trap - EXIT

echo "=== [2/5] ECR 로그인: ${ECR_REGISTRY} ==="
# ⚠️ 이 단계를 빼면 다음 pull 이 "no basic auth credentials" 로 실패한다.
# 자격증명은 EC2 IAM 롤에서 온다 — 액세스 키를 EC2 에 두지 않는다.
aws ecr get-login-password --region "${REGION}" \
  | docker login --username AWS --password-stdin "${ECR_REGISTRY}"

echo "=== [3/5] 이미지 pull: ${AI_IMAGE} ==="
docker compose -f "${COMPOSE_FILE}" pull ai qdrant

echo "=== [4/5] 컨테이너 교체 ==="
docker compose -f "${COMPOSE_FILE}" up -d --remove-orphans ai

echo "=== [5/5] 헬스체크 대기 ==="
# 무인증 /health 를 쓴다. AI-10(/internal/health)은 X-Internal-Token 이 필요해서
# 헬스체크로 쓸 수 없다(Dockerfile HEALTHCHECK 와 같은 이유).
for _ in $(seq 1 30); do
  if curl -fsS "http://localhost:${APP_PORT}/health" >/dev/null 2>&1; then
    echo "배포 성공 — 헬스체크 통과 (커밋 ${IMAGE_TAG})"
    exit 0
  fi
  sleep 2
done

echo "헬스체크 실패 — 로그 확인 필요" >&2
docker logs --tail 50 "${CONTAINER}" >&2 || true
exit 1
