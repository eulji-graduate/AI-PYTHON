# 잇다(Z) · AI 계층 서버

회의 분석 파이프라인의 **AI EC2** 쪽. VAD 절단점, Gemini 계층(L1.5~L5), 벡터(Qdrant)를 담당한다.
Spring(BACKEND 리포)이 이 서버를 부르는 **단방향**이다.

담당 이태연 · 도메인 A(캡처 파이프라인) 내부 API 10개

---

## 이 서버가 하지 않는 것

| 하지 않는다 | 어디서 하나 | 왜 |
|---|---|---|
| 업무 DB 접속 | Spring | 상태 없는 계산 서버로 두면 계층을 특화 모델로 갈아끼워도 계약이 안 깨진다 |
| STT 호출 | Spring | AWS Transcribe IAM 롤이 Spring EC2 에 있다 |
| ffmpeg 조립 | Spring | 오디오 원본을 다루는 쪽이 한 곳이어야 한다 |
| 화자 귀속(L1) · 규칙 검사(L6) · 자동확정 게이트 | Spring | 코드 판정이다. LLM 을 쓰지 않는다 |
| 작업 큐 소비 | Spring (SQS) | 이 서버는 HTTP 동기 호출만 받는다 |

오디오는 항상 **S3 키**로 주고받는다 — 두 인스턴스 사이에 공유 볼륨이 없다.

---

## 엔드포인트

전부 `X-Internal-Token` 헤더 필수. 미구현 계층은 **501** 로 거절한다(200 + 빈 결과로 두면
Spring 이 "계층 정상 완료, 산출물 없음"으로 기록해 미구현이 품질 문제로 위장된다).
지금은 열한 개가 다 붙어 있어 501 을 내는 경로가 없다.

| ID | 경로 | 상태 |
|---|---|---|
| **AI-01** | `POST /internal/vad/cutpoint` | **구현됨** — 계약은 제안 상태(아래) |
| **AI-02** | `POST /internal/layers/l1-5/resolve-reference` | **구현됨** |
| **AI-03** | `POST /internal/layers/l2/segment-topics` | **구현됨** |
| **AI-04** | `POST /internal/layers/l3/summarize-topic` | **구현됨** |
| **AI-05** | `POST /internal/layers/l3-5/gate` | **구현됨** |
| **AI-06** | `POST /internal/layers/l4/extract-tuples` | **구현됨** |
| **AI-07** | `POST /internal/layers/l5/verify` | **구현됨** |
| **AI-08** | `POST /internal/vector/upsert` | **구현됨** |
| **AI-09** | `POST /internal/similar` | **구현됨** |
| **AI-10** | `GET /internal/health` | **구현됨** |
| **AI-11** | `POST /internal/layers/overview/summarize-meeting` | **구현됨** |

AI-10 이 돌려주는 `implemented` 목록과 실제 라우팅이 어긋나지 않는지 테스트가 검증한다
(`test_internal_auth.py`). 계층을 붙이고 목록을 잊으면 워커가 미구현 계층을 부른다.

`GET /health` 는 무인증 — ALB·컨테이너 liveness 용이다. 계층 수용 가능 여부는 AI-10 이 답한다.

### few-shot (AI-08 · AI-09)

계층은 `/internal/similar` 를 거치지 않고 `few_shot.lookup` 을 **프로세스 안에서** 부른다.
같은 인스턴스에 있는 벡터를 네트워크로 왕복시킬 이유가 없다. 그 엔드포인트는 같은 조회를
밖에서 떼어 볼 수 있게 열어 둔 것이다 — few-shot 이 이상할 때 계층 전체를 돌리지 않고
검색만 확인할 수 있어야 한다.

    저장  근거 발화 → 벡터(key) + payload = 확정 tuple      AI-08
    검색  새 발화   → 벡터(query) → 가장 가까운 key → payload 를 few-shot 으로   AI-09

⚠ 임베딩 대상은 **근거 발화 원문**이지 확정 tuple 이 아니다. 검색 시점에 손에 있는 것은
tuple 이 아니라 새 발화이므로, tuple 을 임베딩하면 쿼리와 키가 다른 공간에 놓여 유사도가
망가진다(V5.10 주석).

⚠ 조회 실패는 계층을 세우지 않는다. few-shot 은 정확도를 올리는 재료이지 계층의 입력이
아니라서, Qdrant 가 내려갔다고 여섯 계층이 전부 실패하면 파이프라인이 인덱스 하나에 인질로
잡힌다. 대신 로그로 크게 남긴다 — 빈 예시 목록은 "없다"와 "실패했다" 둘 다로 읽힌다.

---

## 계층 하나 추가하는 법

L4(`app/layers/l4.py`)가 나머지 계층의 원본 틀이다. 복제해서 **세 곳만** 바꾼다.

1. **`SPEC`** — 프롬프트 파일명과 `prompt_version`
2. **`build_response_schema()`** — 참석자·발화를 enum 으로 박은 응답 스키마
3. **후처리** — 파싱된 dict → 계층 DTO

호출·재시도·토큰 집계는 `app/layers/runner.py` 가, 참석자·발화 포맷과 값 되돌리기는
`app/layers/formatting.py` 가 공통으로 갖는다. 계층마다 따로 쓰면 같은 버그를 아홉 번
고치게 되고, 실제로는 아홉 번째를 빠뜨린다.

라우팅을 붙였으면 `internal.py` 의 `IMPLEMENTED` 도 함께 고친다 — 워커가 그 목록을 보고
호출 여부를 정한다.

### 후처리가 프롬프트보다 강한 자리들

계층마다 "프롬프트가 부탁하던 것"을 코드가 강제로 바꾼 지점이 하나씩 있다. 새 계층을
만들 때도 같은 것을 찾아볼 것 — 여기가 이 서버에서 정확도가 실제로 만들어지는 곳이다.

| 계층 | 부탁 대신 강제한 것 |
|---|---|
| L1.5 | `surface` 가 그 발화 문자열에 실제로 있는지 확인. 없으면 버린다 — 근거 강제를 문자열 수준까지 |
| L2 | 모델에게 **시작점만** 받고 구간은 코드가 계산. 겹침·구멍이 생길 방법 자체가 없다. 오버랩 3발화도 코드가 붙인다 |
| L3 | 분류가 깨지면 버리지 않고 `DISCUSSION` 으로 내린다 — 버리면 내용이 사라지고, 내리면 사람이 한 번 더 볼 뿐이다 |
| L3.5 | **판정이 없는 항목은 `DISCUSSED`.** 누락이 통과가 되지 않는다 — precision 우선의 실체 |
| L4 | 참석자·근거 발화를 enum 으로 박고 후처리에서 한 번 더 검증 |
| L5 | 한 관점이 실패하면 `agree=false`. 둘 다 실패하면 **계층 실패로 던진다** — 검증이 안 돈 것을 '갈렸다'로 기록하지 않는다 |

### 정확도 4원칙은 프롬프트가 아니라 스키마로 걸린다

| 원칙 | 어떻게 강제되나 |
|---|---|
| 닫힌 목록 | `assigneeCandidatePersonId` 를 참석자 id 의 문자열 enum 으로. `unknown_person` 탈출구 포함 |
| 근거 강제 | `evidenceUtteranceId` 를 전달된 발화 id enum 으로 + `required`. 후처리에서 한 번 더 검증 |
| 기권 우선 | `dueDate` 는 `required` 아님 + 형식 깨지면 `None`. 틀린 값보다 빈 값 |
| 한 계층 한 목표 | 계층당 프롬프트 1개. 요약·배정·분류를 섞지 않는다 |

> 구조화 출력의 enum 은 **문자열만** 지원한다. 그래서 personId·utteranceId 를 문자열로 받고
> 후처리에서 정수로 되돌린다. 이 제약이 `unknown_person` 을 문자열 탈출구로 두는 이유이기도 하다.

> 모델이 스스로 말한 `confidence` 숫자는 받지 않는다. 자기보고 신뢰도는 85~95 에 몰리고
> 실제 정확도와 맞지 않는다. 자동 확정은 Spring 이 코드로 판정하는 4조건으로 한다.

---

## 로컬 실행

전제: [uv](https://docs.astral.sh/uv/) (파이썬 설치까지 uv 가 한다)

```bash
uv sync
cp .env.example .env      # INTERNAL_TOKEN · GEMINI_API_KEY 채우기
uv run uvicorn app.main:app --reload --port 8000
```

`DRY_RUN=1` 이면 Gemini 를 부르지 않고 고정 스텁을 돌려준다 — 라우팅·스키마만 볼 때 쓴다.

```bash
uv run pytest        # 테스트
uv run ruff check .  # 린트
```

### 모델 고정

`GEMINI_MODEL` 기본값은 BACKEND 리포 `GeminiModels.PINNED` 와 같은 구체 버전이다.
`-latest` 별칭은 Google 이 실제 모델을 교체하는 순간 코드가 그대로인데도 결과가 바뀌어
품질 지표를 신뢰할 수 없게 만든다.

---

## 실패 정책

| 분류 | 예 | 처리 | HTTP |
|---|---|---|---|
| `TRANSIENT` | 타임아웃 · 5xx · 순단 | 백오프 3회 (2s · 8s · 30s, 지터 ±20%) | 503 |
| `RATE_LIMIT` | 429 | `Retry-After` 존중 | 503 |
| `PERMANENT` | 스키마 위반 · 컨텍스트 초과 · 빈 응답 | 즉시 실패 | 422 |

`retryable` 을 응답 본문에 넣는다. Spring 이 메시지 문자열로 재시도를 추측하게 두면
영구 실패를 세 번 재시도해 토큰만 태운다 — 판정은 실패를 만든 쪽이 한다.

**실패 사유에 제공자 응답 본문을 싣지 않는다.** L5 의 `results[].error` 는 오류 코드까지만
담는다(`RATE_LIMITED` 등) — 본문에 무엇이 들어 있을지 보증할 수 없고, 그게 우리 API 응답을
타고 나가면 되돌릴 수 없다. 본문이 필요한 진단은 서버 로그를 본다.

---

## 이미지 · 배포

```
PR       docker build 만 (자격증명 없음) — Dockerfile 이 깨졌는지 확인
develop  ECR z-ai 로 커밋 SHA 태그 푸시
main     빌드·푸시 + SSM 으로 AI EC2 배포 (tag:Role=ai · /opt/z-ai-worker/deploy.sh)
```

| 항목 | 값 |
|---|---|
| 레지스트리 | Private ECR **`z-ai`** (ap-northeast-2) |
| 태그 | **커밋 SHA 만.** `latest` 안 씀 · Immutable(재푸시 불가) |
| 보관 | Lifecycle 로 최신 20개 |
| 자격증명 | **GitHub OIDC 만.** 액세스 키·Docker Hub 토큰을 저장하지 않는다 |
| 리포 변수 | `AWS_DEPLOY_ROLE_ARN` · `AWS_REGION` · `ECR_REPOSITORY` |

> ⚠️ **이 프로젝트는 `pip` + `requirements.txt` 가 아니라 `uv` + `uv.lock` 이다.**
> `pip install -r requirements.txt` 로 이미지를 만들면 그런 파일이 없어서 실패한다.
> Dockerfile 은 `uv sync --locked --no-dev` 로 설치하고 `.venv` 를 실행 스테이지로 복사한다.

Immutable 이라 같은 커밋을 재실행하면 푸시가 실패한다. 그래서 워크플로가 **태그 존재를
먼저 확인하고 건너뛴다** — 재실행이 빨간불이 되면 사람이 빨간불을 무시하기 시작한다.

컨테이너 헬스체크는 무인증 `/health` 를 쓴다. AI-10(`/internal/health`)은 토큰이 필요해서
헬스체크로 쓸 수 없다 — 토큰을 이미지·컨테이너 설정에 심어야 하기 때문이다.

---

## AI-01 · VAD 절단점 계산

녹음을 10분 단위로 잘라 STT 에 넣는데, **아무 데나 자르면 말하는 중간이 잘린다.** 잘린 자리의
단어는 앞뒤 블록 어디서도 온전히 인식되지 않고 그 손실이 정본에 그대로 남아, 뒤 계층 전부가
그 문장을 못 본다. 그래서 경계 근처에서 사람이 말을 쉬는 지점을 찾는다.

    POST /internal/vad/cutpoint
    { "meetingId": 500, "bucket": "z-recordings",
      "s3Key": "org-1/vad/meeting-500/0040.wav",
      "windowStartOffsetMs": 580000, "targetOffsetMs": 600000, "minSilenceMs": 700 }

    → { "cutOffsetMs": 597340, "cutReason": "VAD_SILENCE", "silenceMs": 920 }
    → { "cutOffsetMs": 600000, "cutReason": "FALLBACK_OVERLAP", "silenceMs": null }

> ⚠ **이 계약은 제안이다.** 명세에 AI-01 은 한 줄짜리 표 항목("S3 키로 ±20초만 전달.
> onnxruntime 버전 silero-vad")과 동작 규칙만 있고 요청·응답 예시가 없다. 소비처인 Spring
> 블록 조립 경로(CAP-04·05·07·10 · 김현지)가 아직 없어 맞춰볼 상대도 없다. 합의되면 바뀌는
> 것은 `app/schemas/vad.py` 와 라우터뿐이고, 판정 로직은 그대로다.

**입력은 원본 청크가 아니라 Spring 이 잘라 만든 ±20초 wav 다.** 설계 문서 「전송 포맷 —
VAD 입력만 wav」가 정한 것이고, 자르고 변환하는 것은 Spring 의 ffmpeg 이 한다. 그래서
이 서버에는 ffmpeg 이 없고 탐색 창을 자를 이유도 없다 — 받은 wav 가 이미 그 창이다.

오프셋은 전부 **회의 시작 기준 경과 ms** 다. `windowStartOffsetMs` 가 그 wav 의 첫 샘플이
회의의 어디인지를 알려주고, 그래야 응답의 `cutOffsetMs` 를 Spring 이 stt_block 의
start_ms·end_ms 에 그대로 넣을 수 있다.

### 동작

1. `bucket`/`s3Key` 로 wav 를 받는다 — 본문에 싣지 않는다(명세 「파일 전달: S3 경유」)
2. 표준 라이브러리로 wav 헤더를 읽고 PCM 을 꺼낸다. **16kHz mono 16-bit 이 아니면 거절한다**
3. silero-vad(ONNX)로 프레임(32ms)마다 발화 확률을 낸다
4. `minSilenceMs` 이상 이어진 무음 중 **가장 긴 것**의 한가운데에서 자른다

### 정한 것들

- **무음 한가운데에서 자른다.** 시작·끝에서 자르면 디코딩 오차 몇십 ms 로 앞뒤 블록 중 한쪽이
  첫 음절을 먹는다. 가운데면 그 오차를 양쪽이 나눠 흡수한다.
- **가장 긴 무음을 고른다.** 설계 문서가 정한 규칙이다. 긴 침묵일수록 말이 실제로 끊긴
  자리이고, 절단 오차가 앞뒤 발화를 건드릴 여지도 적다. 길이가 같으면 목표에 가까운 쪽 —
  순서에 맡기면 창 앞쪽이 늘 이겨 블록이 짧아지는 편향이 생긴다.
- **못 찾는 것은 실패가 아니다.** 말이 끊이지 않는 회의에서는 700ms 무음이 없는 것이 정상이라
  목표 지점에서 자르고 `FALLBACK_OVERLAP` 을 남긴다. 여기서 에러를 주면 블록 조립이 멈춰
  그 회의가 STT 를 아예 못 받는다.
- **임계값(700ms)을 요청으로 연다.** 명세가 정한 초기값이지 불변값이 아니고, 서버 상수로
  박으면 조정할 때마다 배포해야 한다. 안 보내면 명세값으로 동작한다. 탐색 창(±20초)은 여기
  값이 아니다 — 창을 자르는 것은 Spring 이다.
- **오디오를 가공하지 않는다.** 형식이 다르면 리샘플링해 주지 않고 거절한다. 고쳐 주기 시작하면
  그 순간 오디오 처리가 두 곳에 생기고, "원본을 다루는 쪽은 한 곳"이라는 경계가 무너진다.

### 필요한 것

| | |
|---|---|
| VAD 입력 wav | Spring 이 만든다 — **최대 40초(±20초) · 16kHz mono 16-bit**. 넘으면 거절한다 |
| silero-vad ONNX | `models/silero_vad.onnx` — 이미지에 함께 넣는다([models/README.md](models/README.md)) |
| S3 권한 | AI EC2 의 IAM 롤. 액세스 키를 `.env` 에 두지 않는다 |
