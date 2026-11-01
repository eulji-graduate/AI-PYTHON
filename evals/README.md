# AI 계층 반복 측정 (`evals/`)

프롬프트·모델·후처리를 고친 뒤 **좋아졌는지**를 재는 틀이다. 시연용 샘플 회의 한 건을 서버에 여러 번
흘려 계층별 점수와 흔들림을 모은다.

## 왜 반복해서 재나

같은 입력에 같은 출력이 나오지 않는다. `app/clients/gemini.py` 주석의 실측(L3.5, temperature 0 +
seed 고정에서도 14건 중 매회 3건이 뒤집힘)이 근거다. 그래서 이 틀의 판단 규칙은 하나다.

> **프롬프트·모델을 바꾼 뒤의 평균 차이가 baseline의 `spread`(최댓값-최솟값)보다 작으면 효과로 읽지 않는다.**

## 구성

| 파일 | 내용 |
|---|---|
| `fixtures/demo_meeting.json` | 샘플 회의 "앱 개발 착수 - 기획 회의" (발화 27, 참석자 3, 회의일 2026-11-09 월) |
| `fixtures/demo_meeting.gold.json` | 정답 라벨 (주제 4, 항목 8, 배정 5, 지시어 3) |
| `scoring.py` | 채점 순수 함수 |
| `run_eval.py` | N번 실행해 요약을 내는 스크립트 |

샘플 회의에는 일부러 이런 것들을 섞었다.

- 본인 지정("제가 ~까지 할게요") / 직접 지목("하윤님이 ~ 맡아 주세요")
- 지시어("그것도", "그건") — 앞 발화를 읽어야 풀린다
- 참석자가 아닌 사람 이름(강서연) — 담당자로 밀어 넣으면 안 된다
- 결정 못 한 논의(소셜 로그인, 테스트 서버 담당 미정) — **배정으로 뽑히면 안 된다**
- 상대 날짜("이번 주 금요일", "다음 주 월요일", "내일") — 회의일 기준으로 계산돼야 한다

## 쓰는 법

서버는 따로 띄운다. 토큰은 환경변수로만 준다(인자·파일에 적지 않는다).

```bash
# 배선 확인 (고정 스텁이라 점수는 의미 없다)
INTERNAL_TOKEN=test DRY_RUN=1 VAD_MODEL_PATH=models/silero_vad.onnx uv run uvicorn app.main:app --port 8000

# 다른 터미널에서
INTERNAL_TOKEN=test uv run python -m evals.run_eval --runs 5 --out evals/out/dry.json
```

실제 품질을 재려면 `.env`(또는 환경변수)에 `GEMINI_API_KEY`와 `INTERNAL_TOKEN`을 넣고 `DRY_RUN` 없이
서버를 띄운 뒤 같은 명령을 돌린다. 결과 파일(`evals/out/`)은 `.gitignore` 대상이다.

옵션: `--runs N` (기본 5), `--layers l4,l5` (일부 계층만), `--base-url`, `--timeout`.

## 무엇을 재나

입력을 앞 계층 출력이 아니라 **정답 라벨**에서 만든다(격리 모드). 앞 계층이 틀려도 뒤 계층 점수가
같이 무너지지 않아서 어느 계층을 고쳐야 하는지 보인다.

| 계층 | 지표 |
|---|---|
| L1.5 지시어 | `found_rate`, `resolved_rate`(해소 문장에 핵심어가 모두 있는가) |
| L2 주제 분할 | `start_precision`, `start_recall` (시작 발화 id 일치) |
| L3.5 게이트 | `accuracy`, **`false_confirmed`**(논의를 확정으로 올린 수), `missed_confirmed` |
| L4 배정 | `precision`, `recall`, `f1`, `source_accuracy`, `due_accuracy` — (담당자, 근거 발화)로 짝을 짓는다 |
| L5 검증 | `positive_agree_rate`(맞는 배정은 합의), `negative_reject_rate`(일부러 틀린 담당자는 불일치) |
| 개요 | 자유 문장이라 점수 없이 원문과 길이만 남긴다 |

`false_confirmed`가 가장 중요한 숫자다. L3.5는 precision 우선이라 논의가 확정으로 올라가면 결정되지도
않은 일이 배정으로 번진다.

마감일은 짝 기준에 넣지 않는다. 날짜만 틀린 배정과 아예 못 뽑은 배정이 같은 점수가 되지 않게 하려는
것이다(`due_accuracy`로 따로 본다).

## 한계

- 정답 라벨은 사람이 판단한 값이다. 애매한 발화("2차 이후로 보죠")는 라벨에서 뺐다.
- 샘플 회의는 한 건이다. 점수는 이 회의에 대한 것이지 일반 성능이 아니다.
- L3(주제별 정리)는 채점하지 않는다.
- `DRY_RUN=1`은 각 계층이 고정 스텁을 돌려주므로 요청 스키마와 배선만 확인된다.
