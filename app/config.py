"""설정 — 환경변수 하나로만 주입받는다.

이 서버는 업무 DB에 접속하지 않는다. 그래서 DB 설정이 여기 없는 것이 정상이다.
상태를 갖지 않는 계산 서버로 두면 계층을 특화 모델로 갈아끼우거나 인스턴스를
늘려도 계약이 깨지지 않는다. 모든 입력은 요청 본문으로 받고 출력은 응답으로 준다.
"""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Spring EC2 에서만 들어오는 호출임을 확인하는 공유 시크릿.
    # 보안그룹으로 인바운드를 좁혀도 이건 별도로 둔다 — 같은 VPC 안의 다른
    # 인스턴스가 뚫렸을 때 방어선이 하나도 없는 상태가 되지 않게.
    internal_token: str = ""

    gemini_api_key: str = ""
    gemini_model: str = "gemini-3.5-flash"

    # 샘플링 시드. temperature=0 과 짝으로 흔들림을 줄이려는 시도다.
    #
    # ⚠ **이걸 넣어도 같은 입력에 같은 출력이 나오지 않는다(2026-08-14 실측).** L3.5 게이트에
    # 같은 프롬프트를 3회 넣어 매회 항목 14건 중 3건의 판정이 뒤집혔고, seed 없이 잰 값(12건 중
    # 2건)과 다르지 않았다. 자세한 수치는 `app/clients/gemini.py` 주석에 있다.
    #
    # 흔들리면 두 가지가 함께 깨진다 —
    #   사용자 쪽: 같은 회의를 재분석하면 판정이 달라진다("AI 가 오늘은 다르게 말한다")
    #   우리 쪽:   프롬프트를 안 고쳤는데 점수가 흔들려 A/B 로 개선을 확인할 수 없다
    # 후자가 지금 상태이므로, **모든 품질 측정은 반복 실행으로 잡음 바닥을 함께 재야 한다.**
    #
    # 값 자체에는 의미가 없다. **바뀌지 않는 것**이 요구사항이라 상수로 둔다. 설정에 두는 이유는
    # 측정할 때 시드를 바꿔가며 잡음 바닥을 재려면 밖에서 줄 수 있어야 하기 때문이다.
    gemini_seed: int = 20260814

    # 동시에 제공자에게 나가는 호출 수 상한.
    #
    # Spring 이 주제 단위 계층을 주제별로 동시에 부르기 시작하면서 필요해졌다 — 회의 하나에
    # 주제가 7개면 그만큼이 한꺼번에 나가고, 분석이 두 건 겹치면 그 두 배다. **제공자
    # 레이트리밋을 우리가 스스로 당기는** 모양이 된다.
    #
    # Spring 쪽 풀 크기로는 못 막는다 — 인스턴스가 늘거나 다른 호출자가 붙으면 그 가정이
    # 깨진다. 제공자로 나가는 문이 이 프로세스 하나뿐이라 여기가 상한을 두기에 맞는 자리다.
    #
    # 4 는 실측 없이 정한 값이다. 계층 호출이 I/O 대기라 CPU 와 무관하고, 늘리면 지연이
    # 줄지만 429 가 가까워진다. 실제 쿼터가 정해지면 그 값으로 다시 잡아야 한다.
    gemini_max_concurrency: int = 4

    # 임베딩 모델(AI-08·09). 생성 모델과 **따로 둔다** — 계층 모델을 특화 모델로 갈아끼울 때
    # 임베딩까지 함께 바뀌면 기존 컬렉션 전체를 못 쓰게 된다. 두 축은 독립적으로 움직인다.
    gemini_embed_model: str = "gemini-embedding-001"

    # 벡터 차원. **컬렉션을 만들 때 박히는 값이다** — 모델 기본값에 맡기면 모델을 바꾸는
    # 순간 기존 컬렉션에 넣지 못한다. 여기서 못박고 output_dimensionality 로 강제한다.
    embed_dim: int = 768

    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str = ""
    # 컬렉션 이름에 차원·모델을 넣지 않는다. 이름이 스키마를 설명하기 시작하면 바꿀 때마다
    # 코드 상수까지 함께 고쳐야 한다. 재색인이 필요하면 이 값을 새 이름으로 준다.
    qdrant_collection: str = "meeting_tuple_vector"

    # few-shot 조회 자체를 끈다. Qdrant 를 아직 안 띄운 환경에서 계층만 돌려볼 때 쓴다 —
    # 켜 둔 채로 두면 계층 호출마다 조회 실패 로그가 쌓인다(동작은 한다).
    few_shot_enabled: bool = True

    # 계층 호출 실패 정책 (명세 「레이어 호출 실패 정책」)
    # 일시적 실패는 지수 백오프 3회, 지터 ±20%.
    retry_delays_sec: tuple[float, ...] = (2.0, 8.0, 30.0)
    retry_jitter_ratio: float = 0.2

    # ── AI-01 VAD ────────────────────────────────────────────────────────────
    # silero-vad ONNX 모델 파일. **이미지에 함께 넣는다** — 런타임에 받아오면 모델이 바뀔 때
    # 절단점이 배포와 무관하게 조용히 달라지고, "어제와 오늘의 블록 경계가 다른" 이유를
    # 아무도 못 찾는다. 없으면 VAD_MODEL_MISSING 으로 명확히 실패한다.
    vad_model_path: str = "/app/models/silero_vad.onnx"

    # 동시에 도는 VAD 추론 수. 기본 실행기(min(32, cpu+4))에 맡기지 않는 이유는 t3.medium
    # (2 vCPU)에 Qdrant 까지 같이 떠 있어, 코어보다 많은 추론이 서로 코어를 뺏으면 헬스체크까지
    # 늦어져 컨테이너가 재시작되기 때문이다. 인스턴스를 키우면 이 값을 올린다.
    vad_max_workers: int = 1

    # S3 리전. 자격증명은 여기 두지 않는다 — AI EC2 의 IAM 롤을 boto3 가 집는다.
    # 키를 .env 에 두면 그 파일이 유출 표면이 되고 자동 회전도 안 된다.
    s3_region: str = ""

    # 1이면 Gemini 를 부르지 않고 고정 스텁을 돌려준다. 라우팅·스키마 확인용.
    dry_run: bool = False


@lru_cache
def get_settings() -> Settings:
    return Settings()
