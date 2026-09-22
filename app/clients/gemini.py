"""Gemini 호출 — 구조화 출력 전용.

프롬프트로 "JSON 으로 답해줘"라고 부탁하지 않는다. 응답 스키마를 붙여 구조로 강제한다.
부탁과 강제는 다르다 — 부탁은 대체로 지켜지지만 '대체로'가 파이프라인 중간에 있으면
뒤 계층 전체가 그 확률만큼 깨진다.

temperature=0 과 seed 를 함께 고정한다. 줄 수 있는 것은 준다.

⚠ **그래도 같은 입력에 같은 출력이 나오지 않는다. 실측으로 확인했다.**

2026-08-14, L3.5 게이트에 같은 발화·같은 참석자 명단·같은 항목을 넣고 반복 호출했다.
few-shot 은 껐다(검색 결과가 흔들리면 그것도 섞인다). 즉 프롬프트 문자열이 완전히 동일했다.

    temperature=0 만          2회 중 항목 12건에서 2건 뒤집힘
    temperature=0 + seed      3회 중 항목 14건에서 매회 3건 뒤집힘

seed 를 넣어도 줄지 않았다. 제공자가 결정성을 약속하지 않으므로 "무시됐다"와 "부족하다"를
우리 쪽에서 구분할 방법도 없다. 그래서 이 두 값은 **흔들림을 줄이려는 시도**이고, 없앤
장치가 아니다.

여기서 나오는 규칙이 하나 있다 — **1회 측정으로 개선을 판단하지 않는다.** 반복 실행으로
잡음 바닥을 함께 재고, 그 폭을 넘는 차이만 효과로 읽는다. 이 주석의 앞 버전은 "같은 입력에
같은 출력이 나와야 한다"고 단정했는데, 그 문장을 믿고 1회 측정으로 판단하는 것이 지금
이 파이프라인에서 가장 위험한 습관이다.
"""

from __future__ import annotations

import asyncio
import json
import random

from app.config import Settings
from app.errors import LayerError, LayerErrorKind, classify_provider_error
from app.schemas.common import Usage


class GeminiClient:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client = None  # 최초 호출 때 만든다 (DRY_RUN 이면 아예 안 만든다)

        # 동시에 제공자에게 나가는 호출 수 상한.
        #
        # ⚠ **이 객체가 프로세스에 하나여야 상한이 성립한다.** 요청마다 새로 만들면 각자
        #   자기 몫의 상한을 갖게 되어 아무것도 제한하지 않는다(routers/internal.py 의
        #   _gemini_client 가 하나로 유지한다).
        #
        # 왜 필요한가 — Spring 이 주제 단위 계층(L3·L3.5·L4)을 주제별로 동시에 부른다
        # (BACKEND 주제 병렬화). 회의 하나에 주제가 7개면 그만큼이 한꺼번에 나가고, 분석이
        # 두 건 겹치면 그 두 배다. 제공자 레이트리밋을 **우리가 스스로 당기는** 모양이 된다.
        #
        # 상한을 여기 두는 이유는 Spring 쪽 풀 크기로는 못 막기 때문이다 — 인스턴스가 늘거나
        # 다른 호출자가 붙으면 그 가정이 깨진다. 제공자로 나가는 문이 여기 하나뿐이다.
        self._limiter = asyncio.Semaphore(settings.gemini_max_concurrency)

    def _ensure_client(self):
        if self._client is None:
            if not self._settings.gemini_api_key:
                raise LayerError(
                    LayerErrorKind.PERMANENT,
                    "GEMINI_KEY_MISSING",
                    "GEMINI_API_KEY 가 설정되지 않았습니다.",
                )
            from google import genai  # 지연 임포트 — DRY_RUN 로컬 실행에 SDK 가 필요 없게

            self._client = genai.Client(api_key=self._settings.gemini_api_key)
        return self._client

    async def generate_json(self, *, prompt: str, response_schema: dict) -> tuple[dict, Usage]:
        """실패 정책을 여기 한 곳에 둔다. 계층 코드가 재시도를 각자 구현하면
        어떤 계층은 재시도하고 어떤 계층은 안 하는 상태가 조용히 생긴다."""
        from google.genai import types

        client = self._ensure_client()
        config = types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=response_schema,
            temperature=0,
            # temperature 와 짝이다. 하나만 두면 흔들림이 남는다(모듈 주석 참고).
            seed=self._settings.gemini_seed,
        )

        attempts = len(self._settings.retry_delays_sec) + 1
        last: LayerError | None = None

        for attempt in range(attempts):
            try:
                # ⚠ 상한은 **호출 하나만** 감싼다. 아래 백오프 sleep 까지 잡고 있으면 기다리는
                #   동안 남의 자리를 막는다 — 그때 제공자는 놀고 우리만 줄을 선다.
                async with self._limiter:
                    response = await client.aio.models.generate_content(
                        model=self._settings.gemini_model,
                        contents=prompt,
                        config=config,
                    )
                return self._parse(response)
            except LayerError:
                raise
            # SDK 예외 계층에 기대지 않고 상태코드·메시지로 분류한다 — SDK 버전이 올라가며
            # 예외 클래스가 바뀌어도 재시도 판정이 조용히 무너지지 않게.
            except Exception as exc:
                last = classify_provider_error(exc)
                if not last.retryable or attempt == attempts - 1:
                    raise last from exc
                await asyncio.sleep(self._delay_for(attempt, last))

        raise last if last else LayerError(LayerErrorKind.TRANSIENT, "PROVIDER_UNAVAILABLE", "재시도 소진")

    def _delay_for(self, attempt: int, error: LayerError) -> float:
        # 429 는 서버가 알려준 대기시간을 존중한다. 우리 백오프를 우선하면
        # 레이트리밋이 안 풀린 상태로 다시 때려 쿼터만 더 태운다.
        # `is not None` 으로 본다 — Retry-After: 0 은 "바로 다시 보내도 된다"는
        # 서버의 답인데, 참/거짓으로 보면 0 이 falsy 라 30초를 기다린다.
        if error.kind is LayerErrorKind.RATE_LIMIT and error.retry_after_sec is not None:
            return max(0.0, error.retry_after_sec)

        base = self._settings.retry_delays_sec[attempt]
        # 지터 — 여러 계층이 동시에 실패하면 재시도가 같은 순간에 겹쳐 다시 실패한다.
        ratio = self._settings.retry_jitter_ratio
        return base * random.uniform(1 - ratio, 1 + ratio)

    @staticmethod
    def _parse(response) -> tuple[dict, Usage]:
        meta = getattr(response, "usage_metadata", None)
        usage = Usage(
            tokens_in=getattr(meta, "prompt_token_count", 0) or 0,
            tokens_out=getattr(meta, "candidates_token_count", 0) or 0,
        )

        text = (getattr(response, "text", None) or "").strip()
        if not text:
            # 스키마를 붙였는데도 빈 응답이면 세이프티 차단 등 입력 자체의 문제다.
            # 같은 입력으로 다시 보내도 같은 결과라 재시도하지 않는다.
            raise LayerError(LayerErrorKind.PERMANENT, "EMPTY_RESPONSE", "구조화 응답이 비어 있습니다.")
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as exc:
            raise LayerError(LayerErrorKind.PERMANENT, "SCHEMA_INVALID", f"JSON 파싱 실패: {exc}") from exc

        if not isinstance(parsed, dict):
            raise LayerError(LayerErrorKind.PERMANENT, "SCHEMA_INVALID", "최상위가 객체가 아닙니다.")
        return parsed, usage
