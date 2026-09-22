"""임베딩 — AI-08 저장과 AI-09 조회가 **같은 공간**에 놓이게 하는 한 곳.

저장과 검색이 서로 다른 모델·차원을 쓰면 유사도가 조용히 망가진다. 검색 결과가 비는 게
아니라 **엉뚱한 예시가 그럴듯한 점수와 함께** 나오므로, 프롬프트를 봐도 무엇이 잘못됐는지
드러나지 않는다. 그래서 두 경로가 이 모듈 하나만 부른다.

<h2>task_type 을 나눈다</h2>
Gemini 임베딩은 저장용(RETRIEVAL_DOCUMENT)과 질의용(RETRIEVAL_QUERY)을 구분한다. 같은
텍스트라도 "찾아지는 쪽"과 "찾는 쪽"의 최적 표현이 다르기 때문이다. 둘 다 DOCUMENT 로
넣어도 동작은 하지만 검색 품질이 떨어진다 — 티가 안 나는 종류의 손해라 처음부터 맞춘다.

<h2>차원을 고정한다</h2>
컬렉션을 만들 때 차원이 박히므로, 모델 기본값에 맡기면 모델을 바꾸는 순간 기존 컬렉션에
넣지 못한다. `output_dimensionality` 로 못박아 두면 모델 교체가 재색인 하나로 끝난다.
"""

from __future__ import annotations

import hashlib
import struct

from app.config import Settings
from app.errors import LayerError, LayerErrorKind, classify_provider_error

# 저장되는 쪽(근거 발화)과 찾는 쪽(새 발화)의 task_type.
TASK_DOCUMENT = "RETRIEVAL_DOCUMENT"
TASK_QUERY = "RETRIEVAL_QUERY"


class EmbeddingClient:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client = None  # 최초 호출 때 만든다 (DRY_RUN 이면 아예 안 만든다)

    @property
    def dim(self) -> int:
        return self._settings.embed_dim

    @property
    def model_name(self) -> str:
        return self._settings.gemini_embed_model

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

    async def embed(self, texts: list[str], *, task_type: str) -> list[list[float]]:
        """여러 개를 한 번에 보낸다. 저장은 배치로 들어오고, 배치마다 왕복하면
        그 수만큼 지연이 곱해진다."""
        if not texts:
            return []

        if self._settings.dry_run:
            return [_stub_vector(text, self.dim) for text in texts]

        from google.genai import types

        client = self._ensure_client()
        try:
            response = await client.aio.models.embed_content(
                model=self._settings.gemini_embed_model,
                contents=texts,
                config=types.EmbedContentConfig(
                    task_type=task_type,
                    output_dimensionality=self._settings.embed_dim,
                ),
            )
        except LayerError:
            raise
        # 계층 호출과 같은 분류를 쓴다 — 임베딩만 다른 재시도 규칙을 가지면
        # 어느 실패가 왜 재시도됐는지 로그에서 갈린다.
        except Exception as exc:
            raise classify_provider_error(exc) from exc

        vectors = [list(item.values) for item in (response.embeddings or [])]
        if len(vectors) != len(texts):
            # 개수가 다르면 어느 텍스트의 벡터인지 짝지을 수 없다. 순서로 짝짓는 코드가
            # 뒤에 있으므로, 여기서 막지 않으면 **다른 발화의 벡터가 저장된다.**
            raise LayerError(
                LayerErrorKind.PERMANENT,
                "EMBEDDING_COUNT_MISMATCH",
                f"임베딩 개수가 입력과 다릅니다: 입력={len(texts)} 응답={len(vectors)}",
            )
        return vectors


def _stub_vector(text: str, dim: int) -> list[float]:
    """DRY_RUN 용 고정 벡터.

    **의미가 없다.** 같은 텍스트에 같은 값이 나온다는 것만 지킨다 — 라우팅과 저장·조회
    배선을 키 없이 확인하기 위한 것이고, 이걸로 유사도를 평가하면 안 된다.
    """
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    # 차원 하나당 4바이트가 필요하다. 해시(32바이트)를 **필요한 바이트 수 기준**으로 반복한다 —
    # 차원 기준으로 반복하면 dim 이 커질수록 모자란다(768차원에 800바이트만 나온다).
    need = dim * 4
    raw = (digest * (need // len(digest) + 1))[:need]
    values = struct.unpack(f"{dim}i", raw)
    return [v / 2**31 for v in values]
