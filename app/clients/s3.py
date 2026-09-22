"""S3 — 오디오는 본문이 아니라 키로 온다.

명세 「내부 API 규칙」이 못박은 것이다 — *"파일 전달: S3 경유. 로컬 볼륨 공유가 없다 —
모든 오디오 참조는 S3 키로 넘긴다."* 두 EC2 사이에 공유 디스크가 없고, 20초짜리 오디오를
HTTP 본문에 실으면 Spring 이 먼저 받아 디코딩까지 해야 해서 같은 일을 두 곳에서 하게 된다.

<h2>자격증명을 설정으로 받지 않는다</h2>
AI EC2 의 IAM 롤을 boto3 가 알아서 집는다. 액세스 키를 .env 에 두면 그 파일이 유출 표면이
되고, 롤과 달리 자동으로 돌지도 않는다. 로컬 개발에서는 평소 쓰던 AWS 프로필이 그대로 잡힌다.
"""

from __future__ import annotations

import asyncio
from functools import lru_cache

from app.errors import LayerError, LayerErrorKind


@lru_cache
def _client(region: str):
    try:
        import boto3
    except ImportError as exc:  # pragma: no cover - 의존성 누락은 배포 문제다
        raise LayerError(
            LayerErrorKind.PERMANENT,
            "S3_RUNTIME_MISSING",
            "boto3 가 설치되어 있지 않습니다.",
        ) from exc

    return boto3.client("s3", region_name=region or None)


async def fetch(bucket: str, key: str, region: str = "", max_bytes: int | None = None) -> bytes:
    """객체를 받아온다. 상한을 넘으면 받다 말고 거절한다.

    boto3 는 동기라 스레드로 뺀다 — 이벤트 루프에서 그대로 부르면 다운로드 동안 다른 요청이
    전부 멈춘다. 계층 서버는 한 번에 여러 회의를 받는다.

    <h2>범위 요청으로 상한을 건다</h2>
    **이 서버는 무엇을 받을지 고를 수 없다.** s3Key 는 요청이 정하고, 실수든 아니든 10분짜리
    원본을 가리키면 그 전부가 메모리에 올라온다. t3.medium 한 대에 Qdrant 까지 같이 떠 있어
    그건 곧 인스턴스 전체의 정지다.

    상한+1 바이트를 요청해서, 그만큼 다 오면 "더 있다"는 뜻이므로 거절한다. HeadObject 로
    크기를 먼저 묻는 방법도 있지만 왕복이 하나 늘고, 그 사이에 객체가 바뀌면 검사한 크기와
    받은 크기가 갈린다.
    """
    client = _client(region)

    def _get() -> bytes:
        kwargs: dict = {"Bucket": bucket, "Key": key}
        if max_bytes is not None:
            kwargs["Range"] = f"bytes=0-{max_bytes}"
        return client.get_object(**kwargs)["Body"].read()

    try:
        data = await asyncio.to_thread(_get)
    except Exception as exc:
        raise _classify(exc, bucket, key) from exc

    if max_bytes is not None and len(data) > max_bytes:
        # 잘라서 넘기지 않는다. 잘린 wav 는 헤더가 말하는 길이와 실제가 어긋나 판정이
        # 조용히 이상해진다 — 거절해서 보내는 쪽이 고치게 하는 편이 낫다.
        raise LayerError(
            LayerErrorKind.PERMANENT,
            "AUDIO_TOO_LARGE",
            f"VAD 입력이 상한({max_bytes} bytes)을 넘습니다: s3://{bucket}/{key}",
        )
    return data


def _classify(exc: Exception, bucket: str, key: str) -> LayerError:
    """S3 실패를 재시도 가능/불가로 나눈다.

    없는 키를 세 번 더 물어봐야 여전히 없다. 반대로 순단·스로틀링은 기다리면 풀린다 —
    둘을 섞으면 한쪽은 헛돌고 다른 한쪽은 회복할 기회를 잃는다(errors.py 와 같은 판단).
    """
    code = ""
    response = getattr(exc, "response", None)
    if isinstance(response, dict):
        code = str(response.get("Error", {}).get("Code", ""))

    if code in {"NoSuchKey", "NoSuchBucket", "404", "AccessDenied", "403"}:
        return LayerError(
            LayerErrorKind.PERMANENT,
            "AUDIO_NOT_FOUND",
            f"오디오를 읽을 수 없습니다(s3://{bucket}/{key}): {code or exc}",
        )

    if code in {"SlowDown", "RequestLimitExceeded", "Throttling", "ThrottlingException"}:
        return LayerError(LayerErrorKind.RATE_LIMIT, "S3_THROTTLED", str(exc))

    return LayerError(LayerErrorKind.TRANSIENT, "S3_UNAVAILABLE", str(exc))
