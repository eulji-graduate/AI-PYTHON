"""내부 호출 인증 — `X-Internal-Token` 공유 시크릿.

Spring EC2 와 AI EC2 는 같은 호스트가 아니다. localhost 신뢰로 넘길 수 없어
토큰을 필수로 둔다.

비교는 `secrets.compare_digest` 로 한다. `==` 는 앞에서부터 다른 문자를 만나면
즉시 반환하므로 응답 시간 차이로 토큰을 한 글자씩 알아낼 수 있다.
"""

import secrets

from fastapi import Depends, Header, HTTPException, status

from app.config import Settings, get_settings


async def require_internal_token(
    x_internal_token: str | None = Header(default=None),
    settings: Settings = Depends(get_settings),
) -> None:
    # 토큰이 설정돼 있지 않으면 인증이 조용히 통과하는 상태가 된다 — 그게 더 위험하므로 막는다.
    if not settings.internal_token:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="INTERNAL_TOKEN 이 설정되지 않았습니다. 인증 없이 기동할 수 없습니다.",
        )

    if x_internal_token is None or not secrets.compare_digest(x_internal_token, settings.internal_token):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="유효하지 않은 내부 토큰입니다.",
        )
