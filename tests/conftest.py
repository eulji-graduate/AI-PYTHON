"""테스트 전역 설정 — 개발자 로컬 `.env`가 테스트 결과에 섞이지 않게 막는다 (이슈 #11).

`Settings`는 `env_file=".env"`로 작업 디렉터리의 파일을 읽는다. 서버 실행과 `evals/`에는 의도된 동작이다.
테스트에서는 그 파일을 읽지 않는다 — 읽으면 같은 테스트가 개발자 `.env`에 따라 통과하거나 실패한다(CI에는 `.env`가 없다).

각 테스트가 필요한 설정은 `monkeypatch.setenv`로 직접 정한다. 환경변수는 이 격리와 무관하게 그대로 먹는다.
"""

import pytest

from app.config import Settings, get_settings


@pytest.fixture(autouse=True)
def _isolate_dotenv(monkeypatch):
    monkeypatch.setitem(Settings.model_config, "env_file", None)
    # get_settings 는 캐시된다. 격리 전에 만들어진 값이 남아 있으면 이전 `.env` 값이 새어 나온다.
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()
