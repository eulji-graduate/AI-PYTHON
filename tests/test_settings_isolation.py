"""테스트가 개발자 로컬 `.env`를 읽지 않는지 확인한다 (이슈 #11).

`Settings`는 `env_file=".env"`로 작업 디렉터리의 파일을 읽는다. 서버에게는 맞는 동작이다.
하지만 테스트가 그 파일을 읽으면 같은 코드가 개발자 `.env`에 따라 통과하거나 실패한다 —
CI에는 `.env`가 없어서 로컬에서만 빨간 불이 켜지고, 진짜 회귀와 구분되지 않는다.

`tests/conftest.py`의 autouse 픽스처가 이 격리를 맡는다. 이 파일은 그 격리가 **실제로 걸려 있는지**를 본다.
"""

from pathlib import Path

from app.config import Settings, get_settings


def _write_dotenv_disabling_few_shot(directory: Path) -> Path:
    path = directory / ".env"
    path.write_text("FEW_SHOT_ENABLED=0\n", encoding="utf-8")
    return path


def test_dotenv_파일은_명시하면_읽힌다(tmp_path):
    """대조군 — 이 환경에서 `.env` 읽기 자체는 동작한다.

    아래 테스트가 "읽을 `.env`가 없어서" 통과한 것이 아님을 보이기 위해 둔다.
    """
    dotenv = _write_dotenv_disabling_few_shot(tmp_path)

    assert Settings(_env_file=dotenv).few_shot_enabled is False


def test_작업_디렉터리의_dotenv는_읽지_않는다(tmp_path, monkeypatch):
    _write_dotenv_disabling_few_shot(tmp_path)
    monkeypatch.chdir(tmp_path)
    # 개발자 셸에 같은 이름의 환경변수가 있어도 결과가 달라지지 않게 비운다.
    monkeypatch.delenv("FEW_SHOT_ENABLED", raising=False)
    get_settings.cache_clear()

    try:
        assert get_settings().few_shot_enabled is True
    finally:
        get_settings.cache_clear()
