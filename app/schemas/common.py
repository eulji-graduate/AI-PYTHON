"""계층 공통 DTO.

API 표면은 camelCase 다(명세 기준). 파이썬 쪽은 snake_case 로 쓰고 alias 로 변환한다.
`populate_by_name=True` 라 테스트에서는 snake_case 로도 만들 수 있다.
"""

from pydantic import BaseModel, ConfigDict, Field
from pydantic.alias_generators import to_camel

# 명단 밖 대상을 가리키는 탈출구. 닫힌 목록에 이게 없으면 모델이 참석자 중 아무나
# 하나를 억지로 고르고, 그 값은 검증도 안 된다. 계층이 공유하므로 여기 둔다 —
# 계층마다 따로 두면 한쪽만 고쳐졌을 때 enum 과 후처리가 어긋난다.
UNKNOWN_PERSON = "unknown_person"

# 판정 불가를 나타내는 enum 값. nullable + enum 조합은 구조화 출력에서 제공자마다
# 처리가 갈리므로, 값으로 받고 후처리에서 None 으로 바꾼다.
UNKNOWN = "UNKNOWN"


class CamelModel(BaseModel):
    model_config = ConfigDict(
        alias_generator=to_camel,
        populate_by_name=True,
        extra="forbid",
    )


class Usage(CamelModel):
    """계층별 토큰. 선택 항목이 아니다 — 없으면 QLTY-03(비용)이 성립하지 않고
    특화 모델 전환의 손익분기점을 계산할 수 없다."""

    tokens_in: int = 0
    tokens_out: int = 0


class Participant(CamelModel):
    """닫힌 목록의 원소. person_id 가 None 인 항목이 `unknown_person` 탈출구다
    (게스트·외부 참석자). 탈출구가 없으면 모델이 명단 안에서 억지로 하나를 고른다."""

    person_id: int | None = None
    name: str


class Utterance(CamelModel):
    """근거 발화 후보. 이 목록의 id 만 evidence 로 반환될 수 있다(근거 강제)."""

    utterance_id: int
    speaker_id: int | None = None
    start_ms: int | None = None
    text: str


class FewShotExample(CamelModel):
    """AI-09 가 돌려준 예시. Spring 이 따로 조회하지 않고 Python 이 내부에서 부르므로,
    무엇을 썼는지 응답에 실어 보내야 review_log.input_context 가 온전해진다."""

    input_text: str
    payload: dict
    score: float | None = None


class LayerMeta(CamelModel):
    """모든 계층 응답에 공통으로 붙는다. 프롬프트를 바꾼 뒤 정확도 변화를 추적하는 키다."""

    usage: Usage = Field(default_factory=Usage)
    model: str
    prompt_version: str
