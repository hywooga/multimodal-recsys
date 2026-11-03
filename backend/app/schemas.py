from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

EventType = Literal["view", "click", "add_to_cart", "purchase", "search"]
SearchType = Literal["text", "image", "hybrid"]


class ItemOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    source: str
    brand_id: int | None
    brand_name: str | None = None          # brands.name JOIN 결과 — 카드에 "BRAND #10" 대신 실제 브랜드명 노출
    title: str
    category: str
    price: float | None
    image_path: str | None


class ItemDetailOut(ItemOut):
    """상품 상세 페이지용. 카드용 ItemOut 에 설명 · 외부 상품 id 를 더한다.
    배치 조회(GET /items)는 응답 가벼운 ItemOut 유지, 단건 조회(GET /items/{id})만 이 스키마로 바꾼다."""
    description: str = ""
    source_item_id: str


class SearchHit(ItemOut):
    score: float                      # 결합 점수 (merge 방식에 따라 척도가 다르다. 순서만 의미 있음)


class SearchResultOut(BaseModel):
    """프로젝트 명세 2-5 "검색 API 필수 응답 필드" 의 results[] 아이템 규격."""
    product_id: str                   # 서버 내부 int id 를 문자열로 노출 (명세 샘플 "P12345")
    name: str                         # 명세 요구명 — 내부적으로는 Item.title 과 동일
    score: float
    price: float | None


class SearchOut(BaseModel):
    # ───── 명세(2-5) 필수 응답 필드 ─────
    search_type: SearchType           # text · image · hybrid 중 무엇으로 돌았는지
    results: list[SearchResultOut]    # 명세 필수: product_id · name · score · price 만
    latency_ms: float                 # 핸들러 시작부터 응답 직전까지 벽시계 시간
    total_count: int                  # 반환된 결과 수 (페이지네이션 없음)

    # ───── 데모 프론트/디버깅용 확장 필드 (명세 외) ─────
    query: str | None                 # 받은 검색어
    query_en: str | None              # 인코더에 넣은 영어 검색어 (한국어면 사전 번역 결과)
    untranslated: list[str]           # 사전에 없어 번역하지 못한 한국어 단어 (화면 안내용)
    weights: dict[str, float]         # text (MiniLM → 상품 텍스트) · text_image (Marqo 텍스트 → 상품 이미지) · image (업로드 이미지)
    merge: str
    items: list[SearchHit]            # 프론트가 쓰는 전체 상품 상세 (brand/image_path 등)


class EventIn(BaseModel):
    user_id: int
    type: EventType
    item_id: int | None = None
    query: str | None = None
    model_version_id: int | None = None
    experiment_id: int | None = None
    variant: str | None = None
    # 클라이언트가 생성한 재전송 멱등 키. 네트워크 재시도 · 시뮬레이터 재실행에 쓴다. 생략 시 서버가 UUID4 생성.
    event_id: UUID | None = None
    # 신규 고객의 초기 이력 주입. True이면 A/B · CTR 집계에서 제외.
    is_setup: bool = False


class EventOut(EventIn):
    model_config = ConfigDict(from_attributes=True)
    id: int
    event_id: UUID
    created_at: datetime


class UserIn(BaseModel):
    # (source, source_user_id)가 같으면 같은 user를 돌려주는 멱등 생성. source_user_id가 None이면 익명 신규.
    source: str = "sim"
    source_user_id: str | None = None
    persona: str | None = None


class UserOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    source: str
    source_user_id: str | None
    persona: str | None


class RecommendationOut(BaseModel):
    """프로젝트 명세 2-5 "추천 API 필수 응답 필드" 의 recommendations[] 아이템 규격."""
    product_id: str                   # 서버 내부 int id 를 문자열로 노출
    score: float                      # 모델이 돌려준 점수 (score_kind="model") 또는 순위 placeholder (score_kind="rank")
    score_kind: Literal["model", "rank"]  # 자동 채점·리뷰에서 "모델 점수"로 오해되지 않도록 출처 명시
    reason: str                       # 모델별 간단 라벨 (popular · brand_match · personalized)
    is_exploration: bool              # MAB 탐색 슬롯 여부 — 미구현이라 항상 False


class PipelineLatency(BaseModel):
    """명세 요구 pipeline_latency. Multi-Stage(Candidate→Ranking→Re-ranking) 가 아직 쪼개져 있지 않아
    candidate_ms 에 전체 시간을, ranking/reranking 에 0 을 넣는다. Ranking 모델 구현되면 쪼개서 측정."""
    candidate_ms: float
    ranking_ms: float
    reranking_ms: float
    total_ms: float


class SessionContext(BaseModel):
    """명세 요구 session_context. 지금은 해당 고객의 최근 click 이벤트 상품 id 만 돌려준다.
    session_interest(단기 관심 라벨)는 세션 인코더 미구현이라 None."""
    recent_clicks: list[str]
    session_interest: str | None


class RecommendOut(BaseModel):
    user_id: int
    model: str
    model_version_id: int | None
    experiment_id: int | None = None
    variant: str | None = None

    # ───── 명세(2-5) 필수 응답 필드 ─────
    recommendations: list[RecommendationOut]
    pipeline_latency: PipelineLatency
    session_context: SessionContext

    # ───── 하위호환 확장 필드 (시뮬레이터 bridge · 데모 프론트가 읽음) ─────
    items: list[int]                  # recommendations[].product_id 와 같은 순서의 int id 목록


class ModelVersionIn(BaseModel):
    name: str
    version: str = "v1"
    params: dict = Field(default_factory=dict)


class ModelVersionOut(ModelVersionIn):
    model_config = ConfigDict(from_attributes=True)
    id: int
    status: str


class Variant(BaseModel):
    name: str
    model_version_id: int
    weight: float = Field(gt=0)


class ExperimentIn(BaseModel):
    name: str
    variants: list[Variant] = Field(min_length=2)


class ExperimentOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    name: str
    status: str
    variants: list[Variant]


class VariantMetrics(BaseModel):
    variant: str
    users: int
    impressions: int
    clicks: int
    purchases: int
    ctr: float
    ctr_ci_95: tuple[float, float] = (0.0, 1.0)   # Wilson score 95% CI (CTR 자체의 신뢰구간)


class ABComparison(BaseModel):
    """두 변량의 지표 비교 (CTR · CVR). baseline 대비 treatment 의 차이와 통계 검정 결과."""
    metric: Literal["ctr", "cvr"]
    baseline: str
    treatment: str
    baseline_rate: float
    treatment_rate: float
    diff: float                                   # treatment_rate - baseline_rate
    lift: float | None                            # diff / baseline_rate (baseline=0 이면 None)
    diff_ci_95: tuple[float, float]               # 차이의 Wald 95% CI
    p_value: float                                # 양측 두 비율 Z-test
    test: Literal["two_proportion_z"]
    significant: bool                             # p_value < 0.05


class ExperimentStats(BaseModel):
    """실험 전체 요약: 그룹별 지표 + 그룹 간 통계 검정 결과."""
    experiment_id: int
    experiment_name: str
    variants: list[VariantMetrics]
    comparisons: list[ABComparison]
