"""관리자 기능: 모델 레지스트리(등록 · 교체 버튼), A/B 실험(생성 · 중지 · 그룹별 지표 · 통계 검정)."""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from .. import recommenders
from ..db import get_db
from ..models import Event, Experiment, ModelVersion
from ..schemas import (ABComparison, ExperimentIn, ExperimentOut, ExperimentStats,
                       ModelVersionIn, ModelVersionOut, VariantMetrics)
from ..services.stats import two_proportion_z, wilson_ci_95

SIG_LEVEL = 0.05                            # 유의수준 (양측). significant = p_value < SIG_LEVEL

router = APIRouter(tags=["admin"])


@router.get("/models", response_model=list[ModelVersionOut])
def list_models(db: Session = Depends(get_db)):
    return db.scalars(select(ModelVersion).order_by(ModelVersion.id)).all()


@router.get("/models/available")
def available_models():
    return sorted(recommenders.REGISTRY)


@router.post("/models", response_model=ModelVersionOut, status_code=201)
def create_model(m: ModelVersionIn, db: Session = Depends(get_db)):
    if m.name not in recommenders.REGISTRY:
        raise HTTPException(422, f"등록되지 않은 모델: {m.name} (가능: {sorted(recommenders.REGISTRY)})")
    mv = ModelVersion(**m.model_dump())
    db.add(mv)
    db.commit()
    db.refresh(mv)
    return mv


@router.post("/models/{model_id}/activate", response_model=ModelVersionOut)
def activate_model(model_id: int, db: Session = Depends(get_db)):
    """대시보드 '이 모델로 교체' 버튼. 기존 active는 candidate로 내린다."""
    mv = db.get(ModelVersion, model_id)
    if mv is None:
        raise HTTPException(404, f"모델 없음: {model_id}")
    for other in db.scalars(select(ModelVersion).where(ModelVersion.status == "active")):
        other.status = "candidate"
    mv.status = "active"
    db.commit()
    db.refresh(mv)
    return mv


@router.post("/experiments", response_model=ExperimentOut, status_code=201)
def create_experiment(e: ExperimentIn, db: Session = Depends(get_db)):
    for v in e.variants:
        if db.get(ModelVersion, v.model_version_id) is None:
            raise HTTPException(422, f"모델 없음: {v.model_version_id}")
    for running in db.scalars(select(Experiment).where(Experiment.status == "running")):
        running.status = "stopped"           # 동시에 하나만 진행
    exp = Experiment(name=e.name, variants=[v.model_dump() for v in e.variants])
    db.add(exp)
    db.commit()
    db.refresh(exp)
    return exp


@router.get("/experiments", response_model=list[ExperimentOut])
def list_experiments(db: Session = Depends(get_db)):
    return db.scalars(select(Experiment).order_by(Experiment.id.desc())).all()


@router.post("/experiments/{exp_id}/stop", response_model=ExperimentOut)
def stop_experiment(exp_id: int, db: Session = Depends(get_db)):
    exp = db.get(Experiment, exp_id)
    if exp is None:
        raise HTTPException(404, f"실험 없음: {exp_id}")
    exp.status = "stopped"
    db.commit()
    db.refresh(exp)
    return exp


def _variant_counts(db: Session, exp_id: int) -> list[VariantMetrics]:
    """실험 그룹별 집계 — is_setup=True 이벤트(신규 고객 초기 이력 주입)는 제외.
    사전조건: 실험이 존재한다는 호출자 측 보장."""
    base = (Event.experiment_id == exp_id) & (Event.is_setup == False)   # noqa: E712
    rows = db.execute(select(Event.variant, Event.type, func.count())
                      .where(base).group_by(Event.variant, Event.type)).all()
    agg: dict[str | None, dict[str, int]] = {}
    for variant, typ, n in rows:
        agg.setdefault(variant, {})[typ] = n
    users = dict(db.execute(select(Event.variant, func.count(func.distinct(Event.user_id)))
                            .where(base).group_by(Event.variant)).all())
    out: list[VariantMetrics] = []
    for variant, c in sorted(agg.items(), key=lambda x: str(x[0])):
        imp, clk = c.get("view", 0), c.get("click", 0)
        out.append(VariantMetrics(
            variant=str(variant), users=users.get(variant, 0),
            impressions=imp, clicks=clk, purchases=c.get("purchase", 0),
            ctr=clk / imp if imp else 0.0,
            ctr_ci_95=wilson_ci_95(clk, imp),
        ))
    return out


@router.get("/experiments/{exp_id}/metrics", response_model=list[VariantMetrics])
def experiment_metrics(exp_id: int, db: Session = Depends(get_db)):
    """그룹별 노출(view) · 클릭 · 구매 · CTR 과 CTR 의 Wilson 95% CI.
    통계 검정 (p-value · lift · diff CI) 은 /experiments/{id}/stats 에서."""
    if db.get(Experiment, exp_id) is None:
        raise HTTPException(404, f"실험 없음: {exp_id}")
    return _variant_counts(db, exp_id)


def _comparisons(variants: list[VariantMetrics]) -> list[ABComparison]:
    """2개 이상 변량이 있으면 알파벳 순 첫 변량을 baseline 으로, 나머지와 CTR · CVR 각각 2-prop z-test.
    CVR 은 purchases/impressions. 변량 하나뿐이거나 impressions=0 조합은 건너뜀."""
    if len(variants) < 2:
        return []
    baseline, *rest = variants
    comps: list[ABComparison] = []
    for treatment in rest:
        for metric, b_num, t_num in (
            ("ctr", baseline.clicks, treatment.clicks),
            ("cvr", baseline.purchases, treatment.purchases),
        ):
            b_den, t_den = baseline.impressions, treatment.impressions
            if b_den == 0 or t_den == 0:
                continue                     # 노출이 없으면 비교 자체가 성립 안 함 — 건너뜀
            res = two_proportion_z(b_num, b_den, t_num, t_den)
            comps.append(ABComparison(
                metric=metric,
                baseline=baseline.variant, treatment=treatment.variant,
                baseline_rate=b_num / b_den, treatment_rate=t_num / t_den,
                diff=res["diff"], lift=res["lift"], diff_ci_95=res["diff_ci_95"],
                p_value=res["p_value"], test="two_proportion_z",
                significant=res["p_value"] < SIG_LEVEL,
            ))
    return comps


@router.get("/experiments/{exp_id}/stats", response_model=ExperimentStats)
def experiment_stats(exp_id: int, db: Session = Depends(get_db)):
    """A/B 통계 검정 (명세 4.5). 그룹별 집계 + 그룹 간 두 비율 Z-test (CTR · CVR).
    각 비교는 diff · lift · 차이의 95% CI · p-value · significant (p<0.05) 를 돌려준다.

    baseline 은 변량명 알파벳 순 첫 번째. 변량이 셋 이상이면 baseline 과 각각 1:1 비교.
    is_setup=True (초기 이력) 는 집계에서 제외."""
    exp = db.get(Experiment, exp_id)
    if exp is None:
        raise HTTPException(404, f"실험 없음: {exp_id}")
    variants = _variant_counts(db, exp_id)
    return ExperimentStats(
        experiment_id=exp.id, experiment_name=exp.name,
        variants=variants, comparisons=_comparisons(variants),
    )
