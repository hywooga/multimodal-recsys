"""A/B 통계 검정 — 명세 4.5 "통계적 유의성 검정 · p-value · 95% CI".

두 비율 Z-test 로 CTR · CVR 를 비교.
   /experiments/{exp_id}/metrics 응답에 각 변량의 CTR Wilson 95% CI 가 포함되는지
   /experiments/{exp_id}/stats 가 baseline 대비 변량들의 diff · lift · p-value · 유의성을 돌려주는지
   is_setup 이벤트가 통계 집계에 섞이지 않는지 (PR #30 와 공유되는 가드)
"""
import math

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import sessionmaker

from app import db as app_db
from app.config import settings
from app.main import app
from app.services.stats import norm_sf_two_sided, two_proportion_z, wilson_ci_95
from scripts.seed_sample import seed


@pytest.fixture(autouse=True)
def _no_encoders(monkeypatch):
    monkeypatch.setattr(settings, "load_image_encoder", False)
    monkeypatch.setattr(settings, "load_text_encoder", False)


@pytest.fixture()
def client(tmp_path):
    engine = app_db.make_engine(f"sqlite:///{tmp_path / 'test.db'}")
    Session = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)
    app_db.init_db(engine)
    with Session() as s:
        seed(s)

    def override():
        s = Session()
        try:
            yield s
        finally:
            s.close()

    app.dependency_overrides[app_db.get_db] = override
    yield TestClient(app)
    app.dependency_overrides.clear()


def _create_experiment(client) -> dict:
    ids = [m["id"] for m in client.get("/models").json()]
    return client.post("/experiments", json={"name": "pop vs brand", "variants": [
        {"name": "A", "model_version_id": ids[0], "weight": 1},
        {"name": "B", "model_version_id": ids[1], "weight": 1}]}).json()


def _log(client, user_id: int, variant: str, exp_id: int, views: int, clicks: int, purchases: int = 0):
    """해당 변량에 노출 · 클릭 · 구매 이벤트를 지정한 수만큼 심는다 (item_id=3 고정)."""
    for i in range(views):
        client.post("/events", json={"user_id": user_id, "type": "view", "item_id": 3,
                                     "experiment_id": exp_id, "variant": variant})
    for i in range(clicks):
        client.post("/events", json={"user_id": user_id, "type": "click", "item_id": 3,
                                     "experiment_id": exp_id, "variant": variant})
    for i in range(purchases):
        client.post("/events", json={"user_id": user_id, "type": "purchase", "item_id": 3,
                                     "experiment_id": exp_id, "variant": variant})


# ─── 순수 math 헬퍼 ─────────────────────────────────────────────────────────

def test_norm_sf_two_sided_matches_known_values():
    # 표준정규 양측 p-value — 교재 표준값
    assert abs(norm_sf_two_sided(1.96) - 0.05) < 1e-3
    assert abs(norm_sf_two_sided(2.576) - 0.01) < 1e-3
    assert abs(norm_sf_two_sided(0.0) - 1.0) < 1e-9


def test_wilson_ci_within_zero_one_and_contains_p():
    for s, n in [(0, 10), (5, 10), (10, 10), (50, 100), (99, 100), (0, 0)]:
        low, high = wilson_ci_95(s, n)
        assert 0.0 <= low <= high <= 1.0
        if n > 0:
            # p 가 CI 안에 있어야 — s=n (p=1) 또는 s=0 (p=0) 극단에서 float 노이즈로 1e-16 어긋나는 경계는 허용.
            eps = 1e-9
            assert low - eps <= s / n <= high + eps


def test_two_proportion_z_large_effect_is_significant():
    # 95/100 vs 5/100 — 극단적 차이. p < 1e-20 수준
    res = two_proportion_z(5, 100, 95, 100)
    assert res["p_value"] < 0.001
    assert res["diff"] == pytest.approx(0.9, abs=1e-9)
    assert res["lift"] == pytest.approx(18.0, abs=1e-9)
    assert res["diff_ci_95"][0] > 0.8


def test_two_proportion_z_tiny_difference_is_not_significant():
    # 50/100 vs 51/100 — 거의 같음
    res = two_proportion_z(50, 100, 51, 100)
    assert res["p_value"] > 0.5
    assert res["diff"] == pytest.approx(0.01, abs=1e-9)
    assert res["diff_ci_95"][0] < 0 < res["diff_ci_95"][1]           # CI 가 0 을 포함


def test_two_proportion_z_handles_empty_samples():
    # 분모 0 → p=1, diff=0 (크래시 없음)
    assert two_proportion_z(0, 0, 5, 10)["p_value"] == 1.0
    assert two_proportion_z(5, 10, 0, 0)["p_value"] == 1.0


# ─── /experiments/{id}/metrics 에 CI 추가 ──────────────────────────────────

def test_metrics_includes_ctr_wilson_ci(client):
    exp = _create_experiment(client)
    _log(client, user_id=1, variant="A", exp_id=exp["id"], views=100, clicks=20)
    _log(client, user_id=2, variant="B", exp_id=exp["id"], views=100, clicks=40)

    m = {row["variant"]: row for row in client.get(f"/experiments/{exp['id']}/metrics").json()}
    for v in ("A", "B"):
        low, high = m[v]["ctr_ci_95"]
        assert 0.0 <= low <= m[v]["ctr"] <= high <= 1.0


# ─── /experiments/{id}/stats ────────────────────────────────────────────────

def test_stats_404_on_missing_experiment(client):
    assert client.get("/experiments/999/stats").status_code == 404


def test_stats_response_shape_and_significant_difference(client):
    """20/100 vs 40/100 CTR 차이가 유의하게 잡히는지. 기본 유의수준 0.05."""
    exp = _create_experiment(client)
    _log(client, user_id=1, variant="A", exp_id=exp["id"], views=100, clicks=20, purchases=5)
    _log(client, user_id=2, variant="B", exp_id=exp["id"], views=100, clicks=40, purchases=15)

    body = client.get(f"/experiments/{exp['id']}/stats").json()
    assert body["experiment_id"] == exp["id"]
    assert body["experiment_name"] == exp["name"]
    assert {v["variant"] for v in body["variants"]} == {"A", "B"}

    comps = {c["metric"]: c for c in body["comparisons"]}
    assert set(comps) == {"ctr", "cvr"}
    ctr = comps["ctr"]
    assert ctr["baseline"] == "A" and ctr["treatment"] == "B"
    assert ctr["baseline_rate"] == pytest.approx(0.2)
    assert ctr["treatment_rate"] == pytest.approx(0.4)
    assert ctr["diff"] == pytest.approx(0.2)
    assert ctr["lift"] == pytest.approx(1.0)                        # (0.4 - 0.2) / 0.2
    assert ctr["p_value"] < 0.05 and ctr["significant"] is True
    assert ctr["test"] == "two_proportion_z"
    low, high = ctr["diff_ci_95"]
    assert low < ctr["diff"] < high                                  # CI 가 diff 를 포함
    assert low > 0                                                   # 유의한 양의 차이


def test_stats_no_significant_difference_when_rates_match(client):
    exp = _create_experiment(client)
    _log(client, user_id=1, variant="A", exp_id=exp["id"], views=100, clicks=30)
    _log(client, user_id=2, variant="B", exp_id=exp["id"], views=100, clicks=31)

    comps = {c["metric"]: c for c in client.get(f"/experiments/{exp['id']}/stats").json()["comparisons"]}
    assert comps["ctr"]["significant"] is False
    low, high = comps["ctr"]["diff_ci_95"]
    assert low < 0 < high                                            # CI 가 0 을 가로지름


def test_stats_skips_comparison_when_no_impressions(client):
    """어떤 변량이 노출 0 이면 비율 비교가 성립 안 함 → 그 metric 은 응답에서 생략."""
    exp = _create_experiment(client)
    _log(client, user_id=1, variant="A", exp_id=exp["id"], views=10, clicks=5)
    # B 는 아무 이벤트도 안 심음 → impressions=0, 집계에 등장조차 안 함
    body = client.get(f"/experiments/{exp['id']}/stats").json()
    assert body["comparisons"] == []                                 # 비교 상대 없음


def test_stats_is_setup_events_excluded(client):
    """is_setup=True 이벤트는 통계 집계에서도 제외돼야 한다 (PR #30 와 같은 규칙)."""
    exp = _create_experiment(client)
    # live: A=5/100, B=50/100 (B가 유의하게 좋음)
    _log(client, user_id=1, variant="A", exp_id=exp["id"], views=100, clicks=5)
    _log(client, user_id=2, variant="B", exp_id=exp["id"], views=100, clicks=50)
    # setup 로 A 에 "클릭 폭탄" 을 넣어도 통계가 바뀌면 안 된다
    for _ in range(100):
        client.post("/events", json={"user_id": 1, "type": "click", "item_id": 3,
                                     "experiment_id": exp["id"], "variant": "A", "is_setup": True})

    body = client.get(f"/experiments/{exp['id']}/stats").json()
    ctr = next(c for c in body["comparisons"] if c["metric"] == "ctr")
    assert ctr["baseline_rate"] == pytest.approx(0.05)              # setup 100건을 세면 1.05 가 됐을 것
    assert ctr["significant"] is True


def test_stats_lift_none_when_baseline_rate_is_zero(client):
    """baseline CTR=0 이면 lift(= Δ/baseline) 는 정의 불능 → None."""
    exp = _create_experiment(client)
    _log(client, user_id=1, variant="A", exp_id=exp["id"], views=50, clicks=0)
    _log(client, user_id=2, variant="B", exp_id=exp["id"], views=50, clicks=10)
    comps = {c["metric"]: c for c in client.get(f"/experiments/{exp['id']}/stats").json()["comparisons"]}
    assert comps["ctr"]["baseline_rate"] == 0.0
    assert comps["ctr"]["lift"] is None
