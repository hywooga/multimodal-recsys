"""A/B 테스트 유의성 검정 — 두 비율 비교 (CTR · CVR 등).

명세 4.5 "통계적 유의성 검정 (Chi-square 또는 Z-test)" 을 두 비율 Z-test 로 구현한다.
2변량 Chi-square 와 수학적으로 동치지만(z^2 = χ²), API 응답에 CI 가 포함돼야 해서 Z-test 가 자연스럽다.

외부 의존성 없이 math 표준 라이브러리만 쓴다 (scipy 추가 피함).
"""
import math

Z_95 = 1.959964                             # 양측 95% 임계값 (= scipy.stats.norm.ppf(0.975))


def norm_sf_two_sided(z: float) -> float:
    """표준정규분포 양측 생존함수. P(|Z| > |z|) = 2 * (1 - Φ(|z|))."""
    return 2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(z) / math.sqrt(2.0))))


def wilson_ci_95(successes: int, trials: int) -> tuple[float, float]:
    """Wilson score 95% 신뢰구간 — 표본이 작거나 비율이 0/1 에 가까울 때 Normal approx 보다 정확.
    trials=0 이면 (0, 1) 로 폴백 (아는 게 없음)."""
    if trials <= 0:
        return (0.0, 1.0)
    z = Z_95
    p = successes / trials
    denom = 1.0 + z * z / trials
    center = (p + z * z / (2.0 * trials)) / denom
    half = z * math.sqrt(p * (1.0 - p) / trials + z * z / (4.0 * trials * trials)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def two_proportion_z(baseline_success: int, baseline_n: int,
                     treatment_success: int, treatment_n: int) -> dict:
    """두 비율 비교 Z-test. baseline 과 treatment 의 성공 수 · 시행 수를 받아
    diff(treatment-baseline), lift, p-value, 차이의 Wald 95% CI 를 돌려준다.

    p-value 는 양측 검정, pooled standard error 를 쓴다 (교재 표준).
    CI 는 unpooled Wald interval (각 비율 분산을 더함) — 소표본에서 overcoverage 가능하나
    Z-test 와 분리된 질문(차이의 범위) 이라 흔히 쓰는 조합.

    표본이 0 이거나 성공/실패가 전부 없는 경우: p_value=1.0, diff=0, CI=(0,0).
    """
    if baseline_n <= 0 or treatment_n <= 0:
        return {"diff": 0.0, "lift": None, "z": 0.0, "p_value": 1.0, "diff_ci_95": (0.0, 0.0)}
    p1 = baseline_success / baseline_n
    p2 = treatment_success / treatment_n
    diff = p2 - p1

    # 유의성 검정 — pooled SE
    p_pool = (baseline_success + treatment_success) / (baseline_n + treatment_n)
    se_pool = math.sqrt(p_pool * (1.0 - p_pool) * (1.0 / baseline_n + 1.0 / treatment_n))
    if se_pool == 0.0:
        p_value, z = 1.0, 0.0
    else:
        z = diff / se_pool
        p_value = norm_sf_two_sided(z)

    # 차이의 CI — unpooled Wald interval
    se_wald = math.sqrt(p1 * (1.0 - p1) / baseline_n + p2 * (1.0 - p2) / treatment_n)
    half = Z_95 * se_wald
    ci = (diff - half, diff + half)

    lift = (diff / p1) if p1 > 0 else None
    return {"diff": diff, "lift": lift, "z": z, "p_value": p_value, "diff_ci_95": ci}
