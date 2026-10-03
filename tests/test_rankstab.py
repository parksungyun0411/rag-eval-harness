"""리더보드 순위 안정성: 해석적으로 답을 아는 경우로 검증한다."""

import math

import numpy as np
import pytest
from scipy import stats

from rag_eval.rankstab import (
    approximation_check,
    exact_n_required,
    flip_probability,
    kendall_tau_b,
    normal_hold_prob,
    normal_n_required,
    pair_analysis,
    paired_sum_probs,
    rank_stability,
)

# 시스템 a가 질문 3건 중 2건에서 b를 이김: d = [+1, +1, -1]
TWO = np.array([[1.0, 1.0, 0.0], [0.0, 0.0, 1.0]])


def test_paired_sum_probs_known():
    # n=1: (p+, p0, p-)
    assert paired_sum_probs(1, 0.5, 0.2) == pytest.approx((0.5, 0.3, 0.2))
    # p+=2/3, p-=1/3, n=3: P(S>0)=20/27, P(S<0)=7/27
    hold, tie, rev = paired_sum_probs(3, 2 / 3, 1 / 3)
    assert hold == pytest.approx(20 / 27) and rev == pytest.approx(7 / 27) and tie == 0
    # n=2: 동점 = 2·(2/3)(1/3) = 4/9
    assert paired_sum_probs(2, 2 / 3, 1 / 3)[1] == pytest.approx(4 / 9)
    # 불일치가 없으면 항상 동점
    assert paired_sum_probs(10, 0, 0) == (0.0, 1.0, 0.0)


def test_paired_sum_probs_matches_brute_force():
    rng = np.random.default_rng(0)
    p_pos, p_neg, n = 0.12, 0.07, 9
    draws = rng.choice([1, -1, 0], p=[p_pos, p_neg, 1 - p_pos - p_neg], size=(200000, n)).sum(1)
    hold, tie, rev = paired_sum_probs(n, p_pos, p_neg)
    assert (draws > 0).mean() == pytest.approx(hold, abs=0.004)
    assert (draws == 0).mean() == pytest.approx(tie, abs=0.004)
    assert hold + tie + rev == pytest.approx(1.0)


def test_normal_approximations():
    # δ=0.1, 불일치 0.3 → σ²=0.29, n* = (1.96·√0.29 / 0.1)²
    assert normal_n_required(0.1, 0.3) == pytest.approx((1.959964 * math.sqrt(0.29) / 0.1) ** 2)
    assert normal_n_required(0.0, 0.3) == math.inf
    # 큰 n에서 정규 근사와 정확해가 가까워짐
    assert normal_hold_prob(2000, 0.02, 0.2) == pytest.approx(
        paired_sum_probs(2000, 0.11, 0.09)[0], abs=0.01
    )


def _brute_stays_from(p_pos, p_neg, target=0.975, horizon=600):
    ok = [paired_sum_probs(n, p_pos, p_neg)[0] >= target for n in range(1, horizon + 1)]
    last_fail = max((n for n, v in enumerate(ok, start=1) if not v), default=0)
    return last_fail + 1


@pytest.mark.parametrize("p_pos,p_neg", [(2 / 3, 1 / 3), (0.3, 0.1), (0.12, 0.05), (0.5, 0.0)])
def test_exact_n_required_matches_brute_force(p_pos, p_neg):
    # 짝수 n의 동점 때문에 P(S>0)가 단조가 아니므로 "이 n부터 계속 유지" 기준으로 비교
    assert exact_n_required(p_pos, p_neg) == _brute_stays_from(p_pos, p_neg)


def test_exact_n_required_none():
    assert exact_n_required(0.1, 0.2) is None
    assert exact_n_required(0.1, 0.1) is None


def test_flip_probability():
    r = flip_probability(3, gap=1 / 3, discordance=1.0)
    # p+=2/3, p-=1/3 → 역전 7/27, 동점 0
    assert r["p_reverse"] == pytest.approx(7 / 27) and r["p_flip"] == pytest.approx(7 / 27)
    assert r["discordance_assumed_independent"] is False
    r = flip_probability(2, gap=1 / 3, discordance=1.0)
    # 역전 (1/3)² = 1/9, 동점 4/9 → p_flip = 1/9 + 2/9
    assert r["p_flip"] == pytest.approx(1 / 9 + 2 / 9)
    # 독립 가정: pa=0.8, pb=0.6 → 불일치 = 0.8·0.4 + 0.6·0.2 = 0.44
    r = flip_probability(50, gap=0.2, accuracy=0.8)
    assert r["discordance"] == pytest.approx(0.44)
    assert r["discordance_assumed_independent"] is True
    # 같은 gap이라도 불일치율이 낮으면(짝지은 비교) 덜 뒤집힌다
    assert (
        flip_probability(50, 0.05, discordance=0.06)["p_flip"]
        < flip_probability(50, 0.05, discordance=0.5)["p_flip"]
    )
    with pytest.raises(ValueError):
        flip_probability(10, gap=0.3, discordance=0.2)
    with pytest.raises(ValueError):
        flip_probability(10, gap=0.0)


def test_kendall_tau_b_matches_scipy():
    rng = np.random.default_rng(1)
    full = rng.random(7)
    boot = np.round(rng.random((7, 30)), 1)  # 동점 포함
    ours = kendall_tau_b(full, boot)
    ref = [stats.kendalltau(full, boot[:, b]).statistic for b in range(30)]
    assert np.allclose(ours, ref)


def test_rank_stability_two_systems_known():
    res = rank_stability(TWO, ["a", "b"], ns=[1, 3], n_boot=20000, seed=3)
    by_n = {r["n"]: r for r in res["curve"]}
    assert by_n[1]["top1_held"] == pytest.approx(2 / 3, abs=0.015)
    assert by_n[3]["top1_held"] == pytest.approx(20 / 27, abs=0.015)
    assert by_n[3]["top1_lo"] <= 20 / 27 <= by_n[3]["top1_hi"]
    # 두 시스템이면 순위 전체 유지 = 1위 유지, tau 평균 = 2·P(유지) - 1 (n 홀수, 동점 없음)
    assert by_n[3]["order_held"] == by_n[3]["top1_held"]
    assert by_n[3]["tau_mean"] == pytest.approx(2 * by_n[3]["top1_held"] - 1)
    assert res["pair_hold"][0][1] == pytest.approx(20 / 27, abs=0.015)


def test_ties_are_not_held():
    # n=2에서 동점 확률 4/9 → 엄격 기준 유지 확률 = (2/3)² = 4/9
    res = rank_stability(TWO, ["a", "b"], ns=[2], n_boot=20000, seed=4)
    assert res["curve"][0]["top1_held"] == pytest.approx(4 / 9, abs=0.015)


def test_separated_systems_and_topk():
    X = np.vstack([np.ones(40), np.full(40, 0.5), np.zeros(40), np.full(40, 0.25)])
    res = rank_stability(X, ["a", "b", "c", "d"], ns=[5, 40], n_boot=200, topks=(2,))
    for r in res["curve"]:
        assert r["top1_held"] == r["order_held"] == r["top2_held"] == 1.0
        assert r["tau_mean"] == 1.0
    assert res["pairs"] == [("a", "b"), ("b", "d"), ("d", "c")]


def test_without_replacement_full_set():
    res = rank_stability(TWO, ["a", "b"], ns=[3], n_boot=50, replace=False)
    assert res["curve"][0]["order_held"] == 1.0


def test_best_share_splits_ties():
    X = np.array([[1.0, 0.0], [1.0, 0.0]])  # 항상 동점
    res = rank_stability(X, ["a", "b"], ns=[1], n_boot=100)
    assert res["curve"][0]["best_share"] == pytest.approx([0.5, 0.5])
    assert res["curve"][0]["top1_held"] == 0.0


def test_pair_analysis_and_approx_check():
    res = rank_stability(TWO, ["a", "b"], ns=[1, 3, 101], n_boot=4000, seed=5)
    pa = pair_analysis(TWO, ["a", "b"], res)[0]
    assert pa["gap"] == pytest.approx(1 / 3) and pa["discordance"] == pytest.approx(1.0)
    assert pa["n_exact"] == exact_n_required(2 / 3, 1 / 3)
    # 격자 101에서 이미 0.975를 넘음 → 시뮬레이션 최소 n은 격자값 101
    assert paired_sum_probs(101, 2 / 3, 1 / 3)[0] > 0.975 and pa["n_sim_grid"] == 101
    rows = approximation_check(TWO, res)
    assert [r["n"] for r in rows] == [1, 3, 101]
    for r in rows:
        assert r["sim"] == pytest.approx(r["exact"], abs=0.03)


def test_normal_n_required_power():
    # 검정력 50%는 z_power = 0 → 기존 (1.96·σ/δ)² 와 같음
    assert normal_n_required(0.1, 0.3, power=0.5) == pytest.approx(normal_n_required(0.1, 0.3))
    # 검정력 80%: δ=0.1, 불일치 0.3 → σ²=0.29, n = ((1.959964 + 0.841621)·√0.29 / 0.1)² ≈ 227.6
    want = ((1.959963984540054 + 0.8416212335729143) * math.sqrt(0.29) / 0.1) ** 2
    assert normal_n_required(0.1, 0.3, power=0.8) == pytest.approx(want)
    assert normal_n_required(0.1, 0.3, power=0.8) == pytest.approx(227.6175, abs=1e-3)
    with pytest.raises(ValueError):
        normal_n_required(0.1, 0.3, power=1.0)
    r = flip_probability(300, gap=0.01, discordance=0.05)
    assert r["n_power80_normal"] == pytest.approx(normal_n_required(0.01, 0.05, power=0.8))


def test_flip_probability_float_rounding_regression():
    # accuracy=1.0, gap=0.1 → 독립 가정 불일치율이 부동소수로 gap보다 아주 조금 작게 나올 수 있다
    r = flip_probability(10, 0.1, accuracy=1.0)
    assert r["discordance"] == pytest.approx(0.1)
    # 불일치 = gap 이면 아래 쪽만 맞힌 질문이 없다 → 역전 불가능, 동점은 불일치 질문이 0개일 때만
    assert r["p_reverse"] == 0.0
    assert r["p_tie"] == pytest.approx(0.9**10)
    flip_probability(10, 0.1, discordance=0.1 - 1e-15)  # 허용 오차 안
    with pytest.raises(ValueError):
        flip_probability(10, 0.1, discordance=0.1 - 1e-6)


def test_pair_analysis_mcnemar_known():
    # a만 맞힘 9문항, b만 맞힘 6문항, 둘 다 맞힘 5문항 → McNemar 정확 p = 2·Σ_{k≤6} C(15,k) / 2^15
    a = np.array([1.0] * 9 + [0.0] * 6 + [1.0] * 5)
    b = np.array([0.0] * 9 + [1.0] * 6 + [1.0] * 5)
    X = np.vstack([a, b])
    res = rank_stability(X, ["a", "b"], ns=[20], n_boot=50)
    pa = pair_analysis(X, ["a", "b"], res)[0]
    want = 2 * sum(math.comb(15, k) for k in range(7)) / 2**15
    assert (pa["n_pos"], pa["n_neg"]) == (9, 6)
    assert pa["mcnemar_p"] == pytest.approx(want)
    assert pa["mcnemar_p"] == pytest.approx(0.6072, abs=1e-4)
    assert pa["n_power80_normal"] == pytest.approx(normal_n_required(3 / 20, 15 / 20, power=0.8))
