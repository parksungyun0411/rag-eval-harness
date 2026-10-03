"""통계 손계산 검증."""

import pytest

from rag_eval.stats import kendall_tau, mcnemar_test, spearman_rho, wilson_interval


def test_wilson_8_of_10():
    # p=0.8, n=10, z=1.96: center = (0.8 + 0.19208)/1.38416, half = 1.96*sqrt(0.025604)/1.38416
    lo, hi = wilson_interval(8, 10)
    assert lo == pytest.approx(0.4902, abs=1e-4)
    assert hi == pytest.approx(0.9433, abs=1e-4)


def test_wilson_edges():
    lo, hi = wilson_interval(0, 20)
    assert lo == 0.0 and 0 < hi < 0.2
    lo, hi = wilson_interval(20, 20)
    assert hi == 1.0 and 0.8 < lo < 1.0
    with pytest.raises(ValueError):
        wilson_interval(3, 0)


def test_mcnemar_exact():
    # 불일치 b=10, c=2 → 정확 이항: p = 2 * (C(12,0)+C(12,1)+C(12,2)) / 2^12 = 158/4096
    a = [1] * 10 + [0] * 2 + [1] * 5 + [0] * 3
    b = [0] * 10 + [1] * 2 + [1] * 5 + [0] * 3
    res = mcnemar_test(a, b)
    assert (res.b, res.c, res.method) == (10, 2, "exact")
    assert res.p_value == pytest.approx(158 / 4096)


def test_mcnemar_chi2():
    # b=20, c=10 → (|20-10|-1)^2/30 = 2.7, df=1 → p ≈ 0.10035
    a = [1] * 20 + [0] * 10
    b = [0] * 20 + [1] * 10
    res = mcnemar_test(a, b)
    assert res.method == "chi2-cc"
    assert res.statistic == pytest.approx(2.7)
    assert res.p_value == pytest.approx(0.10035, abs=1e-4)


def test_mcnemar_no_discordance():
    res = mcnemar_test([1, 0, 1], [1, 0, 1])
    assert res.p_value == 1.0 and res.method == "none"


def test_rank_correlations():
    # 6쌍 중 (2,3) 한 쌍만 불일치 → tau = (5-1)/6
    tau, _ = kendall_tau([1, 2, 3, 4], [1, 3, 2, 4])
    assert tau == pytest.approx(4 / 6)
    # d = [0, 1, -1, 0], sum d^2 = 2 → rho = 1 - 6*2/(4*15) = 0.8
    rho, _ = spearman_rho([1, 2, 3, 4], [1, 3, 2, 4])
    assert rho == pytest.approx(0.8)
    tau, _ = kendall_tau([1, 2, 3], [3, 2, 1])
    assert tau == pytest.approx(-1.0)
