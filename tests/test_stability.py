"""순위 안정성: 해석적으로 답을 아는 합성 점수 행렬로 검증한다."""

import numpy as np
import pytest

from rag_eval.stability import (
    _order_keys,
    default_ns,
    min_n_for,
    ranking_stability,
    score_matrix,
    stability_from_rows,
)

# a는 질의 3건 중 2건에서 b를 이긴다: 질의별 차이 d = [+1, +1, -1], 전체 순위 a > b
TWO = np.array([[1.0, 1.0, 0.0], [0.0, 0.0, 1.0]])


def test_order_keys_ties_break_by_row():
    ranks = _order_keys(np.array([[0.5], [0.9], [0.5]]))
    assert ranks[:, 0].tolist() == [1, 0, 2]


def test_well_separated_configs_always_match():
    m = np.array([[0.9] * 20, [0.5] * 20, [0.1] * 20])
    res = ranking_stability(m, ["a", "b", "c"], ns=[1, 5, 20], n_boot=200)
    assert res["full_order"] == ["a", "b", "c"]
    for r in res["results"]:
        assert r["exact"] == 1.0 and r["top1"] == 1.0 and r["pairwise"] == 1.0


def test_binomial_known_answer():
    # 복원추출 n건에서 a가 앞설 확률 = P(+1 개수 > -1 개수), p = 2/3
    # n=1: 2/3,  n=3: 3·(2/3)²·(1/3) + (2/3)³ = 20/27
    res = ranking_stability(TWO, ["a", "b"], ns=[1, 3], n_boot=20000, seed=1)
    by_n = {r["n"]: r for r in res["results"]}
    assert by_n[1]["exact"] == pytest.approx(2 / 3, abs=0.015)
    assert by_n[3]["exact"] == pytest.approx(20 / 27, abs=0.015)
    lo, hi = by_n[3]["exact_ci"]
    assert lo <= 20 / 27 <= hi
    # 설정이 2개면 exact = top1 = pairwise
    assert by_n[3]["top1"] == by_n[3]["exact"] == pytest.approx(by_n[3]["pairwise"])


def test_tie_breaks_toward_name_order():
    # n=2에서 (+1,-1) 조합이면 평균이 같아 이름순(a 먼저)으로 깨지므로 일치로 센다
    # P = (2/3)² + 2·(2/3)(1/3) = 8/9
    res = ranking_stability(TWO, ["a", "b"], ns=[2], n_boot=20000, seed=2)
    assert res["results"][0]["exact"] == pytest.approx(8 / 9, abs=0.015)


def test_pairwise_three_configs():
    # c는 항상 꼴찌. (a,b) 쌍만 흔들린다 → n=1에서 pairwise 기댓값 = (2/3 + 1 + 1) / 3
    m = np.vstack([TWO, [[-1.0, -1.0, -1.0]]])
    res = ranking_stability(m, ["a", "b", "c"], ns=[1], n_boot=20000, seed=3)
    r = res["results"][0]
    assert r["pairwise"] == pytest.approx(8 / 9, abs=0.01)
    assert r["exact"] == pytest.approx(2 / 3, abs=0.015)
    assert r["top1"] == pytest.approx(2 / 3, abs=0.015)
    lo, hi = r["pairwise_interval"]
    assert lo == pytest.approx(2 / 3) and hi == pytest.approx(1.0)


def test_without_replacement_full_set_is_exact():
    res = ranking_stability(TWO, ["a", "b"], ns=[3], n_boot=50, replace=False)
    assert res["results"][0]["exact"] == 1.0
    with pytest.raises(ValueError):
        ranking_stability(TWO, ["a", "b"], ns=[4], replace=False)


def test_deterministic_with_seed():
    a = ranking_stability(TWO, ["a", "b"], ns=[3], n_boot=300, seed=7)
    b = ranking_stability(TWO, ["a", "b"], ns=[3], n_boot=300, seed=7)
    assert a["results"] == b["results"]


def test_input_validation():
    with pytest.raises(ValueError):
        ranking_stability(TWO, ["b", "a"], ns=[1])
    with pytest.raises(ValueError):
        ranking_stability(TWO[:1], ["a"], ns=[1])


def test_min_n_for():
    res = {
        "results": [
            {"n": 10, "exact_ci": [0.5, 0.7]},
            {"n": 50, "exact_ci": [0.91, 0.97]},
            {"n": 30, "exact_ci": [0.85, 0.95]},
        ]
    }
    assert min_n_for(res, 0.9) == 50
    assert min_n_for(res, 0.95) is None


def test_score_matrix_and_rows():
    rows = [
        {"golden": "real", "config": c, "qid": q, "m": v}
        for c, vals in {"b": [0, 0, 1, 5], "a": [1, 1, 0]}.items()
        for q, v in zip(["q1", "q2", "q3", "q4"], vals, strict=False)
    ] + [{"golden": "other", "config": "a", "qid": "q1", "m": 9}]
    configs, qids, m = score_matrix(rows, "real", "m")
    # q4는 a에 없으므로 빠진다
    assert configs == ["a", "b"] and qids == ["q1", "q2", "q3"]
    assert m.tolist() == TWO.tolist()
    res = stability_from_rows(rows, "real", "m", ns=[1], n_boot=10)
    assert res["metric"] == "m" and res["n_queries"] == 3
    with pytest.raises(ValueError):
        score_matrix(rows, "other", "m")


def test_default_ns():
    assert default_ns(150) == [10, 20, 30, 50, 75, 100, 150]
    assert default_ns(8) == [8]
