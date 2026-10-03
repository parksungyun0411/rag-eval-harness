"""지표 손계산 검증.

ranked = [d1, d2, d3, d4, d5], 관련 = {d2: 2, d4: 1, d9: 1} (d9는 검색되지 않음, 관련 3건)
- hit@1 = 0, hit@2 = 1
- recall@2 = 1/3, recall@5 = 2/3
- MRR = 1/2 (첫 관련 문서가 2위)
- DCG@5 = 2/log2(3) + 1/log2(5); IDCG@5 = 2/log2(2) + 1/log2(3) + 1/log2(4)
- AP = (1/2 + 2/4) / 3 = 1/3
"""

import math

import pytest

from rag_eval.metrics import (
    average_precision,
    evaluate_query,
    hit_at_k,
    mrr,
    ndcg_at_k,
    recall_at_k,
)

RANKED = ["d1", "d2", "d3", "d4", "d5"]
RELS = {"d2": 2, "d4": 1, "d9": 1}


def test_hit():
    assert hit_at_k(RANKED, RELS, 1) == 0.0
    assert hit_at_k(RANKED, RELS, 2) == 1.0


def test_recall():
    assert recall_at_k(RANKED, RELS, 1) == 0.0
    assert recall_at_k(RANKED, RELS, 2) == pytest.approx(1 / 3)
    assert recall_at_k(RANKED, RELS, 5) == pytest.approx(2 / 3)


def test_mrr():
    assert mrr(RANKED, RELS) == pytest.approx(0.5)
    assert mrr(RANKED, RELS, k=1) == 0.0
    assert mrr(["x", "y"], {"z": 1}) == 0.0


def test_ndcg_hand_computed():
    dcg5 = 2 / math.log2(3) + 1 / math.log2(5)
    idcg5 = 2 / 1 + 1 / math.log2(3) + 1 / math.log2(4)
    assert ndcg_at_k(RANKED, RELS, 5) == pytest.approx(dcg5 / idcg5)
    assert ndcg_at_k(RANKED, RELS, 5) == pytest.approx(0.540586, abs=1e-6)
    # k=2: DCG = 2/log2(3), IDCG = 2 + 1/log2(3)
    assert ndcg_at_k(RANKED, RELS, 2) == pytest.approx(0.479625, abs=1e-6)


def test_ndcg_perfect_and_zero():
    assert ndcg_at_k(["d2", "d4", "d9"], RELS, 3) == pytest.approx(1.0)
    assert ndcg_at_k(["x", "y"], RELS, 2) == 0.0


def test_average_precision():
    assert average_precision(RANKED, RELS) == pytest.approx(1 / 3)
    # 관련 2건을 1·3위에서 찾음: (1/1 + 2/3) / 2
    assert average_precision(["a", "x", "b"], {"a": 1, "b": 1}) == pytest.approx(5 / 6)
    # 중복 docid는 한 번만 센다
    assert average_precision(["a", "a", "b"], {"a": 1, "b": 1}) == pytest.approx((1 + 2 / 3) / 2)


def test_zero_grade_is_not_relevant():
    rels = {"d1": 0, "d2": 1}
    assert hit_at_k(RANKED, rels, 1) == 0.0
    assert recall_at_k(RANKED, rels, 2) == 1.0


def test_undefined_without_relevant():
    with pytest.raises(ValueError):
        recall_at_k(RANKED, {"d1": 0}, 5)
    with pytest.raises(ValueError):
        hit_at_k(RANKED, RELS, 0)


def test_evaluate_query_keys():
    out = evaluate_query(RANKED, RELS, ks=(1, 5))
    assert set(out) == {"hit@1", "recall@1", "ndcg@1", "hit@5", "recall@5", "ndcg@5", "mrr", "map"}
    assert out["recall@5"] == pytest.approx(2 / 3)
