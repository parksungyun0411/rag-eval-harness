"""도메인 독립 검색·RAG 평가 하네스."""

from rag_eval.metrics import (
    average_precision,
    evaluate_query,
    hit_at_k,
    mrr,
    ndcg_at_k,
    recall_at_k,
)
from rag_eval.stats import kendall_tau, mcnemar_test, spearman_rho, wilson_interval
from rag_eval.types import GoldenSet, Qrels, Run

__all__ = [
    "GoldenSet",
    "Qrels",
    "Run",
    "average_precision",
    "evaluate_query",
    "hit_at_k",
    "kendall_tau",
    "mcnemar_test",
    "mrr",
    "ndcg_at_k",
    "recall_at_k",
    "spearman_rho",
    "wilson_interval",
]
__version__ = "0.1.0"
