"""골든셋 2종 비교: 같은 검색 설정들을 두 골든셋(예: 실제 질의 대 LLM 합성 질의)으로 쟀을 때
(1) 설정별 점수가 얼마나 다른지(차이), (2) 설정 순위가 얼마나 유지되는지(순위 상관)를 계산한다.

합성 골든셋이 점수를 부풀리더라도 설정 순위가 같다면 "설정 선택"에는 쓸 수 있다는 뜻이고,
순위까지 바뀐다면 합성 골든셋으로 고른 설정이 실제 질의에서 최선이 아닐 수 있다는 뜻이다.
설정 수가 적으면 순위 상관이 가질 수 있는 값이 몇 개 안 되므로 p값과 설정 수를 같이 남긴다.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from rag_eval.matrix import summarize
from rag_eval.stats import kendall_tau, spearman_rho

DEFAULT_METRICS = ("hit@10", "recall@10", "mrr", "ndcg@10", "map")


def _ranks(values: Sequence[float]) -> list[float]:
    """내림차순 순위(1이 최고). 동점은 평균 순위."""
    order = sorted(range(len(values)), key=lambda i: -values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        for t in range(i, j + 1):
            ranks[order[t]] = (i + j) / 2 + 1
        i = j + 1
    return ranks


def compare_golden_sets(
    rows: Sequence[Mapping[str, Any]],
    a: str,
    b: str,
    metrics: Sequence[str] = DEFAULT_METRICS,
) -> dict[str, Any]:
    """질의별 행에서 골든셋 ``a`` 와 ``b`` 를 비교한다. 두 골든셋에 모두 있는 설정만 쓴다."""
    summary = {(s["golden"], s["config"]): s for s in summarize(rows)}
    configs = sorted({c for g, c in summary if g == a} & {c for g, c in summary if g == b})
    if len(configs) < 2:
        raise ValueError(f"두 골든셋에 공통 설정이 2개 이상 필요합니다: {configs}")
    result: dict[str, Any] = {"a": a, "b": b, "configs": configs, "metrics": {}}
    for m in metrics:
        va = [float(summary[(a, c)][m]) for c in configs]
        vb = [float(summary[(b, c)][m]) for c in configs]
        tau, tau_p = kendall_tau(va, vb)
        rho, rho_p = spearman_rho(va, vb)
        ra, rb = _ranks(va), _ranks(vb)
        result["metrics"][m] = {
            "per_config": [
                {
                    "config": c,
                    a: va[i],
                    b: vb[i],
                    "gap": vb[i] - va[i],
                    f"rank_{a}": ra[i],
                    f"rank_{b}": rb[i],
                }
                for i, c in enumerate(configs)
            ],
            "mean_gap": sum(vb[i] - va[i] for i in range(len(configs))) / len(configs),
            "kendall_tau": tau,
            "kendall_p": tau_p,
            "spearman_rho": rho,
            "spearman_p": rho_p,
            "best_config": {a: configs[ra.index(min(ra))], b: configs[rb.index(min(rb))]},
            "n_configs": len(configs),
        }
    return result
