"""질의 수에 따른 설정 순위 안정성.

"평가 질의가 n건뿐이었다면, 설정들의 순위가 전체 질의로 매긴 순위와 얼마나 자주 같았을까"를
부트스트랩으로 추정한다. 평가셋을 얼마나 크게 만들어야 설정 비교 결론이 흔들리지 않는지
판단하는 데 쓴다.

절차 (지표 하나, 골든셋 하나 기준)
1. 설정 × 질의 점수 행렬 M (C × Q)을 만든다. 모든 설정에 결과가 있는 질의만 쓴다.
2. 전체 순위: 질의 평균 내림차순, 평균이 같으면 설정 이름순.
3. 각 n에 대해 B번 반복: 질의 n건을 뽑아(기본 복원추출) 평균을 내고 같은 규칙으로 순위를 매긴다.
4. 반복마다 다음을 기록한다.
   - exact: 순위 전체가 전체 순위와 같으면 1
   - top1: 1위 설정이 같으면 1
   - pairwise: 설정 쌍 중 순서(우열·동점)가 전체와 같은 쌍의 비율
5. exact·top1은 B번 중 일치 비율과 Wilson 구간(부트스트랩 반복 수에 따른 추정 오차),
   pairwise는 반복값의 평균과 백분위 구간(표본에 따른 변동 폭)을 낸다.

복원추출에서는 n = Q여도 일치율이 1이 아니다. "같은 크기의 평가셋을 다시 뽑았다면"을 뜻한다.
비복원추출(``replace=False``)은 "지금 평가셋의 부분집합만 썼다면"을 뜻하고 n = Q에서 1이 된다.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from rag_eval.stats import wilson_interval


def score_matrix(
    rows: Sequence[Mapping[str, Any]], golden: str, metric: str
) -> tuple[list[str], list[str], np.ndarray]:
    """질의별 행 → (설정 목록, 질의 목록, C × Q 행렬). 모든 설정에 있는 질의만 남긴다."""
    table: dict[str, dict[str, float]] = {}
    for r in rows:
        if r["golden"] == golden:
            table.setdefault(r["config"], {})[r["qid"]] = float(r[metric])
    if len(table) < 2:
        raise ValueError(f"골든셋 {golden}에 설정이 2개 이상 필요합니다")
    configs = sorted(table)
    qids = sorted(set.intersection(*(set(v) for v in table.values())))
    if not qids:
        raise ValueError("모든 설정에 공통인 질의가 없습니다")
    return configs, qids, np.array([[table[c][q] for q in qids] for c in configs])


def _order_keys(means: np.ndarray) -> np.ndarray:
    """(C, B) 평균 → 각 열의 순위(0이 1위). 동점은 행 순서(=이름순)로 깬다."""
    # 안정 정렬이므로 같은 값이면 앞선 행(이름이 빠른 설정)이 먼저 온다
    order = np.argsort(-means, axis=0, kind="stable")
    ranks = np.empty_like(order)
    rows = np.arange(means.shape[0])[:, None]
    np.put_along_axis(ranks, order, np.broadcast_to(rows, order.shape), axis=0)
    return ranks


def ranking_stability(
    matrix: np.ndarray,
    configs: Sequence[str],
    ns: Sequence[int],
    n_boot: int = 1000,
    seed: int = 0,
    replace: bool = True,
    interval: float = 0.95,
) -> dict[str, Any]:
    """``matrix`` 는 C × Q 점수 행렬, ``configs`` 는 행 이름.

    ``configs`` 가 이름순이어야 동점을 이름순으로 깨는 규칙이 성립한다.
    """
    matrix = np.asarray(matrix, dtype=np.float64)
    n_cfg, n_q = matrix.shape
    if n_cfg != len(configs) or n_cfg < 2:
        raise ValueError("설정 수와 행렬 행 수가 다르거나 2 미만입니다")
    if list(configs) != sorted(configs):
        raise ValueError("configs는 이름순으로 정렬돼 있어야 합니다")
    rng = np.random.default_rng(seed)
    full = matrix.mean(axis=1)
    full_rank = _order_keys(full[:, None])[:, 0]
    full_order = [configs[i] for i in np.argsort(full_rank)]
    iu, ju = np.triu_indices(n_cfg, k=1)
    full_sign = np.sign(full[iu] - full[ju])
    alpha = (1 - interval) / 2
    results = []
    for n in ns:
        if n < 1 or (not replace and n > n_q):
            raise ValueError(f"n={n}이(가) 유효하지 않습니다 (질의 {n_q}건)")
        if replace:
            idx = rng.integers(0, n_q, size=(n_boot, n))
        else:
            idx = np.argsort(rng.random((n_boot, n_q)), axis=1)[:, :n]
        means = matrix[:, idx].mean(axis=2)  # (C, B)
        ranks = _order_keys(means)
        exact = np.all(ranks == full_rank[:, None], axis=0)
        top1 = ranks[np.argmin(full_rank)] == 0
        boot_sign = np.sign(means[iu] - means[ju])  # (pairs, B)
        pairwise = (boot_sign == full_sign[:, None]).mean(axis=0)
        e_lo, e_hi = wilson_interval(int(exact.sum()), n_boot)
        t_lo, t_hi = wilson_interval(int(top1.sum()), n_boot)
        results.append(
            {
                "n": int(n),
                "exact": float(exact.mean()),
                "exact_ci": [e_lo, e_hi],
                "top1": float(top1.mean()),
                "top1_ci": [t_lo, t_hi],
                "pairwise": float(pairwise.mean()),
                "pairwise_interval": [
                    float(np.quantile(pairwise, alpha)),
                    float(np.quantile(pairwise, 1 - alpha)),
                ],
            }
        )
    return {
        "configs": list(configs),
        "n_queries": n_q,
        "full_order": full_order,
        "full_means": {c: float(m) for c, m in zip(configs, full, strict=True)},
        "n_boot": n_boot,
        "replace": replace,
        "seed": seed,
        "results": results,
    }


def min_n_for(result: Mapping[str, Any], target: float = 0.9, key: str = "exact") -> int | None:
    """일치율 신뢰구간 하한이 ``target`` 이상이 되는 가장 작은 n. 없으면 None."""
    for r in sorted(result["results"], key=lambda r: r["n"]):
        if r[f"{key}_ci"][0] >= target:
            return int(r["n"])
    return None


def default_ns(n_queries: int) -> list[int]:
    """10건부터 전체 질의 수까지 대략 균등한 격자."""
    base = [10, 20, 30, 50, 75, 100, 150, 200, 300, 500, 750, 1000]
    ns = [n for n in base if n < n_queries]
    return ns + [n_queries]


def stability_from_rows(
    rows: Sequence[Mapping[str, Any]],
    golden: str,
    metric: str,
    ns: Sequence[int] | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    configs, qids, m = score_matrix(rows, golden, metric)
    res = ranking_stability(m, configs, ns or default_ns(len(qids)), **kwargs)
    res.update({"golden": golden, "metric": metric})
    return res
