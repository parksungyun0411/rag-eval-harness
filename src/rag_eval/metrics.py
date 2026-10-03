"""랭킹 지표.

모든 함수는 질의 하나에 대한 값이다. ``ranked`` 는 순위순 docid 목록,
``rels`` 는 ``{docid: grade}`` 이며 grade >= 1 을 관련으로 본다.
관련 문서가 없는 질의는 지표가 정의되지 않으므로 ``ValueError`` 를 낸다.

정의
- hit@k: 상위 k 안에 관련 문서가 하나라도 있으면 1
- recall@k: 상위 k 안의 관련 문서 수 / 전체 관련 문서 수
- MRR(@k): 첫 관련 문서 순위의 역수, k 안에 없으면 0
- nDCG@k: 선형 이득(gain = grade), 할인 1/log2(rank+1), 이상 순위는 grade 내림차순
- AP(@k): 관련 문서가 나온 순위마다의 precision 합 / 전체 관련 문서 수 (trec_eval 방식)
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence


def _relevant(rels: Mapping[str, int]) -> set[str]:
    relevant = {d for d, g in rels.items() if g > 0}
    if not relevant:
        raise ValueError("관련 문서가 없는 질의는 지표가 정의되지 않습니다")
    return relevant


def _check_k(k: int) -> None:
    if k < 1:
        raise ValueError(f"k는 1 이상이어야 합니다: {k}")


def hit_at_k(ranked: Sequence[str], rels: Mapping[str, int], k: int) -> float:
    _check_k(k)
    relevant = _relevant(rels)
    return 1.0 if any(d in relevant for d in ranked[:k]) else 0.0


def recall_at_k(ranked: Sequence[str], rels: Mapping[str, int], k: int) -> float:
    _check_k(k)
    relevant = _relevant(rels)
    return len(relevant.intersection(ranked[:k])) / len(relevant)


def mrr(ranked: Sequence[str], rels: Mapping[str, int], k: int | None = None) -> float:
    relevant = _relevant(rels)
    cutoff = ranked if k is None else ranked[:k]
    for i, d in enumerate(cutoff, start=1):
        if d in relevant:
            return 1.0 / i
    return 0.0


def _dcg(gains: Iterable[float]) -> float:
    return sum(g / math.log2(i + 1) for i, g in enumerate(gains, start=1))


def ndcg_at_k(ranked: Sequence[str], rels: Mapping[str, int], k: int) -> float:
    _check_k(k)
    _relevant(rels)
    dcg = _dcg(max(rels.get(d, 0), 0) for d in ranked[:k])
    ideal = sorted((g for g in rels.values() if g > 0), reverse=True)[:k]
    return dcg / _dcg(ideal)


def average_precision(
    ranked: Sequence[str], rels: Mapping[str, int], k: int | None = None
) -> float:
    relevant = _relevant(rels)
    cutoff = ranked if k is None else ranked[:k]
    hits = 0
    total = 0.0
    seen: set[str] = set()
    for i, d in enumerate(cutoff, start=1):
        if d in relevant and d not in seen:
            hits += 1
            total += hits / i
        seen.add(d)
    return total / len(relevant)


def evaluate_query(
    ranked: Sequence[str], rels: Mapping[str, int], ks: Sequence[int] = (1, 5, 10, 20)
) -> dict[str, float]:
    """질의 하나에 대한 전체 지표. 키는 ``hit@10`` 같은 형식."""
    out: dict[str, float] = {}
    for k in ks:
        out[f"hit@{k}"] = hit_at_k(ranked, rels, k)
        out[f"recall@{k}"] = recall_at_k(ranked, rels, k)
        out[f"ndcg@{k}"] = ndcg_at_k(ranked, rels, k)
    out["mrr"] = mrr(ranked, rels)
    out["map"] = average_precision(ranked, rels)
    return out
