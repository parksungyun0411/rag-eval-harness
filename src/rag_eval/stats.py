"""설정 비교용 통계.

- Wilson 점수 구간: 질의 수가 150건 수준이면 정규근사 구간이 0·1 근처에서 깨지므로
  hit@k 같은 비율 지표의 신뢰구간은 Wilson으로 낸다.
- McNemar 검정: 같은 질의에 대한 두 설정의 성공/실패(0/1)를 짝지어 비교한다.
  불일치 쌍(b + c)이 25 미만이면 정확 이항검정, 이상이면 연속성 보정 카이제곱.
- 순위 상관: 두 골든셋에서 설정들의 순위가 얼마나 일치하는지 Kendall tau-b와 Spearman rho.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from scipy import stats as _st


def wilson_interval(successes: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if n <= 0:
        raise ValueError("n은 1 이상이어야 합니다")
    if not 0 <= successes <= n:
        raise ValueError("successes는 0 이상 n 이하여야 합니다")
    p = successes / n
    z2 = z * z
    denom = 1 + z2 / n
    center = (p + z2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / denom
    return max(0.0, center - half), min(1.0, center + half)


@dataclass(frozen=True)
class McNemarResult:
    b: int  # A 성공, B 실패
    c: int  # A 실패, B 성공
    statistic: float
    p_value: float
    method: str


def mcnemar_test(a: Sequence[float], b: Sequence[float], exact_below: int = 25) -> McNemarResult:
    """같은 질의 순서로 정렬된 두 설정의 0/1 결과를 비교한다."""
    if len(a) != len(b):
        raise ValueError("두 결과의 질의 수가 다릅니다")
    n01 = sum(1 for x, y in zip(a, b, strict=True) if x >= 0.5 > y)
    n10 = sum(1 for x, y in zip(a, b, strict=True) if y >= 0.5 > x)
    disc = n01 + n10
    if disc == 0:
        return McNemarResult(n01, n10, 0.0, 1.0, "none")
    if disc < exact_below:
        p = _st.binomtest(min(n01, n10), disc, 0.5).pvalue
        return McNemarResult(n01, n10, float(min(n01, n10)), float(min(1.0, p)), "exact")
    chi2 = (abs(n01 - n10) - 1) ** 2 / disc
    p = float(_st.chi2.sf(chi2, 1))
    return McNemarResult(n01, n10, chi2, p, "chi2-cc")


def kendall_tau(x: Sequence[float], y: Sequence[float]) -> tuple[float, float]:
    """Kendall tau-b와 p값."""
    if len(x) != len(y) or len(x) < 2:
        raise ValueError("길이가 같고 2 이상이어야 합니다")
    res = _st.kendalltau(x, y)
    return float(res.statistic), float(res.pvalue)


def spearman_rho(x: Sequence[float], y: Sequence[float]) -> tuple[float, float]:
    if len(x) != len(y) or len(x) < 2:
        raise ValueError("길이가 같고 2 이상이어야 합니다")
    res = _st.spearmanr(x, y)
    return float(res.statistic), float(res.pvalue)
