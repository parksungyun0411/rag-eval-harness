"""리더보드 순위 안정성: 질문이 몇 개면 순위를 믿을 수 있는가.

입력은 시스템 × 질문 점수 행렬 X (S × Q, 정답 여부면 0/1)이다. 질문 집합을 부트스트랩해
n건짜리 평가셋을 흉내 내고, 전체 질문으로 매긴 순위와 비교한다. 이 방법은 정보검색의
topic set size 연구(평가 주제 수와 시스템 순위 안정성의 관계)에서 쓰이는 표준적인 방식이며,
이 모듈은 그것을 한국어 RAG 리더보드 데이터에 적용하고 계산기로 묶은 것이다.

용어
- 전체 순위: 전체 질문의 평균 점수 내림차순.
- tau: 부트스트랩 평균과 전체 평균 사이 Kendall tau-b.
- 1위 유지: 전체 1위(동률이면 그 집합 중 하나)가 다른 모든 시스템보다 **엄격하게** 높음.
  동점은 유지로 세지 않는다(보수적).
- 순위 전체 유지: 전체 순위에서 우열이 있는 모든 쌍이 같은 방향으로 **엄격하게** 갈림.
- top-k 유지: 전체 상위 k개 집합의 최솟값이 나머지의 최댓값보다 엄격하게 높음.
- 쌍 분리(95%): 두 시스템 차이의 부트스트랩 분포에서 P(차이 > 0) ≥ 0.975
  (양측 95% 백분위 구간이 0을 포함하지 않음과 같은 기준).

해석적 교차 검증
- 정확해: 복원추출 n건에서 두 시스템 차이의 합 S는 질문별 차이 d ∈ {+1, 0, -1}의
  경험 분포 (p+, p0, p-)를 따르는 삼항 분포의 합이다. P(S > 0)를 정확히 계산할 수 있으므로
  시뮬레이션 값과의 차이는 몬테카를로 오차뿐이어야 한다.
- 정규 근사(McNemar식 검정력 근사): δ = p+ - p-, σ² = (p+ + p-) - δ² 일 때
  P(S > 0) ≈ 1 - Φ((0.5 - nδ) / (σ√n)), 필요한 n ≈ (z·σ / δ)².
  질문 수가 작거나 불일치율이 낮으면 정규 근사가 어긋난다.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np
from scipy import stats as _st

from rag_eval.stats import wilson_interval

Z975 = 1.959963984540054


# ---------------------------------------------------------------- 정확해·근사


def paired_sum_probs(n: int, p_pos: float, p_neg: float) -> tuple[float, float, float]:
    """질문별 차이 d가 +1(p_pos), -1(p_neg), 0(나머지)인 n건 독립 추출의 합 S에 대해
    (P(S > 0), P(S = 0), P(S < 0)) 를 정확히 계산한다."""
    if n < 0 or p_pos < 0 or p_neg < 0 or p_pos + p_neg > 1 + 1e-12:
        raise ValueError("확률 또는 n이 유효하지 않습니다")
    pi = p_pos + p_neg
    if n == 0 or pi <= 0:
        return 0.0, 1.0, 0.0
    q = min(1.0, p_pos / pi)
    k = np.arange(n + 1)
    pk = _st.binom.pmf(k, n, pi)  # 불일치 질문 수
    half = k // 2
    gt = _st.binom.sf(half, k, q)  # X > k/2
    lt = _st.binom.cdf(np.ceil(k / 2) - 1, k, q)  # X < k/2
    lt = np.where(k == 0, 0.0, lt)
    eq = np.where(k % 2 == 0, _st.binom.pmf(half, k, q), 0.0)
    return float(pk @ gt), float(pk @ eq), float(pk @ lt)


def normal_hold_prob(n: int, delta: float, discordance: float) -> float:
    """정규 근사(연속성 보정) P(S > 0)."""
    var = discordance - delta * delta
    if var <= 0:
        return 1.0 if delta > 0 else 0.0
    return float(1 - _st.norm.cdf((0.5 - n * delta) / math.sqrt(var * n)))


def normal_n_required(delta: float, discordance: float, z: float = Z975) -> float:
    """두 시스템을 양측 95%로 가르는 데 필요한 질문 수의 정규 근사 (z·σ/δ)²."""
    if delta <= 0:
        return math.inf
    var = discordance - delta * delta
    return (z * math.sqrt(max(var, 0.0)) / delta) ** 2


def _first_parity(p_pos: float, p_neg: float, target: float, parity: int, n_max: int) -> int | None:
    """홀수(parity=1) 또는 짝수(parity=0) n = 2m + parity 만 볼 때 P(S > 0) ≥ target 인 첫 n.
    같은 홀짝 안에서는 P(S > 0)가 n에 대해 증가한다고 보고 이분 탐색한다."""

    def ok(m: int) -> bool:
        return paired_sum_probs(2 * m + parity, p_pos, p_neg)[0] >= target

    lo = hi = 0 if parity else 1
    limit = n_max // 2
    while not ok(hi):
        if hi >= limit:
            return None
        lo, hi = hi, min(max(1, hi * 2), limit)
    while lo < hi:
        mid = (lo + hi) // 2
        if ok(mid):
            hi = mid
        else:
            lo = mid + 1
    return 2 * lo + parity


def exact_n_required(
    p_pos: float, p_neg: float, target: float = 0.975, n_max: int = 20000
) -> int | None:
    """정확해 기준으로 이 n부터는 항상 P(S > 0) ≥ target 인 가장 작은 n. n_max 안에 없으면 None.

    짝수 n에서는 동점 확률 때문에 P(S > 0)가 바로 앞 홀수 n보다 낮을 수 있어 n에 대해 단조가
    아니다. 홀수·짝수를 따로 찾은 뒤 둘 다 기준을 넘는 지점부터를 답으로 한다.
    """
    if p_pos <= p_neg:
        return None
    first_odd = _first_parity(p_pos, p_neg, target, 1, n_max)
    first_even = _first_parity(p_pos, p_neg, target, 0, n_max)
    if first_odd is None or first_even is None:
        return None
    n = max(first_odd, first_even)
    return n - 1 if n - 1 >= min(first_odd, first_even) else n


def flip_probability(
    n: int,
    gap: float,
    discordance: float | None = None,
    accuracy: float | None = None,
) -> dict[str, Any]:
    """질문 n건, 정확도 차이 gap(> 0, 실제로 앞선 쪽 기준)일 때 순위가 뒤집힐 확률.

    - ``discordance``: 두 시스템 중 하나만 맞힌 질문의 비율. 같은 질문 집합으로 둘을 채점하는
      리더보드에서는 이 값이 분산을 결정한다. 주지 않으면 두 시스템이 질문별로 독립이라고 보고
      ``accuracy`` (앞선 쪽 정확도, 기본 0.5 + gap/2)로부터 계산한다.
    - 반환: p_reverse = P(뒤진 쪽이 엄격히 앞섬), p_tie = P(동점), p_flip = p_reverse + p_tie/2
      (동점은 반반으로 갈린다고 봄), p_flip_normal = 정규 근사, n_95 = 양측 95%로 가르는 데
      필요한 n의 정확해·정규 근사.
    """
    if n < 1:
        raise ValueError("n은 1 이상이어야 합니다")
    if not 0 < gap <= 1:
        raise ValueError("gap은 (0, 1] 범위여야 합니다")
    assumed = discordance is None
    if discordance is None:
        pa = 0.5 + gap / 2 if accuracy is None else accuracy
        pb = pa - gap
        if not (0 <= pb and pa <= 1):
            raise ValueError("accuracy와 gap 조합이 [0, 1]을 벗어납니다")
        discordance = pa * (1 - pb) + pb * (1 - pa)
    if not gap <= discordance <= 1:
        raise ValueError("discordance는 gap 이상 1 이하여야 합니다")
    p_pos = (discordance + gap) / 2
    p_neg = (discordance - gap) / 2
    hold, tie, rev = paired_sum_probs(n, p_pos, p_neg)
    var = discordance - gap * gap
    flip_normal = float(_st.norm.cdf(-gap * math.sqrt(n / var))) if var > 0 else 0.0
    return {
        "n": n,
        "gap": gap,
        "discordance": discordance,
        "discordance_assumed_independent": assumed,
        "p_reverse": rev,
        "p_tie": tie,
        "p_flip": rev + tie / 2,
        "p_flip_normal": flip_normal,
        "n_95_exact": exact_n_required(p_pos, p_neg),
        "n_95_normal": normal_n_required(gap, discordance),
    }


# ---------------------------------------------------------------- 부트스트랩


def kendall_tau_b(full: np.ndarray, boot: np.ndarray) -> np.ndarray:
    """full (S,) 과 boot (S, B) 각 열 사이 Kendall tau-b. 동점 보정 포함."""
    iu, ju = np.triu_indices(len(full), k=1)
    a = np.sign(full[iu] - full[ju])[:, None]
    b = np.sign(boot[iu] - boot[ju])
    num = (a * b).sum(axis=0)
    den = np.sqrt((a != 0).sum(axis=0) * (b != 0).sum(axis=0))
    return np.where(den > 0, num / np.where(den > 0, den, 1), 0.0)


def _winner_held(full: np.ndarray, boot: np.ndarray) -> np.ndarray:
    top = full == full.max()
    if top.all():
        return np.zeros(boot.shape[1], dtype=bool)
    return boot[top].max(axis=0) > boot[~top].max(axis=0)


def _topk_held(full: np.ndarray, boot: np.ndarray, k: int) -> np.ndarray | None:
    order = np.argsort(-full, kind="stable")
    if k >= len(full) or full[order[k - 1]] == full[order[k]]:
        return None  # 전체 순위에서 k위 경계가 동점이면 정의하지 않음
    top = np.zeros(len(full), dtype=bool)
    top[order[:k]] = True
    return boot[top].min(axis=0) > boot[~top].max(axis=0)


def _order_held(full: np.ndarray, boot: np.ndarray) -> np.ndarray:
    iu, ju = np.triu_indices(len(full), k=1)
    a = np.sign(full[iu] - full[ju])
    keep = a != 0
    b = np.sign(boot[iu][keep] - boot[ju][keep])
    return np.all(b == a[keep][:, None], axis=0)


def _best_share(boot: np.ndarray) -> np.ndarray:
    """시스템별 '부트스트랩에서 1위' 비율. 공동 1위는 1/k씩 나눈다."""
    is_max = boot == boot.max(axis=0, keepdims=True)
    return (is_max / is_max.sum(axis=0, keepdims=True)).mean(axis=1)


def _bootstrap_means(
    X: np.ndarray, n: int, n_boot: int, rng: np.random.Generator, replace: bool, chunk: int = 250
) -> np.ndarray:
    q = X.shape[1]
    out = np.empty((X.shape[0], n_boot))
    for start in range(0, n_boot, chunk):
        b = min(chunk, n_boot - start)
        if replace:
            idx = rng.integers(0, q, size=(b, n))
        else:
            idx = np.argsort(rng.random((b, q)), axis=1)[:, :n]
        out[:, start : start + b] = X[:, idx].mean(axis=2)
    return out


def adjacent_pairs(full: np.ndarray) -> list[tuple[int, int]]:
    order = np.argsort(-full, kind="stable")
    return [(int(order[i]), int(order[i + 1])) for i in range(len(order) - 1)]


def rank_stability(
    X: np.ndarray,
    systems: Sequence[str],
    ns: Sequence[int],
    n_boot: int = 2000,
    seed: int = 0,
    replace: bool = True,
    topks: Sequence[int] = (3, 5),
) -> dict[str, Any]:
    """n별 순위 안정성 곡선과 인접 쌍 분리 확률."""
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2 or X.shape[0] != len(systems) or X.shape[0] < 2:
        raise ValueError("X는 (시스템 수 ≥ 2, 질문 수) 행렬이어야 합니다")
    rng = np.random.default_rng(seed)
    full = X.mean(axis=1)
    pairs = adjacent_pairs(full)
    curve = []
    pair_hold = np.zeros((len(pairs), len(ns)))
    for t, n in enumerate(ns):
        if n < 1 or (not replace and n > X.shape[1]):
            raise ValueError(f"n={n} 이(가) 유효하지 않습니다")
        boot = _bootstrap_means(X, n, n_boot, rng, replace)
        tau = kendall_tau_b(full, boot)
        row: dict[str, Any] = {
            "n": int(n),
            "tau_mean": float(tau.mean()),
            "tau_lo": float(np.quantile(tau, 0.025)),
            "tau_hi": float(np.quantile(tau, 0.975)),
        }
        for name, held in (
            ("top1", _winner_held(full, boot)),
            ("order", _order_held(full, boot)),
            *((f"top{k}", _topk_held(full, boot, k)) for k in topks),
        ):
            if held is None:
                continue
            s = int(held.sum())
            lo, hi = wilson_interval(s, n_boot)
            row[f"{name}_held"] = s / n_boot
            row[f"{name}_lo"] = lo
            row[f"{name}_hi"] = hi
        row["best_share"] = _best_share(boot).tolist()
        curve.append(row)
        for p, (i, j) in enumerate(pairs):
            pair_hold[p, t] = float((boot[i] > boot[j]).mean())
    return {
        "systems": list(systems),
        "n_questions": X.shape[1],
        "full_mean": full.tolist(),
        "ns": [int(n) for n in ns],
        "n_boot": n_boot,
        "seed": seed,
        "replace": replace,
        "curve": curve,
        "pairs": [(systems[i], systems[j]) for i, j in pairs],
        "pair_index": pairs,
        "pair_hold": pair_hold.tolist(),
    }


def pair_analysis(
    X: np.ndarray, systems: Sequence[str], result: dict[str, Any], target: float = 0.975
) -> list[dict[str, Any]]:
    """인접 쌍별: 정확도 차이, 불일치율, 시뮬레이션 최소 n(격자), 정확해 최소 n, 정규 근사 n."""
    X = np.asarray(X, dtype=np.float64)
    out = []
    for p, (i, j) in enumerate(result["pair_index"]):
        d = X[i] - X[j]
        p_pos, p_neg = float((d > 0).mean()), float((d < 0).mean())
        delta, disc = p_pos - p_neg, p_pos + p_neg
        # 격자에서 이 n부터 끝까지 계속 기준을 넘는 가장 작은 n
        sim_n = None
        for n, h in reversed(list(zip(result["ns"], result["pair_hold"][p], strict=True))):
            if h < target:
                break
            sim_n = n
        out.append(
            {
                "upper": systems[i],
                "lower": systems[j],
                "gap": delta,
                "discordance": disc,
                "p_pos": p_pos,
                "p_neg": p_neg,
                "n_sim_grid": sim_n,
                "n_exact": exact_n_required(p_pos, p_neg, target),
                "n_normal": normal_n_required(delta, disc),
            }
        )
    return out


def approximation_check(X: np.ndarray, result: dict[str, Any]) -> list[dict[str, Any]]:
    """인접 쌍 × n 마다 시뮬레이션·정확해·정규 근사의 P(위 시스템이 엄격히 앞섬)."""
    X = np.asarray(X, dtype=np.float64)
    rows = []
    for p, (i, j) in enumerate(result["pair_index"]):
        d = X[i] - X[j]
        p_pos, p_neg = float((d > 0).mean()), float((d < 0).mean())
        for t, n in enumerate(result["ns"]):
            exact = paired_sum_probs(n, p_pos, p_neg)[0]
            rows.append(
                {
                    "pair": p,
                    "upper": result["systems"][i],
                    "lower": result["systems"][j],
                    "n": n,
                    "sim": result["pair_hold"][p][t],
                    "exact": exact,
                    "normal": normal_hold_prob(n, p_pos - p_neg, p_pos + p_neg),
                }
            )
    return rows


def domain_winners(full: np.ndarray, systems: Sequence[str]) -> list[str]:
    return [s for s, v in zip(systems, full, strict=True) if v == full.max()]
