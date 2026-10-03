// 순위 역전 확률 계산기 (브라우저·Node 공용). rag_eval.rankstab.flip_probability 와 같은 식이다.
// 질문별 차이 d ∈ {+1, 0, -1} 가 (p+, p0, p-) 를 따르는 n건 독립 추출의 합 S 에 대해
// P(S > 0), P(S = 0), P(S < 0) 을 정확히 계산한다.
(function (root) {
  "use strict";

  // Lanczos 근사 log Γ(x)
  const G = 7;
  const C = [
    0.99999999999980993, 676.5203681218851, -1259.1392167224028, 771.32342877765313,
    -176.61502916214059, 12.507343278686905, -0.13857109526572012, 9.9843695780195716e-6,
    1.5056327351493116e-7,
  ];
  function lgamma(x) {
    if (x < 0.5) return Math.log(Math.PI / Math.sin(Math.PI * x)) - lgamma(1 - x);
    x -= 1;
    let a = C[0];
    const t = x + G + 0.5;
    for (let i = 1; i < G + 2; i++) a += C[i] / (x + i);
    return 0.5 * Math.log(2 * Math.PI) + (x + 0.5) * Math.log(t) - t + Math.log(a);
  }

  function logBinomPmf(k, n, p) {
    if (p <= 0) return k === 0 ? 0 : -Infinity;
    if (p >= 1) return k === n ? 0 : -Infinity;
    return lgamma(n + 1) - lgamma(k + 1) - lgamma(n - k + 1) + k * Math.log(p) + (n - k) * Math.log(1 - p);
  }

  function pairedSumProbs(n, pPos, pNeg) {
    const pi = pPos + pNeg;
    if (n === 0 || pi <= 0) return [0, 1, 0];
    const q = Math.min(1, pPos / pi);
    let gt = 0, eq = 0, lt = 0;
    for (let k = 0; k <= n; k++) {
      const lpk = logBinomPmf(k, n, pi);
      if (lpk < -40) continue; // 기여가 1e-17 미만인 k는 건너뜀
      const pk = Math.exp(lpk);
      if (k === 0) { eq += pk; continue; }
      // X ~ Binom(k, q) 의 pmf 를 로그 공간 점화식으로 계산 (x마다 lgamma 를 부르지 않음)
      let g = 0, e = 0, l = 0;
      if (q >= 1) g = 1;
      else if (q <= 0) l = 1; // k > 0 이고 모두 아래 쪽만 맞힘
      else {
        const lr = Math.log(q) - Math.log(1 - q);
        let lp = k * Math.log(1 - q);
        for (let x = 0; x <= k; x++) {
          if (x > 0) lp += Math.log((k - x + 1) / x) + lr;
          const px = Math.exp(lp);
          if (2 * x > k) g += px; else if (2 * x === k) e += px; else l += px;
        }
      }
      gt += pk * g; eq += pk * e; lt += pk * l;
    }
    return [gt, eq, lt];
  }

  // 표준정규 누적분포 (Abramowitz–Stegun 7.1.26 기반 erf)
  function normCdf(z) {
    const t = 1 / (1 + 0.3275911 * Math.abs(z) / Math.SQRT2);
    const y = 1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t *
      Math.exp(-(z * z) / 2);
    return z >= 0 ? 0.5 * (1 + y) : 0.5 * (1 - y);
  }

  function firstParity(pPos, pNeg, target, parity, nMax) {
    const ok = (m) => pairedSumProbs(2 * m + parity, pPos, pNeg)[0] >= target;
    let lo = parity ? 0 : 1, hi = lo;
    const limit = Math.floor(nMax / 2);
    while (!ok(hi)) {
      if (hi >= limit) return null;
      lo = hi; hi = Math.min(Math.max(1, hi * 2), limit);
    }
    while (lo < hi) {
      const mid = Math.floor((lo + hi) / 2);
      if (ok(mid)) hi = mid; else lo = mid + 1;
    }
    return 2 * lo + parity;
  }

  // 이 n부터는 항상 P(S > 0) ≥ target 인 가장 작은 n (rag_eval.rankstab.exact_n_required 와 같은 정의)
  function exactNRequired(pPos, pNeg, target = 0.975, nMax = 20000) {
    if (pPos <= pNeg) return null;
    const o = firstParity(pPos, pNeg, target, 1, nMax);
    const e = firstParity(pPos, pNeg, target, 0, nMax);
    if (o === null || e === null) return null;
    const n = Math.max(o, e);
    return n - 1 >= Math.min(o, e) ? n - 1 : n;
  }

  const EPS = 1e-12; // rag_eval.rankstab.EPS 와 같은 값
  const Z975 = 1.959963984540054, Z80 = 0.8416212335729143;
  // 정규 근사 n95 가 이 값을 넘으면 정확해 탐색을 건너뛰고 근사값만 보여 준다(화면 응답 시간 보호)
  const N95_EXACT_MAX = 5000;

  function flipProbability(n, gap, discordance, accuracy) {
    if (!(n >= 1)) throw new Error("n은 1 이상이어야 합니다");
    if (!(gap > 0 && gap <= 1)) throw new Error("gap은 (0, 1] 범위여야 합니다");
    let assumed = false;
    if (discordance === null || discordance === undefined || Number.isNaN(discordance)) {
      assumed = true;
      const pa = accuracy === null || accuracy === undefined || Number.isNaN(accuracy) ? 0.5 + gap / 2 : accuracy;
      const pb = pa - gap;
      if (pb < 0 || pa > 1) throw new Error("accuracy와 gap 조합이 [0, 1]을 벗어납니다");
      discordance = pa * (1 - pb) + pb * (1 - pa);
    }
    if (!(discordance >= gap - EPS && discordance <= 1 + EPS)) throw new Error("불일치율은 gap 이상 1 이하여야 합니다");
    discordance = Math.min(Math.max(discordance, gap), 1);
    const pPos = (discordance + gap) / 2, pNeg = Math.max((discordance - gap) / 2, 0);
    const [hold, tie, rev] = pairedSumProbs(n, pPos, pNeg);
    const v = discordance - gap * gap;
    const n95Normal = v > 0 ? Math.pow(Z975 * Math.sqrt(v) / gap, 2) : 0;
    const skip = n95Normal > N95_EXACT_MAX;
    return {
      n, gap, discordance,
      discordanceAssumedIndependent: assumed,
      pReverse: rev, pTie: tie, pFlip: rev + tie / 2, pHold: hold,
      pFlipNormal: v > 0 ? normCdf(-gap * Math.sqrt(n / v)) : 0,
      // n95: 관측 gap·불일치율이 참값이라고 가정한 양측 95% 분리 n (검정력 약 50% 기준)
      n95Exact: skip ? null : exactNRequired(pPos, pNeg),
      n95ExactSkipped: skip,
      n95Normal,
      nPower80Normal: v > 0 ? Math.pow((Z975 + Z80) * Math.sqrt(v) / gap, 2) : 0,
    };
  }

  const api = { pairedSumProbs, flipProbability, exactNRequired, normCdf, N95_EXACT_MAX };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.Flip = api;
})(typeof window !== "undefined" ? window : globalThis);
