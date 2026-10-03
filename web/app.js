// 데모 화면. data/results.json 만으로 동작하고(정적 배포 가능), FastAPI 서버가 있으면
// 계산기 결과를 /api/flip 정확해와 나란히 보여 준다.
(function () {
  "use strict";
  const $ = (id) => document.getElementById(id);
  const fmt = (x, d = 3) => (x === null || x === undefined || Number.isNaN(x) ? "—" : Number(x).toFixed(d));
  const pct = (x, d = 1) => (x === null || x === undefined ? "—" : (100 * x).toFixed(d) + "%");
  const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
  let DATA = null;
  let API = false;

  function renderMeta(m) {
    $("meta").textContent =
      `데이터: ${m.dataset} @ ${m.revision.slice(0, 12)} · 복원추출 ${m.n_boot.toLocaleString()}회 · seed ${m.seed} · 생성 ${m.generated}`;
    $("footer").textContent = `재현 명령: ${m.command || "rag-eval rankstab"} · 원자료와 보고서: reports/REPORT.md`;
  }

  function renderLeaderboard() {
    const pairs = DATA.pairs;
    const doms = DATA.domains;
    const head =
      "<tr><th>순위</th><th class='sys'>시스템</th><th>정답</th><th class='sys'>정확도 · 95% CI</th>" +
      doms.map((d) => `<th>${d}</th>`).join("") + "<th>아래 순위와 구분</th></tr>";
    const rows = DATA.leaderboard.map((r, i) => {
      const x = (v) => (v * 180).toFixed(1);
      const bar =
        `<svg width="190" height="14" aria-hidden="true">` +
        `<rect x="0" y="4" width="${x(r.accuracy)}" height="6" fill="#9db9e3"/>` +
        `<line x1="${x(r.ci[0])}" x2="${x(r.ci[1])}" y1="7" y2="7" stroke="#1d5fbf" stroke-width="1.5"/>` +
        `</svg> ${fmt(r.accuracy)}`;
      let sep = "";
      if (i < pairs.length) {
        const n = pairs[i].n_exact;
        sep = n !== null && n <= DATA.leaderboard[0].n
          ? `<td class="sep-ok">${n}문항</td>`
          : `<td class="sep-no">${n === null ? "&gt;20,000" : n.toLocaleString()}문항</td>`;
      } else sep = "<td></td>";
      return `<tr><td>${r.rank}</td><td class="sys">${esc(r.system)}</td><td>${r.correct}/${r.n}</td>` +
        `<td class="sys bar">${bar}</td>` + doms.map((d) => `<td>${fmt(r.domains[d], 2)}</td>`).join("") + sep + "</tr>";
    });
    $("leaderboard").innerHTML = head + rows.join("");
  }

  function lineChart(series, opts) {
    const W = 960, H = 340, L = 52, R = 16, T = 14, B = 40;
    const xs = series[0].points.map((p) => p.x);
    const xmax = Math.max(...xs), xmin = 0;
    const X = (v) => L + ((v - xmin) / (xmax - xmin)) * (W - L - R);
    const Y = (v) => T + (1 - v) * (H - T - B);
    let s = `<svg viewBox="0 0 ${W} ${H}" xmlns="http://www.w3.org/2000/svg">`;
    for (let g = 0; g <= 5; g++) {
      const v = g / 5;
      s += `<line x1="${L}" x2="${W - R}" y1="${Y(v)}" y2="${Y(v)}" stroke="#e8ebef"/>` +
        `<text x="${L - 8}" y="${Y(v) + 4}" font-size="11" text-anchor="end" fill="#5f6b7a">${v.toFixed(1)}</text>`;
    }
    xs.forEach((v) => {
      s += `<text x="${X(v)}" y="${H - B + 16}" font-size="11" text-anchor="middle" fill="#5f6b7a">${v}</text>`;
    });
    s += `<text x="${(L + W - R) / 2}" y="${H - 6}" font-size="12" text-anchor="middle" fill="#5f6b7a">${opts.xlabel}</text>`;
    series.forEach((sr) => {
      if (sr.band) {
        const up = sr.points.map((p) => `${X(p.x)},${Y(p.hi)}`).join(" ");
        const dn = sr.points.slice().reverse().map((p) => `${X(p.x)},${Y(p.lo)}`).join(" ");
        s += `<polygon points="${up} ${dn}" fill="${sr.color}" opacity="0.13"/>`;
      }
      s += `<polyline fill="none" stroke="${sr.color}" stroke-width="2" points="${sr.points.map((p) => `${X(p.x)},${Y(p.y)}`).join(" ")}"/>`;
      sr.points.forEach((p) => { s += `<circle cx="${X(p.x)}" cy="${Y(p.y)}" r="3" fill="${sr.color}"/>`; });
    });
    series.forEach((sr, i) => {
      const lx = L + 12 + i * 190;
      s += `<rect x="${lx}" y="${T + 4}" width="12" height="3" fill="${sr.color}"/>` +
        `<text x="${lx + 18}" y="${T + 9}" font-size="12" fill="#1f2933">${sr.label}</text>`;
    });
    return s + "</svg>";
  }

  function renderChart() {
    const scope = $("scope").value;
    const curve = scope === "overall" ? DATA.overall_curve : DATA.domain_curves[scope];
    const pts = (k) => curve.filter((c) => c[k + "_held"] !== undefined)
      .map((c) => ({ x: c.n, y: c[k + "_held"], lo: c[k + "_lo"], hi: c[k + "_hi"] }));
    const series = [
      { label: "Kendall tau (95% 구간)", color: "#1d5fbf", band: true,
        points: curve.map((c) => ({ x: c.n, y: c.tau_mean, lo: c.tau_lo, hi: c.tau_hi })) },
      { label: "1위 유지 확률", color: "#c2410c", band: true, points: pts("top1") },
    ];
    if (curve[0].top3_held !== undefined) series.push({ label: "top-3 집합 유지", color: "#0f766e", points: pts("top3") });
    series.push({ label: "순위 전체 유지", color: "#6b7280", points: pts("order") });
    $("chart").innerHTML = lineChart(series, { xlabel: "질문 수 n (복원추출)" });
    const last = curve[curve.length - 1];
    const winners = scope === "overall" ? DATA.leaderboard[0].system : DATA.domain_winners[scope];
    $("chart-note").textContent =
      `${scope === "overall" ? "전체" : scope} 1위: ${winners}. n=${last.n}에서 Kendall tau ${fmt(last.tau_mean)} ` +
      `[${fmt(last.tau_lo)}, ${fmt(last.tau_hi)}], 1위 유지 ${fmt(last.top1_held)}, 순위 전체 유지 ${fmt(last.order_held)}.` +
      " 유지는 엄격한 우위만 세며 동점은 유지로 보지 않습니다.";
  }

  function fillPair() {
    const a = $("sysA").value, b = $("sysB").value;
    if (!a || !b || a === b) return;
    const ba = DATA.correct_bits[a], bb = DATA.correct_bits[b];
    let pos = 0, neg = 0, ca = 0, cb = 0;
    for (let i = 0; i < ba.length; i++) {
      const x = +ba[i], y = +bb[i];
      ca += x; cb += y;
      if (x > y) pos++; else if (y > x) neg++;
    }
    const q = ba.length;
    const gap = Math.abs(ca - cb) / q;
    $("gap").value = (100 * gap).toFixed(2);
    $("disc").value = (100 * (pos + neg) / q).toFixed(2);
    $("acc").value = "";
    if (gap === 0) $("calc-out").innerHTML = `<p class="err">두 시스템의 정답 수가 같습니다(${ca}/${q}). 역전 확률이 정의되지 않습니다.</p>`;
    else compute();
  }

  function stat(k, v, s, main) {
    return `<div class="stat${main ? " main" : ""}"><div class="k">${k}</div><div class="v">${v}</div>${s ? `<div class="s">${s}</div>` : ""}</div>`;
  }

  async function compute() {
    const n = parseInt($("n").value, 10);
    const gap = parseFloat($("gap").value) / 100;
    const discRaw = $("disc").value.trim();
    const accRaw = $("acc").value.trim();
    const disc = discRaw === "" ? null : parseFloat(discRaw) / 100;
    const acc = accRaw === "" ? null : parseFloat(accRaw) / 100;
    let r;
    try {
      r = Flip.flipProbability(n, gap, disc, acc);
    } catch (e) {
      $("calc-out").innerHTML = `<p class="err">${esc(e.message)}</p>`;
      return;
    }
    let html =
      stat("순위가 뒤집혀 보일 확률", pct(r.pFlip), "동점은 반반으로 계산", true) +
      stat("엄격한 역전", pct(r.pReverse)) +
      stat("동점", pct(r.pTie)) +
      stat("정규 근사", pct(r.pFlipNormal), "McNemar식 근사") +
      stat("95%로 가르는 데 필요한 n", r.n95Exact === null ? "&gt;20,000" : r.n95Exact.toLocaleString(),
        `정규 근사 ${Math.ceil(r.n95Normal).toLocaleString()}`) +
      stat("사용한 불일치율", pct(r.discordance), r.discordanceAssumedIndependent ? "독립 가정으로 계산" : "입력값");
    if (API) {
      const qs = new URLSearchParams({ n, gap });
      if (disc !== null) qs.set("discordance", disc);
      if (acc !== null) qs.set("accuracy", acc);
      try {
        const res = await fetch(`api/flip?${qs}`);
        const j = await res.json();
        html += res.ok
          ? stat("서버 /api/flip", pct(j.p_flip), `n95 ${j.n_95_exact === null ? "&gt;20,000" : j.n_95_exact}`)
          : stat("서버 /api/flip", "오류", esc(j.detail || res.status));
      } catch (e) { /* 정적 배포: 서버 없음 */ }
    }
    $("calc-out").innerHTML = html;
  }

  async function init() {
    const res = await fetch("data/results.json");
    DATA = await res.json();
    try { API = (await fetch("api/health")).ok; } catch (e) { API = false; }
    renderMeta(DATA.meta);
    renderLeaderboard();
    $("scope").innerHTML = ["overall", ...DATA.domains]
      .map((d) => `<option value="${d}">${d === "overall" ? "전체 (300문항)" : d + " (60문항)"}</option>`).join("");
    $("scope").addEventListener("change", renderChart);
    renderChart();
    const opts = '<option value="">—</option>' +
      DATA.leaderboard.map((r) => `<option>${esc(r.system)}</option>`).join("");
    $("sysA").innerHTML = opts; $("sysB").innerHTML = opts;
    $("sysA").value = DATA.leaderboard[0].system; $("sysB").value = DATA.leaderboard[1].system;
    ["sysA", "sysB"].forEach((id) => $(id).addEventListener("change", fillPair));
    ["n", "gap", "disc", "acc"].forEach((id) => $(id).addEventListener("input", compute));
    $("n").value = DATA.leaderboard[0].n;
    fillPair();
  }

  init().catch((e) => { $("meta").textContent = "결과를 불러오지 못했습니다: " + e.message; });
})();
