"""보고서 생성: 마크다운 표와 그림을 ``reports/`` 에 쓴다.

숫자는 모두 질의별 원자료(per_query.jsonl)·비교 결과(JSON)·판정 결과에서 다시 계산한다.
보고서 파일에 손으로 숫자를 적지 않는다.
"""

from __future__ import annotations

import datetime as _dt
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from rag_eval.matrix import summarize  # noqa: E402
from rag_eval.stats import mcnemar_test, wilson_interval  # noqa: E402

TABLE_METRICS = ("hit@1", "hit@10", "recall@10", "mrr", "ndcg@10", "map")


def _fmt(x: float, digits: int = 3) -> str:
    return f"{x:.{digits}f}"


def _table(header: Sequence[str], body: Sequence[Sequence[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(row) + " |" for row in body]
    return "\n".join(lines)


def _stamp() -> str:
    return _dt.datetime.now().strftime("%Y-%m-%d %H:%M")


def _k_values(rows: Sequence[Mapping[str, Any]], prefix: str) -> list[int]:
    return sorted({int(k.split("@")[1]) for k in rows[0] if k.startswith(prefix + "@")})


def render_matrix_report(
    rows: Sequence[Mapping[str, Any]],
    out_dir: str | Path,
    baseline: str | None = None,
    paired_metric: str = "hit@10",
    config_order: Sequence[str] | None = None,
) -> Path:
    """설정별 지표 표, Wilson 구간, 기준 설정 대비 McNemar 표, recall@k 그림."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    summary = summarize(rows)
    goldens = sorted({s["golden"] for s in summary})
    parts = [
        "# 검색 설정별 결과",
        "",
        f"생성: {_stamp()} · `rag_eval.report` 가 질의별 원자료에서 자동 생성",
        "",
    ]
    for g in goldens:
        sums = {s["config"]: s for s in summary if s["golden"] == g}
        order = [c for c in (config_order or sorted(sums)) if c in sums]
        n = sums[order[0]]["n"]
        parts += [f"## 골든셋 `{g}` (질의 {n}건)", ""]
        body = []
        for c in order:
            s = sums[c]
            succ = sum(
                1 for r in rows if r["golden"] == g and r["config"] == c and r[paired_metric] >= 0.5
            )
            lo, hi = wilson_interval(succ, s["n"])
            body.append(
                [f"`{c}`"]
                + [_fmt(s[m]) for m in TABLE_METRICS]
                + [f"[{_fmt(lo)}, {_fmt(hi)}]", _fmt(s["latency_p95_ms"], 1)]
            )
        parts += [
            _table(
                ["설정", *TABLE_METRICS, f"{paired_metric} 95% CI (Wilson)", "p95 지연(ms)"],
                body,
            ),
            "",
        ]
        if baseline and baseline in sums:
            per = {
                c: {
                    r["qid"]: r[paired_metric]
                    for r in rows
                    if r["golden"] == g and r["config"] == c
                }
                for c in order
            }
            qids = sorted(per[baseline])
            body = []
            for c in order:
                if c == baseline:
                    continue
                res = mcnemar_test([per[c][q] for q in qids], [per[baseline][q] for q in qids])
                body.append([f"`{c}`", str(res.b), str(res.c), _fmt(res.p_value, 4), res.method])
            parts += [
                f"### `{baseline}` 대비 {paired_metric} McNemar 검정",
                "",
                "b = 해당 설정만 성공, c = 기준 설정만 성공",
                "",
                _table(["설정", "b", "c", "p", "방법"], body),
                "",
            ]
        ks = _k_values(rows, "recall")
        fig, ax = plt.subplots(figsize=(7, 4))
        for c in order:
            ax.plot(ks, [sums[c][f"recall@{k}"] for k in ks], marker="o", label=c)
        ax.set_xlabel("k")
        ax.set_ylabel("recall@k")
        ax.set_xticks(ks)
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        ax.set_title(f"recall@k — {g}")
        fig.tight_layout()
        img = f"recall_at_k_{g}.png"
        fig.savefig(out / img, dpi=150)
        plt.close(fig)
        parts += [f"![recall@k {g}]({img})", ""]
    path = out / "retrieval.md"
    path.write_text("\n".join(parts), encoding="utf-8")
    return path


def render_golden_report(
    comparison: Mapping[str, Any], out_dir: str | Path, plot_metric: str = "ndcg@10"
) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    a, b = comparison["a"], comparison["b"]
    parts = [
        f"# 골든셋 비교: `{a}` 대 `{b}`",
        "",
        f"생성: {_stamp()} · 설정 {len(comparison['configs'])}개 · gap = `{b}` − `{a}`",
        "",
        "## 순위 상관",
        "",
    ]
    body = []
    for m, r in comparison["metrics"].items():
        body.append(
            [
                m,
                _fmt(r["mean_gap"]),
                f"{_fmt(r['kendall_tau'])} (p={_fmt(r['kendall_p'], 3)})",
                f"{_fmt(r['spearman_rho'])} (p={_fmt(r['spearman_p'], 3)})",
                f"`{r['best_config'][a]}`",
                f"`{r['best_config'][b]}`",
            ]
        )
    parts += [
        _table(
            [
                "지표",
                "평균 gap",
                "Kendall tau-b",
                "Spearman rho",
                f"최고 설정({a})",
                f"최고 설정({b})",
            ],
            body,
        ),
        "",
    ]
    for m, r in comparison["metrics"].items():
        parts += [f"## {m}", ""]
        body = [
            [
                f"`{p['config']}`",
                _fmt(p[a]),
                _fmt(p[b]),
                _fmt(p["gap"]),
                f"{p[f'rank_{a}']:g}",
                f"{p[f'rank_{b}']:g}",
            ]
            for p in r["per_config"]
        ]
        parts += [_table(["설정", a, b, "gap", f"순위({a})", f"순위({b})"], body), ""]
    if plot_metric in comparison["metrics"]:
        pc = comparison["metrics"][plot_metric]["per_config"]
        fig, ax = plt.subplots(figsize=(5, 5))
        xs, ys = [p[a] for p in pc], [p[b] for p in pc]
        ax.scatter(xs, ys)
        for p in pc:
            ax.annotate(
                p["config"], (p[a], p[b]), fontsize=7, xytext=(3, 3), textcoords="offset points"
            )
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel(f"{plot_metric} ({a})")
        ax.set_ylabel(f"{plot_metric} ({b})")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        img = f"golden_scatter_{plot_metric.replace('@', '_at_')}.png"
        fig.savefig(out / img, dpi=150)
        plt.close(fig)
        parts += [f"![{plot_metric} 산점도]({img})", ""]
    path = out / "golden_compare.md"
    path.write_text("\n".join(parts), encoding="utf-8")
    return path


def render_judge_report(aggregate: Mapping[str, Any], out_dir: str | Path) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    parts = [
        "# 답변 LLM 판정 (표본)",
        "",
        f"생성: {_stamp()} · 표본 {aggregate['n']}건, 파싱 성공 {aggregate['n_valid']}건",
        "",
    ]
    body = [
        [
            c,
            _fmt(v["mean"], 2),
            _fmt(v["pass_rate"]),
            f"[{_fmt(v['pass_ci'][0])}, {_fmt(v['pass_ci'][1])}]",
        ]
        for c, v in aggregate["criteria"].items()
    ]
    parts += [_table(["항목", "평균(1~5)", "통과율(≥4)", "95% CI (Wilson)"], body), ""]
    path = out / "judge.md"
    path.write_text("\n".join(parts), encoding="utf-8")
    return path


def render_stability_report(result: Mapping[str, Any], out_dir: str | Path) -> Path:
    """질의 수별 순위 일치율 표와 곡선."""
    from rag_eval.stability import min_n_for

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    sampling = "복원추출(부트스트랩)" if result["replace"] else "비복원추출(부분집합)"
    need = min_n_for(result, 0.9)
    parts = [
        f"# 질의 수에 따른 설정 순위 안정성 — `{result['golden']}` · {result['metric']}",
        "",
        f"생성: {_stamp()} · 설정 {len(result['configs'])}개 · 전체 질의 {result['n_queries']}건 · "
        f"{sampling} {result['n_boot']}회 · seed {result['seed']}",
        "",
        "전체 질의 순위: " + " > ".join(f"`{c}`" for c in result["full_order"]),
        "",
        "exact = 순위 전체 일치 비율, top1 = 1위 일치 비율 (95% Wilson 구간), "
        "pairwise = 설정 쌍 순서 일치 비율의 평균 (95% 백분위 구간)",
        "",
    ]
    body = [
        [
            str(r["n"]),
            _fmt(r["exact"]),
            f"[{_fmt(r['exact_ci'][0])}, {_fmt(r['exact_ci'][1])}]",
            _fmt(r["top1"]),
            f"[{_fmt(r['top1_ci'][0])}, {_fmt(r['top1_ci'][1])}]",
            _fmt(r["pairwise"]),
            f"[{_fmt(r['pairwise_interval'][0])}, {_fmt(r['pairwise_interval'][1])}]",
        ]
        for r in result["results"]
    ]
    parts += [
        _table(["n", "exact", "exact CI", "top1", "top1 CI", "pairwise", "pairwise 구간"], body),
        "",
        "exact 일치율 구간 하한이 0.9 이상이 되는 최소 n: "
        + (f"{need}" if need is not None else "격자 안에 없음"),
        "",
    ]
    ns = [r["n"] for r in result["results"]]
    fig, ax = plt.subplots(figsize=(7, 4))
    for key, label in (("exact", "exact"), ("top1", "top1")):
        ys = [r[key] for r in result["results"]]
        lo = [r[f"{key}_ci"][0] for r in result["results"]]
        hi = [r[f"{key}_ci"][1] for r in result["results"]]
        ax.plot(ns, ys, marker="o", label=label)
        ax.fill_between(ns, lo, hi, alpha=0.2)
    ax.plot(ns, [r["pairwise"] for r in result["results"]], marker="s", label="pairwise")
    ax.set_xlabel("n (queries)")
    ax.set_ylabel("agreement with full-set ordering")
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    slug = f"{result['golden']}_{result['metric'].replace('@', '_at_')}"
    img = f"stability_{slug}.png"
    fig.savefig(out / img, dpi=150)
    plt.close(fig)
    parts += [f"![순위 안정성]({img})", ""]
    path = out / f"stability_{slug}.md"
    path.write_text("\n".join(parts), encoding="utf-8")
    return path
