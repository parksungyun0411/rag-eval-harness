"""한국어 RAG 리더보드 순위 안정성 실험 (allganize/RAG-Evaluation-Dataset-KO).

    rag-eval rankstab --data data/raw/rag_evaluation_result.csv --out reports

산출물 (모두 같은 실행에서 생성, 숫자를 손으로 적지 않는다)
- reports/REPORT.md
- reports/rank_stability/*.csv  (표의 원자료)
- reports/rank_stability/*.png  (그림)
- web/data/results.json          (데모 화면용)
"""

from __future__ import annotations

import csv
import datetime as _dt
import json
import math
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy import stats as _st  # noqa: E402

from rag_eval.datasets import allganize_ko  # noqa: E402
from rag_eval.datasets.allganize_ko import CorrectnessMatrix  # noqa: E402
from rag_eval.rankstab import (  # noqa: E402
    approximation_check,
    domain_winners,
    pair_analysis,
    rank_stability,
)
from rag_eval.stats import wilson_interval  # noqa: E402

OVERALL_NS = (10, 20, 30, 40, 50, 60, 80, 100, 150, 200, 250, 300)
DOMAIN_NS = (10, 20, 30, 40, 50, 60)
SUB = "rank_stability"


def _write_csv(path: Path, header: Sequence[str], rows: Sequence[Sequence[Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            w.writerow([f"{v:.6f}" if isinstance(v, float) else v for v in r])


def _f(x: float | None, d: int = 3) -> str:
    if x is None or (isinstance(x, float) and math.isinf(x)):
        return "—"
    return f"{x:.{d}f}"


def _ci(lo: float, hi: float, d: int = 3) -> str:
    return f"[{lo:.{d}f}, {hi:.{d}f}]"


def _table(header: Sequence[str], body: Sequence[Sequence[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(r) + " |" for r in body]
    return "\n".join(lines)


CURVE_KEYS = ("top1", "order", "top3", "top5")


def _curve_rows(curve: list[dict[str, Any]], prefix: Sequence[Any] = ()) -> list[list[Any]]:
    rows = []
    for c in curve:
        row: list[Any] = [*prefix, c["n"], c["tau_mean"], c["tau_lo"], c["tau_hi"]]
        for k in CURVE_KEYS:
            row += [c.get(f"{k}_held", ""), c.get(f"{k}_lo", ""), c.get(f"{k}_hi", "")]
        rows.append(row)
    return rows


CURVE_HEADER = ["n", "tau_mean", "tau_lo", "tau_hi"] + [
    f"{k}_{s}" for k in CURVE_KEYS for s in ("held", "lo", "hi")
]


def run(
    data_path: str | Path,
    out_dir: str | Path = "reports",
    web_dir: str | Path | None = "web/data",
    n_boot: int = 2000,
    seed: int = 0,
    command: str = "",
    log: Callable[[str], None] = print,
) -> dict[str, Any]:
    out = Path(out_dir)
    sub = out / SUB
    sub.mkdir(parents=True, exist_ok=True)
    cm = allganize_ko.load(data_path)
    sha = allganize_ko.sha256_of(data_path)
    cm.to_csv(sub / "correctness_matrix.csv")
    S, Q = cm.X.shape
    acc = cm.accuracy()
    order = np.argsort(-acc, kind="stable")
    log(f"[rankstab] 시스템 {S}개 × 질문 {Q}건")

    # 리더보드
    dom_acc = {d: cm.subset(d).accuracy() for d in allganize_ko.DOMAINS}
    lb_rows = []
    for rank, i in enumerate(order, start=1):
        c = int(cm.X[i].sum())
        lo, hi = wilson_interval(c, Q)
        lb_rows.append(
            [
                rank,
                cm.systems[i],
                c,
                Q,
                acc[i],
                lo,
                hi,
                *(dom_acc[d][i] for d in allganize_ko.DOMAINS),
            ]
        )
    _write_csv(
        sub / "leaderboard.csv",
        [
            "rank",
            "system",
            "correct",
            "n",
            "accuracy",
            "wilson_lo",
            "wilson_hi",
            *allganize_ko.DOMAINS,
        ],
        lb_rows,
    )

    # 전체 순위 안정성
    overall = rank_stability(cm.X, cm.systems, OVERALL_NS, n_boot=n_boot, seed=seed)
    _write_csv(sub / "curve_overall.csv", CURVE_HEADER, _curve_rows(overall["curve"]))
    pairs = pair_analysis(cm.X, cm.systems, overall)
    _write_csv(
        sub / "adjacent_pairs.csv",
        [
            "upper",
            "lower",
            "gap",
            "discordance",
            "p_pos",
            "p_neg",
            "n_sim_grid",
            "n_exact",
            "n_normal",
            "n_power80_normal",
            "n_upper_only",
            "n_lower_only",
            "mcnemar_p",
        ],
        [
            [
                p["upper"],
                p["lower"],
                p["gap"],
                p["discordance"],
                p["p_pos"],
                p["p_neg"],
                p["n_sim_grid"] if p["n_sim_grid"] is not None else "",
                p["n_exact"] if p["n_exact"] is not None else "",
                p["n_normal"],
                p["n_power80_normal"],
                p["n_pos"],
                p["n_neg"],
                p["mcnemar_p"],
            ]
            for p in pairs
        ],
    )
    approx = approximation_check(cm.X, overall)
    _write_csv(
        sub / "approx_check.csv",
        ["upper", "lower", "n", "sim", "exact", "normal"],
        [[r["upper"], r["lower"], r["n"], r["sim"], r["exact"], r["normal"]] for r in approx],
    )

    # 도메인별
    domain_curves: dict[str, Any] = {}
    domain_rows: list[list[Any]] = []
    share_rows: list[list[Any]] = []
    corr_rows: list[list[Any]] = []
    for k, d in enumerate(allganize_ko.DOMAINS):
        dm = cm.subset(d)
        res = rank_stability(dm.X, dm.systems, DOMAIN_NS, n_boot=n_boot, seed=seed + 1 + k)
        domain_curves[d] = res
        domain_rows += _curve_rows(res["curve"], prefix=(d,))
        full = dm.accuracy()
        share = res["curve"][-1]["best_share"]  # n = 60 (도메인 전체 크기)
        for i, s in enumerate(cm.systems):
            share_rows.append([d, s, full[i], share[i]])
        tau, p = _st.kendalltau(acc, full)
        corr_rows.append([d, tau, p, " / ".join(domain_winners(full, cm.systems)), full.max()])
    _write_csv(sub / "curve_domain.csv", ["domain", *CURVE_HEADER], domain_rows)
    _write_csv(
        sub / "domain_best_share.csv",
        ["domain", "system", "accuracy", "best_share_n60"],
        share_rows,
    )
    _write_csv(
        sub / "domain_rank_corr.csv",
        ["domain", "kendall_tau_vs_overall", "p_value", "winners", "winner_accuracy"],
        corr_rows,
    )

    _plots(cm, overall, pairs, approx, domain_curves, sub)
    meta = {
        "dataset": allganize_ko.REPO,
        "revision": allganize_ko.REVISION,
        "file": allganize_ko.FILENAME,
        "sha256": sha,
        "n_boot": n_boot,
        "seed": seed,
        "sampling": "복원추출(부트스트랩)",
        "overall_ns": list(OVERALL_NS),
        "domain_ns": list(DOMAIN_NS),
        "command": command,
        "generated": _dt.datetime.now().strftime("%Y-%m-%d %H:%M"),
    }
    report = _render_report(cm, lb_rows, overall, pairs, approx, domain_curves, corr_rows, meta)
    (out / "REPORT.md").write_text(report, encoding="utf-8")
    log(f"[rankstab] {out / 'REPORT.md'}")
    if web_dir is not None:
        _write_web(Path(web_dir), cm, lb_rows, overall, pairs, domain_curves, corr_rows, meta)
    return {"overall": overall, "pairs": pairs, "approx": approx, "meta": meta}


# ---------------------------------------------------------------- 그림


def _plots(cm, overall, pairs, approx, domain_curves, sub: Path) -> None:
    ns = [c["n"] for c in overall["curve"]]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(ns, [c["tau_mean"] for c in overall["curve"]], marker="o", label="overall")
    ax.fill_between(
        ns,
        [c["tau_lo"] for c in overall["curve"]],
        [c["tau_hi"] for c in overall["curve"]],
        alpha=0.2,
    )
    for d, res in domain_curves.items():
        ax.plot(
            [c["n"] for c in res["curve"]],
            [c["tau_mean"] for c in res["curve"]],
            linestyle="--",
            label=d,
        )
    ax.set_xlabel("n (questions, bootstrap)")
    ax.set_ylabel("Kendall tau-b vs full-set ranking")
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(sub / "tau_vs_n.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    for key, label in (
        ("top1", "rank-1 held"),
        ("top3", "top-3 set held"),
        ("top5", "top-5 set held"),
        ("order", "full order held"),
    ):
        if f"{key}_held" not in overall["curve"][0]:
            continue
        ys = [c[f"{key}_held"] for c in overall["curve"]]
        ax.plot(ns, ys, marker="o", label=label)
        ax.fill_between(
            ns,
            [c[f"{key}_lo"] for c in overall["curve"]],
            [c[f"{key}_hi"] for c in overall["curve"]],
            alpha=0.15,
        )
    ax.set_xlabel("n (questions, bootstrap)")
    ax.set_ylabel("probability (strict, ties = not held)")
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(sub / "held_vs_n.png", dpi=150)
    plt.close(fig)

    labels = [f"{i + 1}-{i + 2}" for i in range(len(pairs))]
    exact = [p["n_exact"] if p["n_exact"] is not None else np.nan for p in pairs]
    normal = [p["n_normal"] for p in pairs]
    power80 = [p["n_power80_normal"] for p in pairs]
    x = np.arange(len(pairs))
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(x - 0.27, exact, width=0.27, label="~50% power, exact")
    ax.bar(x, normal, width=0.27, label="~50% power, normal approx")
    ax.bar(x + 0.27, power80, width=0.27, label="80% power, normal approx")
    ax.axhline(
        cm.X.shape[1],
        color="gray",
        linestyle="--",
        linewidth=1,
        label=f"available questions ({cm.X.shape[1]})",
    )
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_xlabel("adjacent pair (overall rank i vs i+1)")
    ax.set_ylabel("questions to separate at 95%\n(observed gap taken as true)")
    ax.grid(alpha=0.3, axis="y")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(sub / "pairs_n_required.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter([r["exact"] for r in approx], [r["sim"] for r in approx], s=8, label="simulation")
    ax.scatter(
        [r["exact"] for r in approx],
        [r["normal"] for r in approx],
        s=8,
        marker="x",
        label="normal approx",
    )
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=1)
    ax.set_xlabel("exact P(upper strictly ahead)")
    ax.set_ylabel("estimate")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(sub / "approx_check.png", dpi=150)
    plt.close(fig)

    order = np.argsort(-cm.accuracy(), kind="stable")
    doms = list(domain_curves)
    mat = np.array([[domain_curves[d]["curve"][-1]["best_share"][i] for d in doms] for i in order])
    fig, ax = plt.subplots(figsize=(7.5, 8))
    im = ax.imshow(mat, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(doms)))
    ax.set_xticklabels(doms)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([cm.systems[i] for i in order], fontsize=7)
    for r in range(mat.shape[0]):
        for c in range(mat.shape[1]):
            if mat[r, c] >= 0.05:
                ax.text(c, r, f"{mat[r, c]:.2f}", ha="center", va="center", fontsize=6)
    fig.colorbar(im, ax=ax, label="P(rank 1) under bootstrap, n=60")
    fig.tight_layout()
    fig.savefig(sub / "domain_best_share.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- 보고서


def _curve_at(curve: list[dict[str, Any]], n: int) -> dict[str, Any]:
    return next(c for c in curve if c["n"] == n)


def _render_report(cm, lb_rows, overall, pairs, approx, domain_curves, corr_rows, meta) -> str:
    S, Q = cm.X.shape
    B = meta["n_boot"]
    p = [
        "# 한국어 RAG 리더보드 순위 안정성 — 실험 보고서",
        "",
        f"생성: {meta['generated']} · `rag_eval.leaderboard_experiment` 가 자동 생성 (숫자를 손으로 적지 않음)",
        "",
        "## 재현 정보",
        "",
        _table(
            ["항목", "값"],
            [
                ["데이터셋", f"`{meta['dataset']}` (Hugging Face, MIT)"],
                ["리비전", f"`{meta['revision']}`"],
                ["파일 · SHA-256", f"`{meta['file']}` · `{meta['sha256']}`"],
                ["규모", f"시스템 {S}개 × 질문 {Q}건 (도메인 5개 × 60건)"],
                ["정답 라벨", "데이터셋의 `*_ox` 열 (작성자가 LLM 평가기 4종 투표로 판정)"],
                ["추출", f"{meta['sampling']} {B}회, seed {meta['seed']} (도메인별은 seed+1…+5)"],
                ["n 격자", f"전체 {meta['overall_ns']}, 도메인 {meta['domain_ns']}"],
                ["명령", f"`{meta['command']}`" if meta["command"] else "—"],
            ],
        ),
        "",
        "## 1. 리더보드 (전체 300문항)",
        "",
        "정확도 옆 구간은 Wilson 95% 신뢰구간입니다. 원자료: `rank_stability/leaderboard.csv`",
        "",
    ]
    p.append(
        _table(
            ["순위", "시스템", "정답", "정확도", "95% CI", *allganize_ko.DOMAINS],
            [
                [
                    str(r[0]),
                    r[1],
                    f"{r[2]}/{r[3]}",
                    _f(r[4]),
                    _ci(r[5], r[6]),
                    *(_f(v, 2) for v in r[7:]),
                ]
                for r in lb_rows
            ],
        )
    )
    p += [
        "",
        "## 2. 질문 수에 따른 전체 순위 안정성",
        "",
        f"질문 n건을 {B}회 복원추출해 매긴 순위를 300문항 순위와 비교했습니다. "
        "tau 구간은 반복값의 2.5~97.5 백분위입니다. "
        "확률 옆 구간은 반복 횟수에서 오는 몬테카를로 오차(Wilson 95%)일 뿐이며, "
        "질문 표본 자체의 불확실성을 나타내지 않습니다. "
        "'유지'는 엄격한 우위만 셉니다(동점은 유지 아님). 원자료: `rank_stability/curve_overall.csv`",
        "",
    ]
    body = []
    for c in overall["curve"]:
        body.append(
            [
                str(c["n"]),
                f"{_f(c['tau_mean'])} {_ci(c['tau_lo'], c['tau_hi'])}",
                f"{_f(c['top1_held'])} {_ci(c['top1_lo'], c['top1_hi'])}",
                _f(1 - c["top1_held"]),
                f"{_f(c['top3_held'])}" if "top3_held" in c else "—",
                f"{_f(c['top5_held'])}" if "top5_held" in c else "—",
                f"{_f(c['order_held'], 4)}",
            ]
        )
    p += [
        _table(
            [
                "n",
                "Kendall tau-b [95%]",
                "1위 유지 [반복 오차 95%]",
                "1위 바뀜",
                "top-3 집합 유지",
                "top-5 집합 유지",
                "순위 전체 유지",
            ],
            body,
        ),
        "",
        "![tau](rank_stability/tau_vs_n.png)",
        "",
        "![held](rank_stability/held_vs_n.png)",
        "",
        "## 3. 인접 순위 쌍을 가르는 데 필요한 질문 수",
        "",
        "전체 순위에서 바로 이웃한 두 시스템마다, 관측된 차이와 불일치율이 참값이라고 가정했을 때 "
        "P(위 시스템이 엄격히 앞섬) ≥ 0.975가 되는(양측 95%로 갈리는) 질문 수입니다. "
        "이 기준은 δ√n/σ ≈ 1.96 인 지점이라 검정력 약 50%에 해당합니다(그 n으로 새로 평가하면 절반쯤만 유의하게 갈림). "
        "검정력 80% 기준은 ((1.96 + 0.84)·σ/δ)² 정규 근사로 따로 적었습니다. "
        "'격자'는 시뮬레이션 격자에서 그 n부터 끝까지 기준을 넘는 첫 값, '정확해'는 삼항 분포로 계산한 값입니다(탐색 상한 20,000). "
        "300을 넘는 값은 같은 질문 분포를 가정한 외삽입니다. "
        "'위만/아래만'은 300문항 중 한쪽만 맞힌 질문 수이고, McNemar p는 그 불일치 쌍에 대한 양측 정확 검정입니다. "
        "원자료: `rank_stability/adjacent_pairs.csv`",
        "",
    ]
    body = []
    for k, pr in enumerate(pairs, start=1):
        body.append(
            [
                f"{k}-{k + 1}",
                pr["upper"],
                pr["lower"],
                _f(pr["gap"]),
                _f(pr["discordance"]),
                f"{pr['n_pos']}/{pr['n_neg']}",
                "<0.0001" if pr["mcnemar_p"] < 1e-4 else _f(pr["mcnemar_p"], 4),
                str(pr["n_sim_grid"]) if pr["n_sim_grid"] is not None else f">{Q}",
                str(pr["n_exact"]) if pr["n_exact"] is not None else ">20000",
                f"{pr['n_normal']:.0f}" if math.isfinite(pr["n_normal"]) else "—",
                f"{pr['n_power80_normal']:.0f}" if math.isfinite(pr["n_power80_normal"]) else "—",
            ]
        )
    p += [
        _table(
            [
                "순위",
                "위",
                "아래",
                "정확도 차",
                "불일치율",
                "위만/아래만",
                "McNemar p",
                "검정력 50%: 격자(≤300)",
                "검정력 50%: 정확해",
                "검정력 50%: 정규 근사",
                "검정력 80%: 정규 근사",
            ],
            body,
        ),
        "",
        "![pairs](rank_stability/pairs_n_required.png)",
        "",
    ]
    sep = sum(1 for pr in pairs if pr["n_exact"] is not None and pr["n_exact"] <= Q)
    p += [f"300문항 안에서 정확해 기준으로 갈리는 인접 쌍: {sep}/{len(pairs)}", ""]

    # 근사 비교
    diffs_sim = [abs(r["sim"] - r["exact"]) for r in approx]
    diffs_norm = [abs(r["normal"] - r["exact"]) for r in approx]
    worst = sorted(approx, key=lambda r: -abs(r["normal"] - r["exact"]))[:8]
    p += [
        "## 4. 시뮬레이션 · 정확해 · 정규 근사 비교",
        "",
        f"인접 쌍 {len(pairs)}개 × n {len(overall['ns'])}개 = {len(approx)}개 지점의 P(위 시스템이 엄격히 앞섬)입니다. "
        "복원추출 시뮬레이션은 이론상 정확해와 같은 분포를 따르므로 차이는 몬테카를로 오차입니다. "
        "원자료: `rank_stability/approx_check.csv`",
        "",
        _table(
            ["비교", "최대 절대 차", "평균 절대 차", "차 > 0.05 지점 수"],
            [
                [
                    "시뮬레이션 − 정확해",
                    _f(max(diffs_sim)),
                    _f(float(np.mean(diffs_sim))),
                    str(sum(d > 0.05 for d in diffs_sim)),
                ],
                [
                    "정규 근사 − 정확해",
                    _f(max(diffs_norm)),
                    _f(float(np.mean(diffs_norm))),
                    str(sum(d > 0.05 for d in diffs_norm)),
                ],
            ],
        ),
        "",
        "정규 근사가 정확해와 가장 많이 어긋난 지점:",
        "",
        _table(
            ["위", "아래", "n", "정확해", "시뮬레이션", "정규 근사"],
            [
                [r["upper"], r["lower"], str(r["n"]), _f(r["exact"]), _f(r["sim"]), _f(r["normal"])]
                for r in worst
            ],
        ),
        "",
        "![approx](rank_stability/approx_check.png)",
        "",
        "## 5. 도메인별 (각 60문항)",
        "",
        "도메인 안에서의 1위와, 도메인 순위와 전체 순위의 Kendall tau입니다. "
        "전체 300문항에 그 도메인 60문항이 포함되어 있어, 이 tau는 부분과 전체의 상관입니다. "
        "'1위 확률'은 도메인 60문항을 복원추출했을 때 각 시스템이 1위가 되는 비율(공동 1위는 나눔)입니다. "
        "원자료: `rank_stability/domain_rank_corr.csv`, `rank_stability/domain_best_share.csv`, `rank_stability/curve_domain.csv`",
        "",
    ]
    body = []
    for row in corr_rows:
        d = row[0]
        res = domain_curves[d]
        c60 = _curve_at(res["curve"], 60)
        share = c60["best_share"]
        top = sorted(range(len(share)), key=lambda i: -share[i])[:3]
        body.append(
            [
                d,
                row[3],
                _f(row[4]),
                _f(row[1]),
                " · ".join(f"{cm.systems[i]} {share[i]:.2f}" for i in top),
                _f(c60["tau_mean"]),
            ]
        )
    p += [
        _table(
            [
                "도메인",
                "1위(동률 포함)",
                "1위 정확도",
                "전체 순위와 tau",
                "1위 확률 상위 3 (n=60)",
                "tau (n=60 재추출)",
            ],
            body,
        ),
        "",
        "![domain](rank_stability/domain_best_share.png)",
        "",
    ]
    return "\n".join(p) + "\n"


def _write_web(web_dir: Path, cm, lb_rows, overall, pairs, domain_curves, corr_rows, meta) -> None:
    web_dir.mkdir(parents=True, exist_ok=True)
    data = {
        "meta": meta,
        "domains": list(allganize_ko.DOMAINS),
        "leaderboard": [
            {
                "rank": r[0],
                "system": r[1],
                "correct": r[2],
                "n": r[3],
                "accuracy": r[4],
                "ci": [r[5], r[6]],
                "domains": dict(zip(allganize_ko.DOMAINS, r[7:], strict=True)),
            }
            for r in lb_rows
        ],
        "overall_curve": [
            {k: v for k, v in c.items() if k != "best_share"} for c in overall["curve"]
        ],
        "domain_curves": {
            d: [{k: v for k, v in c.items() if k != "best_share"} for c in res["curve"]]
            for d, res in domain_curves.items()
        },
        "domain_winners": {row[0]: row[3] for row in corr_rows},
        # 화면의 계산기가 임의의 두 시스템 불일치율을 구할 수 있도록 정답 여부를 문자열로 싣는다
        "question_domains": cm.domains,
        "correct_bits": {
            s: "".join(str(int(v)) for v in cm.X[i]) for i, s in enumerate(cm.systems)
        },
        "pairs": [
            {
                **p,
                "n_normal": p["n_normal"] if math.isfinite(p["n_normal"]) else None,
                "n_power80_normal": (
                    p["n_power80_normal"] if math.isfinite(p["n_power80_normal"]) else None
                ),
            }
            for p in pairs
        ],
    }
    (web_dir / "results.json").write_text(
        json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8"
    )


def load_matrix_csv(path: str | Path) -> CorrectnessMatrix:
    return CorrectnessMatrix.from_csv(path)
