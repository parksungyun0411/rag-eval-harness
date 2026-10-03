"""``rag-eval`` 명령행.

rag-eval synth   --corpus corpus.jsonl --out data/golden/synthetic --n 100
rag-eval compare --per-query per_query.jsonl --a real --b synthetic --out results/golden.json
rag-eval judge   --answers answers.jsonl --out results/judgments.jsonl --n 50
rag-eval report  --per-query ... [--golden results/golden.json] [--judgments ...] --out reports/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rag_eval.types import read_jsonl, write_jsonl


def _cmd_synth(args: argparse.Namespace) -> None:
    from rag_eval.llm import get_provider
    from rag_eval.synth import DEFAULT_TEMPLATE, generate_synthetic_golden

    corpus = list(read_jsonl(args.corpus))
    template = (
        Path(args.template).read_text(encoding="utf-8") if args.template else DEFAULT_TEMPLATE
    )
    gs = generate_synthetic_golden(
        corpus, get_provider(), n=args.n, seed=args.seed, template=template, name=args.name
    )
    gs.save(args.out)
    print(f"[synth] {len(gs.queries)}건 → {args.out}")


def _cmd_compare(args: argparse.Namespace) -> None:
    from rag_eval.golden import compare_golden_sets

    rows = list(read_jsonl(args.per_query))
    result = compare_golden_sets(rows, args.a, args.b)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[compare] {args.a} 대 {args.b} → {args.out}")


def _cmd_judge(args: argparse.Namespace) -> None:
    from rag_eval.judge import judge_answers
    from rag_eval.llm import get_provider

    answers = list(read_jsonl(args.answers))
    judgments = judge_answers(answers, get_provider(), n=args.n, seed=args.seed)
    write_jsonl(args.out, judgments)
    print(f"[judge] {len(judgments)}건 → {args.out}")


def _cmd_report(args: argparse.Namespace) -> None:
    from rag_eval.judge import aggregate_judgments
    from rag_eval.report import render_golden_report, render_judge_report, render_matrix_report

    written = []
    if args.per_query:
        rows = list(read_jsonl(args.per_query))
        order = args.order.split(",") if args.order else None
        written.append(
            render_matrix_report(rows, args.out, baseline=args.baseline, config_order=order)
        )
    if args.golden:
        comparison = json.loads(Path(args.golden).read_text(encoding="utf-8"))
        written.append(render_golden_report(comparison, args.out))
    if args.judgments:
        written.append(
            render_judge_report(aggregate_judgments(list(read_jsonl(args.judgments))), args.out)
        )
    if not written:
        raise SystemExit("--per-query, --golden, --judgments 중 하나 이상이 필요합니다")
    for p in written:
        print(f"[report] {p}")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="rag-eval")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("synth", help="LLM 합성 질의 골든셋 생성")
    s.add_argument("--corpus", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--n", type=int, default=100)
    s.add_argument("--seed", type=int, default=42)
    s.add_argument("--name", default="synthetic")
    s.add_argument("--template", help="프롬프트 템플릿 파일({text} 자리표시자)")
    s.set_defaults(func=_cmd_synth)

    c = sub.add_parser("compare", help="골든셋 2종 점수 차·순위 상관")
    c.add_argument("--per-query", required=True)
    c.add_argument("--a", default="real")
    c.add_argument("--b", default="synthetic")
    c.add_argument("--out", required=True)
    c.set_defaults(func=_cmd_compare)

    j = sub.add_parser("judge", help="답변 표본 LLM 판정")
    j.add_argument("--answers", required=True)
    j.add_argument("--out", required=True)
    j.add_argument("--n", type=int, default=50)
    j.add_argument("--seed", type=int, default=42)
    j.set_defaults(func=_cmd_judge)

    r = sub.add_parser("report", help="마크다운 표·그림 생성")
    r.add_argument("--per-query")
    r.add_argument("--golden")
    r.add_argument("--judgments")
    r.add_argument("--baseline")
    r.add_argument("--order", help="설정 표시 순서, 쉼표 구분")
    r.add_argument("--out", default="reports")
    r.set_defaults(func=_cmd_report)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
