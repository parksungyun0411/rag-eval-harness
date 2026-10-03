"""매트릭스 실행 → 골든셋 비교 → 보고서 → 합성·판정까지, 가짜 검색기·가짜 LLM으로 연결 검증."""

import json
from pathlib import Path

import pytest

from rag_eval.cli import main as cli_main
from rag_eval.golden import _ranks, compare_golden_sets
from rag_eval.judge import aggregate_judgments, judge_answers, parse_judgment
from rag_eval.matrix import run_matrix, summarize
from rag_eval.report import render_golden_report, render_judge_report, render_matrix_report
from rag_eval.synth import generate_synthetic_golden
from rag_eval.types import GoldenSet, read_jsonl

FIXTURES = Path(__file__).parent / "fixtures"


class FixedRetriever:
    """질의 문자열마다 정해진 순위를 돌려주는 가짜 검색기."""

    def __init__(self, table: dict[str, list[str]]):
        self.table = table

    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        docs = self.table.get(query, [])[:k]
        return [(d, 1.0 / (i + 1)) for i, d in enumerate(docs)]


class FakeLLM:
    name = "fake"

    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.calls = 0

    def complete(self, system: str, prompt: str, max_tokens: int = 1024) -> str:
        self.calls += 1
        return self.outputs[(self.calls - 1) % len(self.outputs)]


@pytest.fixture
def golden() -> GoldenSet:
    return GoldenSet.load(FIXTURES / "synthetic_golden", name="real")


def test_golden_load_and_judged(golden, tmp_path):
    assert golden.judged_qids() == ["q1", "q2", "q3"]
    golden.save(tmp_path / "g")
    again = GoldenSet.load(tmp_path / "g", name="real")
    assert again.qrels == golden.qrels and again.queries == golden.queries


def test_run_matrix_writes_rows(golden, tmp_path):
    good = FixedRetriever({"alpha beta": ["d1", "d2"], "gamma": ["d3"], "delta epsilon": ["d4"]})
    bad = FixedRetriever({"alpha beta": ["x", "d2"], "gamma": ["x"], "delta epsilon": ["x"]})
    built = []

    def factory():
        built.append(1)
        return bad

    rows = run_matrix(
        {"good": good, "bad": factory},
        [golden],
        ks=(1, 2),
        depth=5,
        out_dir=tmp_path,
        log=lambda _: None,
    )
    assert len(rows) == 6 and built == [1]
    s = {x["config"]: x for x in summarize(rows)}
    assert s["good"]["hit@1"] == 1.0
    # bad: q1만 2위에서 d2(grade1) 적중 → hit@2 = 1/3, mrr = (1/2)/3
    assert s["bad"]["hit@2"] == pytest.approx(1 / 3)
    assert s["bad"]["mrr"] == pytest.approx(1 / 6)
    assert len(list(read_jsonl(tmp_path / "per_query.jsonl"))) == 6
    run = list(read_jsonl(tmp_path / "runs" / "real" / "good.jsonl"))
    assert run[0]["docids"] == ["d1", "d2"]
    with pytest.raises(ValueError):
        run_matrix({"good": good}, [golden], ks=(10,), depth=5)


def _rows(golden_name, values):
    """설정별 hit@10 값을 질의 하나짜리 행으로 만든다."""
    return [
        {"golden": golden_name, "config": c, "qid": "q", "latency_ms": 1.0, "hit@10": v, "mrr": v}
        for c, v in values.items()
    ]


def test_ranks_ties():
    assert _ranks([0.9, 0.5, 0.9, 0.1]) == [1.5, 3.0, 1.5, 4.0]


def test_compare_golden_sets():
    rows = _rows("real", {"a": 0.2, "b": 0.4, "c": 0.6}) + _rows(
        "synthetic", {"a": 0.5, "b": 0.9, "c": 0.7}
    )
    res = compare_golden_sets(rows, "real", "synthetic", metrics=("hit@10",))
    m = res["metrics"]["hit@10"]
    # 순위: real c>b>a, synthetic b>c>a → 3쌍 중 (b,c)만 불일치 → tau = 1/3
    assert m["kendall_tau"] == pytest.approx(1 / 3)
    # 순위 차 d = [0, 1, -1] → rho = 1 - 6*2/(3*8) = 0.5
    assert m["spearman_rho"] == pytest.approx(0.5)
    assert m["mean_gap"] == pytest.approx((0.3 + 0.5 + 0.1) / 3)
    assert m["best_config"] == {"real": "c", "synthetic": "b"}
    with pytest.raises(ValueError):
        compare_golden_sets(
            _rows("real", {"a": 1}) + _rows("synthetic", {"a": 1}), "real", "synthetic"
        )


def test_reports(golden, tmp_path):
    good = FixedRetriever({"alpha beta": ["d1", "d2"], "gamma": ["d3"], "delta epsilon": ["d4"]})
    bad = FixedRetriever({})
    rows = run_matrix(
        {"good": good, "bad": bad}, [golden], ks=(1, 10), depth=10, log=lambda _: None
    )
    p = render_matrix_report(rows, tmp_path, baseline="bad")
    text = p.read_text(encoding="utf-8")
    assert "McNemar" in text and "`good`" in text
    assert (tmp_path / "recall_at_k_real.png").exists()

    rows2 = _rows("real", {"a": 0.2, "b": 0.4, "c": 0.6}) + _rows(
        "synthetic", {"a": 0.5, "b": 0.9, "c": 0.7}
    )
    comp = compare_golden_sets(rows2, "real", "synthetic", metrics=("hit@10",))
    g = render_golden_report(comp, tmp_path, plot_metric="hit@10")
    assert "Kendall" in g.read_text(encoding="utf-8")
    assert (tmp_path / "golden_scatter_hit_at_10.png").exists()

    agg = aggregate_judgments([{"groundedness": 5, "correctness": 3, "helpfulness": 4}])
    j = render_judge_report(agg, tmp_path)
    assert "groundedness" in j.read_text(encoding="utf-8")


def test_synth():
    corpus = [{"docid": f"d{i}", "text": f"문서 {i}"} for i in range(5)]
    llm = FakeLLM(['질문: "주민등록 이전은 어떻게 하나요?"', ""])
    gs = generate_synthetic_golden(corpus, llm, n=4, seed=0, log=lambda _: None)
    # 빈 출력 2건은 건너뜀
    assert len(gs.queries) == 2
    qid = next(iter(gs.queries))
    assert gs.queries[qid] == "주민등록 이전은 어떻게 하나요?"
    assert gs.qrels[qid] == {gs.meta[qid]["source_docid"]: 2}
    with pytest.raises(ValueError):
        generate_synthetic_golden(corpus, llm, n=10)


def test_parse_judgment():
    ok = parse_judgment(
        '설명 {"groundedness": 4, "correctness": 2, "helpfulness": 5, "rationale": "r"}'
    )
    assert ok == {"groundedness": 4, "correctness": 2, "helpfulness": 5, "rationale": "r"}
    assert parse_judgment("no json") is None
    assert parse_judgment('{"groundedness": 9, "correctness": 2, "helpfulness": 5}') is None


def test_judge_and_aggregate():
    answers = [
        {"qid": f"q{i}", "question": "q", "answer": "a", "reference": "r", "contexts": ["c"]}
        for i in range(4)
    ]
    llm = FakeLLM(
        [
            '{"groundedness": 5, "correctness": 4, "helpfulness": 3}',
            '{"groundedness": 2, "correctness": 4, "helpfulness": 3}',
            "깨진 출력",
        ]
    )
    js = judge_answers(answers, llm, n=None, log=lambda _: None)
    assert len(js) == 4 and "parse_error" in js[2]
    agg = aggregate_judgments(js)
    assert agg["n"] == 4 and agg["n_valid"] == 3
    # groundedness: 5, 2, 5 → 평균 4, 통과 2/3
    assert agg["criteria"]["groundedness"]["mean"] == pytest.approx(4.0)
    assert agg["criteria"]["groundedness"]["pass_rate"] == pytest.approx(2 / 3)


def test_cli_compare_and_report(tmp_path):
    rows = _rows("real", {"a": 0.2, "b": 0.4, "c": 0.6}) + _rows(
        "synthetic", {"a": 0.5, "b": 0.9, "c": 0.7}
    )
    for r in rows:
        r.update({"recall@10": r["hit@10"], "ndcg@10": r["hit@10"], "map": r["hit@10"]})
    pq = tmp_path / "per_query.jsonl"
    pq.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    cli_main(["compare", "--per-query", str(pq), "--out", str(tmp_path / "golden.json")])
    comp = json.loads((tmp_path / "golden.json").read_text(encoding="utf-8"))
    assert comp["metrics"]["hit@10"]["n_configs"] == 3
    cli_main(["report", "--golden", str(tmp_path / "golden.json"), "--out", str(tmp_path / "rep")])
    assert (tmp_path / "rep" / "golden_compare.md").exists()
