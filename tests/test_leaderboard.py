"""데이터셋 로더 · 실험 실행 · 데모 API · JS 계산기 일치. 데이터는 모두 합성 픽스처다."""

import csv
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from rag_eval.datasets import allganize_ko
from rag_eval.datasets.allganize_ko import CorrectnessMatrix
from rag_eval.leaderboard_experiment import run
from rag_eval.rankstab import flip_probability

ROOT = Path(__file__).parent.parent
FIX = Path(__file__).parent / "fixtures" / "allganize_synthetic" / "rag_evaluation_result.csv"


def test_load_synthetic_fixture():
    cm = allganize_ko.load(FIX)
    assert cm.systems == ["Alli (claude3.5-sonnet)", "Alli (gpt-4o)", "Cohere (command-r)"]
    assert cm.X.shape == (3, 10)
    assert cm.accuracy().tolist() == [0.9, 0.7, 0.3]
    assert cm.qids[:3] == ["finance-001", "finance-002", "public-001"]
    assert cm.subset("law").X.tolist() == [[1, 1], [1, 0], [0, 0]]


def test_load_rejects_missing_values(tmp_path):
    rows = list(csv.reader(open(FIX, encoding="utf-8")))
    rows[3][7] = ""  # 첫 시스템 _ox 결측
    bad = tmp_path / "bad.csv"
    with open(bad, "w", encoding="utf-8", newline="") as f:
        csv.writer(f).writerows(rows)
    with pytest.raises(ValueError, match="O/X가 아닌 값"):
        allganize_ko.load(bad)


def test_matrix_csv_roundtrip(tmp_path):
    cm = allganize_ko.load(FIX)
    cm.to_csv(tmp_path / "m.csv")
    again = CorrectnessMatrix.from_csv(tmp_path / "m.csv")
    assert again.systems == cm.systems and again.domains == cm.domains
    assert np.array_equal(again.X, cm.X)


def test_download_verifies_hash(tmp_path):
    url = FIX.resolve().as_uri()
    good = allganize_ko.sha256_of(FIX)
    path = allganize_ko.download(tmp_path / "a.csv", url=url, expected_sha256=good)
    assert allganize_ko.sha256_of(path) == good
    with pytest.raises(RuntimeError, match="SHA-256"):
        allganize_ko.download(tmp_path / "b.csv", url=url, expected_sha256="0" * 64)
    assert not (tmp_path / "b.csv").exists()


def test_pinned_revision_constants():
    assert len(allganize_ko.REVISION) == 40 and allganize_ko.REVISION in allganize_ko.URL
    assert len(allganize_ko.DISPLAY_NAMES) == 23


def test_run_experiment_outputs(tmp_path):
    res = run(FIX, tmp_path / "reports", tmp_path / "web", n_boot=200, seed=0, log=lambda _: None)
    sub = tmp_path / "reports" / "rank_stability"
    for name in (
        "leaderboard.csv",
        "curve_overall.csv",
        "curve_domain.csv",
        "adjacent_pairs.csv",
        "approx_check.csv",
        "domain_best_share.csv",
        "domain_rank_corr.csv",
        "correctness_matrix.csv",
        "tau_vs_n.png",
        "held_vs_n.png",
        "pairs_n_required.png",
        "approx_check.png",
        "domain_best_share.png",
    ):
        assert (sub / name).exists(), name
    report = (tmp_path / "reports" / "REPORT.md").read_text(encoding="utf-8")
    assert allganize_ko.REVISION in report and "200회" in report
    lb = list(csv.DictReader(open(sub / "leaderboard.csv", encoding="utf-8")))
    assert [r["system"] for r in lb] == [
        "Alli (claude3.5-sonnet)",
        "Alli (gpt-4o)",
        "Cohere (command-r)",
    ]
    web = json.loads((tmp_path / "web" / "results.json").read_text(encoding="utf-8"))
    assert web["correct_bits"]["Alli (gpt-4o)"] == "1111111000"
    assert len(res["pairs"]) == 2


# ---------------------------------------------------------------- 데모 API


@pytest.fixture
def client(tmp_path):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from rag_eval.demo.app import create_app

    web = tmp_path / "web"
    shutil.copytree(ROOT / "web", web, ignore=shutil.ignore_patterns("data"))
    run(FIX, tmp_path / "reports", web / "data", n_boot=100, seed=0, log=lambda _: None)
    return TestClient(create_app(web))


def test_api_endpoints(client):
    assert client.get("/api/health").json() == {"ok": True, "results": True}
    lb = client.get("/api/leaderboard").json()
    assert lb["leaderboard"][0]["system"] == "Alli (claude3.5-sonnet)"
    st = client.get("/api/stability").json()
    assert set(st) == {"meta", "overall_curve", "domain_curves", "domain_winners", "pairs"}
    r = client.get("/api/flip", params={"n": 3, "gap": 1 / 3, "discordance": 1.0}).json()
    assert r["p_flip"] == pytest.approx(7 / 27)
    assert (
        client.get("/api/flip", params={"n": 10, "gap": 0.3, "discordance": 0.2}).status_code == 422
    )
    assert client.get("/api/flip", params={"n": 0, "gap": 0.1}).status_code == 422
    # 부동소수 반올림 회귀: JS는 받는데 API가 422를 내던 입력
    ok = client.get("/api/flip", params={"n": 10, "gap": 0.1, "accuracy": 1.0})
    assert ok.status_code == 200 and ok.json()["p_reverse"] == 0.0
    assert "n_power80_normal" in ok.json()
    page = client.get("/")
    assert page.status_code == 200 and "순위 역전 확률 계산기" in page.text


def test_api_without_results(tmp_path):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from rag_eval.demo.app import create_app

    (tmp_path / "web").mkdir()
    c = TestClient(create_app(tmp_path / "web"))
    assert c.get("/api/health").json()["results"] is False
    assert c.get("/api/leaderboard").status_code == 503


# ---------------------------------------------------------------- JS 계산기


@pytest.mark.skipif(shutil.which("node") is None, reason="node 없음")
@pytest.mark.parametrize(
    "n,gap,disc,acc",
    [
        (3, 1 / 3, 1.0, None),
        (2, 1 / 3, 1.0, None),
        (300, 0.01, 0.05, None),
        (50, 0.2, None, 0.8),
        (120, 0.05, None, None),
    ],
)
def test_js_calculator_matches_python(n, gap, disc, acc):
    js = (
        "const F=require(process.argv[1]);"
        "const a=process.argv.slice(2).map(v=>v==='null'?null:Number(v));"
        "console.log(JSON.stringify(F.flipProbability(...a)));"
    )
    args = [
        str(n),
        str(gap),
        "null" if disc is None else str(disc),
        "null" if acc is None else str(acc),
    ]
    out = subprocess.run(
        ["node", "-e", js, str(ROOT / "web" / "flip.js"), *args],
        capture_output=True,
        text=True,
        check=True,
    )
    got = json.loads(out.stdout)
    want = flip_probability(n, gap, discordance=disc, accuracy=acc)
    assert got["pFlip"] == pytest.approx(want["p_flip"], abs=1e-9)
    assert got["pReverse"] == pytest.approx(want["p_reverse"], abs=1e-9)
    assert got["pTie"] == pytest.approx(want["p_tie"], abs=1e-9)
    assert got["n95Exact"] == want["n_95_exact"]
    assert got["pFlipNormal"] == pytest.approx(want["p_flip_normal"], abs=1e-6)
    assert got["nPower80Normal"] == pytest.approx(want["n_power80_normal"], rel=1e-9)


def _node(js: str, *args: str) -> str:
    out = subprocess.run(
        ["node", "-e", js, str(ROOT / "web" / "flip.js"), *args],
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    return out.stdout


@pytest.mark.skipif(shutil.which("node") is None, reason="node 없음")
def test_js_worst_case_is_fast():
    # 예전에는 gap 1%p·불일치율 50%에서 정확해 n95 탐색이 약 12초 걸렸다
    js = (
        "const F=require(process.argv[1]);const t=Date.now();"
        "const r=F.flipProbability(20000,0.01,0.5,null);"
        "console.log(JSON.stringify({ms:Date.now()-t,skip:r.n95ExactSkipped,n95:r.n95Exact,"
        "p:r.pFlip}));"
    )
    got = json.loads(_node(js))
    assert got["ms"] < 2000
    assert got["skip"] is True and got["n95"] is None
    want = flip_probability(20000, 0.01, discordance=0.5)
    assert got["p"] == pytest.approx(want["p_flip"], abs=1e-9)


@pytest.mark.skipif(shutil.which("node") is None, reason="node 없음")
def test_js_and_python_accept_float_rounding_case():
    js = (
        "const F=require(process.argv[1]);"
        "console.log(JSON.stringify(F.flipProbability(10,0.1,null,1.0)));"
    )
    got = json.loads(_node(js))
    want = flip_probability(10, 0.1, accuracy=1.0)
    assert got["pFlip"] == pytest.approx(want["p_flip"], abs=1e-12)
