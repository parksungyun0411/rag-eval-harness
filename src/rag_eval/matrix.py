"""실험 매트릭스: (골든셋 × 검색 설정) 조합을 모두 돌려 질의별 지표를 남긴다.

검색 설정은 ``Retriever`` 프로토콜만 맞추면 되고, 모델 적재가 무거운 설정을 위해
인스턴스 대신 인자 없는 팩토리를 넘길 수 있다(설정마다 한 번만 만든다).
결과는 질의 단위 행(``per_query.jsonl``)으로 저장해, 평균·신뢰구간·대응 검정을
나중에 같은 원자료에서 다시 계산할 수 있게 한다.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from rag_eval.metrics import evaluate_query
from rag_eval.types import GoldenSet, write_jsonl


@runtime_checkable
class Retriever(Protocol):
    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        """점수 내림차순 (docid, score) 목록."""
        ...


RetrieverSpec = Retriever | Callable[[], Retriever]


def _materialize(spec: RetrieverSpec) -> Retriever:
    if isinstance(spec, Retriever):
        return spec
    return spec()


def run_matrix(
    retrievers: Mapping[str, RetrieverSpec],
    golden_sets: Sequence[GoldenSet],
    ks: Sequence[int] = (1, 5, 10, 20),
    depth: int = 100,
    out_dir: str | Path | None = None,
    log: Callable[[str], None] = print,
) -> list[dict[str, Any]]:
    """모든 조합을 실행하고 질의별 행을 반환한다.

    ``out_dir`` 가 주어지면 ``runs/<골든셋>/<설정>.jsonl`` 과 ``per_query.jsonl`` 을 쓴다.
    """
    if depth < max(ks):
        raise ValueError("depth는 가장 큰 k 이상이어야 합니다")
    rows: list[dict[str, Any]] = []
    out = Path(out_dir) if out_dir is not None else None
    for config, spec in retrievers.items():
        retriever = _materialize(spec)
        for gs in golden_sets:
            qids = gs.judged_qids()
            run_rows = []
            for qid in qids:
                t0 = time.perf_counter()
                hits = retriever.search(gs.queries[qid], depth)
                latency_ms = (time.perf_counter() - t0) * 1000
                docids = [d for d, _ in hits]
                run_rows.append(
                    {"qid": qid, "docids": docids, "scores": [float(s) for _, s in hits]}
                )
                rows.append(
                    {
                        "golden": gs.name,
                        "config": config,
                        "qid": qid,
                        "latency_ms": latency_ms,
                        **evaluate_query(docids, gs.qrels[qid], ks),
                    }
                )
            if out is not None:
                write_jsonl(out / "runs" / gs.name / f"{config}.jsonl", run_rows)
            log(f"[matrix] {gs.name} × {config}: {len(qids)}개 질의")
    if out is not None:
        write_jsonl(out / "per_query.jsonl", rows)
    return rows


def summarize(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """(골든셋, 설정)별 평균. 지표 키는 행에서 숫자형 컬럼을 그대로 쓴다."""
    groups: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for r in rows:
        groups.setdefault((r["golden"], r["config"]), []).append(r)
    out = []
    for (golden, config), rs in groups.items():
        metric_keys = [k for k in rs[0] if k not in ("golden", "config", "qid")]
        summary: dict[str, Any] = {"golden": golden, "config": config, "n": len(rs)}
        for k in metric_keys:
            summary[k] = sum(float(r[k]) for r in rs) / len(rs)
        lat = sorted(float(r["latency_ms"]) for r in rs)
        summary["latency_p95_ms"] = lat[min(len(lat) - 1, int(round(0.95 * (len(lat) - 1))))]
        out.append(summary)
    return out
