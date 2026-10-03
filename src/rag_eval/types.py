"""골든셋·런 자료형과 JSONL 입출력.

파일 규약 (소비 프로젝트와 하네스 사이의 계약):
- ``queries.jsonl``: ``{"qid": str, "text": str, ...}``
- ``qrels.jsonl``:   ``{"qid": str, "docid": str, "grade": int}`` (grade >= 1 이 관련)
- 런 파일:           ``{"qid": str, "docids": [str], "scores": [float]}`` (순위순)
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# qid -> {docid: grade}
Qrels = dict[str, dict[str, int]]
# qid -> 순위순 docid 목록
Run = dict[str, list[str]]


def read_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


@dataclass
class GoldenSet:
    """평가 질의 묶음과 관련성 판정."""

    name: str
    queries: dict[str, str]
    qrels: Qrels
    meta: dict[str, dict[str, Any]] = field(default_factory=dict)

    @classmethod
    def load(cls, directory: str | Path, name: str | None = None) -> GoldenSet:
        directory = Path(directory)
        queries: dict[str, str] = {}
        meta: dict[str, dict[str, Any]] = {}
        for row in read_jsonl(directory / "queries.jsonl"):
            queries[row["qid"]] = row["text"]
            meta[row["qid"]] = {k: v for k, v in row.items() if k not in ("qid", "text")}
        qrels: Qrels = {}
        for row in read_jsonl(directory / "qrels.jsonl"):
            qrels.setdefault(row["qid"], {})[row["docid"]] = int(row["grade"])
        return cls(name=name or directory.name, queries=queries, qrels=qrels, meta=meta)

    def save(self, directory: str | Path) -> None:
        directory = Path(directory)
        write_jsonl(
            directory / "queries.jsonl",
            ({"qid": q, "text": t, **self.meta.get(q, {})} for q, t in self.queries.items()),
        )
        write_jsonl(
            directory / "qrels.jsonl",
            (
                {"qid": q, "docid": d, "grade": g}
                for q, docs in self.qrels.items()
                for d, g in docs.items()
            ),
        )

    def judged_qids(self) -> list[str]:
        """관련 문서가 하나 이상 있는 질의만 (지표가 정의되는 질의)."""
        return [q for q in self.queries if any(g > 0 for g in self.qrels.get(q, {}).values())]
