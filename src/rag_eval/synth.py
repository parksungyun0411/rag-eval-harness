"""LLM 합성 질의 골든셋 생성.

코퍼스에서 문서를 표본 추출해, 그 문서로 답할 수 있는 질의를 LLM이 쓰게 한다.
정답은 출처 문서 하나(grade 2)다. 소비 프로젝트는 실제 질의 골든셋과 같은 라벨 규칙을
합성 질의에도 적용해 qrels를 다시 만들 수 있다(질의 출처만 다르게 두기 위해).

코퍼스 형식: ``{"docid": str, "text": str, ...}`` JSONL
"""

from __future__ import annotations

import random
from collections.abc import Callable, Sequence
from typing import Any

from rag_eval.llm import LLMProvider
from rag_eval.types import GoldenSet

DEFAULT_SYSTEM = "당신은 검색 평가용 질의를 만드는 도우미입니다. 질의 한 개만 출력합니다."

DEFAULT_TEMPLATE = """다음 문서를 읽고, 이 문서가 답이 되는 사용자 질문을 하나 작성하세요.
- 문서의 문장을 그대로 베끼지 말고, 실제 사용자가 검색창이나 문의 게시판에 쓸 법한 말투로 씁니다.
- 질문만 출력합니다.

[문서]
{text}
"""


def _clean(output: str) -> str:
    text = output.strip()
    for prefix in ("질문:", "질의:", "Q:"):
        if text.startswith(prefix):
            text = text[len(prefix) :].strip()
    return text.strip('"').strip()


def generate_synthetic_golden(
    corpus: Sequence[dict[str, Any]],
    provider: LLMProvider,
    n: int = 100,
    seed: int = 42,
    template: str = DEFAULT_TEMPLATE,
    system: str = DEFAULT_SYSTEM,
    max_chars: int = 3000,
    name: str = "synthetic",
    doc_filter: Callable[[dict[str, Any]], bool] | None = None,
    log: Callable[[str], None] = print,
) -> GoldenSet:
    pool = [d for d in corpus if doc_filter is None or doc_filter(d)]
    if len(pool) < n:
        raise ValueError(f"표본 {n}건을 뽑기에 문서가 부족합니다: {len(pool)}")
    sample = random.Random(seed).sample(pool, n)
    queries: dict[str, str] = {}
    qrels: dict[str, dict[str, int]] = {}
    meta: dict[str, dict[str, Any]] = {}
    for i, doc in enumerate(sample, start=1):
        out = _clean(provider.complete(system, template.format(text=doc["text"][:max_chars])))
        if not out:
            log(f"[synth] 빈 출력, 건너뜀: {doc['docid']}")
            continue
        qid = f"syn-{doc['docid']}"
        queries[qid] = out
        qrels[qid] = {doc["docid"]: 2}
        meta[qid] = {"source_docid": doc["docid"], "generator": provider.name}
        if i % 10 == 0:
            log(f"[synth] {i}/{n}")
    return GoldenSet(name=name, queries=queries, qrels=qrels, meta=meta)
