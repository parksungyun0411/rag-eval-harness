"""LLM 판정: 생성된 답변 표본을 기준 답변·근거 문맥과 대조해 채점한다.

입력 형식(answers.jsonl): ``{"qid", "question", "answer", "reference", "contexts": [str]}``
항목별 1~5점이며, 4점 이상을 "통과"로 보고 통과율에 Wilson 구간을 붙인다.
판정 출력이 JSON으로 파싱되지 않으면 그 건은 ``parse_error`` 로 남기고 집계에서 뺀다.
"""

from __future__ import annotations

import json
import random
import re
from collections.abc import Callable, Sequence
from typing import Any

from rag_eval.llm import LLMProvider
from rag_eval.stats import wilson_interval

CRITERIA = ("groundedness", "correctness", "helpfulness")

JUDGE_SYSTEM = "당신은 답변 품질을 엄격하게 채점하는 평가자입니다. 지정한 JSON만 출력합니다."

JUDGE_TEMPLATE = """질문, 기준 답변(공식 답변), 검색된 근거 문맥, 평가 대상 답변이 주어집니다.
아래 세 항목을 1~5 정수로 채점하세요.
- groundedness: 평가 대상 답변의 주장이 근거 문맥으로 뒷받침되는가 (근거 없는 주장이 있으면 감점)
- correctness: 기준 답변과 사실·안내 내용이 일치하는가
- helpfulness: 질문자가 다음에 무엇을 하면 되는지 알 수 있는가

[질문]
{question}

[기준 답변]
{reference}

[근거 문맥]
{contexts}

[평가 대상 답변]
{answer}

출력 형식: {{"groundedness": int, "correctness": int, "helpfulness": int, "rationale": str}}
"""

_JSON_RE = re.compile(r"\{.*\}", re.S)


def parse_judgment(output: str) -> dict[str, Any] | None:
    m = _JSON_RE.search(output)
    if not m:
        return None
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    scores = {}
    for c in CRITERIA:
        v = data.get(c)
        if not isinstance(v, int | float) or not 1 <= v <= 5:
            return None
        scores[c] = int(v)
    scores["rationale"] = str(data.get("rationale", ""))
    return scores


def judge_answers(
    answers: Sequence[dict[str, Any]],
    provider: LLMProvider,
    n: int | None = 50,
    seed: int = 42,
    max_context_chars: int = 6000,
    log: Callable[[str], None] = print,
) -> list[dict[str, Any]]:
    sample = list(answers)
    if n is not None and n < len(sample):
        sample = random.Random(seed).sample(sample, n)
    out = []
    for i, a in enumerate(sample, start=1):
        contexts = "\n---\n".join(a.get("contexts", []))[:max_context_chars]
        raw = provider.complete(
            JUDGE_SYSTEM,
            JUDGE_TEMPLATE.format(
                question=a["question"],
                reference=a["reference"],
                contexts=contexts,
                answer=a["answer"],
            ),
        )
        parsed = parse_judgment(raw)
        row: dict[str, Any] = {"qid": a["qid"], "judge": provider.name}
        if parsed is None:
            row["parse_error"] = raw[:500]
        else:
            row.update(parsed)
        out.append(row)
        if i % 10 == 0:
            log(f"[judge] {i}/{len(sample)}")
    return out


def aggregate_judgments(judgments: Sequence[dict[str, Any]], pass_score: int = 4) -> dict[str, Any]:
    valid = [j for j in judgments if "parse_error" not in j]
    result: dict[str, Any] = {"n": len(judgments), "n_valid": len(valid), "criteria": {}}
    if not valid:
        return result
    for c in CRITERIA:
        scores = [j[c] for j in valid]
        passed = sum(1 for s in scores if s >= pass_score)
        lo, hi = wilson_interval(passed, len(scores))
        result["criteria"][c] = {
            "mean": sum(scores) / len(scores),
            "pass_rate": passed / len(scores),
            "pass_ci": [lo, hi],
        }
    return result
