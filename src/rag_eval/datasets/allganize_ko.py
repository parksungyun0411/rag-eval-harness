"""allganize/RAG-Evaluation-Dataset-KO (Hugging Face, MIT) 로더.

질문 300건(금융·공공·의료·법률·커머스 각 60건)에 대해 RAG 시스템 23종의 답변과
정답 여부(``*_ox``: "O"/"X")가 들어 있다. 정답 여부는 데이터셋 작성자가 LLM 평가기 4종
(TonicAI answer_similarity, MLflow answer_similarity·answer_correctness, Allganize Eval)의
투표로 정한 값이다(데이터셋 카드 기준).

원본 CSV(약 5.7MB, 각 시스템의 답변 전문 포함)는 저장소에 넣지 않고, 고정 리비전에서 받아
SHA-256으로 확인한다. 분석에는 시스템 × 질문 0/1 행렬만 쓴다.
"""

from __future__ import annotations

import csv
import hashlib
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO = "allganize/RAG-Evaluation-Dataset-KO"
# 2024-11-22 커밋. 이 리비전의 데이터셋 카드 표와 아래 CSV의 O 개수가 시스템·도메인별로 일치함을 확인했다.
REVISION = "35db50ff8739d78b11ee68b6f4ef04a95862a504"
FILENAME = "rag_evaluation_result.csv"
SHA256 = "b78fd2bc6b5fd7d36d9be9e23db6fe43677de5a9f6431a7b2826c03992938049"
URL = f"https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/{FILENAME}"
DOMAINS = ("finance", "public", "medical", "law", "commerce")

# 열 이름 → 데이터셋 카드 표의 표기. 카드와 열 이름이 다른 것은
# alli_Llama-3-Ko-8B-Finance-Evol 하나이며, 카드의 "Alli (Llama-3-Alpha-Ko-8B-Instruct-Pro)"와
# 도메인별 정답 수가 모두 같아 같은 시스템으로 본다.
DISPLAY_NAMES = {
    "alli_claude-3.5-sonnet": "Alli (claude3.5-sonnet)",
    "alli_claude3-opus": "Alli (claude3-opus)",
    "alli_gpt-4o": "Alli (gpt-4o)",
    "alli_gpt-4": "Alli (gpt-4)",
    "alli_gpt-4-turbo": "Alli (gpt-4-turbo)",
    "alli_alpha-ko-202411-32B": "Alli (alpha-ko-202411-32B)",
    "alli_gpt-4o-mini": "Alli (gpt-4o-mini)",
    "upstage_gpt-4-turbo": "Upstage (gpt-4-turbo)",
    "openai_assistant_gpt-4-turbo": "OpenAI Assistant (gpt-4-turbo)",
    "openai_assistant_gpt-4": "OpenAI Assistant (gpt-4)",
    "upstage_gpt-4": "Upstage (gpt-4)",
    "alli_Llama-3-Ko-8B-Finance-Evol": "Alli (Llama-3-Alpha-Ko-8B-Instruct-Pro)",
    "alli_KONI-Llama3-8B-Instruct-20240729": "Alli (KONI-Llama3-8B-Instruct)",
    "upstage_solar": "Upstage (solar)",
    "langchain_gpt-4-turbo": "Langchain (gpt-4-turbo)",
    "cohere_command-r-plus": "Cohere (command-r-plus)",
    "cohere_command-r": "Cohere (command-r)",
    "upstage_gpt-3.5-turbo": "Upstage (gpt-3.5-turbo)",
    "alli_Llama-3-Alpha-Ko-8B-Instruct": "Alli (Llama-3-Alpha-Ko-8B-Instruct)",
    "langchain_gpt-3.5-turbo": "Langchain (gpt-3.5-turbo)",
    "anything_llm_gpt-4-turbo": "Anything LLM (gpt-4-turbo)",
    "anything_llm_claude3-opus": "Anything LLM (claude3-opus)",
    "anything_llm_gpt-3.5-turbo": "Anything LLM (gpt-3.5-turbo)",
}


@dataclass
class CorrectnessMatrix:
    """시스템 × 질문 정답 행렬 (1 = O, 0 = X)."""

    systems: list[str]
    qids: list[str]
    domains: list[str]  # 질문별 도메인
    X: np.ndarray  # (시스템 수, 질문 수)

    def subset(self, domain: str) -> CorrectnessMatrix:
        idx = [i for i, d in enumerate(self.domains) if d == domain]
        if not idx:
            raise ValueError(f"도메인 없음: {domain}")
        return CorrectnessMatrix(
            self.systems, [self.qids[i] for i in idx], [domain] * len(idx), self.X[:, idx]
        )

    def accuracy(self) -> np.ndarray:
        return self.X.mean(axis=1)

    def to_csv(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["qid", "domain", *self.systems])
            for j, q in enumerate(self.qids):
                w.writerow([q, self.domains[j], *(int(v) for v in self.X[:, j])])

    @classmethod
    def from_csv(cls, path: str | Path) -> CorrectnessMatrix:
        with open(path, encoding="utf-8") as f:
            rows = list(csv.reader(f))
        systems = rows[0][2:]
        body = rows[1:]
        X = np.array([[int(v) for v in r[2:]] for r in body], dtype=np.float64).T
        return cls(systems, [r[0] for r in body], [r[1] for r in body], X)


def sha256_of(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(dest: str | Path, url: str = URL, expected_sha256: str = SHA256) -> Path:
    """고정 리비전 CSV를 받아 해시를 확인한다. 이미 있고 해시가 맞으면 다시 받지 않는다."""
    dest = Path(dest)
    if dest.exists() and sha256_of(dest) == expected_sha256:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".part")
    urllib.request.urlretrieve(url, tmp)  # noqa: S310 - 고정 https URL
    got = sha256_of(tmp)
    if got != expected_sha256:
        tmp.unlink()
        raise RuntimeError(f"SHA-256 불일치: {got} (기대 {expected_sha256})")
    tmp.replace(dest)
    return dest


def load(path: str | Path) -> CorrectnessMatrix:
    """원본 CSV → 정답 행렬. 값이 O/X가 아니면 오류를 낸다(결측을 조용히 0으로 두지 않는다)."""
    with open(path, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError("빈 CSV")
    cols = [c for c in rows[0] if c.endswith("_ox")]
    keys = [c[: -len("_ox")] for c in cols]
    X = np.zeros((len(cols), len(rows)))
    for j, r in enumerate(rows):
        for i, c in enumerate(cols):
            v = (r[c] or "").strip()
            if v not in ("O", "X"):
                raise ValueError(f"{j}행 {c}: O/X가 아닌 값 {v!r}")
            X[i, j] = 1.0 if v == "O" else 0.0
    domains = [r["domain"].strip() for r in rows]
    unknown = set(domains) - set(DOMAINS)
    if unknown:
        raise ValueError(f"알 수 없는 도메인: {unknown}")
    # 질문 ID: 도메인 내 순번 (원본에 ID 열이 없다)
    counters: dict[str, int] = {}
    qids = []
    for d in domains:
        counters[d] = counters.get(d, 0) + 1
        qids.append(f"{d}-{counters[d]:03d}")
    systems = [DISPLAY_NAMES.get(k, k) for k in keys]
    return CorrectnessMatrix(systems, qids, domains, X)
