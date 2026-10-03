"""설정 파일(YAML) → 검색기 팩토리. 데이터셋과 무관하다.

소비 프로젝트는 ``docids``·``texts`` (문서 순서 고정)와 설정 파일만 넘기면
dense / 어휘 / RRF 하이브리드 / 리랭커 조합을 이름별 팩토리로 받는다.
모델은 설정마다가 아니라 별칭마다 한 번만 적재한다(같은 임베딩 모델을 dense·hybrid가 공유).
테스트에서는 ``embedder_factory``·``scorer_factory``·``tokenizer`` 로 가짜 모델을 주입한다.

설정 예시는 ``configs/matrix.example.yaml``.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from rag_eval.matrix import Retriever
from rag_eval.retrieval.dense import (
    DenseIndex,
    DenseRetriever,
    Embedder,
    SentenceTransformerEmbedder,
)
from rag_eval.retrieval.fusion import (
    CrossEncoderScorer,
    HybridRetriever,
    PairScorer,
    RerankRetriever,
)
from rag_eval.retrieval.lexical import BM25Index, BM25Retriever, KiwiTokenizer, Tokenizer

LEXICAL_BACKENDS = ("kiwi_bm25", "opensearch")


@dataclass
class MatrixConfig:
    configs: list[dict[str, Any]]
    models: dict[str, str]
    ks: list[int] = field(default_factory=lambda: [1, 5, 10, 20])
    depth: int = 100
    rrf_k: int = 60
    rerank_candidates: int = 50
    baseline: str | None = None
    lexical_backend: str = "kiwi_bm25"
    opensearch_url: str = "http://localhost:9200"
    opensearch_index: str = "rag-eval"
    # 소비 프로젝트 전용 키(예: answer_config)는 여기에 그대로 남긴다
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> MatrixConfig:
        known = set(cls.__dataclass_fields__) - {"extra"}
        cfg = cls(
            **{k: v for k, v in raw.items() if k in known},
            extra={k: v for k, v in raw.items() if k not in known},
        )
        cfg.validate()
        return cfg

    @classmethod
    def load(cls, path: str | Path) -> MatrixConfig:
        return cls.from_dict(yaml.safe_load(Path(path).read_text(encoding="utf-8")))

    def validate(self) -> None:
        if self.lexical_backend not in LEXICAL_BACKENDS:
            raise ValueError(f"알 수 없는 lexical_backend: {self.lexical_backend}")
        names = [c["name"] for c in self.configs]
        if len(names) != len(set(names)):
            raise ValueError("설정 이름이 중복됩니다")
        for c in self.configs:
            for key in ("dense", "rerank"):
                if c.get(key) and c[key] not in self.models:
                    raise ValueError(f"{c['name']}: 알 수 없는 모델 별칭 {c[key]}")
            if not c.get("dense") and not c.get("lexical"):
                raise ValueError(f"{c['name']}: dense나 lexical 중 하나는 있어야 합니다")
        if self.baseline and self.baseline not in names:
            raise ValueError(f"baseline {self.baseline} 이(가) 설정 목록에 없습니다")

    def names(self) -> list[str]:
        return [c["name"] for c in self.configs]

    def dense_aliases(self) -> list[str]:
        return sorted({c["dense"] for c in self.configs if c.get("dense")})

    def uses_lexical(self) -> bool:
        return any(c.get("lexical") for c in self.configs)


def dense_path(index_dir: Path, alias: str) -> Path:
    return index_dir / f"dense__{alias}.npy"


class RetrievalStack:
    """색인 디렉터리 하나와 설정 하나로 이름별 검색기를 만든다."""

    def __init__(
        self,
        cfg: MatrixConfig,
        index_dir: str | Path,
        doc_texts: Mapping[str, str],
        device: str = "",
        embedder_factory: Callable[[str], Embedder] | None = None,
        scorer_factory: Callable[[str], PairScorer] | None = None,
        tokenizer: Tokenizer | None = None,
    ):
        self.cfg = cfg
        self.index_dir = Path(index_dir)
        self.doc_texts = doc_texts
        self._embedder_factory = embedder_factory or (
            lambda name: SentenceTransformerEmbedder(name, device)
        )
        self._scorer_factory = scorer_factory or (lambda name: CrossEncoderScorer(name, device))
        self._tokenizer = tokenizer
        self._embedders: dict[str, Embedder] = {}
        self._dense: dict[str, DenseRetriever] = {}
        self._scorers: dict[str, PairScorer] = {}
        self._lexical: Retriever | None = None
        self._docids: list[str] | None = None

    @property
    def docids(self) -> list[str]:
        if self._docids is None:
            self._docids = json.loads((self.index_dir / "docids.json").read_text(encoding="utf-8"))
        return self._docids

    def tokenizer(self) -> Tokenizer:
        if self._tokenizer is None:
            self._tokenizer = KiwiTokenizer()
        return self._tokenizer

    def embedder(self, alias: str) -> Embedder:
        if alias not in self._embedders:
            self._embedders[alias] = self._embedder_factory(self.cfg.models[alias])
        return self._embedders[alias]

    def dense(self, alias: str) -> DenseRetriever:
        if alias not in self._dense:
            index = DenseIndex.load(dense_path(self.index_dir, alias), self.docids)
            self._dense[alias] = DenseRetriever(self.embedder(alias), index)
        return self._dense[alias]

    def lexical(self) -> Retriever:
        if self._lexical is None:
            if self.cfg.lexical_backend == "opensearch":
                from rag_eval.retrieval.opensearch import OpenSearchRetriever

                self._lexical = OpenSearchRetriever(
                    self.cfg.opensearch_url, self.cfg.opensearch_index
                )
            else:
                bm25 = BM25Index.load(self.index_dir / "bm25.pkl")
                self._lexical = BM25Retriever(bm25, self.tokenizer())
        return self._lexical

    def scorer(self, alias: str) -> PairScorer:
        if alias not in self._scorers:
            self._scorers[alias] = self._scorer_factory(self.cfg.models[alias])
        return self._scorers[alias]

    def build(self, spec: Mapping[str, Any]) -> Retriever:
        parts: list[Retriever] = []
        if spec.get("lexical"):
            parts.append(self.lexical())
        if spec.get("dense"):
            parts.append(self.dense(spec["dense"]))
        base = (
            parts[0] if len(parts) == 1 else HybridRetriever(parts, self.cfg.rrf_k, self.cfg.depth)
        )
        if spec.get("rerank"):
            return RerankRetriever(
                base, self.scorer(spec["rerank"]), self.doc_texts, self.cfg.rerank_candidates
            )
        return base

    def build_named(self, name: str) -> Retriever:
        return self.build(next(c for c in self.cfg.configs if c["name"] == name))

    def factories(self) -> dict[str, Callable[[], Retriever]]:
        return {c["name"]: (lambda c=c: self.build(c)) for c in self.cfg.configs}


def build_indexes(
    cfg: MatrixConfig,
    index_dir: str | Path,
    docids: Sequence[str],
    texts: Sequence[str],
    device: str = "",
    embedder_factory: Callable[[str], Embedder] | None = None,
    tokenizer: Tokenizer | None = None,
    log: Callable[[str], None] = print,
) -> None:
    """설정에 쓰인 dense 모델마다 임베딩을, 어휘 검색을 쓰면 BM25(또는 OpenSearch) 색인을 만든다."""
    if len(docids) != len(texts):
        raise ValueError("docids와 texts 길이가 다릅니다")
    out = Path(index_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "docids.json").write_text(json.dumps(list(docids)), encoding="utf-8")
    make_embedder = embedder_factory or (lambda name: SentenceTransformerEmbedder(name, device))
    for alias in cfg.dense_aliases():
        path = dense_path(out, alias)
        if path.exists():
            log(f"[index] {alias}: 기존 색인 사용 ({path})")
            continue
        log(f"[index] {alias}: {len(texts)}건 임베딩 ({cfg.models[alias]})")
        DenseIndex(docids, make_embedder(cfg.models[alias]).encode(list(texts))).save(path)
    if cfg.uses_lexical():
        if cfg.lexical_backend == "opensearch":
            from rag_eval.retrieval.opensearch import build_index

            n = build_index(
                cfg.opensearch_url, cfg.opensearch_index, zip(docids, texts, strict=True)
            )
            log(f"[index] OpenSearch {n}건")
        else:
            tok = tokenizer or KiwiTokenizer()
            BM25Index().fit(docids, [tok(t) for t in texts]).save(out / "bm25.pkl")
            log(f"[index] BM25 {len(docids)}건")
