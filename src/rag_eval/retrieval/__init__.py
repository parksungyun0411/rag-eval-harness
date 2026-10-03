"""데이터셋과 무관한 검색 구성 요소.

- ``lexical``: Kiwi 토크나이저(선택 의존성 ``kiwi``), 프로세스 내 BM25
- ``dense``: 임베더 프로토콜, sentence-transformers 임베더(선택 의존성 ``models``), 전수 탐색 색인
- ``fusion``: RRF 하이브리드, 크로스인코더 리랭커
- ``opensearch``: OpenSearch+nori 어휘 백엔드(선택 의존성 ``opensearch``, 미검증)
- ``stack``: YAML 설정 → 이름별 검색기 팩토리, 색인 생성

무거운 의존성(kiwipiepy, sentence-transformers, opensearch-py)은 해당 클래스를
만들 때만 불러오므로, 이 패키지를 import 하는 것만으로는 설치가 필요하지 않다.
"""

from rag_eval.retrieval.fusion import HybridRetriever, RerankRetriever, rrf
from rag_eval.retrieval.lexical import BM25Index, BM25Retriever, whitespace_tokenizer
from rag_eval.retrieval.stack import MatrixConfig, RetrievalStack, build_indexes

__all__ = [
    "BM25Index",
    "BM25Retriever",
    "HybridRetriever",
    "MatrixConfig",
    "RerankRetriever",
    "RetrievalStack",
    "build_indexes",
    "rrf",
    "whitespace_tokenizer",
]
