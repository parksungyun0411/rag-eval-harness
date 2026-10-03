# RAG 평가 하네스

[![CI](https://github.com/parksungyun0411/rag-eval-harness/actions/workflows/ci.yml/badge.svg)](https://github.com/parksungyun0411/rag-eval-harness/actions/workflows/ci.yml) ![Python](https://img.shields.io/badge/python-3.12-3776AB?logo=python&logoColor=white) [![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

검색·RAG 평가 코드를 특정 도메인에서 분리한 패키지입니다. 랭킹 지표, 설정 비교용 통계, 실험 매트릭스 실행기, 보고서 생성기를 담고 있으며, [minwon-rag](https://github.com/parksungyun0411/minwon-rag)가 이 패키지를 의존성으로 사용합니다.

이 저장소에는 "골든셋 2종 비교" 실험도 들어 있습니다. 같은 검색 설정들을 실제 시민 질의 골든셋과 LLM이 합성한 질의 골든셋으로 각각 측정하고, 점수 차이와 설정 순위의 일치 정도(Kendall·Spearman 순위 상관)를 계산합니다. 합성 골든셋이 점수를 부풀리더라도 설정 순위가 같다면 설정 선택에는 쓸 수 있고, 순위까지 바뀐다면 합성 골든셋으로 고른 설정이 실제 질의에서는 최선이 아닐 수 있습니다.

현재 패키지 코드와 단위 테스트까지 작성했습니다. 공공데이터 인증키와 LLM 키를 받기 전이라 실험은 실행하지 않았고, 평가 표는 비어 있습니다.

## 구현 범위

- 랭킹 지표: hit@k, recall@k, MRR, nDCG@k, mAP
- 통계: Wilson 점수 신뢰구간, McNemar 대응 검정(정확 이항 / 연속성 보정 카이제곱), Kendall tau-b, Spearman rho
- 실험 매트릭스 실행기: (골든셋 × 검색 설정) 전 조합 실행, 질의별 지표·지연 시간과 런 파일 저장
- 골든셋 비교: 설정별 점수 차, 두 골든셋에서의 설정 순위와 순위 상관, 골든셋별 최고 설정
- 보고서 생성기: 마크다운 표와 그림(recall@k 곡선, 골든셋 산점도)을 `reports/`에 생성
- LLM 공급자 인터페이스: Anthropic, OpenAI 호환 서버(로컬 vLLM·Ollama 포함)
- 합성 질의 골든셋 생성, 답변 표본 LLM 판정(근거성·정확성·유용성 1~5점, 통과율 Wilson 구간)
- `rag-eval` 명령: `synth`, `compare`, `judge`, `report`

## 아키텍처

<img src="docs/diagrams/architecture.png" alt="rag-eval-harness 아키텍처 — 소비 프로젝트, 생성, 실행, 분석" width="900">

하네스는 검색기 구현을 알지 못합니다. `search(query, k) -> [(docid, score)]` 하나만 맞추면 어떤 검색기든 매트릭스에 넣을 수 있고, 골든셋·런·답변은 아래 JSONL 규약으로 주고받습니다.

| 파일 | 형식 |
|---|---|
| `queries.jsonl` | `{"qid", "text", ...}` |
| `qrels.jsonl` | `{"qid", "docid", "grade"}` (grade ≥ 1이 관련) |
| 런 파일 | `{"qid", "docids": [...], "scores": [...]}` |
| `per_query.jsonl` | `{"golden", "config", "qid", "latency_ms", "hit@10", "mrr", ...}` |
| `answers.jsonl` | `{"qid", "question", "answer", "reference", "contexts": [...]}` |

다이어그램 원본은 [docs/diagrams/architecture.html](docs/diagrams/architecture.html)입니다.

## 기술 스택

| 영역 | 기술 |
|---|---|
| 언어 | ![Python](https://img.shields.io/badge/Python_3.12-3776AB?logo=python&logoColor=white) ![uv](https://img.shields.io/badge/uv-DE5FE9) |
| 계산·통계 | ![NumPy](https://img.shields.io/badge/NumPy-013243?logo=numpy&logoColor=white) ![SciPy](https://img.shields.io/badge/SciPy-8CAAE6?logo=scipy&logoColor=white) |
| 시각화 | ![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C) |
| LLM | ![Anthropic](https://img.shields.io/badge/Anthropic_API-191919?logo=anthropic&logoColor=white) ![OpenAI 호환](https://img.shields.io/badge/OpenAI_호환-5B6770) |
| 코드 품질 | ![pytest](https://img.shields.io/badge/pytest-0A9EDC?logo=pytest&logoColor=white) ![Ruff](https://img.shields.io/badge/Ruff-D7FF64?logo=ruff&logoColor=black) ![GitHub Actions](https://img.shields.io/badge/GitHub_Actions-2088FF?logo=githubactions&logoColor=white) |

## 실행 방법

필요한 도구는 Python 3.12와 uv입니다. 단위 테스트는 키 없이 실행됩니다.

```bash
uv sync
make test
make lint
```

### 라이브러리로 사용

```python
from rag_eval import GoldenSet
from rag_eval.matrix import run_matrix
from rag_eval.report import render_matrix_report

golden = GoldenSet.load("data/golden/real", name="real")
rows = run_matrix(
    {"bm25": bm25, "hybrid": lambda: build_hybrid()},
    [golden],
    ks=(1, 5, 10, 20),
    depth=100,
    out_dir="results",
)
render_matrix_report(rows, "reports", baseline="bm25")
```

```bash
rag-eval synth   --corpus corpus.jsonl --out data/golden/synthetic --n 100
rag-eval compare --per-query results/per_query.jsonl --a real --b synthetic --out results/golden.json
rag-eval judge   --answers results/answers.jsonl --out results/judgments.jsonl --n 50
rag-eval report  --golden results/golden.json --judgments results/judgments.jsonl --out reports
```

### 골든셋 2종 비교 실험

minwon-rag를 옆 디렉터리에 두고 실행합니다. `collect`·`index`는 minwon-rag의 타깃을 그대로 부르고, `eval`은 두 골든셋으로 검색 매트릭스를 실행한 뒤 비교와 판정을 진행합니다. 공공데이터포털 인증키(minwon-rag `.env`)와 LLM 키(양쪽 `.env`의 `LLM_PROVIDER`, `LLM_MODEL`, API 키)가 필요합니다.

```bash
git clone https://github.com/parksungyun0411/minwon-rag ../minwon-rag
cp .env.example .env
make setup
make collect    # 실제 질의 수집·분할 + 합성 질의 100건
make index
make eval       # 매트릭스 → 점수 차·순위 상관 → 답변 50건 판정
make report
```

## 평가

측정 전 — `make eval` 실행 후 자동 생성

아래 표의 숫자는 `make report`가 `results/`의 원자료에서 다시 계산해 `reports/`에 쓴 값을 옮겨 적습니다. 지금은 측정 전이라 모든 칸이 비어 있습니다.

골든셋 2종 비교 — 실제 질의 대 합성 질의 (`reports/golden_compare.md`):

| 지표 | 평균 gap(합성 − 실제) | Kendall tau-b | Spearman rho | 최고 설정(실제) | 최고 설정(합성) |
|---|---:|---:|---:|---|---|
| hit@10 | | | | | |
| recall@10 | | | | | |
| MRR | | | | | |
| nDCG@10 | | | | | |

답변 표본 LLM 판정 (`reports/judge.md`):

| 항목 | 평균(1~5) | 통과율(≥4) | 95% CI (Wilson) |
|---|---:|---:|---|
| 근거성 | | | |
| 정확성 | | | |
| 유용성 | | | |

설정별 nDCG@10 산점도(x: 실제, y: 합성)는 `reports/golden_scatter_ndcg_at_10.png`에 생성됩니다.

지표와 통계 함수 자체는 손으로 계산한 값과 비교하는 단위 테스트로 검증합니다(`tests/test_metrics.py`, `tests/test_stats.py`).

## 프로젝트 구조

```text
rag-eval-harness/
├── src/rag_eval/
│   ├── metrics.py   hit@k, recall@k, MRR, nDCG, mAP
│   ├── stats.py     Wilson, McNemar, Kendall, Spearman
│   ├── types.py     GoldenSet, JSONL 입출력
│   ├── matrix.py    실험 매트릭스 실행기, 요약
│   ├── golden.py    골든셋 2종 비교
│   ├── report.py    마크다운 표·그림
│   ├── llm.py       LLM 공급자
│   ├── synth.py     합성 질의 골든셋
│   ├── judge.py     답변 LLM 판정
│   └── cli.py       rag-eval 명령
├── docs/diagrams/   아키텍처 다이어그램(HTML 원본, PNG)
├── tests/           손계산 지표 테스트, 가짜 검색기·LLM 연결 테스트
├── reports/         make report 결과
└── results/         비교·판정 원자료
```

## 구현하면서 신경 쓴 점

- 평균이 아니라 질의별 원자료를 남깁니다. 매트릭스 실행기는 질의 한 건마다 지표와 지연 시간을 기록하고, 평균·신뢰구간·대응 검정·골든셋 비교를 모두 같은 `per_query.jsonl`에서 계산합니다. 보고서를 다시 만들 때 검색을 다시 실행하지 않아도 됩니다.
- 비율 지표의 신뢰구간은 Wilson 방식으로 계산합니다. 평가 질의가 150건 수준이면 hit@k가 0이나 1 근처일 때 정규근사 구간은 [0, 1]을 벗어나거나 폭이 0이 됩니다.
- 설정 간 비교에는 McNemar 대응 검정을 씁니다. 두 설정이 같은 질의를 푸는 구조라 독립 표본 검정이 맞지 않습니다. 불일치 쌍이 25건 미만이면 정확 이항검정, 이상이면 연속성 보정 카이제곱을 적용합니다.
- 지표 정의는 `metrics.py` 첫머리에 적어 고정했습니다. nDCG는 선형 이득(gain = grade), mAP는 trec_eval과 같이 전체 관련 문서 수로 나눕니다. 관련 문서가 없는 질의는 지표가 정의되지 않으므로 오류를 내고, 매트릭스 실행기는 이런 질의를 처음부터 제외합니다.
- 순위 상관에는 p값과 설정 수를 함께 남깁니다. 비교하는 설정이 7개 안팎이면 Kendall tau가 가질 수 있는 값의 수가 적습니다. 상관계수 하나로 결론을 내지 않도록 보고서에 설정별 점수·순위 표를 같이 씁니다.
- 합성 질의에도 실제 질의와 같은 라벨 규칙을 적용합니다. 하네스의 합성 기본값은 출처 문서 하나만 정답으로 두지만, minwon-rag는 출처 문서(grade 2)에 더해 실제 질의와 같은 인용 법령 규칙의 라벨을 붙입니다. 두 골든셋의 차이를 질의 출처 하나로 좁히기 위해서입니다.
- LLM 호출은 `complete(system, prompt)` 프로토콜 뒤에 두었습니다. 공급자와 모델은 환경 변수로 고르고, 테스트는 가짜 공급자를 넣어 키 없이 실행됩니다. 판정 출력이 JSON으로 파싱되지 않은 건은 집계에서 빼고, 표본 수와 유효 건수를 보고서에 함께 기록합니다.

## 한계

- 실험을 실행하지 않았습니다. 평가 표가 비어 있는 이유입니다.
- 비교하는 검색 설정이 7개 안팎이라 순위 상관의 통계적 검정력이 낮습니다. 설정을 늘리면 서로 비슷한 설정이 섞여 순위가 동점에 가까워지는 문제가 생깁니다.
- LLM 판정을 사람 판정과 대조해 보정하지 않았습니다. 판정 모델의 편향(긴 답변 선호 등)이 결과에 그대로 들어갑니다.
- 합성 질의는 100건, 판정 표본은 50건이라 신뢰구간이 넓습니다.
- 사람이 직접 쓴 질의로 만든 세 번째 골든셋과 인용 법령 F1은 이번 범위에서 제외했습니다.
- 하네스 단독으로는 데이터를 만들지 않습니다. `collect`·`index`·`eval` 타깃은 소비 프로젝트(minwon-rag)의 Makefile에 의존합니다.

## 라이선스

코드는 [MIT](LICENSE) 라이선스를 따릅니다.
