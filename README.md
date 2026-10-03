# RAG 평가 하네스

[![CI](https://github.com/parksungyun0411/rag-eval-harness/actions/workflows/ci.yml/badge.svg)](https://github.com/parksungyun0411/rag-eval-harness/actions/workflows/ci.yml) ![Python](https://img.shields.io/badge/python-3.12-3776AB?logo=python&logoColor=white) [![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

검색·RAG 평가 코드를 특정 데이터셋에서 분리한 패키지입니다. 랭킹 지표와 설정 비교용 통계, 실험 매트릭스 실행기, 보고서 생성기를 중심으로, 교체 가능한 검색 구성 요소(Kiwi+BM25, dense, RRF 하이브리드, 크로스인코더 리랭커)를 함께 담았습니다. 데이터셋은 소비 프로젝트가 JSONL 규약에 맞춰 넘기기만 하면 됩니다.

평가 결과를 해석할 때 필요한 분석 두 가지를 더했습니다.

- **질의 수에 따른 순위 안정성**: 평가 질의가 n건뿐이었다면 검색 설정들의 순위가 전체 질의로 매긴 순위와 얼마나 자주 같았을지를 부트스트랩으로 추정합니다. 평가셋을 몇 건으로 만들어야 설정 비교 결론이 흔들리지 않는지 판단하는 근거가 됩니다.
- **골든셋 2종 비교**: 같은 검색 설정들을 실제 사용자 질의 골든셋과 LLM이 합성한 질의 골든셋으로 각각 측정하고, 점수 차이와 설정 순위의 일치 정도(Kendall·Spearman 순위 상관)를 계산합니다. 합성 골든셋이 점수를 부풀리더라도 설정 순위가 같다면 설정 선택에는 쓸 수 있고, 순위까지 바뀐다면 합성 골든셋으로 고른 설정이 실제 질의에서는 최선이 아닐 수 있습니다.

현재 패키지 코드와 단위 테스트까지 작성했습니다. 실데이터 실험은 실행하지 않았고, 평가 표는 비어 있습니다.

## 구현 범위

- 랭킹 지표: hit@k, recall@k, MRR, nDCG@k, mAP
- 통계: Wilson 점수 신뢰구간, McNemar 대응 검정(정확 이항 / 연속성 보정 카이제곱), Kendall tau-b, Spearman rho
- 실험 매트릭스 실행기: (골든셋 × 검색 설정) 전 조합 실행, 질의별 지표·지연 시간과 런 파일 저장
- 순위 안정성: 질의 수 n별 순위 전체 일치율·1위 일치율(Wilson 구간)과 설정 쌍 순서 일치율(백분위 구간), 목표 일치율을 만족하는 최소 n
- 골든셋 비교: 설정별 점수 차, 두 골든셋에서의 설정 순위와 순위 상관, 골든셋별 최고 설정
- 검색 구성 요소(`rag_eval.retrieval`): Kiwi 형태소 + 프로세스 내 BM25, sentence-transformers dense 전수 탐색, RRF 하이브리드, 크로스인코더 리랭커, OpenSearch+nori 어휘 백엔드(미검증), YAML 설정 → 이름별 검색기 팩토리
- 보고서 생성기: 마크다운 표와 그림(recall@k 곡선, 순위 안정성 곡선, 골든셋 산점도)을 `reports/`에 생성
- LLM 공급자 인터페이스(Anthropic, OpenAI 호환 서버), 합성 질의 골든셋 생성, 답변 표본 LLM 판정
- `rag-eval` 명령: `synth`, `compare`, `stability`, `judge`, `report`

## 아키텍처

<img src="docs/diagrams/architecture.png" alt="rag-eval-harness 아키텍처 — 소비 프로젝트, 검색 구성 요소, 생성, 실행, 분석" width="900">

매트릭스 실행기는 검색기 구현을 알지 못합니다. `search(query, k) -> [(docid, score)]` 하나만 맞추면 어떤 검색기든 넣을 수 있고, `rag_eval.retrieval`의 구성 요소는 이 규약을 따르는 기본 구현입니다. 골든셋·런·답변은 아래 JSONL 규약으로 주고받습니다.

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
| 검색 | ![Kiwi](https://img.shields.io/badge/Kiwi-5B6770) ![BM25](https://img.shields.io/badge/BM25-scipy.sparse-5B6770) ![sentence-transformers](https://img.shields.io/badge/sentence--transformers-5B6770) ![OpenSearch](https://img.shields.io/badge/OpenSearch-선택-005EB8?logo=opensearch&logoColor=white) |
| 시각화 | ![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C) |
| LLM | ![Anthropic](https://img.shields.io/badge/Anthropic_API-191919?logo=anthropic&logoColor=white) ![OpenAI 호환](https://img.shields.io/badge/OpenAI_호환-5B6770) |
| 코드 품질 | ![pytest](https://img.shields.io/badge/pytest-0A9EDC?logo=pytest&logoColor=white) ![Ruff](https://img.shields.io/badge/Ruff-D7FF64?logo=ruff&logoColor=black) ![GitHub Actions](https://img.shields.io/badge/GitHub_Actions-2088FF?logo=githubactions&logoColor=white) |

## 실행 방법

필요한 도구는 Python 3.12와 uv입니다. 단위 테스트는 키와 모델 없이 실행됩니다.

```bash
uv sync --extra kiwi
make test
make lint
```

선택 의존성은 쓰는 기능만 설치합니다.

| extra | 내용 |
|---|---|
| `kiwi` | Kiwi 형태소 분석기(BM25 토크나이저) |
| `models` | sentence-transformers, torch(dense 임베딩, 크로스인코더) |
| `llm` | Anthropic SDK, httpx(합성 질의, LLM 판정) |
| `opensearch` | opensearch-py(OpenSearch 어휘 백엔드, 미검증) |

### 라이브러리로 사용

```python
from rag_eval import GoldenSet
from rag_eval.matrix import run_matrix
from rag_eval.report import render_matrix_report
from rag_eval.retrieval import MatrixConfig, RetrievalStack, build_indexes

cfg = MatrixConfig.load("configs/matrix.example.yaml")
build_indexes(cfg, "data/index", docids, texts)  # dense 임베딩 + BM25 색인
stack = RetrievalStack(cfg, "data/index", dict(zip(docids, texts)))

golden = GoldenSet.load("data/golden/real", name="real")
rows = run_matrix(stack.factories(), [golden], ks=cfg.ks, depth=cfg.depth, out_dir="results")
render_matrix_report(rows, "reports", baseline=cfg.baseline, config_order=cfg.names())
```

```bash
rag-eval stability --per-query results/per_query.jsonl --golden real --metric ndcg@10 --out results/stability.json
rag-eval synth     --corpus corpus.jsonl --out data/golden/synthetic --n 100
rag-eval compare   --per-query results/per_query.jsonl --a real --b synthetic --out results/golden.json
rag-eval judge     --answers results/answers.jsonl --out results/judgments.jsonl --n 50
rag-eval report    --stability results/stability.json --golden results/golden.json --out reports
```

### 소비 프로젝트와 함께 실행

`collect`·`index`·`eval`은 `CONSUMER_DIR`(기본값 `../minwon-rag`)의 Makefile 타깃을 불러 데이터·색인·검색 매트릭스를 만들고, 하네스는 그 결과로 순위 안정성·골든셋 비교·판정을 실행합니다. 소비 프로젝트는 `collect`·`synth`·`index`·`eval`·`answers` 타깃과 `results/per_query.jsonl`·`results/answers.jsonl` 산출만 맞추면 됩니다. 합성 질의와 판정에는 `.env`의 `LLM_PROVIDER`, `LLM_MODEL`, API 키가 필요합니다.

```bash
cp .env.example .env
make setup
make collect CONSUMER_DIR=../<소비 프로젝트>
make index   CONSUMER_DIR=../<소비 프로젝트>
make eval    CONSUMER_DIR=../<소비 프로젝트>
make report
```

## 평가

측정 전 — `make eval` 실행 후 자동 생성

아래 표의 숫자는 `make report`가 `results/`의 원자료에서 다시 계산해 `reports/`에 쓴 값을 옮겨 적습니다. 지금은 측정 전이라 모든 칸이 비어 있습니다.

질의 수에 따른 설정 순위 안정성 — 실제 질의 골든셋, nDCG@10, 복원추출 1,000회 (`reports/stability_real_ndcg_at_10.md`):

| n | 순위 전체 일치율 | 95% CI | 1위 일치율 | 95% CI | 쌍 순서 일치율 |
|---:|---:|---|---:|---|---:|
| 10 | | | | | |
| 30 | | | | | |
| 50 | | | | | |
| 100 | | | | | |
| 전체 | | | | | |

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

지표·통계·순위 안정성·BM25·RRF 함수는 손으로 계산하거나 해석적으로 구한 값과 비교하는 단위 테스트로 검증합니다(`tests/test_metrics.py`, `tests/test_stats.py`, `tests/test_stability.py`, `tests/test_retrieval.py`).

## 프로젝트 구조

```text
rag-eval-harness/
├── src/rag_eval/
│   ├── metrics.py     hit@k, recall@k, MRR, nDCG, mAP
│   ├── stats.py       Wilson, McNemar, Kendall, Spearman
│   ├── stability.py   질의 수에 따른 순위 안정성(부트스트랩)
│   ├── types.py       GoldenSet, JSONL 입출력
│   ├── matrix.py      실험 매트릭스 실행기, 요약
│   ├── golden.py      골든셋 2종 비교
│   ├── report.py      마크다운 표·그림
│   ├── retrieval/     BM25·Kiwi, dense, RRF·리랭커, OpenSearch, 설정 → 검색기
│   ├── llm.py         LLM 공급자
│   ├── synth.py       합성 질의 골든셋
│   ├── judge.py       답변 LLM 판정
│   └── cli.py         rag-eval 명령
├── configs/           검색 설정 매트릭스 예시
├── docs/diagrams/     아키텍처 다이어그램(HTML 원본, PNG)
├── tests/             손계산·해석해 테스트, 가짜 검색기·임베딩·LLM 연결 테스트
├── reports/           make report 결과
└── results/           비교·판정·안정성 원자료
```

## 구현하면서 신경 쓴 점

- 평균이 아니라 질의별 원자료를 남깁니다. 매트릭스 실행기는 질의 한 건마다 지표와 지연 시간을 기록하고, 평균·신뢰구간·대응 검정·순위 안정성·골든셋 비교를 모두 같은 `per_query.jsonl`에서 계산합니다. 분석을 추가하거나 다시 실행할 때 검색을 다시 돌리지 않아도 됩니다.
- 순위 안정성은 두 가지 불확실성을 나눠 표시합니다. 순위 전체 일치율과 1위 일치율의 Wilson 구간은 부트스트랩 반복 수에서 오는 추정 오차이고, 쌍 순서 일치율의 백분위 구간은 질의 표본이 바뀔 때 결과가 흔들리는 폭입니다. 두 구간의 의미가 달라 보고서에 따로 적습니다.
- 순위 안정성의 추출 방식을 명시합니다. 기본값인 복원추출은 "같은 모집단에서 n건짜리 평가셋을 새로 뽑았다면"을 뜻해 n이 전체 질의 수와 같아도 일치율이 1이 되지 않습니다. 비복원추출(`--subsample`)은 "지금 평가셋의 일부만 썼다면"을 뜻하며 n이 전체일 때 1이 됩니다. 평균이 같은 동점은 두 경우 모두 설정 이름순으로 깹니다.
- 순위 안정성 함수는 답을 해석적으로 아는 합성 점수 행렬로 테스트합니다. 예를 들어 설정 a가 질의 3건 중 2건에서 b를 이기는 행렬에서, 복원추출 3건일 때 a가 앞설 확률은 이항분포로 20/27이며 테스트는 부트스트랩 추정치가 이 값에 수렴하는지 확인합니다.
- 비율 지표의 신뢰구간은 Wilson 방식으로 계산합니다. 평가 질의가 150건 수준이면 hit@k가 0이나 1 근처일 때 정규근사 구간은 [0, 1]을 벗어나거나 폭이 0이 됩니다.
- 설정 간 비교에는 McNemar 대응 검정을 씁니다. 두 설정이 같은 질의를 푸는 구조라 독립 표본 검정이 맞지 않습니다. 불일치 쌍이 25건 미만이면 정확 이항검정, 이상이면 연속성 보정 카이제곱을 적용합니다.
- 지표 정의는 `metrics.py` 첫머리에 적어 고정했습니다. nDCG는 선형 이득(gain = grade), mAP는 trec_eval과 같이 전체 관련 문서 수로 나눕니다. 관련 문서가 없는 질의는 지표가 정의되지 않으므로 오류를 내고, 매트릭스 실행기는 이런 질의를 처음부터 제외합니다.
- 검색 구성 요소는 데이터셋을 모르게 만들었습니다. 색인 생성은 `docids`와 `texts`만 받고, 설정 파일의 모델 별칭마다 모델을 한 번만 적재해 dense·하이브리드·리랭커 설정이 공유합니다. 무거운 의존성(kiwipiepy, sentence-transformers, opensearch-py)은 해당 클래스를 만들 때만 불러오므로, 지표·통계만 쓰는 경우 설치가 필요하지 않습니다.
- 하이브리드 결합은 RRF를 썼습니다. BM25 점수와 코사인 유사도는 척도가 달라 정규화 방식에 따라 결과가 바뀌는데, 순위만 쓰는 RRF는 이 선택 자체를 없앱니다. 리랭커는 상위 후보만 다시 매기고 그 뒤 순위는 기반 검색 순서를 유지해, recall@k를 계산하는 검색 깊이가 줄지 않게 했습니다.
- LLM 호출은 `complete(system, prompt)` 프로토콜 뒤에 두었습니다. 공급자와 모델은 환경 변수로 고르고, 테스트는 가짜 공급자를 넣어 키 없이 실행됩니다. 판정 출력이 JSON으로 파싱되지 않은 건은 집계에서 빼고, 표본 수와 유효 건수를 보고서에 함께 기록합니다.

## 한계

- 실데이터 실험을 실행하지 않았습니다. 평가 표가 비어 있는 이유입니다.
- 순위 안정성 분석은 평가 질의가 모집단에서 독립적으로 뽑혔다고 가정합니다. 질의가 특정 시기나 주제에 몰려 있으면 실제 변동보다 작게 추정됩니다. 라벨 자체의 오류로 생기는 불확실성은 반영하지 않습니다.
- 순위 안정성은 전체 질의 수보다 큰 n으로 외삽하지 않습니다. 기본 격자는 전체 질의 수에서 끝납니다.
- 골든셋 비교에 쓰는 검색 설정이 7개 안팎이면 순위 상관의 통계적 검정력이 낮습니다. 설정을 늘리면 서로 비슷한 설정이 섞여 순위가 동점에 가까워집니다.
- LLM 판정을 사람 판정과 대조해 보정하지 않았습니다. 판정 모델의 편향(긴 답변 선호 등)이 결과에 그대로 들어갑니다.
- OpenSearch 어휘 백엔드는 작성만 했고 실행하지 않았습니다. 개발 환경에 Docker가 없었습니다. dense 임베더와 크로스인코더도 단위 테스트에서는 가짜 모델로만 실행했고, 실제 모델 적재는 확인하지 않았습니다.
- 하네스 단독으로는 데이터를 만들지 않습니다. `collect`·`index`·`eval` 타깃은 소비 프로젝트의 Makefile에 의존합니다.
- 사람이 직접 쓴 질의로 만든 세 번째 골든셋과 인용 법령 F1은 이번 범위에서 제외했습니다.

## 라이선스

코드는 [MIT](LICENSE) 라이선스를 따릅니다.
