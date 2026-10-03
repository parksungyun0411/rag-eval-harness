# 대표 실험: 한국어 RAG 리더보드 순위 안정성 (allganize/RAG-Evaluation-Dataset-KO, 키 불필요)
#   collect → index → eval → report 순서. eval이 REPORT.md·CSV·그림·web/data/results.json을 함께 만든다.
# 보조 실험: 실제 질의 대 합성 질의 골든셋 비교 (golden-* 타깃, 소비 프로젝트 CONSUMER_DIR 필요)
DATA ?= data/raw/rag_evaluation_result.csv
N_BOOT ?= 10000
SEED ?= 0
PORT ?= 8000
RUN = uv run $(if $(wildcard .env),--env-file .env,)

.PHONY: setup test lint collect index eval report serve \
	golden-collect golden-index golden-eval golden-report

setup:
	uv sync --extra kiwi --extra demo --extra llm --extra models

test:
	uv run pytest -q

lint:
	uv run ruff check .
	uv run ruff format --check .

collect:  ## 고정 리비전 CSV 다운로드 + SHA-256 확인 (원본은 커밋하지 않음)
	uv run rag-eval rankstab-fetch --out $(DATA)

index: collect  ## 시스템 × 질문 정답 행렬
	uv run python -c "from rag_eval.datasets import allganize_ko as a; \
		a.load('$(DATA)').to_csv('reports/rank_stability/correctness_matrix.csv')"

eval: index  ## 부트스트랩·정확해·정규 근사 → reports/REPORT.md, reports/rank_stability/, web/data/
	uv run rag-eval rankstab --data $(DATA) --n-boot $(N_BOOT) --seed $(SEED)

report:  ## REPORT.md는 eval이 함께 생성한다. 여기서는 보조 실험 결과가 있으면 보고서로 만든다
	@test -f reports/REPORT.md || (echo "reports/REPORT.md 없음 — make eval 먼저" && exit 1)
	$(if $(wildcard results/golden.json)$(wildcard results/judgments.jsonl)$(wildcard results/stability.json), \
		uv run rag-eval report --out reports \
		$(if $(wildcard results/stability.json),--stability results/stability.json,) \
		$(if $(wildcard results/golden.json),--golden results/golden.json,) \
		$(if $(wildcard results/judgments.jsonl),--judgments results/judgments.jsonl,), \
		@echo "보조 실험 결과 없음 — reports/REPORT.md 만 있음")

serve:  ## 데모 화면 + API (http://localhost:$(PORT))
	uv run uvicorn rag_eval.demo.app:app --port $(PORT)

# ---------------------------------------------------------------- 보조 실험
# 데이터·색인·검색 실행은 소비 프로젝트가 맡는다.
# 소비 프로젝트 규약: make collect / synth / index / eval / answers 타깃,
#                     results/per_query.jsonl, results/answers.jsonl 산출
CONSUMER_DIR ?= ../minwon-rag
PER_QUERY ?= $(CONSUMER_DIR)/results/per_query.jsonl
ANSWERS ?= $(CONSUMER_DIR)/results/answers.jsonl
JUDGE_N ?= 50

golden-collect:
	$(MAKE) -C $(CONSUMER_DIR) collect synth

golden-index:
	$(MAKE) -C $(CONSUMER_DIR) index

golden-eval:
	$(MAKE) -C $(CONSUMER_DIR) eval answers
	$(RUN) rag-eval stability --per-query $(PER_QUERY) --golden real --metric ndcg@10 \
		--out results/stability.json
	$(RUN) rag-eval compare --per-query $(PER_QUERY) --a real --b synthetic --out results/golden.json
	$(RUN) rag-eval judge --answers $(ANSWERS) --out results/judgments.jsonl --n $(JUDGE_N)

golden-report: report
