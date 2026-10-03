# 골든셋 2종 비교 실험 (P1-E).
# 데이터·인덱스·검색 실행은 소비 프로젝트(minwon-rag)가 맡고, 이 하네스는 비교·판정·보고서를 맡는다.
CONSUMER_DIR ?= ../minwon-rag
PER_QUERY ?= $(CONSUMER_DIR)/results/per_query.jsonl
ANSWERS ?= $(CONSUMER_DIR)/results/answers.jsonl
JUDGE_N ?= 50
RUN = uv run $(if $(wildcard .env),--env-file .env,)

.PHONY: setup test lint collect index eval report

setup:
	uv sync --extra llm

test:
	uv run pytest -q

lint:
	uv run ruff check .
	uv run ruff format --check .

collect:  ## 실제 질의 골든셋(수집·시점 분할) + LLM 합성 골든셋
	$(MAKE) -C $(CONSUMER_DIR) collect synth

index:
	$(MAKE) -C $(CONSUMER_DIR) index

eval:  ## 두 골든셋으로 검색 매트릭스 실행 → 점수 차·순위 상관 → 답변 표본 판정
	$(MAKE) -C $(CONSUMER_DIR) eval answers
	$(RUN) rag-eval compare --per-query $(PER_QUERY) --a real --b synthetic --out results/golden.json
	$(RUN) rag-eval judge --answers $(ANSWERS) --out results/judgments.jsonl --n $(JUDGE_N)

report:
	uv run rag-eval report --golden results/golden.json --judgments results/judgments.jsonl --out reports
