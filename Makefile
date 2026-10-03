# 하네스 자체: setup / test / lint
# 실험: 데이터·색인·검색 실행은 소비 프로젝트(CONSUMER_DIR)가 맡고,
#       이 하네스는 순위 안정성·골든셋 비교·판정·보고서를 맡는다.
# 소비 프로젝트 규약: make collect / synth / index / eval / answers 타깃,
#                     results/per_query.jsonl, results/answers.jsonl 산출
CONSUMER_DIR ?= ../minwon-rag
PER_QUERY ?= $(CONSUMER_DIR)/results/per_query.jsonl
ANSWERS ?= $(CONSUMER_DIR)/results/answers.jsonl
JUDGE_N ?= 50
STABILITY_GOLDEN ?= real
STABILITY_METRIC ?= ndcg@10
STABILITY_BOOT ?= 1000
RUN = uv run $(if $(wildcard .env),--env-file .env,)

.PHONY: setup test lint collect index eval report

setup:
	uv sync --extra llm --extra kiwi --extra models

test:
	uv run pytest -q

lint:
	uv run ruff check .
	uv run ruff format --check .

collect:  ## 실제 질의 골든셋(수집·분할) + LLM 합성 골든셋
	$(MAKE) -C $(CONSUMER_DIR) collect synth

index:
	$(MAKE) -C $(CONSUMER_DIR) index

eval:  ## 매트릭스 → 순위 안정성 → 골든셋 점수 차·순위 상관 → 답변 표본 판정
	$(MAKE) -C $(CONSUMER_DIR) eval answers
	$(RUN) rag-eval stability --per-query $(PER_QUERY) --golden $(STABILITY_GOLDEN) \
		--metric $(STABILITY_METRIC) --n-boot $(STABILITY_BOOT) --out results/stability.json
	$(RUN) rag-eval compare --per-query $(PER_QUERY) --a real --b synthetic --out results/golden.json
	$(RUN) rag-eval judge --answers $(ANSWERS) --out results/judgments.jsonl --n $(JUDGE_N)

report:  ## results/ 에 있는 원자료만 골라 reports/ 생성
	uv run rag-eval report --out reports \
		$(if $(wildcard results/stability.json),--stability results/stability.json,) \
		$(if $(wildcard results/golden.json),--golden results/golden.json,) \
		$(if $(wildcard results/judgments.jsonl),--judgments results/judgments.jsonl,)
