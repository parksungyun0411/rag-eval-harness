# reports/

측정 전 — `make eval` 실행 후 자동 생성

`make report` 가 `results/` 에 있는 원자료만 골라 아래 파일을 이 디렉터리에 씁니다. 손으로 숫자를 적지 않습니다.

| 파일 | 내용 | 원자료 |
|---|---|---|
| `stability_<골든셋>_<지표>.md` | 질의 수 n별 설정 순위 일치율(exact·top1, Wilson 구간)과 쌍별 일치율(백분위 구간) | `results/stability.json` |
| `stability_<골든셋>_<지표>.png` | 위 일치율의 n에 따른 곡선 | 같음 |
| `golden_compare.md` | 골든셋 2종(실제 질의 대 합성 질의) 설정별 점수·차이·순위 상관 | `results/golden.json` |
| `golden_scatter_ndcg_at_10.png` | 설정별 nDCG@10 산점도(x: 실제, y: 합성) | 같음 |
| `judge.md` | 답변 표본 LLM 판정 항목별 평균·통과율·Wilson 구간 | `results/judgments.jsonl` |
