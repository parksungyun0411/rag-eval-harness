# reports/

## 대표 실험: 한국어 RAG 리더보드 순위 안정성 (측정 완료)

`make eval` (= `rag-eval rankstab --data data/raw/rag_evaluation_result.csv --n-boot 10000 --seed 0`) 이
아래 파일을 한 번에 생성합니다. 숫자를 손으로 적지 않습니다. 같은 seed로 다시 실행하면 CSV가 바이트 단위로 같습니다.

| 파일 | 내용 |
|---|---|
| `REPORT.md` | 재현 정보(데이터셋 리비전·SHA-256·seed·반복 수·명령)와 모든 표·그림 |
| `rank_stability/correctness_matrix.csv` | 시스템 23 × 질문 300 정답 행렬(1 = O). 원본 데이터셋(MIT)에서 파생 |
| `rank_stability/leaderboard.csv` | 정확도, Wilson 95% 구간, 도메인별 정확도 |
| `rank_stability/curve_overall.csv` | 질문 수 n별 Kendall tau, 1위·top-3·top-5·순위 전체 유지 확률 |
| `rank_stability/curve_domain.csv` | 도메인별 같은 곡선 (n ≤ 60) |
| `rank_stability/adjacent_pairs.csv` | 인접 순위 쌍별 정확도 차, 불일치율, 95%로 가르는 데 필요한 n(격자·정확해·정규 근사) |
| `rank_stability/approx_check.csv` | 쌍 × n 마다 시뮬레이션·정확해·정규 근사 |
| `rank_stability/domain_best_share.csv` | 도메인별 각 시스템의 1위 확률 (n = 60 복원추출) |
| `rank_stability/domain_rank_corr.csv` | 도메인 순위와 전체 순위의 Kendall tau, 도메인 1위 |
| `rank_stability/*.png` | 위 CSV로 그린 그림 |

### 데이터 출처와 라이선스

`correctness_matrix.csv` 는 [allganize/RAG-Evaluation-Dataset-KO](https://huggingface.co/datasets/allganize/RAG-Evaluation-Dataset-KO)
(리비전 `35db50ff8739d78b11ee68b6f4ef04a95862a504`, MIT 라이선스)의 `rag_evaluation_result.csv` 에서
`*_ox` 열만 0/1로 옮긴 파생 자료입니다. 질문·정답·각 시스템의 답변 전문이 든 원본 CSV(약 5.7MB)는
저장소에 넣지 않고 `make collect` 가 고정 리비전에서 받아 SHA-256을 확인합니다.

## 보조 실험: 실제 질의 대 합성 질의 골든셋 (측정 전)

측정 전 — `make golden-eval` 실행 후 자동 생성

| 파일 | 내용 | 원자료 |
|---|---|---|
| `stability_<골든셋>_<지표>.md`, `.png` | 검색 설정 순위의 질문 수별 일치율 | `results/stability.json` |
| `golden_compare.md`, `golden_scatter_ndcg_at_10.png` | 골든셋 2종 설정별 점수·차이·순위 상관 | `results/golden.json` |
| `judge.md` | 답변 표본 LLM 판정 항목별 평균·통과율·Wilson 구간 | `results/judgments.jsonl` |
