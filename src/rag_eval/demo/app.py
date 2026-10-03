"""데모 서버.

    uv run --extra demo uvicorn rag_eval.demo.app:app --port 8000

- ``GET /api/leaderboard``: 리더보드(정확도, Wilson 구간, 도메인별 정확도)
- ``GET /api/stability``: 전체·도메인별 안정성 곡선, 인접 쌍 분석
- ``GET /api/flip?n=&gap=&discordance=&accuracy=``: 순위 역전 확률 계산기
- ``GET /``: 정적 화면(web/). 화면은 서버 없이도 ``web/data/results.json`` 만으로 동작한다.

결과 데이터는 ``rag-eval rankstab`` 이 만든 ``web/data/results.json`` 을 그대로 읽는다.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.staticfiles import StaticFiles

from rag_eval.rankstab import flip_probability

DEFAULT_WEB = Path(__file__).resolve().parents[3] / "web"


def _clean(obj: Any) -> Any:
    """JSON에 넣을 수 없는 inf/nan을 None으로."""
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_clean(v) for v in obj]
    return obj


def create_app(web_dir: str | Path | None = None) -> FastAPI:
    web = Path(web_dir or os.environ.get("RAG_EVAL_WEB_DIR") or DEFAULT_WEB)
    results_path = web / "data" / "results.json"
    app = FastAPI(title="RAG 리더보드 순위 안정성", version="0.1.0")
    cache: dict[str, Any] = {}

    def results() -> dict[str, Any]:
        if "data" not in cache:
            if not results_path.exists():
                raise HTTPException(503, "results.json 없음 — rag-eval rankstab 을 먼저 실행하세요")
            cache["data"] = json.loads(results_path.read_text(encoding="utf-8"))
        return cache["data"]

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        return {"ok": True, "results": results_path.exists()}

    @app.get("/api/leaderboard")
    def leaderboard() -> dict[str, Any]:
        r = results()
        return {"meta": r["meta"], "domains": r["domains"], "leaderboard": r["leaderboard"]}

    @app.get("/api/stability")
    def stability() -> dict[str, Any]:
        r = results()
        keys = ("meta", "overall_curve", "domain_curves", "domain_winners", "pairs")
        return {k: r[k] for k in keys}

    @app.get("/api/flip")
    def flip(
        n: int = Query(..., ge=1, le=100000),
        gap: float = Query(..., gt=0, le=1),
        discordance: float | None = Query(None, gt=0, le=1),
        accuracy: float | None = Query(None, ge=0, le=1),
    ) -> dict[str, Any]:
        try:
            return _clean(flip_probability(n, gap, discordance=discordance, accuracy=accuracy))
        except ValueError as e:
            raise HTTPException(422, str(e)) from e

    app.mount("/", StaticFiles(directory=web, html=True), name="web")
    return app


app = create_app()
