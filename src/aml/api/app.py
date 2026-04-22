from __future__ import annotations

import logging
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import PlainTextResponse

from aml.contracts import AlertResolution, TransactionEvent
from aml.runtime.bootstrap import RuntimeContext, build_runtime

logger = logging.getLogger(__name__)


def create_app(runtime: RuntimeContext | None = None) -> FastAPI:
    runtime = runtime or build_runtime()
    app = FastAPI(title="AML E2E Product", version="1.0.0")
    app.state.runtime = runtime

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/ready")
    def ready() -> dict[str, str]:
        bundle_path = runtime.settings.inference_bundle_path
        if not Path(bundle_path).exists():
            raise HTTPException(status_code=503, detail="Inference bundle is missing")
        try:
            runtime.repository.monitoring_summary()
        except Exception as exc:
            raise HTTPException(status_code=503, detail=f"Operational backend is not ready: {exc}") from exc
        return {"status": "ready"}

    @app.get("/metrics", response_class=PlainTextResponse)
    def metrics() -> str:
        return runtime.metrics.render_prometheus()

    @app.post("/score")
    def score(event: TransactionEvent):
        try:
            return runtime.score_use_case.execute(event)
        except Exception as exc:
            runtime.metrics.record_error()
            runtime.repository.save_dlq(event.model_dump(mode="json"), str(exc))
            logger.exception("Scoring failed", extra={"event_id": event.event_id, "status": "error"})
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.get("/events/{event_id}")
    def get_event(event_id: str):
        bundle = runtime.repository.fetch_event_bundle(event_id)
        if bundle is None:
            raise HTTPException(status_code=404, detail="Event not found")
        return bundle

    @app.get("/alerts")
    def list_alerts(status: str | None = Query(default=None), limit: int = Query(default=50, le=500)):
        return runtime.repository.list_alerts(status=status, limit=limit)

    @app.post("/alerts/{alert_id}/resolution")
    def resolve_alert(alert_id: str, resolution: AlertResolution):
        return runtime.resolve_alert_use_case.execute(alert_id, resolution)

    @app.get("/monitoring/summary")
    def monitoring_summary():
        return runtime.repository.monitoring_summary()

    return app


app = create_app()


def main() -> None:
    import uvicorn

    uvicorn.run("aml.api.app:app", host="0.0.0.0", port=8000, reload=False)


if __name__ == "__main__":
    main()
