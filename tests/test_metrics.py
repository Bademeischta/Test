import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "serve", Path(__file__).resolve().parents[1] / "scripts" / "serve.py"
)
serve = importlib.util.module_from_spec(spec)
spec.loader.exec_module(serve)
app = serve.app


def test_metrics_endpoint_returns_prometheus_format():
    with app.test_client() as client:
        resp = client.get("/metrics")
        assert resp.status_code == 200
        # Prometheus metrics output should include HELP or TYPE lines
        assert b"# TYPE" in resp.data or b"# HELP" in resp.data
