from flask import Flask, Response, request
from chess_ai.policy_value_net import PolicyValueNet
from chess_ai.game_environment import GameEnvironment
from prometheus_client import Counter, generate_latest, CONTENT_TYPE_LATEST

# Simple counter for Prometheus
REQUEST_COUNT = Counter("requests_total", "HTTP requests", ["endpoint"])

app = Flask(__name__)

@app.get("/health")
def health():
    REQUEST_COUNT.labels(endpoint="health").inc()
    return {"status": "ok"}

@app.get("/metrics")
def metrics():
    REQUEST_COUNT.labels(endpoint="metrics").inc()
    return Response(generate_latest(), mimetype=CONTENT_TYPE_LATEST)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
