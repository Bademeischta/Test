from flask import Flask
from chess_ai.policy_value_net import PolicyValueNet
from chess_ai.game_environment import GameEnvironment

app = Flask(__name__)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/metrics")
def metrics():
    # placeholder metric
    return {"games_played": 0}

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
