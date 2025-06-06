# Chess AI Skeleton

Dieses Repository enthält ein Beispielprojekt für eine selbstlernende Schach-KI.
Es vereint eine minimalistische Python-Implementierung mit einigen
C++-Komponenten. Ziel ist es, die grundlegenden Bausteine zum Training einer
Engine nach Vorbild von AlphaZero oder NNUE-basierter Stockfish-Varianten zu
demonstrieren.

## Projektstruktur

* `chess_ai/` – Python-Modul mit allen Kernklassen
  * `game_environment.py` – Anbindung an `python-chess`, Board-Encoding und die
    Methode `is_quiet_move()` zum Erkennen ruhiger Züge
  * `policy_value_net.py` – Residual-Netzwerk für Politik- und Wertschätzung
  * `mcts.py` – Monte-Carlo-Tree-Search (MCTS) basierend auf den Netzwerkausgaben
  * `self_play.py` – Generierung von Trainingsdaten via Selbstspiel
  * `evaluation.py` – Duelle zweier Netze zum Leistungsvergleich
  * `replay_buffer.py` – einfacher Speicher für Spielzüge
  * `trainer.py` – Trainingsroutine für Policy und Value
  * `config.py` – zentrale Konfiguration (u.a. `FILTER_QUIET_POSITIONS`)
* `superengine/` – prototypische C++‑Engine mit Bitboards und NNUE‑Interface
* `tests/` – Pytest-Suite für Kernfunktionen
* `requirements.txt` – minimale Abhängigkeiten

## Installation

```bash
./scripts/install.sh
```

Dies installiert alle Python-Abhängigkeiten und baut standardmäßig auch die C++
Engine in `superengine/`.
Wer nur die Python-Funktionalität benötigt, kann stattdessen einfach

```bash
pip install -r requirements.txt
```
ausführen und den Engine-Build überspringen.

## Verwendung

Kurzes Selbstspiel und Evaluierung:

```bash
python -m chess_ai.self_play
python -m chess_ai.evaluation
```

Beispielausgabe von `python -m chess_ai.self_play`:

```text
Step 0: value=0.00
Step 1: value=-0.00
Step 2: value=0.00
Self-play finished successfully.
```

Alle Parameter befinden sich in `chess_ai/config.py`. Das Flag
`FILTER_QUIET_POSITIONS` bewirkt, dass nur Stellungen gespeichert werden, in
denen der gewählte Zug weder eine Figur schlägt noch ein Schach bietet. So
entsteht ein rauschärmerer Datensatz.

## Tests




Die Tests prüfen das Board-Encoding, MCTS, Replay Buffer und das Verhalten von
`is_quiet_move()`.

## Hintergrund und Ausblick

Die Codebasis orientiert sich an modernen Engines, die klassische Alpha-Beta-
Suche mit neuronalen Bewertungsfunktionen kombinieren. Ein möglicher Weg zur
Überlegenheit gegenüber Stockfish besteht aus:

1. Ausbau des NNUE/Policy-Netzes (mehr Filter/Schichten, ggf. GPU-Beschleunigung)
2. Nutzung hochwertiger Trainingsdaten aus ruhigen Positionen
3. Hybridansatz aus Alpha-Beta und MCTS, um taktische Tiefe und strategische
   Weitsicht zu verbinden
4. Intensive Parallelisierung sowie Optimierung von Speicherstrukturen

Dieses Repository liefert nur eine schlanke Demonstration – es fehlen beispielsweise
fortgeschrittene Trainingsinfrastruktur, umfangreiche Datensätze und echte
Stockfish-Integration. Dennoch bildet es ein kompaktes Fundament, um mit
selbstlernenden Schachprogrammen zu experimentieren.

## Roadmap und Checkliste

Nachfolgend eine schrittweise Roadmap mit allen noch offenen Arbeiten. Jeder
Punkt besitzt ein ankreuzbares Kästchen, sodass du den Fortschritt direkt im
Repository verfolgen kannst.

### 1. C++‑Engine (superengine)

#### 1.1 Move‑Generation & Legalität

- [ ] Sliding‑Moves (Bishop, Rook, Queen) implementieren
- [ ] `Position::in_check(Color)` komplettieren
- [ ] En‑Passant‑Logik ergänzen
- [ ] King‑Moves und Rochade umsetzen
- [ ] `generate_all_pseudo` und `generate_legal_moves` vereinen

#### 1.2 Suchkern (Alpha‑Beta, Quiescence, TT, LMR, Null‑Move)

- [ ] Transposition Table implementieren
- [ ] Quiescence Search einbauen
- [ ] Negamax/PVS mit LMR und Null‑Move
- [ ] Search‑Entrypoint samt Zeitmanagement
- [ ] Unit‑Tests (Perft, Mate‑in‑1)

#### 1.3 NNUE‑Integration

- [ ] `extract_features` (Half‑KP) implementieren
- [ ] Vorwärtsdurchlauf (Eval) fertigstellen
- [ ] Python‑Tools zum Trainieren und Quantisieren
- [ ] Catch2‑Tests für geladene Netze

#### 1.4 Tests & CI

- [ ] Perft‑Tests (Depth 1–5)
- [ ] clang‑format & Sanitizer in der CI
- [ ] Coverage‑Reporting (optional)

### 2. Python‑Komponente (chess_ai)

- [ ] TensorBoard/W&B Logging integrieren
- [ ] LMDB‑ReplayBuffer für große Datensätze
- [ ] Linting (flake8/black/isort) in der CI

### 3. CI/CD & Deployment

- [ ] Dockerfiles für C++‑ und Python‑Teil
- [ ] Kubernetes‑Manifeste (Deployment, CronJob)
- [ ] Monitoring und Health‑Checks
