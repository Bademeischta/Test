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
Engine in `superengine/`. Zusätzlich wird das Projekt selbst im
Entwicklungsmodus installiert.
Wer nur die Python-Funktionalität benötigt, kann stattdessen einfach

```bash
pip install -r requirements.txt
pip install -e .
```
ausführen und den Engine-Build überspringen. Das zusätzliche `pip install -e .`
stellt sicher, dass das `chess_ai`-Paket korrekt eingebunden ist.

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

### Training

Ein komplettes Trainingsspielzeug befindet sich unter `scripts/train.py`. Die
Länge des Trainings kannst du über die Kommandozeilenargumente selbst
bestimmen:

```bash
python scripts/train.py --games 5 --epochs 3 --simulations 50
```

* `--games` legt fest, wie viele Selbstpartien generiert werden.
* `--epochs` gibt die Anzahl der Trainingsdurchläufe über den Buffer an.
* `--simulations` steuert die MCTS-Suchtiefe pro Zug.

Während des Trainings erscheinen nun kurze Statistiken zu jedem Epoch.

### Gegen die KI spielen

Nach dem Training kannst du mit folgendem Skript gegen das neueste Netz
antreten:

```bash
python scripts/play_vs_ai.py --simulations 100
```

Per `--play-white` wählst du deine Farbe.

### C++-Engine nutzen

Die Verzeichnisse unter `superengine/` enthalten eine experimentelle
C++-Engine. Sie wird beim Aufruf von `./scripts/install.sh` automatisch mit
`cmake` und `make` gebaut. Anschließend kannst du in `superengine/build` die
erzeugten Testprogramme ausführen, z.B.:

```bash
cd superengine/build
ctest
```

Die Datei `main.cpp` demonstriert zudem eine einfache UCI-Schnittstelle, die du
nach einem eigenen Build als Einstieg für weitere Experimente verwenden kannst.

Alle Parameter befinden sich in `chess_ai/config.py`. Das Flag
`FILTER_QUIET_POSITIONS` bewirkt, dass nur Stellungen gespeichert werden, in
denen der gewählte Zug weder eine Figur schlägt noch ein Schach bietet. So
entsteht ein rauschärmerer Datensatz.

## Tests




Die Tests prüfen das Board-Encoding, MCTS, Replay Buffer und das Verhalten von
`is_quiet_move()`.

## Verbesserungen in dieser Version

* Caching der Netzwerk-Auswertungen im MCTS für schnellere Simulationen
* Prioritized Experience Replay im ReplayBuffer
* Gradienten-Clipping im Trainer
* Temperaturgesteuerte Zugauswahl beim Selbstspiel
* Helfer zum Laden von Checkpoints im NetworkManager
* LMDB-basierter ReplayBuffer f\xC3\xBCr gro\xC3\x9Fe Datens\xC3\xA4tze

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

- [x] Sliding‑Moves (Bishop, Rook, Queen) implementieren
- [x] `Position::in_check(Color)` komplettieren
- [x] En‑Passant‑Logik ergänzen
- [x] King‑Moves und Rochade umsetzen
- [x] `generate_all_pseudo` und `generate_legal_moves` vereinen -> `generate_moves`

#### 1.2 Suchkern (Alpha‑Beta, Quiescence, TT, LMR, Null‑Move)

- [x] Transposition Table implementieren
- [x] Quiescence Search einbauen
- [x] Negamax/PVS mit LMR und Null‑Move
- [x] Search‑Entrypoint samt Zeitmanagement
- [x] Unit‑Tests (Perft, Mate‑in‑1)

#### 1.3 NNUE‑Integration

- [ ] `extract_features` (Half‑KP) implementieren
- [ ] Vorwärtsdurchlauf (Eval) fertigstellen
- [ ] Python‑Tools zum Trainieren und Quantisieren
- [ ] Catch2‑Tests für geladene Netze

#### 1.4 Tests & CI

- [x] Perft‑Tests (Depth 1–5)
- [ ] clang‑format & Sanitizer in der CI
- [ ] Coverage‑Reporting (optional)

### 2. Python‑Komponente (chess_ai)

- [ ] TensorBoard/W&B Logging integrieren
- [x] LMDB‑ReplayBuffer für große Datensätze
- [ ] Linting (flake8/black/isort) in der CI

### 3. CI/CD & Deployment

- [ ] Dockerfiles für C++‑ und Python‑Teil
- [ ] Kubernetes‑Manifeste (Deployment, CronJob)
- [ ] Monitoring und Health‑Checks
