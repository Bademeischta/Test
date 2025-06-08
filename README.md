# Chess AI Skeleton
[![Coverage](https://codecov.io/gh/example/chess-ai/graph/badge.svg)](https://codecov.io/gh/example/chess-ai)

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

### GPU Setup

Falls deine GPU von der offiziellen PyTorch-Distribution nicht unterstützt wird,
musst du PyTorch selbst kompilieren. Setze dazu beispielsweise

```bash
export TORCH_CUDA_ARCH_LIST="12.0;10.0;8.6;8.0;7.5;7.0;6.1"
git clone --recursive https://github.com/pytorch/pytorch
cd pytorch && python setup.py install
```

Danach erkennt `torch.cuda.is_available()` die RTX 5070 korrekt und das Training
nutzt die GPU.

### Gegen die KI spielen

Nach dem Training kannst du mit folgendem Skript gegen das neueste Netz
antreten:

```bash
python scripts/play_vs_ai.py --simulations 100
```

Per `--play-white` wählst du deine Farbe.

### C++-Engine nutzen

Die Verzeichnisse unter `superengine/` enthalten eine experimentelle
C++-Engine. Beim Aufruf von `./scripts/install.sh` wird sie nun
über ein kleines `Makefile` gebaut. Das erzeugte Binary findest du
anschließend im Ordner `superengine/make-build`.

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

## Trainer-Optimierung mit PyTorch Lightning

Das Policy- und NNUE-Training verwendet in den Beispielskripten weitgehend die
Standardwerte des Lightning-Trainers. Mit den folgenden Ansatzpunkten kannst du
den Trainingsablauf deutlich robuster und effizienter gestalten:

1. **Lernraten-Scheduler** – statt fester Lernrate einen Scheduler wie
   OneCycleLR oder CosineAnnealing einsetzen, um zu Beginn sanft anzulernen und
   am Ende besser zu konvergieren.
2. **Callbacks aktivieren** – ModelCheckpoint und EarlyStopping sichern die
   besten Modelle automatisch, ein Profiler deckt Bottlenecks auf.
3. **Gradient Accumulation & Clipping** – \`accumulate_grad_batches\` erlaubt
   gro\xC3\x9Fe effektive Batch-Gr\xC3\xB6\xC3\x9Fen ohne zus\xC3\xA4tzlichen GPU-Speicher,
   \`gradient_clip_val\` stabilisiert das Training.
4. **Mixed Precision & verteiltes Training** – sowohl f\xC3\xBCr Policy als auch
   NNUE \`precision="bf16-mixed"\` nutzen und bei mehreren GPUs DDP oder FSDP
   einschalten.
5. **Datenpipeline optimieren** – den LMDB-Buffer per DataModule einbinden und
   mit \`persistent_workers=True\` sowie erh\xC3\xB6htem \`prefetch_factor\` den
   Datenfluss beschleunigen.
6. **Hyperparameter-Suche** – \`Tuner\` zusammen mit Optuna oder Ray Tune
   automatisiert die besten Parameter finden lassen.
7. **Curriculum Learning** – im Selbstspiel zun\xC3\xA4chst gegen leichtere Gegner
   antreten und die Schwierigkeit progressiv steigern.
8. **Monitoring** – TensorBoard oder Weights & Biases f\xC3\xBCr Live-Logging der
   Metriken und Lernraten verwenden.

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
- [x] Move‑Ordering‑Heuristiken (MVV/LVA, Killer, History)

#### 1.3 NNUE‑Integration

 - [x] `extract_features` (Half‑KP) implementieren
 - [x] Vorwärtsdurchlauf (Eval) fertigstellen
- [ ] Python‑Tools zum Trainieren und Quantisieren
 - [x] Catch2‑Tests für geladene Netze

#### 1.4 Tests & CI

- [x] Perft‑Tests (Depth 1–5)
 - [x] clang‑format & Sanitizer in der CI
 - [x] Coverage‑Reporting (optional)

### 2. Python‑Komponente (chess_ai)

 - [x] TensorBoard/W&B Logging integrieren
- [x] LMDB‑ReplayBuffer für große Datensätze
 - [x] Linting (flake8/black/isort) in der CI

### 3. CI/CD & Deployment

 - [x] Dockerfiles für C++‑ und Python‑Teil
 - [x] Kubernetes‑Manifeste (Deployment, CronJob)
 - [x] Monitoring und Health‑Checks
