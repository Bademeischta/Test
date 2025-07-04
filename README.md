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

Der Parameter ``Config.DEVICE`` wählt nun automatisch ``"cuda:0"`` aus, wenn
eine kompatible GPU verfügbar ist. Damit läuft das Training direkt auf der RTX
5070 oder anderen CUDA-Geräten.

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
C++-Engine. Zum Bauen wird das native [ONNX Runtime](https://onnxruntime.ai)
benötigt. Lade dazu das vorgefertigte C++ SDK herunter oder kompiliere es aus
den Quellen und setze die Umgebungsvariable `ONNXRuntime_DIR` auf den Ordner,
der die Datei `onnxruntimeConfig.cmake` enthält. Anschließend wird beim Aufruf
von `./scripts/install.sh` die Engine automatisch mittels `cmake` kompiliert.
Für einen manuellen Build wechselst du in das Verzeichnis `superengine` und
führst aus:

```bash
cd superengine
cmake -B build -S .

cmake --build build --config Debug --parallel $(nproc)
cd build
ctest -C Debug
```

Damit ist der Build abgeschlossen und die Unit-Tests werden ausgeführt. Bei
Multi-Config-Generatoren wie Visual Studio muss dabei die Konfiguration
angegeben werden. Obiges Beispiel nutzt `Debug`; bei einem Release-Build
verwendest du entsprechend `--config Release` und `ctest -C Release`.

### Superengine Kommandos

Nach erfolgreichem Build befindet sich das Binary `superengine` im
`build`-Ordner. Es kennt zwei Modi:

1. **UCI-Modus** – ohne weitere Argumente gestartet. Unterstützte Befehle:
   - `uci` – meldet den Namen der Engine und bestätigt mit `uciok`.
   - `isready` – gibt `readyok` zurück.
   - `position startpos [moves ...]` – setzt die Startposition und spielt
     optional Züge ab.
   - `position fen <FEN>` – lädt eine Stellung aus FEN.
   - `go` – berechnet einen Zug und antwortet mit `bestmove`.
   - `quit` – beendet die Engine.

  Beispielaufruf:

```bash
./build/superengine
uci
isready
position startpos
go
quit
```

2. **Selfplay-Modus** – zum Erzeugen von Trainingspartien:

```bash
./build/superengine selfplay --seed <Zahl> --net <pfad.zur.nnue> --games <Anzahl>
```

Die erzeugten PGNs landen im aktuellen Verzeichnis.

Weitere Helferskripte in `superengine/scripts/`:

```bash
python superengine/scripts/prepare_data.py <pgn1> [pgn2 ...]
python superengine/scripts/train_policy.py
python superengine/scripts/train_nnue.py
python superengine/scripts/quantize_nnue.py <model.pth> <output.nnue>

cmake --build build --parallel <Anzahl-der-Jobs>

```


cmake --build build --config Debug --parallel <Anzahl-der-Jobs>

```

Danach lassen sich in `superengine/build` alle Unit-Tests per
`ctest -C Debug` starten. Wird die Engine im Release-Modus gebaut,
ersetzt du `Debug` entsprechend durch `Release`.

### Superengine Kommandos

Nach erfolgreichem Build befindet sich das Binary `superengine` im
`build`-Ordner. Es kennt zwei Modi:

1. **UCI-Modus** – ohne weitere Argumente gestartet. Unterstützte Befehle:
   - `uci` – meldet den Namen der Engine und bestätigt mit `uciok`.
   - `isready` – gibt `readyok` zurück.
   - `position startpos [moves ...]` – setzt die Startposition und spielt
     optional Züge ab.
   - `position fen <FEN>` – lädt eine Stellung aus FEN.
   - `go` – berechnet einen Zug und antwortet mit `bestmove`.
   - `quit` – beendet die Engine.

2. **Selfplay-Modus** – zum Erzeugen von Trainingspartien:

```bash
./superengine selfplay --seed <Zahl> --net <pfad.zur.nnue> --games <Anzahl>
```

Die erzeugten PGNs landen im aktuellen Verzeichnis.

Weitere Helferskripte in `superengine/scripts/`:

- `prepare_data.py <pgn1> [pgn2 ...]` – extrahiert FENs und Bewertungen.
- `train_policy.py` – trainiert das Policy/Value-Netz auf den PGNs in `games/`.
- `train_nnue.py` – Beispiel zum Trainieren eines kleinen NNUE.
- `quantize_nnue.py <model.pth> <output.nnue>` – wandelt PyTorch-Gewichte um.


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

- [x] Bitboards & FEN‑Parser
- [x] Pawn‑Moves
- [x] Knight‑Moves
- [x] Sliding‑Moves (Rook, Bishop, Queen)
- [x] King‑Moves + Castling
- [x] En Passant
- [x] Legal‑Move‑Filter mit `in_check()`
- [x] Quiescence‑Search
- [x] Alpha‑Beta / PVS Search
- [x] Transposition Table (TT)
- [x] Late‑Move‑Reductions (LMR)
- [x] Null‑Move Pruning
- [x] Multi‑Threading (Lazy SMP / Shared TT)
- [x] NNUE‑Feature‑Extraction
- [x] NNUE‑Forward (Eval)
- [x] Python‑Script zum Trainieren/Quantisieren von .nnue
- [x] Perft‑Tests (Depth 1–5)
- [x] clang‑format / Sanitizer / strikte Flags

### 2. Python‑Komponente (chess_ai)

- [x] State‑Encoding (18×8×8)
- [x] `is_quiet_move()`
- [x] ActionIndex (64×64×5)
- [x] Policy/Value‑Net (ResNetPV)
- [x] MCTS mit UCT, VirtualLoss, Policy‑Guidance
- [x] ReplayBuffer & Trainer
- [x] Self‑Play & Evaluation
- [x] TensorBoard / W&B Logging
- [x] LMDB/HDF5‑ReplayBuffer für große Datensätze
- [x] pytest‑CI + Lint/Format (flake8, black, isort)

### 3. CI/CD & Deployment

- [x] Python‑CI: pytest + flake8 + black + isort
- [x] C++‑CI: cmake + ctest + clang‑format + Sanitizer
- [x] Dockerfiles (C++ & Python)
- [x] Kubernetes‑Manifeste (Deployment, CronJob)
- [x] Monitoring (Prometheus, Grafana)

#### Monitoring-Setup

Prometheus liest die Metriken der Flask-App über `/metrics` aus. Die nötigen
Kubernetes-Ressourcen befinden sich in `k8s/prometheus.yaml` und
`k8s/grafana.yaml`.

## Version History

See [CHANGELOG.md](CHANGELOG.md) for a detailed history.
