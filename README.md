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
pip install -r requirements.txt
```

Optional kann die C++-Engine in `superengine/` separat mit CMake gebaut werden.

## Verwendung

Kurzes Selbstspiel und Evaluierung:

```bash
python -m chess_ai.self_play
python -m chess_ai.evaluation
```

Alle Parameter befinden sich in `chess_ai/config.py`. Das Flag
`FILTER_QUIET_POSITIONS` bewirkt, dass nur Stellungen gespeichert werden, in
denen der gewählte Zug weder eine Figur schlägt noch ein Schach bietet. So
entsteht ein rauschärmerer Datensatz.

## Tests

```bash
PYTHONPATH=. pytest -q
```

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

## Ausführliche Roadmap

Die folgende Roadmap beschreibt Schritt für Schritt, wie dieses Projekt von einem kompakten Demonstrations-Code zu einer leistungsfähigen Engine ausgebaut werden kann.

### I. Projektstruktur & Code-Organisation
- **Monorepo-Layout**: Root-Verzeichnis nur mit `python/`, `cpp_engine/`, `infra/`, `benchmarks/`, `docs/` usw.
- **Branch-Strategie**: `main`, `develop` und Feature-Branches. Meilensteine per Git-Tag dokumentieren.

### II. Python-Komponente skalieren
1. **Data-Pipeline**
   - Self-Play-Loop mit Multi-Processing und GPU-Unterstützung
   - Feature-Vektoren per NumPy oder Numba erzeugen, optional in LMDB zwischenspeichern
   - Replay-Buffer als Ringbuffer oder LMDB-Shards, Prioritized Experience Replay möglich
2. **Modelldesign & Training**
   - Größeres NNUE (z.B. 768→1024→512→256→1) und ResNet-Policy-Netze
   - Mixed Precision, optional Transformer-Blöcke
   - Reinforcement-Learning-Schleife nach AlphaZero-Vorbild, Checkpoints und TensorBoard/W&B Logging
3. **Infrastruktur & Hardware**
   - Lokale Docker-Umgebung, Clusterbetrieb via Kubernetes und Ray
   - Gemeinsamer Dateispeicher (NFS/S3) und Monitoring

### III. C++-Engine erweitern
1. **Move-Generierung & Perft**
   - Bitboards mit Zobrist-Hashing, vollständige Pseudo- und Legal-Move-Generatoren
   - Perft-Tests (Depth 1–5) zum Abgleich mit Stockfish
2. **Suchkern**
   - Alpha-Beta/PVS mit Quiescence, Transposition Table, Late-Move-Reductions und Null-Move-Pruning
   - Mehrere Threads (Lazy SMP) für parallele Suche

### IV. Integration Python ↔ C++
- UCI-Modus für die Engine, Self-Play-Aufruf per Subprozess
- Python-Skripte orchestrieren Training, Selbstspiel und Evaluierung

### V. Evaluation & Benchmarking
- cutechess-cli für Elo-Benchmarks, SPRT zum Akzeptieren neuer Netze
- Langzeittracking der Ergebnisse im Verzeichnis `benchmarks/`

### VI. Deployment & Infrastruktur
- Dockerfiles für Python-Training und C++-Engine
- Kubernetes-Manifeste für CronJobs (Self-Play) und Deployment der Engine

### VII. Fortgeschrittene Features
- Opening-Book und Endgame-Tablebases
- Erweiterte Heuristiken (History-Heritstic, adaptive Null-Move)
- Experimente mit MuZero-ähnlichen Netzen oder Transformer-Architekturen

### VIII. Continuous Integration & Tests
- Python: flake8/black/isort sowie PyTest mit hoher Coverage
- C++: clang-format, -Wall/-Wextra, Sanitizer und Catch2-Tests
- Docker-Builds und End-to-End-Smoke-Tests

### IX. Dokumentation & Community
- Architektur-Dokumente unter `docs/` (Mermaid-Diagramme, Sphinx- oder Doxygen-Doku)
- CONTRIBUTING.md, Code of Conduct und Issue-/PR-Vorlagen

### X. Phasenplan (vereinfachte Übersicht)
| Phase | Ziele | Dauer |
| ----- | ----- | ----- |
| I | Repo-Reorganisation, Basis-Perft | 0–2 Monate |
| II | Grundlegende Engine + erstes Training | 2–4 Monate |
| III | GPU-PolicyNet und PVS/LMR | 4–6 Monate |
| IV | Cluster-Training, höhere Spielstärke | 6–9 Monate |
| V | Opening Book, optimierte Suche, Release | 9–12 Monate |
| VI | Fortgeschrittene Forschung (z.B. MuZero) | >12 Monate |

### XI. Checklisten
Eine Reihe von Checklisten hilft dabei, den Fortschritt zu messen (Perft-Werte, TT-Treffer, CI-Status, Trainingserfolge etc.). Diese Punkte sind im ausführlichen Roadmap-Dokument beschrieben und sollten nach und nach abgehakt werden.

