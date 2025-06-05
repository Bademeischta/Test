# SuperEngine Design

This document outlines the very early architecture of SuperEngine.
The engine keeps piece bitboards for each color and generates simple
pseudo legal moves. Evaluation currently uses a trivial material count.
The search implements a minimal alpha beta with a quiescence callback.
