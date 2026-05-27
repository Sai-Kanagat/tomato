# BioMotion-AI Enterprise SDK

Production-ready initialization scaffold for a monocular visual-to-neural-control biomechanics pipeline.

## Repository Layout

- `.github/workflows/` CI validation for tests and benchmark gate checks
- `src/frontend/` ingestion, face-blurring, PromptHMR, PromptHMR-Vid stubs
- `src/kinetics/` diffusion kinetics and hybrid physics consistency stubs
- `src/simulation/` MuscleMimic, MuJoCo Warp CUDA skeletons, synergy/SVK mechanics
- `src/biomarkers/` CNN-BiGRU-Attention KCF estimator scaffold
- `docs/` architecture and benchmark strategy docs
- `tests/` pytest suite for latency/RMSE benchmark threshold checks
- `config.yaml` core performance and security gates
- `ARCHITECTURE_SPEC.md` full technical reference

## Benchmarking Priority Guidance

Start with **desktop GPU scaling benchmarks first** to validate physics + model correctness under high-throughput conditions, then profile and optimize **consumer mobile chip latency** against the 16.6ms runtime gate.

## Quick Start

```bash
python -m pip install -r requirements.txt
pytest -q
```
