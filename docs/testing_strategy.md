# Benchmarking Strategy

For automated testing depth, implement two phases:

1. **Desktop GPU scaling first**: validate simulation consistency, throughput (`10s` trial under `1.0s`), and deterministic benchmark baselines.
2. **Mobile runtime latency second**: once correctness and scaling are stable, profile per-chip latency to enforce `<= 16.6ms` frame-to-output on representative consumer devices.

This ordering reduces false negatives from early mobile profiling and provides stronger optimization targets for downstream edge tuning.
