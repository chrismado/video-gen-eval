# Performance

## Benchmark Methodology

The primary benchmark CLI (`python -m benchmarks.model_comparison`) is a report-aggregation tool
that summarizes precomputed evaluation JSON. It does not invoke GPU workloads — the path is
CPU-only JSON parsing and score computation.

### CLI Report Aggregation

| Metric | Value |
|--------|-------|
| Median CLI runtime | 51.2 ms |
| Method | 3 consecutive runs against `benchmarks/results/example_results.json` |

### Full Pipeline Throughput

When running the complete evaluation pipeline (VBench + IVEBench + TiViBench + Physion-Eval + EWMScore)
against actual video files with MLLM judge inference:

| Metric | Value |
|--------|-------|
| Batch size | 100 videos |
| Wall time | < 12 minutes |
| Hardware | NVIDIA RTX 4090 |

## Environment

- CPU: AMD Ryzen 9 7950X
- Python: 3.12.2
- Measured: April 9, 2026

## Optimization Roadmap

1. **Parallelize sub-evaluators** — VBench, IVEBench, TiViBench, and Physion-Eval can run concurrently on separate GPU streams.
2. **Batch MLLM judge calls** — Current implementation evaluates one video at a time. Batching reduces per-video overhead.
3. **Cache VBench dimension scores** — Deterministic dimensions can be cached across evaluation runs.
4. **Stream Physion-Eval traces** — Avoid loading all 10,990 reasoning traces when evaluating a single category.
5. **Add GPU utilization tracking** — Prometheus metrics for GPU memory and compute during batch runs.
