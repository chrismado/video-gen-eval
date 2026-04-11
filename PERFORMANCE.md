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

### Full Pipeline Throughput Roadmap

The local measured path is report aggregation only. End-to-end GPU throughput
depends on which official benchmark adapters and MLLM judge backends are wired.
Use the table below as a measurement plan rather than a published result:

| Metric | Value |
|--------|-------|
| Batch size | TBD |
| Wall time | TBD |
| Hardware | Record per run |

## Environment

- CPU: AMD Ryzen 9 7950X
- Python: 3.12.2
- Measured: April 9, 2026

## Optimization Roadmap

1. **Wire official benchmark adapters** - Keep VBench, IVEBench, and TiViBench integration code separate from the local scoring framework.
2. **Parallelize sub-evaluators** - Run external adapters and local physics checks independently where dependencies allow.
3. **Batch MLLM judge calls** - Current implementation evaluates one video at a time. Batching reduces per-video overhead.
4. **Cache deterministic dimension scores** - Reuse scores across evaluation runs when the video and evaluator version match.
5. **Add GPU utilization tracking** - Prometheus metrics for GPU memory and compute during batch runs.
