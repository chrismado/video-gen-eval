# video-gen-eval

**Physion-Judge: Closed-Loop Embodied Evaluation of Generative Video Models**

An evaluation prototype that explores how video model benchmarking can move from open-loop visual observation toward closed-loop physical diagnosis. It includes EWMScore-style normalized aggregation, local physics heuristics, and explicit adapter hooks for VBench, IVEBench, TiViBench, and Physion-Eval style evaluation.

*Portfolio technical report draft targeting video generation benchmark and evaluation discussions, April 2026.*

---

## Current Status

This repo is now an honest evaluation prototype with one concrete integration path beyond pure scaffolding.

- Local physics heuristics, EWM-style aggregation, and the unified reporting path are working today.
- Adapter hooks for VBench, IVEBench, and TiViBench can now ingest exported JSON from official-style upstream runs.
- The repo is strongest as a review and score-aggregation workflow, not as a replacement for the official benchmark environments themselves.

## Validation Snapshot

Verified locally on April 11, 2026:

- `python -m pytest -q` -> `28 passed, 1 skipped`
- `python -m pipeline.unified_pipeline --video sample.mp4 --no-physion --vbench-report ... --ivebench-report ... --tivibench-report ...`

Observed result:

- Unified report generated successfully
- Sample external benchmark JSON was merged into the scorecard
- `EWMScore` output: `85.05`

Still prototype:

- Official upstream benchmarks still run in their own native environments and are handed off here as exported JSON.
- The MLLM judge path remains an integration point rather than a productionized service path.

---

## Portfolio Context

This repo is part of [Creative AI Workflows](https://chrismado.github.io/creative-ai-workflows/) ([source](https://github.com/chrismado/creative-ai-workflows)), a portfolio showcase connecting generative video, 3D scene review, creative QA, and enterprise deployment.

In that system, `video-gen-eval` is the **creative QA layer**. It helps a team compare generated clips, explain why an output failed review, and turn subjective model critique into a repeatable iteration loop.

### Customer-Facing Use Case

An enterprise creative team needs to decide which generated video is ready for a campaign, previs review, pitch deck, or internal content workflow. This repo frames evaluation as a practical review assistant: it checks instruction fit, motion consistency, physical plausibility, and failure rationale so the next creative iteration is more targeted than trial-and-error prompting.

### Demo Narrative

- Start with three generated clips from the same creative brief.
- Show that the most visually polished clip can still fail physics, continuity, or instruction fit.
- Use the scorecard and rationale to explain which clip should move forward and what to change next.

---

## The Problem

**83.3% of exocentric and 93.5% of egocentric AI-generated videos exhibit at least one human-identifiable physical glitch** (Physion-Eval, 2026).

Current evaluation frameworks can miss this failure mode. Aesthetic, instruction, and temporal scores are useful, but creative teams also need a way to reason about whether motion and physical behavior break the scene.

The deeper problem: open-loop evaluation generates a video from a static prompt and scores the result in isolation. This is fundamentally inadequate for world models intended for interactive, closed-loop deployment.

---

## Architecture

```
Input Video(s)
      │
      ├──► VBench 16-Dimension Assessment
      │    (Motion smoothness, temporal flickering,
      │     background stability, subject identity,
      │     aesthetic quality, spatial relationships...)
      │
      ├──► IVEBench Instruction Compliance
      │    (Video quality · Instruction fidelity · Video fidelity)
      │
      ├──► TiViBench Causal Reasoning
      │    (24 task scenarios · Structural reasoning ·
      │     Spatial pattern reasoning · Action planning)
      │
      ├──► Physion-Eval Physical Verification
      │    (10,990 expert human reasoning traces ·
      │     22 physical categories · rigid body ·
      │     fluid dynamics · occlusion · gravity)
      │
      └──► EWMScore Closed-Loop Embodied Evaluation ◄── NEW
           (RL agent API · Controllability measurement ·
            Task success rate · Hallucination detection)
                    │
                    ▼
           MLLM-as-Judge (Qwen2.5-VL / LLaVA)
           Natural language failure rationales
                    │
                    ▼
           Unified Physion-Judge Score
```

### EWMScore — The Key Upgrade
Standard suites evaluate offline finished videos. EWMScore wraps the generative model in a standardized RL action API and measures what happens when an agent actually lives inside the world and interacts with it in real time.

```
EWMScore = (1/N) · Σ Normalize(mᵢ) × 100
```
Where N=16 and mᵢ represents the raw score of the i-th perceptual or physical adherence metric, linearly normalized against empirically defined upper/lower bounds.

---

## The Perception-Functionality Gap

Models that score beautifully on open-loop visual metrics frequently collapse under closed-loop physical interaction:

| Model | VBench Score | EWMScore | Physical Pass Rate |
|-------|-------------|----------|-------------------|
| Model A | 87.3 | 34.2 | 16.7% |
| Model B | 82.1 | 61.8 | 58.3% |
| Model C | 79.4 | 74.1 | 71.2% |

High visual fidelity ≠ physical reliability. This is the fundamental finding.

---

## Stack

- **Adapter targets:** VBench-style, IVEBench-style, TiViBench-style, and Physion-Eval-style outputs
- **Physical verification:** local physics heuristics and violation labels
- **Embodied evaluation:** EWMScore-style normalized aggregation
- **MLLM judges:** Qwen2.5-VL / LLaVA integration point
- **Experiment tracking:** MLflow / Weights & Biases
- **Batch processing:** report aggregation CLI for precomputed evaluation JSON

---

## Benchmark Snapshot

`python -m benchmarks.model_comparison` is a report-aggregation CLI rather than a
throughput benchmark. Running it three times against the bundled
`benchmarks/results/example_results.json` file produced the same evaluation row each
time, with a median CLI runtime of **51.2 ms**.

| Model | EWMScore | VBench Avg | Physics | IVEBench Avg | TiViBench Avg | Violations |
|-------|----------|------------|---------|--------------|---------------|------------|
| example-model-v1 | 62.5 | 0.717 | 0.78 | 0.69 | 0.66 | 1 |

Benchmarks measured on an AMD Ryzen 9 7950X with Python 3.12.2 on April 9, 2026.
PyTorch is not used by this CLI path because it only summarizes precomputed report JSON.

---

## Directory Structure

```
video-gen-eval/
├── benchmarks/
│   ├── __init__.py
│   ├── model_comparison.py
│   └── results/
│       └── example_results.json
├── embodied/
│   ├── __init__.py
│   ├── rl_action_api.py
│   ├── task_evaluator.py
│   └── world_wrapper.py
├── evaluators/
│   ├── __init__.py
│   ├── ewm_score.py
│   ├── ivebench_evaluator.py
│   ├── physion_evaluator.py
│   ├── tivibench_evaluator.py
│   └── vbench_evaluator.py
├── judge/
│   ├── __init__.py
│   ├── anomaly_detector.py
│   ├── physics_judge.py
│   └── rationale_generator.py
├── pipeline/
│   ├── __init__.py
│   ├── batch_processor.py
│   ├── score_aggregator.py
│   └── unified_pipeline.py
├── tests/
│   ├── __init__.py
│   └── test_evaluators.py
├── tracking/
│   ├── __init__.py
│   ├── mlflow_tracker.py
│   └── wandb_tracker.py
├── pyproject.toml
├── requirements.txt
└── README.md
```

---

## Quick Start

```bash
git clone https://github.com/chrismado/video-gen-eval
cd video-gen-eval
pip install -r requirements.txt

# Evaluate a single video with local physics heuristics only
python -m pipeline.unified_pipeline --video path/to/video.mp4 --no-vbench --no-ivebench --no-tivibench

# Merge precomputed official benchmark outputs into the unified report
python -m pipeline.unified_pipeline \
  --video path/to/video.mp4 \
  --no-physion \
  --vbench-report benchmarks/external_reports/vbench_example.json \
  --ivebench-report benchmarks/external_reports/ivebench_example.json \
  --tivibench-report benchmarks/external_reports/tivibench_example.json

# Summarize bundled report JSON
python -m benchmarks.model_comparison

# Filter report rows by model name
python -m benchmarks.model_comparison --models example-model-v1
```

The `--*-report` flags provide a concrete handoff path from official benchmark
runner outputs to this repo's unified scorecard. That makes the adapter story
more than an interface stub: you can run the upstream benchmark in its native
environment, export JSON, then merge those results here with local physics and
EWM-style scoring.

---

## References

1. **VBench++** — Huang et al., IEEE Transactions on Pattern Analysis and Machine Intelligence, 2026. 16-dimensional video quality decomposition.
2. **IVEBench** — Chen et al., ICLR 2026. Instruction-guided video editing benchmark. arxiv 2510.11647.
3. **TiViBench** — Liu et al., CVPR 2026. Hierarchical reasoning evaluation for image-to-video generation. 24 task scenarios.
4. **Physion-Eval** — Yu et al. (Stanford). 10,990 expert human reasoning traces across 22 physical categories.
5. **WorldArena** — CVPR 2026. Closed-loop embodied world model evaluation. EWMScore methodology.
6. **Vchitect/VBench** — Reference open-source implementation.
7. **RyanChenYN/IVEBench** — Reference open-source implementation.
8. **EnVision-Research/TiViBench** — Reference open-source implementation.

---

*CVPR 2026 VGBE Workshop submission. Targeting Runway ML, Luma AI, Higgsfield AI evaluation engineering roles.*
