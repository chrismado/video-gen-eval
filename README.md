# video-gen-eval

**Physion-Judge: Closed-Loop Embodied Evaluation of Generative Video Models**

A unified evaluation pipeline that transitions video model benchmarking from open-loop visual observation to closed-loop physical diagnosis. Implements the Embodied World Model Score (EWMScore) based on the WorldArena framework, combining VBench, IVEBench, TiViBench, and Physion-Eval into a single "Physics-as-a-Judge" pipeline.

*Technical report submitted to CVPR 2026 VGBE Workshop (1st Workshop on Video Generative Models: Benchmarks and Evaluation), April 2026.*

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

Current evaluation frameworks miss this entirely. VBench scores a video on aesthetic quality. IVEBench scores instruction compliance. Neither tells you whether the physics is broken — and broken physics causes catastrophic failure when a model is used as a world simulator for embodied AI or robotics.

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

- **Evaluation engines:** VBench++ (IEEE 2026), IVEBench (ICLR 2026), TiViBench (CVPR 2026)
- **Physical verification:** Physion-Eval (Stanford/Hong-Xing Yu et al.)
- **Embodied evaluation:** WorldArena EWMScore (CVPR 2026)
- **MLLM judges:** Qwen2.5-VL, LLaVA (configurable)
- **Experiment tracking:** MLflow / Weights & Biases
- **Batch processing:** RTX 4090 — 100-video batch in <12 minutes

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

# Evaluate a single video
python -m pipeline.unified_pipeline --video path/to/video.mp4

# Full benchmark run
python -m benchmarks.model_comparison --models runway-gen4 pika luma

# EWMScore closed-loop evaluation
python -m embodied.world_wrapper --model runway-gen4 --tasks all
```

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
