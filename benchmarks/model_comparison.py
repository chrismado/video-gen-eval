"""
Multi-model benchmark results table.

Prints a formatted comparison table of model evaluation results.
Does NOT fabricate benchmark numbers -- uses placeholder structure
that is populated from actual evaluation runs.

Usage:
    python -m benchmarks.model_comparison
    python -m benchmarks.model_comparison --results-dir benchmarks/results/
"""
import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional


# Column definitions for the results table
COLUMNS = [
    ("Model",              "{:<22}"),
    ("EWMScore",           "{:>9}"),
    ("VBench",             "{:>8}"),
    ("Physics",            "{:>9}"),
    ("IVEBench",           "{:>9}"),
    ("TiViBench",          "{:>10}"),
    ("Violations",         "{:>11}"),
]


def load_results_from_dir(results_dir: str) -> List[Dict]:
    """Load all .json result files from a directory."""
    results = []
    rdir = Path(results_dir)
    if not rdir.is_dir():
        return results
    for p in sorted(rdir.glob("*.json")):
        try:
            with open(p) as f:
                data = json.load(f)
            results.append(data)
        except (json.JSONDecodeError, OSError):
            continue
    return results


def extract_row(result: Dict) -> Dict:
    """Extract display values from a pipeline report dict."""
    raw = result.get("raw_scores", {})
    physics = result.get("physics_judgment", {})

    model_name = result.get("model_name", Path(result.get("video_path", "unknown")).stem)

    return {
        "model":      model_name[:22],
        "ewm_score":  result.get("ewm_score"),
        "vbench":     _avg_vbench(result),
        "physics":    raw.get("physics_compliance"),
        "ivebench":   _avg_evaluator(result, "ivebench"),
        "tivibench":  _avg_evaluator(result, "tivibench"),
        "violations": physics.get("violation_count"),
    }


def _avg_vbench(result: Dict) -> Optional[float]:
    """Average VBench dimension scores from the evaluator results."""
    for er in result.get("evaluators", []):
        if er.get("name") == "vbench" and er.get("scores"):
            vals = list(er["scores"].values())
            return sum(vals) / len(vals) if vals else None
    return None


def _avg_evaluator(result: Dict, name: str) -> Optional[float]:
    """Average scores from a named evaluator."""
    for er in result.get("evaluators", []):
        if er.get("name") == name and er.get("scores"):
            vals = list(er["scores"].values())
            return sum(vals) / len(vals) if vals else None
    return None


def fmt_val(val, precision: int = 1) -> str:
    """Format a numeric value or return '-' if unavailable."""
    if val is None:
        return "-"
    if isinstance(val, float):
        return f"{val:.{precision}f}"
    return str(val)


def print_table(rows: List[Dict]) -> None:
    """Print a formatted results table to stdout."""
    # Header
    header_parts = []
    for col_name, col_fmt in COLUMNS:
        header_parts.append(col_fmt.format(col_name))
    header = " | ".join(header_parts)
    sep = "-" * len(header)

    print()
    print("Physion-Judge: Multi-Model Benchmark Comparison")
    print("=" * len(header))
    print(header)
    print(sep)

    if not rows:
        print("  No results available. Run evaluations first:")
        print("    python -m pipeline.unified_pipeline --video <path> --all-dimensions")
        print("  Then place the .report.json files in benchmarks/results/")
        print(sep)
        print()
        return

    for row in rows:
        parts = [
            COLUMNS[0][1].format(row["model"]),
            COLUMNS[1][1].format(fmt_val(row["ewm_score"])),
            COLUMNS[2][1].format(fmt_val(row["vbench"])),
            COLUMNS[3][1].format(fmt_val(row["physics"])),
            COLUMNS[4][1].format(fmt_val(row["ivebench"])),
            COLUMNS[5][1].format(fmt_val(row["tivibench"])),
            COLUMNS[6][1].format(fmt_val(row["violations"], 0)),
        ]
        print(" | ".join(parts))

    print(sep)
    print()
    print("Notes:")
    print("  - EWMScore: Embodied World Model Score (0-100, higher = better)")
    print("  - VBench: average across 16 perceptual dimensions")
    print("  - Physics: Physion-Eval compliance score (0-1)")
    print("  - Violations: count of detected physical violations")
    print("  - '-' indicates evaluator was not run or not installed")
    print()


def main():
    parser = argparse.ArgumentParser(description="Print multi-model benchmark results")
    parser.add_argument(
        "--results-dir",
        default="benchmarks/results",
        help="Directory containing .report.json files",
    )
    args = parser.parse_args()

    results = load_results_from_dir(args.results_dir)
    rows = [extract_row(r) for r in results]
    print_table(rows)


if __name__ == "__main__":
    main()
