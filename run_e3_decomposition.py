"""
run_e3_am_decomposition.py — orchestrator for the AGT2I-AM decomposition
(Section 6.7.3 / sec:e3_design, Table 6.7).

Trains ONE TabNet per dataset -- reusing the hyperparameters already tuned
for AG-T2I-attention_map in the main benchmark, for a fair comparison
against that benchmark's own "AM-full" result -- then generates all four
variants (full, flat, 1row, nonorm) from that SAME frozen backbone.

Why share the backbone across variants, when my original guide just said
"loop and let each variant retrain TabNet fresh": if each variant retrained
independently, differences between them would be confounded with ordinary
training-run noise in TabNet, even with a fixed seed (floating-point
non-associativity across CUDA kernel launches can make "deterministic"
training subtly non-identical in practice). Training once and reusing the
same cached attention statistics for all four variants removes that
confound entirely and is also ~4x cheaper.

CNN hyperparameters are reused from AM's own tuned config too, for all four
variants -- the ablation holds CNN capacity fixed so that only the AM
variant factor (attention weighting / row replication / normalisation)
differs between runs.

Usage:
    python run_e3_am_decomposition.py
    python run_e3_am_decomposition.py --datasets Cancer Glass
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "running_all_models"))

from execution.runner import run_step
from running_all_models.hyperparameter_search import (
    ensure_global_preprocessing, load_and_prepare_dataset,
)
from running_all_models.benchmark_parallel import load_agt2i_params

# Soybean is excluded: with 19 classes over ~137 test instances per fold,
# some classes go unrepresented and macro-averaged ROC-AUC is undefined,
# which returned NaN for all four variants. Pass it explicitly with
# --datasets if that is ever resolved.
DEFAULT_DATASETS = ["Cancer", "Card", "Gene", "Heart", "Thyroid"]
VARIANTS = ["full", "flat", "1row", "nonorm"]

CACHE_DIR = PROJECT_ROOT / "cache" / "e3_am_decomposition"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR = PROJECT_ROOT / "running_all_models" / "results" / "e3_am_decomposition"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def load_am_params(dataset_name):
    """AM's own already-tuned TabNet + CNN hyperparameters from the main
    benchmark, so E3's backbone matches what "AM-full" actually used."""
    agt2i_params = load_agt2i_params(dataset_name)
    if not agt2i_params or "attention_map" not in agt2i_params:
        return None, None, (
            f"No tuned parameters found for AG-T2I-attention_map on "
            f"{dataset_name} -- run hyperparameter_search.py --model "
            f"\"AG-T2I-attention_map\" --dataset {dataset_name} first."
        )
    tabnet_params, cnn_params = agt2i_params["attention_map"]
    return tabnet_params, cnn_params, None


def tabnet_params_to_env(tabnet_params):
    """Same mapping used in run_e6_shared_backbone.py -- kept identical
    rather than re-derived, so the two scripts can't silently drift apart."""
    env = {}
    if "n_steps" in tabnet_params:
        env["TABNET_N_STEPS"] = str(tabnet_params["n_steps"])
    if "n_d" in tabnet_params:
        env["TABNET_STEP_DIM"] = str(tabnet_params["n_d"])
    if "n_a" in tabnet_params:
        env["TABNET_ATTN_DIM"] = str(tabnet_params["n_a"])
    if "gamma" in tabnet_params:
        env["TABNET_GAMMA"] = str(tabnet_params["gamma"])
    if "lambda_sparse" in tabnet_params:
        env["TABNET_LAMBDA_SPARSE"] = str(tabnet_params["lambda_sparse"])
    lr = tabnet_params.get("optimizer_params", {}).get("lr", tabnet_params.get("lr"))
    if lr is not None:
        env["TABNET_LEARNING_RATE"] = str(lr)
    return env


def cnn_params_to_env(cnn_params):
    env = {}
    if "lr" in cnn_params:
        env["CNN_LEARNING_RATE"] = str(cnn_params["lr"])
    if "optimizer" in cnn_params:
        env["CNN_OPTIMIZER"] = str(cnn_params["optimizer"])
    if "dropout" in cnn_params:
        env["CNN_DROPOUT"] = str(cnn_params["dropout"])
    if "epochs" in cnn_params:
        env["CNN_EPOCHS"] = str(cnn_params["epochs"])
    return env


def train_shared_backbone(dataset_name, target_col, seed, tabnet_params, n_classes):
    """Train ONE TabNet for this dataset, shared across all 4 AM variants.
    Uses the global (dataset-level) preprocessing split, not a fold-specific
    one -- E3 is a self-contained ablation comparing variants against each
    other, not against a specific main-benchmark fold, so it only needs
    internal consistency across the 4 variants, not exact fold matching."""
    cache_key = f"{dataset_name}_seed{seed}"
    fold_cache_dir = CACHE_DIR / cache_key
    tabnet_out = fold_cache_dir / "tabnet_output"
    step_csv = tabnet_out / "tabnet_step_assignment.csv"

    if step_csv.exists():
        print(f"  Shared AM backbone already trained for {cache_key}, reusing.")
        return tabnet_out

    global_processed = PROJECT_ROOT / "data" / "processed" / dataset_name
    env = {
        "DATASET": dataset_name,
        "TARGET_COL": target_col,
        "SEED": str(seed),
        "N_CLASSES": str(n_classes),
        "PROCESSED_DIR": str(global_processed),
        "OUTPUT_DIR": str(fold_cache_dir),
        **tabnet_params_to_env(tabnet_params),
    }

    ok, out, _ = run_step(
        name=f"E3 TabNet training [{dataset_name}]",
        script_path=PROJECT_ROOT / "tabnet_fs" / "train_tabnet.py",
        env_vars=env,
    )
    if not ok:
        raise RuntimeError(f"E3 TabNet training failed: {out[:500]}")
    if not step_csv.exists():
        raise RuntimeError(f"TabNet training reported success but {step_csv} is missing")

    return tabnet_out


def run_one_variant(dataset_name, target_col, variant, seed, tabnet_out,
                    n_classes, cnn_params):
    output_dir = PROJECT_ROOT / "data" / "processed" / dataset_name / f"e3_seed{seed}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # PROCESSED_DIR is the global (dataset-level) preprocessing output --
    # read-only source arrays. OUTPUT_DIR is this run's own derived output
    # (images, labels) -- same PROCESSED_DIR-vs-OUTPUT_DIR separation used
    # throughout the pipeline, just at the dataset level instead of the
    # per-fold level, since E3 doesn't need outer-fold isolation.
    global_processed = PROJECT_ROOT / "data" / "processed" / dataset_name
    env = {
        "DATASET": dataset_name,
        "TARGET_COL": target_col,
        "MOL_LAYOUT": "attention_map",
        "AM_VARIANT": variant,
        "SEED": str(seed),
        "N_CLASSES": str(n_classes),
        "OUTPUT_DIR": str(output_dir),
        "PROCESSED_DIR": str(global_processed),
        "TABNET_IDX_DIR": str(tabnet_out),
        "TABNET_STEP_CSV_PATH": str(tabnet_out / "tabnet_step_assignment.csv"),
        **cnn_params_to_env(cnn_params),
    }

    ok, out, _ = run_step(
        name=f"E3 image build [{dataset_name}/{variant}]",
        script_path=PROJECT_ROOT / "image_builder" / "tabnet_image_builder.py",
        env_vars=env,
    )
    if not ok:
        return {"status": "failed", "stage": "image_build", "error": out[:500]}

    ok, out, _ = run_step(
        name=f"E3 CNN train [{dataset_name}/{variant}]",
        script_path=PROJECT_ROOT / "cnn" / "train_cnn.py",
        env_vars=env,
    )
    if not ok:
        return {"status": "failed", "stage": "train_cnn", "error": out[:500]}

    ok, out, _ = run_step(
        name=f"E3 CNN eval [{dataset_name}/{variant}]",
        script_path=PROJECT_ROOT / "cnn" / "evaluate_cnn.py",
        env_vars=env,
    )
    if not ok:
        return {"status": "failed", "stage": "evaluate_cnn", "error": out[:500]}

    tag = "attention_map_seed" if variant == "full" else f"attention_map-{variant}_seed"
    results_path = (
        output_dir / f"{tag}{seed}" / "arch_tabnetcnn"
        / "cnn_evaluation_results_attention_map.json"
    )
    if not results_path.exists():
        return {"status": "failed", "stage": "results_missing", "error": str(results_path)}

    with open(results_path) as f:
        metrics = json.load(f)
    return {
        "status": "ok",
        "roc_auc": metrics.get("roc_auc"),
        "f1_macro": metrics.get("f1_macro"),
        "accuracy": metrics.get("accuracy"),
    }


def already_done(dataset_name, seed):
    """True only if a previous run genuinely succeeded on every condition, so
    a partial or failed run can be retried rather than permanently skipped."""
    p = RESULTS_DIR / f"{dataset_name}_s{seed}.json"
    if not p.exists():
        return False
    try:
        with open(p) as f:
            entries = json.load(f).get("variants", {})
        return bool(entries) and all(r.get("status") == "ok" for r in entries.values())
    except (json.JSONDecodeError, OSError):
        return False


def save_results(dataset_name, seed, per_variant):
    with open(RESULTS_DIR / f"{dataset_name}_s{seed}.json", "w") as f:
        json.dump({"dataset": dataset_name, "seed": seed, "variants": per_variant},
                  f, indent=2)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--target-col", default="Class")
    args = parser.parse_args()

    combos = [(ds, sd) for ds in args.datasets for sd in args.seeds]
    print(f"{len(combos)} (dataset, seed) combinations queued, "
          f"each producing {len(VARIANTS)} variant evaluations")

    for i, (ds, sd) in enumerate(combos, 1):
        tag = f"[{i}/{len(combos)}] {ds}/seed{sd}"
        if already_done(ds, sd):
            print(f"{tag}: already done, skipping")
            continue

        print(f"{tag}: starting")
        ensure_global_preprocessing(ds, args.target_col)

        tabnet_params, cnn_params, error = load_am_params(ds)
        if tabnet_params is None:
            print(f"{tag}: SKIPPED -- {error}")
            continue

        _, _, _, n_classes = load_and_prepare_dataset(ds, args.target_col)

        try:
            t0 = time.time()
            tabnet_out = train_shared_backbone(ds, args.target_col, sd, tabnet_params, n_classes)
            print(f"  Shared AM backbone ready in {time.time()-t0:.1f}s")
        except Exception as exc:
            print(f"{tag}: FAILED at shared backbone training: {exc}")
            continue

        per_variant = {}
        for variant in VARIANTS:
            t0 = time.time()
            result = run_one_variant(ds, args.target_col, variant, sd,
                                     tabnet_out, n_classes, cnn_params)
            elapsed = time.time() - t0
            per_variant[variant] = result
            if result["status"] == "ok":
                print(f"    {variant:10s} roc_auc={result.get('roc_auc')}  ({elapsed:.1f}s)")
            else:
                print(f"    {variant:10s} FAILED at {result.get('stage')}: {result.get('error')}")

        save_results(ds, sd, per_variant)
        print(f"{tag}: done")

    print(f"\nResults written to: {RESULTS_DIR}")
    print("Run analyse_ablations.py --only e3 to aggregate into Table 6.7.")


if __name__ == "__main__":
    main()
