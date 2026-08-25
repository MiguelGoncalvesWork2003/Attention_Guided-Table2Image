"""
run_e4_threshold_sensitivity.py — orchestrator for the importance-threshold
sensitivity sweep (Section 4.3.3 / sec:e4_design, Table 6.threshold).

Trains ONE TabNet per dataset -- reusing the hyperparameters already tuned
for AG-T2I-step_row, the representative layout for this sweep -- then
applies each theta value as a pure post-hoc filter on that SAME frozen
attention output. This is a natural fit for the shared-backbone pattern:
theta only affects which features clear the retention bar
(tabnet_image_builder.py's `step_df[step_df["global_importance"] >= theta]`)
after TabNet has already trained; it plays no role in TabNet's own training
at all. So training once and re-filtering is not just cheaper than
retraining per theta, it's a more faithful match to what theta actually
does in the pipeline.

CNN hyperparameters are reused from step_row's own tuned config for every
theta value, holding CNN capacity fixed so that only the retained feature
set differs between runs.

Usage:
    python run_e4_threshold_sensitivity.py
    python run_e4_threshold_sensitivity.py --datasets Diabetes Heart
"""

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "running_all_models"))

from execution.runner import run_step
from running_all_models.hyperparameter_search import (
    ensure_global_preprocessing, load_and_prepare_dataset,
)
from running_all_models.benchmark_parallel import load_agt2i_params

# The four datasets spanning the dimensionality range of the suite
# (Diabetes F=8, Heart F=35, Soybean F=82, Gene F=120), per sec:e4_design.
DEFAULT_DATASETS = ["Diabetes", "Heart", "Soybean", "Gene"]
THETAS = [0.0, 0.001, 0.005, 0.01, 0.02]
LAYOUT = "step_row"  # the representative layout for this sweep

CACHE_DIR = PROJECT_ROOT / "cache" / "e4_threshold_sensitivity"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR = PROJECT_ROOT / "running_all_models" / "results" / "e4_threshold_sensitivity"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def load_step_row_params(dataset_name):
    """step_row's own already-tuned TabNet + CNN hyperparameters, so the
    shared backbone here matches what the main benchmark's step_row result
    actually used."""
    agt2i_params = load_agt2i_params(dataset_name)
    if not agt2i_params or "step_row" not in agt2i_params:
        return None, None, (
            f"No tuned parameters found for AG-T2I-step_row on "
            f"{dataset_name} -- run hyperparameter_search.py --model "
            f"\"AG-T2I-step_row\" --dataset {dataset_name} first."
        )
    tabnet_params, cnn_params = agt2i_params["step_row"]
    return tabnet_params, cnn_params, None


def tabnet_params_to_env(tabnet_params):
    """Identical mapping to run_e3_am_decomposition.py and
    run_e6_shared_backbone.py -- kept the same across all three rather than
    re-derived, so they can't silently drift apart."""
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
    """Train ONE TabNet for this dataset, shared across all 5 theta values."""
    cache_key = f"{dataset_name}_seed{seed}"
    fold_cache_dir = CACHE_DIR / cache_key
    tabnet_out = fold_cache_dir / "tabnet_output"
    step_csv = tabnet_out / "tabnet_step_assignment.csv"

    if step_csv.exists():
        print(f"  Shared backbone already trained for {cache_key}, reusing.")
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
        name=f"E4 TabNet training [{dataset_name}]",
        script_path=PROJECT_ROOT / "tabnet_fs" / "train_tabnet.py",
        env_vars=env,
    )
    if not ok:
        raise RuntimeError(f"E4 TabNet training failed: {out[:500]}")
    if not step_csv.exists():
        raise RuntimeError(f"TabNet training reported success but {step_csv} is missing")

    return tabnet_out


def run_one_theta(dataset_name, target_col, theta, seed, tabnet_out,
                  n_classes, cnn_params):
    theta_tag = str(theta).replace(".", "p")
    output_dir = (
        PROJECT_ROOT / "data" / "processed" / dataset_name
        / f"e4_theta{theta_tag}_seed{seed}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    global_processed = PROJECT_ROOT / "data" / "processed" / dataset_name
    env = {
        "DATASET": dataset_name,
        "TARGET_COL": target_col,
        "MOL_LAYOUT": LAYOUT,
        "IMPORTANCE_CUTOFF": str(theta),
        "SEED": str(seed),
        "N_CLASSES": str(n_classes),
        "OUTPUT_DIR": str(output_dir),
        "PROCESSED_DIR": str(global_processed),
        "TABNET_IDX_DIR": str(tabnet_out),
        "TABNET_STEP_CSV_PATH": str(tabnet_out / "tabnet_step_assignment.csv"),
        **cnn_params_to_env(cnn_params),
    }

    ok, out, _ = run_step(
        name=f"E4 image build [{dataset_name}/theta={theta}]",
        script_path=PROJECT_ROOT / "image_builder" / "tabnet_image_builder.py",
        env_vars=env,
    )
    if not ok:
        return {"status": "failed", "stage": "image_build", "error": out[:500]}

    # |F'|, the retained feature count, comes from the image-builder's own
    # metadata JSON (tabnet_layout_{layout}_seed{seed}.json), not from the
    # CNN stages -- read it here since it's the other half of Table 6.threshold.
    n_features_retained = None
    metadata_path = output_dir / f"tabnet_layout_{LAYOUT}_seed{seed}.json"
    if metadata_path.exists():
        with open(metadata_path) as f:
            n_features_retained = json.load(f).get("n_features_retained")

    ok, out, _ = run_step(
        name=f"E4 CNN train [{dataset_name}/theta={theta}]",
        script_path=PROJECT_ROOT / "cnn" / "train_cnn.py",
        env_vars=env,
    )
    if not ok:
        return {"status": "failed", "stage": "train_cnn", "error": out[:500],
                "n_features_retained": n_features_retained}

    ok, out, _ = run_step(
        name=f"E4 CNN eval [{dataset_name}/theta={theta}]",
        script_path=PROJECT_ROOT / "cnn" / "evaluate_cnn.py",
        env_vars=env,
    )
    if not ok:
        return {"status": "failed", "stage": "evaluate_cnn", "error": out[:500],
                "n_features_retained": n_features_retained}

    results_path = (
        output_dir / f"{LAYOUT}_seed{seed}" / "arch_tabnetcnn"
        / f"cnn_evaluation_results_{LAYOUT}.json"
    )
    if not results_path.exists():
        return {"status": "failed", "stage": "results_missing",
                "error": str(results_path), "n_features_retained": n_features_retained}

    with open(results_path) as f:
        metrics = json.load(f)
    return {
        "status": "ok",
        "roc_auc": metrics.get("roc_auc"),
        "f1_macro": metrics.get("f1_macro"),
        "accuracy": metrics.get("accuracy"),
        "n_features_retained": n_features_retained,
    }


def already_done(dataset_name, seed):
    return (RESULTS_DIR / f"{dataset_name}_s{seed}.json").exists()


def save_results(dataset_name, seed, per_theta):
    with open(RESULTS_DIR / f"{dataset_name}_s{seed}.json", "w") as f:
        json.dump({"dataset": dataset_name, "seed": seed, "thetas": per_theta},
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
          f"each producing {len(THETAS)} theta evaluations")

    for i, (ds, sd) in enumerate(combos, 1):
        tag = f"[{i}/{len(combos)}] {ds}/seed{sd}"
        if already_done(ds, sd):
            print(f"{tag}: already done, skipping")
            continue

        print(f"{tag}: starting")
        ensure_global_preprocessing(ds, args.target_col)

        tabnet_params, cnn_params, error = load_step_row_params(ds)
        if tabnet_params is None:
            print(f"{tag}: SKIPPED -- {error}")
            continue

        _, _, _, n_classes = load_and_prepare_dataset(ds, args.target_col)

        try:
            t0 = time.time()
            tabnet_out = train_shared_backbone(ds, args.target_col, sd, tabnet_params, n_classes)
            print(f"  Shared backbone ready in {time.time()-t0:.1f}s")
        except Exception as exc:
            print(f"{tag}: FAILED at shared backbone training: {exc}")
            continue

        per_theta = {}
        for theta in THETAS:
            t0 = time.time()
            result = run_one_theta(ds, args.target_col, theta, sd,
                                   tabnet_out, n_classes, cnn_params)
            elapsed = time.time() - t0
            per_theta[str(theta)] = result
            if result["status"] == "ok":
                print(f"    theta={theta:<6} |F'|={result.get('n_features_retained')}  "
                      f"roc_auc={result.get('roc_auc')}  ({elapsed:.1f}s)")
            else:
                print(f"    theta={theta:<6} FAILED at {result.get('stage')}: "
                      f"{result.get('error')}")

        save_results(ds, sd, per_theta)
        print(f"{tag}: done")

    print(f"\nResults written to: {RESULTS_DIR}")
    print("These feed Table 6.threshold (sec:threshold_sensitivity) directly:")
    print("  |F'| column from n_features_retained, macro-AUC column from roc_auc.")


if __name__ == "__main__":
    main()