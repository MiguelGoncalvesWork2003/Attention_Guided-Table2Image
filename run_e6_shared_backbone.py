"""
run_e6_shared_backbone.py — shared-backbone ablation (Section 5.8's declared
"necessary refinement"; not yet numbered as a table in the current draft --
propose Table 6.11 or folding into the E1 section, see the note at the end
of this docstring).

THE PROBLEM THIS ADDRESSES
---------------------------
Each of the 5 AGT2I layouts currently gets its own independent Optuna study
that tunes TabNet and CNN hyperparameters JOINTLY (tune_agt2i_layout in
hyperparameter_search.py). This means AGT2I-SR's frozen attention backbone
is, in general, a DIFFERENT TabNet than AGT2I-SS's, AGT2I-PR's, etc. -- not
just a different coordinate-assignment rule applied to the same attention.
A performance difference between two layouts is therefore confounded: it
could reflect genuine geometric differences, or it could just as easily
reflect one layout's Optuna search happening to land on a better-performing
TabNet configuration than another's, for reasons unrelated to geometry.

THE DESIGN CHOICE THIS SCRIPT MAKES (flagged, not silently assumed)
---------------------------------------------------------------------
The thesis does not specify exactly which TabNet configuration should serve
as "the" shared backbone for this ablation -- that choice is genuinely
underspecified, and a different one could reasonably be defended. This
script uses the ALREADY-TUNED STANDALONE TabNet BASELINE's hyperparameters
(best_params/<dataset>.json["TabNet"], tuned via tune_single_model on the
classification task itself, exactly like the TabNet baseline already
reported in Table 6.2) as the one shared backbone for all 5 layouts. This
was chosen because: (a) it requires no new HPO study, reusing a
configuration that is already independently justified and already reported
elsewhere in the thesis; (b) it ties the ablation's backbone to a
classification-quality criterion rather than to any one layout's incidental
optimum. If you'd rather run a fresh, layout-agnostic TabNet-only Optuna
study instead, that is a different, defensible design -- this script does
not do that, and says so here rather than silently picking one.

CNN hyperparameters are NOT shared across layouts: each layout's own tuned
CNN-only hyperparameters (the cnn_* keys already saved under
best_params/<dataset>.json["AG-T2I-{layout}"]) are reused, since the
question under test is specifically whether the BACKBONE needs to be
layout-specific, not whether the CNN does -- different image geometries can
legitimately warrant different CNN hyperparameters without reintroducing
the backbone confound.

WHAT THIS SCRIPT ACTUALLY DOES, per (dataset, fold)
-----------------------------------------------------
  1. Reconstructs the same outer split used elsewhere (StratifiedKFold,
     seed-dependent -- see run_e2_permutation_control.py's docstring for the
     same note about Section 5.5's text not matching the code's actual,
     seed-varying split).
  2. Loads the standalone TabNet baseline's tuned hyperparameters.
  3. Trains ONE TabNet with those hyperparameters on this fold, writing to
     a fold-level (not layout-level) cache directory unique to this
     experiment (cache/e6_shared_backbone/...), never touching or
     overwriting the main benchmark's own per-layout caches.
  4. For each of the 5 layouts: builds images from THAT SAME frozen
     attention output (all 5 layouts read identical
     cnn_train_idx.npy / cnn_val_idx.npy / tabnet_step_assignment.csv this
     time), trains a CNN using that layout's own tuned CNN hyperparameters,
     evaluates it.
  5. Reports the resulting 5 scores per fold, for comparison against the
     main benchmark's per-layout (independently-tuned-backbone) scores.

Usage:
    python run_e6_shared_backbone.py
    python run_e6_shared_backbone.py --datasets Cancer Glass --folds 0 1
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "running_all_models"))

from execution.runner import run_step
from running_all_models.benchmark_parallel import load_agt2i_params

DEFAULT_DATASETS = ["Cancer", "Card", "Gene", "Heart", "Soybean", "Thyroid"]
AGT2I_LAYOUTS = ["step_row", "step_sparse", "packed", "packed_T", "attention_map"]

CACHE_DIR = PROJECT_ROOT / "cache" / "e6_shared_backbone"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR = PROJECT_ROOT / "running_all_models" / "results" / "e6_shared_backbone"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def outer_split(dataset_name, target_col, seed, fold):
    """Same seed-dependent split as benchmark_parallel.py and
    run_e2_permutation_control.py -- see their docstrings for the
    discrepancy this reflects against Section 5.5's text."""
    raw_path = PROJECT_ROOT / "data" / "raw" / f"{dataset_name}.csv"
    df = pd.read_csv(raw_path)
    X = df.drop(columns=[target_col])
    y_encoded = LabelEncoder().fit_transform(df[target_col])
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    splits = list(skf.split(X, y_encoded))
    if fold >= len(splits):
        raise ValueError(f"fold {fold} out of range (5-fold split has 0..4)")
    return splits[fold]


def load_standalone_tabnet_params(dataset_name):
    """The shared backbone's hyperparameters: the already-tuned standalone
    TabNet baseline (Table 6.2), not a fresh or per-layout search."""
    params_file = PROJECT_ROOT / "running_all_models" / "best_params" / f"{dataset_name}.json"
    if not params_file.exists():
        return None, f"{params_file} does not exist"
    with open(params_file) as f:
        all_params = json.load(f)
    if "TabNet" not in all_params:
        return None, (
            f"No tuned 'TabNet' entry in {params_file} -- run "
            f"hyperparameter_search.py --model TabNet --dataset {dataset_name} first."
        )
    return all_params["TabNet"], None


def tabnet_params_to_env(tabnet_params):
    """Map the standalone-TabNet best_params schema (n_d, n_a, n_steps,
    gamma, lambda_sparse, optimizer_params={'lr': X}) onto the TABNET_* env
    vars train_tabnet.py actually reads. See models_factory.py's
    get_model_from_params for the same nested-optimizer_params handling."""
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


def load_cnn_params_for_layout(dataset_name, layout):
    agt2i_params = load_agt2i_params(dataset_name)
    if not agt2i_params or layout not in agt2i_params:
        return {}
    _tabnet_params, cnn_params = agt2i_params[layout]
    return cnn_params


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


def train_shared_backbone(dataset_name, target_col, seed, fold, train_idx, test_idx,
                          tabnet_params, n_classes):
    """Train ONE TabNet for this fold, shared across all 5 layouts.
    Returns the tabnet_output directory, or raises on failure."""
    fold_key = f"{dataset_name}_seed{seed}_fold{fold}"
    fold_cache_dir = CACHE_DIR / fold_key
    tabnet_out = fold_cache_dir / "tabnet_output"
    step_csv = tabnet_out / "tabnet_step_assignment.csv"

    if step_csv.exists():
        print(f"  Shared backbone already trained for {fold_key}, reusing.")
        return tabnet_out

    tmp_dir = fold_cache_dir / "tmp_work"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    idx_dir = fold_cache_dir / "custom_split"
    idx_dir.mkdir(parents=True, exist_ok=True)
    np.save(idx_dir / "train_idx.npy", train_idx)
    np.save(idx_dir / "test_idx.npy", test_idx)

    env = {
        "DATASET": dataset_name,
        "TARGET_COL": target_col,
        "SEED": str(seed),
        "N_CLASSES": str(n_classes),
        "PROCESSED_DIR": str(tmp_dir),
        "USE_CUSTOM_SPLIT": "true",
        "TRAIN_IDX_PATH": str(idx_dir / "train_idx.npy"),
        "TEST_IDX_PATH": str(idx_dir / "test_idx.npy"),
        "DROP_THRESHOLD": "0.5",
        "CAT_MISSING": "explicit",
        "NUM_MISSING": "median",
        "SCALING": "standard",
        "ENCODE_CATEGORICALS": "true",
        **tabnet_params_to_env(tabnet_params),
    }

    ok, out, _ = run_step(
        name=f"E6 preprocessing [{dataset_name}/fold{fold}]",
        script_path=PROJECT_ROOT / "preprocessing" / "run_preprocessing.py",
        env_vars=env,
    )
    if not ok:
        raise RuntimeError(f"E6 preprocessing failed: {out[:500]}")

    env["OUTPUT_DIR"] = str(fold_cache_dir)
    ok, out, _ = run_step(
        name=f"E6 TabNet training [{dataset_name}/fold{fold}]",
        script_path=PROJECT_ROOT / "tabnet_fs" / "train_tabnet.py",
        env_vars=env,
    )
    if not ok:
        raise RuntimeError(f"E6 TabNet training failed: {out[:500]}")

    if not step_csv.exists():
        raise RuntimeError(f"TabNet training reported success but {step_csv} is missing")

    return tabnet_out


def run_one_layout(dataset_name, target_col, layout, seed, fold, train_idx, test_idx,
                   tabnet_out, n_classes, cnn_params):
    output_dir = (
        PROJECT_ROOT / "data" / "processed" / dataset_name
        / f"e6_fold{fold}_seed{seed}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    env = {
        "DATASET": dataset_name,
        "TARGET_COL": target_col,
        "MOL_LAYOUT": layout,
        "SEED": str(seed),
        "N_CLASSES": str(n_classes),
        "OUTPUT_DIR": str(output_dir),
        "PROCESSED_DIR": str(output_dir),
        "TABNET_IDX_DIR": str(tabnet_out),
        "TABNET_STEP_CSV_PATH": str(tabnet_out / "tabnet_step_assignment.csv"),
        **cnn_params_to_env(cnn_params),
    }

    # Preprocessing for the CNN-input arrays under this layout's tag, using
    # the SAME outer split as the shared backbone (passed in, not
    # recomputed) so the two can never drift apart.
    idx_dir = output_dir / "custom_split"
    idx_dir.mkdir(parents=True, exist_ok=True)
    np.save(idx_dir / "train_idx.npy", train_idx)
    np.save(idx_dir / "test_idx.npy", test_idx)
    env.update({
        "USE_CUSTOM_SPLIT": "true",
        "TRAIN_IDX_PATH": str(idx_dir / "train_idx.npy"),
        "TEST_IDX_PATH": str(idx_dir / "test_idx.npy"),
        "DROP_THRESHOLD": "0.5", "CAT_MISSING": "explicit",
        "NUM_MISSING": "median", "SCALING": "standard",
        "ENCODE_CATEGORICALS": "true",
    })

    ok, out, _ = run_step(
        name=f"E6 preprocessing [{dataset_name}/{layout}/fold{fold}]",
        script_path=PROJECT_ROOT / "preprocessing" / "run_preprocessing.py",
        env_vars=env,
    )
    if not ok:
        return {"status": "failed", "stage": "preprocessing", "error": out[:500]}

    ok, out, _ = run_step(
        name=f"E6 image build [{dataset_name}/{layout}/fold{fold}]",
        script_path=PROJECT_ROOT / "image_builder" / "tabnet_image_builder.py",
        env_vars=env,
    )
    if not ok:
        return {"status": "failed", "stage": "image_build", "error": out[:500]}

    ok, out, _ = run_step(
        name=f"E6 CNN train [{dataset_name}/{layout}/fold{fold}]",
        script_path=PROJECT_ROOT / "cnn" / "train_cnn.py",
        env_vars=env,
    )
    if not ok:
        return {"status": "failed", "stage": "train_cnn", "error": out[:500]}

    ok, out, _ = run_step(
        name=f"E6 CNN eval [{dataset_name}/{layout}/fold{fold}]",
        script_path=PROJECT_ROOT / "cnn" / "evaluate_cnn.py",
        env_vars=env,
    )
    if not ok:
        return {"status": "failed", "stage": "evaluate_cnn", "error": out[:500]}

    results_path = (
        output_dir / f"{layout}_seed{seed}" / "arch_tabnetcnn"
        / f"cnn_evaluation_results_{layout}.json"
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


def already_done(dataset_name, seed, fold):
    summary_path = RESULTS_DIR / f"{dataset_name}_s{seed}f{fold}.json"
    return summary_path.exists()


def save_results(dataset_name, seed, fold, per_layout_results):
    summary_path = RESULTS_DIR / f"{dataset_name}_s{seed}f{fold}.json"
    with open(summary_path, "w") as f:
        json.dump({
            "dataset": dataset_name, "seed": seed, "fold": fold,
            "layouts": per_layout_results,
        }, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    parser.add_argument("--folds", nargs="+", type=int, default=[0])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--target-col", default="Class")
    args = parser.parse_args()

    combos = [(ds, sd, fl) for ds in args.datasets for sd in args.seeds for fl in args.folds]
    print(f"{len(combos)} (dataset, seed, fold) combinations queued, "
          f"each producing {len(AGT2I_LAYOUTS)} layout evaluations")

    for i, (ds, sd, fl) in enumerate(combos, 1):
        tag = f"[{i}/{len(combos)}] {ds}/seed{sd}/fold{fl}"
        if already_done(ds, sd, fl):
            print(f"{tag}: already done, skipping")
            continue

        print(f"{tag}: starting")
        tabnet_params, error = load_standalone_tabnet_params(ds)
        if tabnet_params is None:
            print(f"{tag}: SKIPPED -- {error}")
            continue

        raw_path = PROJECT_ROOT / "data" / "raw" / f"{ds}.csv"
        n_classes = pd.read_csv(raw_path, usecols=[args.target_col])[args.target_col].nunique()
        train_idx, test_idx = outer_split(ds, args.target_col, sd, fl)

        try:
            t0 = time.time()
            tabnet_out = train_shared_backbone(
                ds, args.target_col, sd, fl, train_idx, test_idx, tabnet_params, n_classes
            )
            print(f"  Shared backbone ready in {time.time()-t0:.1f}s: {tabnet_out}")
        except Exception as exc:
            print(f"{tag}: FAILED at shared backbone training: {exc}")
            continue

        per_layout = {}
        for layout in AGT2I_LAYOUTS:
            cnn_params = load_cnn_params_for_layout(ds, layout)
            t0 = time.time()
            result = run_one_layout(ds, args.target_col, layout, sd, fl,
                                    train_idx, test_idx, tabnet_out, n_classes, cnn_params)
            elapsed = time.time() - t0
            per_layout[layout] = result
            if result["status"] == "ok":
                print(f"    {layout:14s} roc_auc={result.get('roc_auc')}  ({elapsed:.1f}s)")
            else:
                print(f"    {layout:14s} FAILED at {result.get('stage')}: {result.get('error')}")

        save_results(ds, sd, fl, per_layout)
        print(f"{tag}: done")

    print(f"\n{'='*70}\nResults written to: {RESULTS_DIR}\n{'='*70}")
    print("Compare each layout's roc_auc here against its corresponding value\n"
          "in the main benchmark (independently-tuned backbone) to see whether\n"
          "the spread between layouts narrows once the backbone is held fixed.")


if __name__ == "__main__":
    main()