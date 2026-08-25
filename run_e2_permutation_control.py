"""
run_e2_permutation_control.py — orchestrator for the permutation control
(Section 6.7.2, Table 6.6).

Replaces the earlier inline PowerShell loop with a resumable Python script.
For each (dataset, base_layout, permutation_seed) combination, this:

  1. Reconstructs the EXACT outer train/test split benchmark_parallel.py used
     for that (dataset, seed, fold) -- StratifiedKFold(n_splits=5,
     shuffle=True, random_state=seed), which is seed-dependent (see the note
     below), not a single fixed split reused across seeds.
  2. Looks up the tuned TabNet hyperparameters that AG-T2I-{base_layout}
     actually used for this dataset (best_params/<dataset>.json), via the
     SAME load_agt2i_params() the main benchmark uses.
  3. Computes the SAME cache-directory hash (_fold_id) the main benchmark
     used, to find the ALREADY-TRAINED, frozen TabNet + attention statistics
     for that exact (dataset, seed, fold, tabnet_params) combination.
  4. Builds shuffled images from THOSE SAME frozen attention statistics
     (never retrains TabNet), trains a CNN, and evaluates it.

Steps 2-3 matter for correctness, not just convenience: if each permutation
retrained TabNet independently, differences between permutations would be
confounded with incidental differences in the attention backbone, defeating
the point of the control. Reusing the cached backbone holds it fixed.

IMPORTANT -- outer split is seed-dependent, not a single reused split:
benchmark_parallel.py builds its outer 5-fold split as
    StratifiedKFold(n_splits=5, shuffle=True, random_state=seed).split(X, y)
where `seed` is the TRAINING seed (0, 1, or 2), not a fixed 42. This means
each of the 3 training seeds sees a DIFFERENT fold partition. Section 5.5's
text ("a stratified 5-fold split is generated once per dataset using the
global seed 42 and reused across all methods") does not match this. This
script reproduces the actual code's behaviour (seed-dependent), not the
thesis text's description, since it must find the SAME cached TabNet output
the main benchmark actually produced.

Compute scope (deliberately reduced from full 5-fold x 3-seed replication):
by default, ONE fold and ONE seed per (dataset, base_layout), with 5
permutation seeds providing the actual replication the paired test in
Table 6.6 needs. This is an explicit, documented simplification -- see
--folds / --seeds below to scale up if compute allows. The relevant
randomisation axis for this experiment is the permutation itself, not
training noise across folds/seeds, which the main benchmark already
estimates for the ordered condition.

Usage:
    python run_e2_permutation_control.py
    python run_e2_permutation_control.py --datasets Cancer Glass
    python run_e2_permutation_control.py --datasets Cancer --folds 0 1 --seeds 0 1
    python run_e2_permutation_control.py --resume-only   # skip completed, report gaps
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

from execution.runner import run_step, PipelineStepError
from running_all_models.benchmark_parallel import (
    _fold_id, CACHE_DIR, load_agt2i_params,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_DATASETS = ["Cancer", "Card", "Gene", "Heart", "Soybean", "Thyroid"]
BASE_LAYOUTS = ["step_row", "step_sparse", "packed"]
PERMUTATION_SEEDS = [0, 1, 2, 3, 4]

RESULTS_DIR = PROJECT_ROOT / "running_all_models" / "results" / "e2_permutation"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def outer_split(dataset_name, target_col, seed, fold):
    """Reproduce benchmark_parallel.py's outer split exactly (seed-dependent,
    see the module docstring)."""
    raw_path = PROJECT_ROOT / "data" / "raw" / f"{dataset_name}.csv"
    df = pd.read_csv(raw_path)
    X = df.drop(columns=[target_col])
    y_encoded = LabelEncoder().fit_transform(df[target_col])
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    splits = list(skf.split(X, y_encoded))
    if fold >= len(splits):
        raise ValueError(f"fold {fold} out of range (5-fold split has 0..4)")
    return splits[fold]


def find_cached_tabnet(dataset_name, base_layout, seed, fold, train_idx, test_idx):
    """Locate the exact cache directory the main benchmark used for this
    (dataset, base_layout, seed, fold), via the same _fold_id hash. Returns
    the tabnet_output subdirectory, or None if not found (main benchmark
    hasn't completed this combination yet)."""
    agt2i_params = load_agt2i_params(dataset_name)
    if not agt2i_params or base_layout not in agt2i_params:
        return None, (
            f"No tuned parameters found for AG-T2I-{base_layout} on "
            f"{dataset_name} in best_params/{dataset_name}.json -- run "
            f"hyperparameter_search.py for this layout first."
        )
    tabnet_params, _cnn_params = agt2i_params[base_layout]
    fold_str = _fold_id(dataset_name, seed, train_idx, test_idx, tabnet_params)
    fold_cache_dir = CACHE_DIR / "tabnet_cache" / fold_str
    tabnet_out = fold_cache_dir / "tabnet_output"
    step_csv = tabnet_out / "tabnet_step_assignment.csv"
    if not step_csv.exists():
        return None, (
            f"Expected cached TabNet output at {tabnet_out} but it doesn't "
            f"exist. The main benchmark must complete "
            f"AG-T2I-{base_layout} on {dataset_name} (seed={seed}, "
            f"fold={fold}) before this permutation can reuse its backbone."
        )
    return tabnet_out, None


def run_one_combination(dataset_name, target_col, base_layout, permutation_seed,
                        seed, fold, n_classes, quiet=False):
    """Build shuffled images from the cached backbone, train, evaluate."""
    train_idx, test_idx = outer_split(dataset_name, target_col, seed, fold)
    tabnet_out, error = find_cached_tabnet(
        dataset_name, base_layout, seed, fold, train_idx, test_idx
    )
    if tabnet_out is None:
        return {"status": "skipped", "reason": error}

    output_dir = (
        PROJECT_ROOT / "data" / "processed" / dataset_name
        / f"e2_fold{fold}_seed{seed}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    env = {
        "DATASET": dataset_name,
        "TARGET_COL": target_col,
        "MOL_LAYOUT": "shuffled",
        "BASE_LAYOUT": base_layout,
        "PERMUTATION_SEED": str(permutation_seed),
        "SEED": str(seed),
        "N_CLASSES": str(n_classes),
        "OUTPUT_DIR": str(output_dir),
        "TABNET_IDX_DIR": str(tabnet_out),
        "TABNET_STEP_CSV_PATH": str(tabnet_out / "tabnet_step_assignment.csv"),
        "PROCESSED_DIR": str(output_dir),
    }

    # The image builder needs X_train.npy / X_test.npy / y_*.npy /
    # feature_names.npy in PROCESSED_DIR, produced from THIS fold's exact
    # train/test indices (matching what the main benchmark used), not a
    # fresh independent split. run_preprocessing.py's custom-split branch
    # reads TRAIN_IDX_PATH / TEST_IDX_PATH from disk, so write those first.
    idx_dir = output_dir / "custom_split"
    idx_dir.mkdir(parents=True, exist_ok=True)
    np.save(idx_dir / "train_idx.npy", train_idx)
    np.save(idx_dir / "test_idx.npy", test_idx)
    env["USE_CUSTOM_SPLIT"] = "true"
    env["TRAIN_IDX_PATH"] = str(idx_dir / "train_idx.npy")
    env["TEST_IDX_PATH"] = str(idx_dir / "test_idx.npy")

    ok, out, _ = run_step(
        name=f"E2 preprocessing [{dataset_name}/{base_layout}/p{permutation_seed}]",
        script_path=PROJECT_ROOT / "preprocessing" / "run_preprocessing.py",
        env_vars=env,
    )
    if not ok:
        return {"status": "failed", "stage": "preprocessing", "error": out[:500]}

    ok, out, _ = run_step(
        name=f"E2 image build [{dataset_name}/{base_layout}/p{permutation_seed}]",
        script_path=PROJECT_ROOT / "image_builder" / "tabnet_image_builder.py",
        env_vars=env,
    )
    if not ok:
        return {"status": "failed", "stage": "image_build", "error": out[:500]}

    ok, out, _ = run_step(
        name=f"E2 CNN train [{dataset_name}/{base_layout}/p{permutation_seed}]",
        script_path=PROJECT_ROOT / "cnn" / "train_cnn.py",
        env_vars=env,
    )
    if not ok:
        return {"status": "failed", "stage": "train_cnn", "error": out[:500]}

    ok, out, _ = run_step(
        name=f"E2 CNN eval [{dataset_name}/{base_layout}/p{permutation_seed}]",
        script_path=PROJECT_ROOT / "cnn" / "evaluate_cnn.py",
        env_vars=env,
    )
    if not ok:
        return {"status": "failed", "stage": "evaluate_cnn", "error": out[:500]}

    results_path = (
        output_dir / f"shuffled-{base_layout}-p{permutation_seed}_seed{seed}"
        / "arch_tabnetcnn" / "cnn_evaluation_results_shuffled.json"
    )
    if not results_path.exists():
        return {"status": "failed", "stage": "results_missing",
                "error": f"Expected {results_path}"}

    with open(results_path) as f:
        metrics = json.load(f)

    return {
        "status": "ok",
        "roc_auc": metrics.get("roc_auc"),
        "f1_macro": metrics.get("f1_macro"),
        "accuracy": metrics.get("accuracy"),
    }


def already_done(dataset_name, base_layout, permutation_seed, seed, fold):
    summary_path = RESULTS_DIR / f"{dataset_name}_{base_layout}_p{permutation_seed}_s{seed}f{fold}.json"
    return summary_path.exists()


def save_result(dataset_name, base_layout, permutation_seed, seed, fold, result):
    summary_path = RESULTS_DIR / f"{dataset_name}_{base_layout}_p{permutation_seed}_s{seed}f{fold}.json"
    with open(summary_path, "w") as f:
        json.dump({
            "dataset": dataset_name, "base_layout": base_layout,
            "permutation_seed": permutation_seed, "seed": seed, "fold": fold,
            **result,
        }, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    parser.add_argument("--base-layouts", nargs="+", default=BASE_LAYOUTS)
    parser.add_argument("--permutation-seeds", nargs="+", type=int, default=PERMUTATION_SEEDS)
    parser.add_argument("--folds", nargs="+", type=int, default=[0],
                        help="Outer fold indices to use (default: fold 0 only)")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0],
                        help="Training seeds to use (default: seed 0 only)")
    parser.add_argument("--target-col", default="Class")
    parser.add_argument("--resume-only", action="store_true",
                        help="Don't run anything; report what's done vs missing")
    args = parser.parse_args()

    combos = [
        (ds, bl, ps, sd, fl)
        for ds in args.datasets
        for bl in args.base_layouts
        for ps in args.permutation_seeds
        for sd in args.seeds
        for fl in args.folds
    ]
    print(f"{len(combos)} combinations queued "
          f"({len(args.datasets)} datasets x {len(args.base_layouts)} layouts x "
          f"{len(args.permutation_seeds)} permutations x {len(args.seeds)} seeds x "
          f"{len(args.folds)} folds)")

    done, skipped, failed = 0, 0, 0
    for i, (ds, bl, ps, sd, fl) in enumerate(combos, 1):
        tag = f"[{i}/{len(combos)}] {ds}/{bl}/perm{ps}/seed{sd}/fold{fl}"
        if already_done(ds, bl, ps, sd, fl):
            print(f"{tag}: already done, skipping")
            done += 1
            continue
        if args.resume_only:
            print(f"{tag}: MISSING")
            continue

        raw_path = PROJECT_ROOT / "data" / "raw" / f"{ds}.csv"
        n_classes = pd.read_csv(raw_path, usecols=[args.target_col])[args.target_col].nunique()

        print(f"{tag}: running...")
        t0 = time.time()
        try:
            result = run_one_combination(ds, args.target_col, bl, ps, sd, fl, n_classes)
        except Exception as exc:
            result = {"status": "failed", "stage": "exception", "error": str(exc)}
        elapsed = time.time() - t0

        save_result(ds, bl, ps, sd, fl, result)
        if result["status"] == "ok":
            print(f"{tag}: OK  roc_auc={result.get('roc_auc')}  ({elapsed:.1f}s)")
            done += 1
        elif result["status"] == "skipped":
            print(f"{tag}: SKIPPED  {result['reason']}")
            skipped += 1
        else:
            print(f"{tag}: FAILED at {result.get('stage')}: {result.get('error')}")
            failed += 1

    print(f"\n{'='*70}\nDone={done}  Skipped={skipped}  Failed={failed}  "
          f"Total={len(combos)}\n{'='*70}")
    if skipped:
        print("SKIPPED combinations mean the main benchmark hasn't produced the "
              "matching AG-T2I layout run yet for that dataset/seed/fold -- "
              "run hyperparameter_search.py for those first, then re-run this "
              "script (already-completed combinations are automatically skipped).")


if __name__ == "__main__":
    main()