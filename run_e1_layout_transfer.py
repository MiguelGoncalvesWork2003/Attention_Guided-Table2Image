"""
run_e1_layout_transfer.py — orchestrator for the layout-transfer experiment
(Section 6.8 / sec:e1_design, Table 6.10).

THE CLAIM BEING TESTED
-----------------------
Section~\\ref{sec:formal_composition} defines the layout as independent of
the downstream classifier by construction. The testable consequence is not
"the layout still works" -- it always does, trivially -- but that a layout's
RELATIVE MERIT survives a change of downstream architecture. That is a
statement about rank preservation, which is why analyse_ablations.py scores
it with rank correlation rather than an accuracy delta.

WHY THIS SCRIPT GENERATES ITS OWN IMAGES
-----------------------------------------
The main benchmark writes its images to
    data/processed/{dataset}/{fold_hash}_{layout}/
(a content-hashed per-fold directory, see run_agt2i_fold in
benchmark_parallel.py). E1's analysis globs for
    data/processed/{dataset}/{layout}_seed{n}/arch_{arch}/
so the two do not line up, and E1 cannot simply reuse main-benchmark
images. This script therefore builds its own set once per
(dataset, layout, seed), in the location the analysis expects, and then
trains every architecture against those SAME image files.

That reuse is the entire point of the experiment, not an optimisation:
all four architectures must consume byte-identical images, or a difference
between them could reflect a different image rather than a different
network. train_cnn.py reads from IMAGE_DIR and writes only to
IMAGE_DIR/arch_{CNN_ARCH}/, so the four runs cannot contaminate each
other's inputs.

TabNet hyperparameters are the layout's own tuned values from the main
benchmark, so each layout's backbone matches what its main-benchmark result
used. CNN hyperparameters are likewise the layout's tuned values, held
fixed across all four architectures -- the factor under test is the
architecture itself, so everything else stays constant.

Usage:
    python run_e1_layout_transfer.py
    python run_e1_layout_transfer.py --datasets Cancer Card
    python run_e1_layout_transfer.py --layouts step_row --architectures tabnetcnn pixel_mlp
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

DEFAULT_DATASETS = ["Cancer", "Card", "Gene", "Heart"]
DEFAULT_LAYOUTS = ["step_row", "step_sparse", "packed", "packed_T", "attention_map"]
DEFAULT_ARCHITECTURES = ["tabnetcnn", "deep_cnn", "small_resnet", "pixel_mlp"]

CACHE_DIR = PROJECT_ROOT / "cache" / "e1_layout_transfer"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def tabnet_params_to_env(tabnet_params):
    """Identical mapping to the E3/E4/E6 orchestrators -- kept the same
    across all of them rather than re-derived, so they cannot drift apart."""
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


def load_layout_params(dataset_name, layout):
    agt2i_params = load_agt2i_params(dataset_name)
    if not agt2i_params or layout not in agt2i_params:
        return None, None, (
            f"No tuned parameters for AG-T2I-{layout} on {dataset_name} -- run "
            f"hyperparameter_search.py --dataset {dataset_name} "
            f"--model \"AG-T2I-{layout}\" first."
        )
    tabnet_params, cnn_params = agt2i_params[layout]
    return tabnet_params, cnn_params, None


def image_dir_for(dataset_name, layout, seed):
    """The location analyse_ablations.py's E1 section globs for."""
    return PROJECT_ROOT / "data" / "processed" / dataset_name / f"{layout}_seed{seed}"


def ensure_images(dataset_name, target_col, layout, seed, tabnet_params,
                  n_classes, force=False):
    """Train TabNet and build images ONCE per (dataset, layout, seed).
    Every architecture then trains on these exact files."""
    img_dir = image_dir_for(dataset_name, layout, seed)
    if not force and (img_dir / "X_train_img.npy").exists():
        print(f"    images already present at {img_dir.name}, reusing")
        return img_dir, None

    backbone_dir = CACHE_DIR / f"{dataset_name}_{layout}_seed{seed}"
    tabnet_out = backbone_dir / "tabnet_output"
    step_csv = tabnet_out / "tabnet_step_assignment.csv"

    global_processed = PROJECT_ROOT / "data" / "processed" / dataset_name
    base_env = {
        "DATASET": dataset_name,
        "TARGET_COL": target_col,
        "SEED": str(seed),
        "N_CLASSES": str(n_classes),
        "PROCESSED_DIR": str(global_processed),
        **tabnet_params_to_env(tabnet_params),
    }

    if not step_csv.exists():
        ok, out, _ = run_step(
            name=f"E1 TabNet [{dataset_name}/{layout}]",
            script_path=PROJECT_ROOT / "tabnet_fs" / "train_tabnet.py",
            env_vars={**base_env, "OUTPUT_DIR": str(backbone_dir)},
        )
        if not ok:
            return None, f"TabNet training failed: {out[:400]}"
        if not step_csv.exists():
            return None, f"TabNet reported success but {step_csv} is missing"

    ok, out, _ = run_step(
        name=f"E1 image build [{dataset_name}/{layout}]",
        script_path=PROJECT_ROOT / "image_builder" / "tabnet_image_builder.py",
        env_vars={
            **base_env,
            "MOL_LAYOUT": layout,
            # OUTPUT_DIR is the dataset root: tabnet_image_builder.py appends
            # its own "{layout}_seed{seed}" tag, producing exactly the path
            # train_cnn.py and the E1 analysis both expect.
            "OUTPUT_DIR": str(global_processed),
            "TABNET_IDX_DIR": str(tabnet_out),
            "TABNET_STEP_CSV_PATH": str(step_csv),
        },
    )
    if not ok:
        return None, f"Image build failed: {out[:400]}"
    if not (img_dir / "X_train_img.npy").exists():
        return None, f"Image build reported success but {img_dir} has no images"
    return img_dir, None


def run_one_architecture(dataset_name, target_col, layout, arch, seed,
                         n_classes, cnn_params):
    global_processed = PROJECT_ROOT / "data" / "processed" / dataset_name
    env = {
        "DATASET": dataset_name,
        "TARGET_COL": target_col,
        "MOL_LAYOUT": layout,
        "SEED": str(seed),
        "N_CLASSES": str(n_classes),
        "CNN_ARCH": arch,
        "OUTPUT_DIR": str(global_processed),
        "PROCESSED_DIR": str(global_processed),
        **cnn_params_to_env(cnn_params),
    }

    ok, out, _ = run_step(
        name=f"E1 CNN train [{dataset_name}/{layout}/{arch}]",
        script_path=PROJECT_ROOT / "cnn" / "train_cnn.py",
        env_vars=env,
    )
    if not ok:
        return {"status": "failed", "stage": "train_cnn", "error": out[:400]}

    ok, out, _ = run_step(
        name=f"E1 CNN eval [{dataset_name}/{layout}/{arch}]",
        script_path=PROJECT_ROOT / "cnn" / "evaluate_cnn.py",
        env_vars=env,
    )
    if not ok:
        return {"status": "failed", "stage": "evaluate_cnn", "error": out[:400]}

    results_path = (image_dir_for(dataset_name, layout, seed) / f"arch_{arch}"
                    / f"cnn_evaluation_results_{layout}.json")
    if not results_path.exists():
        return {"status": "failed", "stage": "results_missing",
                "error": str(results_path)}

    with open(results_path) as f:
        metrics = json.load(f)
    return {
        "status": "ok",
        "roc_auc": metrics.get("roc_auc"),
        "f1_macro": metrics.get("f1_macro"),
        "n_parameters": metrics.get("n_parameters"),
    }


def already_done(dataset_name, layout, arch, seed):
    """Results live in the analysis's own glob location, so presence of a
    readable result file with a usable metric is the completion signal."""
    p = (image_dir_for(dataset_name, layout, seed) / f"arch_{arch}"
         / f"cnn_evaluation_results_{layout}.json")
    if not p.exists():
        return False
    try:
        with open(p) as f:
            return json.load(f).get("roc_auc") is not None
    except (json.JSONDecodeError, OSError):
        return False


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    parser.add_argument("--layouts", nargs="+", default=DEFAULT_LAYOUTS)
    parser.add_argument("--architectures", nargs="+", default=DEFAULT_ARCHITECTURES)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--target-col", default="Class")
    parser.add_argument("--rebuild-images", action="store_true",
                        help="Regenerate images even if they already exist")
    args = parser.parse_args()

    combos = [(d, l, s) for d in args.datasets for l in args.layouts for s in args.seeds]
    print(f"{len(combos)} (dataset, layout, seed) combinations queued, "
          f"each training {len(args.architectures)} architectures "
          f"on one shared set of images")

    ok_count = skip_count = fail_count = 0

    for i, (ds, layout, seed) in enumerate(combos, 1):
        tag = f"[{i}/{len(combos)}] {ds}/{layout}/seed{seed}"

        pending = [a for a in args.architectures if not already_done(ds, layout, a, seed)]
        if not pending and not args.rebuild_images:
            print(f"{tag}: all architectures already done, skipping")
            ok_count += len(args.architectures)
            continue

        print(f"{tag}: starting ({len(pending)} architecture(s) pending)")
        ensure_global_preprocessing(ds, args.target_col)

        tabnet_params, cnn_params, error = load_layout_params(ds, layout)
        if tabnet_params is None:
            print(f"{tag}: SKIPPED -- {error}")
            skip_count += len(args.architectures)
            continue

        _, _, _, n_classes = load_and_prepare_dataset(ds, args.target_col)

        t0 = time.time()
        img_dir, error = ensure_images(ds, args.target_col, layout, seed,
                                       tabnet_params, n_classes,
                                       force=args.rebuild_images)
        if img_dir is None:
            print(f"{tag}: FAILED building images -- {error}")
            fail_count += len(args.architectures)
            continue
        print(f"    images ready in {time.time()-t0:.1f}s")

        for arch in args.architectures:
            if already_done(ds, layout, arch, seed) and not args.rebuild_images:
                print(f"    {arch:14s} already done")
                ok_count += 1
                continue
            t0 = time.time()
            result = run_one_architecture(ds, args.target_col, layout, arch,
                                          seed, n_classes, cnn_params)
            elapsed = time.time() - t0
            if result["status"] == "ok":
                params = result.get("n_parameters")
                pstr = f", {params:,} params" if params else ""
                print(f"    {arch:14s} roc_auc={result.get('roc_auc'):.4f}  "
                      f"({elapsed:.1f}s{pstr})")
                ok_count += 1
            else:
                print(f"    {arch:14s} FAILED at {result.get('stage')}: "
                      f"{result.get('error')}")
                fail_count += 1

    print(f"\n{'='*70}")
    print(f"OK={ok_count}  Skipped={skip_count}  Failed={fail_count}")
    print(f"{'='*70}")
    print("Aggregate with:  python analyse_ablations.py --only e1")
    if skip_count:
        print("\nSkipped combinations need their layout tuned first -- the message "
              "above each one gives the exact hyperparameter_search.py command.")


if __name__ == "__main__":
    main()
