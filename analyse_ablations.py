"""
analyse_ablations.py — aggregates E2 (permutation control), E3 (AM
decomposition), and E6 (shared backbone) into printed summaries and
LaTeX-ready table snippets.

E1 (layout transfer) has its own analyse_e1.py from earlier in this
project; this script does not duplicate it.

Usage:
    python analyse_ablations.py                  # all three experiments
    python analyse_ablations.py --only e2
    python analyse_ablations.py --only e3 --datasets Cancer Glass
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, spearmanr, kendalltau

PROJECT_ROOT = Path(__file__).resolve().parent
RUNNING_MODELS = PROJECT_ROOT / "running_all_models"


# =====================================================================
# E1 — Layout transfer across architectures (Section 6.8)
# =====================================================================
#
# The reusability claim is not "the layout still works" -- any layout still
# works in the trivial sense that a network trained on it produces
# predictions. The claim that actually distinguishes AGT2I from HACNet is
# stronger and testable: a layout produced without reference to the
# downstream model retains its RELATIVE MERIT when the downstream model
# changes. That is a statement about rank preservation, so it is measured
# with rank correlation, not an accuracy delta.

E1_REFERENCE_ARCH = "tabnetcnn"

E1_LAYOUT_LABELS = {
    "step_row": "AGT2I-SR", "step_sparse": "AGT2I-SS",
    "packed": "AGT2I-PR", "packed_T": "AGT2I-PC", "attention_map": "AGT2I-AM",
}
E1_ARCH_LABELS = {
    "tabnetcnn": "Reference CNN", "deep_cnn": "Deep CNN",
    "small_resnet": "Small ResNet", "pixel_mlp": "Pixel MLP (spatially blind)",
}


def e1_collect(root: Path, metric: str) -> pd.DataFrame:
    """Walk data/processed/*/<layout>_seed<n>/arch_<arch>/ and gather results."""
    rows = []
    processed = root / "data" / "processed"
    if not processed.exists():
        raise FileNotFoundError(f"{processed} not found; is --root correct?")

    for result_file in processed.glob("*/*_seed*/arch_*/cnn_evaluation_results_*.json"):
        try:
            payload = json.loads(result_file.read_text())
        except (json.JSONDecodeError, OSError) as exc:
            print(f"[WARNING] skipping unreadable {result_file}: {exc}")
            continue

        arch_dir = result_file.parent.name
        layout_seed = result_file.parent.parent.name
        dataset = result_file.parent.parent.parent.name

        arch = payload.get("architecture") or arch_dir.replace("arch_", "")
        layout = payload.get("layout") or layout_seed.rsplit("_seed", 1)[0]
        seed = payload.get("seed")
        if seed is None:
            seed = int(layout_seed.rsplit("_seed", 1)[1])

        value = payload.get(metric)
        if value is None:
            for alt in ("roc_auc", "auroc", "f1_macro", "balanced_accuracy"):
                if payload.get(alt) is not None:
                    value = payload[alt]
                    print(f"[WARNING] {result_file.name}: '{metric}' absent, using '{alt}'")
                    break
        if value is None:
            print(f"[WARNING] {result_file.name}: no usable metric, skipped")
            continue

        # Real filename is tabnet_layout_{layout_seed}.json, written by
        # tabnet_image_builder.py -- height/width nested under "image_shape".
        geometry_file = result_file.parent.parent / f"tabnet_layout_{layout_seed}.json"
        geometry_raw = json.loads(geometry_file.read_text()) if geometry_file.exists() else {}
        image_shape = geometry_raw.get("image_shape", {})

        rows.append({
            "dataset": dataset,
            "layout": layout,
            "layout_label": E1_LAYOUT_LABELS.get(layout, layout),
            "arch": arch,
            "arch_label": E1_ARCH_LABELS.get(arch, arch),
            "seed": int(seed),
            "score": float(value),
            "n_parameters": payload.get("n_parameters"),
            "height": image_shape.get("height"),
            "width": image_shape.get("width"),
            "degenerate_1d": geometry_raw.get("degenerate_1d"),
            "sparsity": geometry_raw.get("sparsity"),
        })

    if not rows:
        raise RuntimeError(
            "No E1 results found. Expected files matching\n"
            "  data/processed/<DS>/<layout>_seed<n>/arch_<arch>/cnn_evaluation_results_*.json"
        )
    return pd.DataFrame(rows)


def e1_performance_matrix(df: pd.DataFrame) -> pd.DataFrame:
    per_dataset = (df.groupby(["dataset", "layout_label", "arch_label"])["score"]
                     .mean().reset_index())
    return per_dataset.pivot_table(index="layout_label", columns="arch_label",
                                   values="score", aggfunc="mean")


def e1_rank_preservation(df: pd.DataFrame) -> pd.DataFrame:
    """Does the layout ordering survive the change of architecture? Computed
    per dataset, then summarised, so one wide-spread dataset can't dominate."""
    out = []
    for dataset, chunk in df.groupby("dataset"):
        means = (chunk.groupby(["layout_label", "arch_label"])["score"]
                      .mean().unstack())
        ref_label = E1_ARCH_LABELS[E1_REFERENCE_ARCH]
        if ref_label not in means.columns:
            continue
        reference = means[ref_label]

        for arch_label in means.columns:
            if arch_label == ref_label:
                continue
            other = means[arch_label]
            paired = pd.concat([reference, other], axis=1).dropna()
            if len(paired) < 3:
                continue
            rho, p_rho = spearmanr(paired.iloc[:, 0], paired.iloc[:, 1])
            tau, p_tau = kendalltau(paired.iloc[:, 0], paired.iloc[:, 1])
            out.append({
                "dataset": dataset, "architecture": arch_label,
                "n_layouts": len(paired), "spearman_rho": rho, "spearman_p": p_rho,
                "kendall_tau": tau, "kendall_p": p_tau,
                "mean_delta": float(other.mean() - reference.mean()),
                "best_layout_reference": reference.idxmax(),
                "best_layout_here": other.idxmax(),
                "best_layout_preserved": reference.idxmax() == other.idxmax(),
            })
    return pd.DataFrame(out)


def e1_variance_decomposition(df: pd.DataFrame) -> pd.DataFrame:
    """Share of variance attributable to layout, architecture, interaction.
    Computed within dataset then averaged (datasets differ enormously in
    difficulty, so pooling them would swamp everything else)."""
    parts = []
    for dataset, chunk in df.groupby("dataset"):
        cell = chunk.groupby(["layout_label", "arch_label"])["score"].mean()
        table = cell.unstack()
        if table.isna().any().any() or table.shape[0] < 2 or table.shape[1] < 2:
            continue

        grand = table.values.mean()
        row_means = table.mean(axis=1).values[:, None]
        col_means = table.mean(axis=0).values[None, :]

        ss_layout = table.shape[1] * ((row_means - grand) ** 2).sum()
        ss_arch = table.shape[0] * ((col_means - grand) ** 2).sum()
        ss_inter = ((table.values - row_means - col_means + grand) ** 2).sum()
        ss_total = ss_layout + ss_arch + ss_inter
        if ss_total <= 0:
            continue

        parts.append({
            "dataset": dataset,
            "pct_layout": 100 * ss_layout / ss_total,
            "pct_architecture": 100 * ss_arch / ss_total,
            "pct_interaction": 100 * ss_inter / ss_total,
        })
    return pd.DataFrame(parts)


def e1_spatial_blindness_check(df: pd.DataFrame) -> pd.DataFrame:
    """How much of each layout's advantage survives a spatially blind model?
    PixelMLP cannot use adjacency, so a comparable spread there means what
    separates the layouts is feature set / pixel scaling, not geometry."""
    out = []
    for dataset, chunk in df.groupby("dataset"):
        means = (chunk.groupby(["layout_label", "arch_label"])["score"]
                      .mean().unstack())
        blind_label = E1_ARCH_LABELS["pixel_mlp"]
        ref_label = E1_ARCH_LABELS[E1_REFERENCE_ARCH]
        if blind_label not in means.columns or ref_label not in means.columns:
            continue
        spread_cnn = means[ref_label].max() - means[ref_label].min()
        spread_blind = means[blind_label].max() - means[blind_label].min()
        out.append({
            "dataset": dataset,
            "layout_spread_reference_cnn": spread_cnn,
            "layout_spread_pixel_mlp": spread_blind,
            "spatial_share": (1 - spread_blind / spread_cnn if spread_cnn > 1e-9 else np.nan),
        })
    return pd.DataFrame(out)


def analyse_e1(root: Path = None, metric: str = "roc_auc"):
    print("\n" + "=" * 70)
    print(f"E1 — LAYOUT REUSE ACROSS ARCHITECTURES (metric: {metric})")
    print("=" * 70)

    root = root or PROJECT_ROOT
    try:
        df = e1_collect(root, metric)
    except (FileNotFoundError, RuntimeError) as exc:
        print(f"No E1 results found yet: {exc}")
        return

    out_dir = RUNNING_MODELS / "results" / "e1_layout_transfer"
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "e1_raw.csv", index=False)

    print(f"{len(df)} runs | {df.dataset.nunique()} datasets | "
          f"{df.layout.nunique()} layouts | {df.arch.nunique()} architectures | "
          f"{df.seed.nunique()} seeds")

    degenerate = df[df.degenerate_1d == True]
    if len(degenerate):
        combos = degenerate[["dataset", "layout_label"]].drop_duplicates()
        print(f"\n[!] {len(combos)} (dataset, layout) pairs produced 1-D images. "
              f"Spatial claims do not apply to these:")
        for _, r in combos.iterrows():
            print(f"      {r.dataset} / {r.layout_label}")

    print("\nPerformance matrix (mean over seeds, then datasets):")
    matrix = e1_performance_matrix(df)
    print(matrix.round(4).to_string())
    matrix.to_csv(out_dir / "e1_performance_matrix.csv")

    print("\nRank preservation vs reference architecture [HEADLINE RESULT]:")
    print("  rho near +1 -> ordering transfers; rho near 0 -> architecture-specific")
    ranks = e1_rank_preservation(df)
    if len(ranks):
        print(ranks.round(4).to_string(index=False))
        ranks.to_csv(out_dir / "e1_rank_preservation.csv", index=False)
        print(f"  Mean Spearman rho: {ranks.spearman_rho.mean():.3f}  |  "
              f"Best layout preserved in {ranks.best_layout_preserved.sum()}/{len(ranks)}")

    print("\nVariance decomposition (layout / architecture / interaction):")
    var = e1_variance_decomposition(df)
    if len(var):
        print(var.round(1).to_string(index=False))
        var.to_csv(out_dir / "e1_variance_decomposition.csv", index=False)

    print("\nSpatial blindness control (PixelMLP):")
    blind = e1_spatial_blindness_check(df)
    if len(blind):
        print(blind.round(3).to_string(index=False))
        blind.to_csv(out_dir / "e1_spatial_blindness.csv", index=False)

    print(f"\nCSV output written to {out_dir}")
    print("Table 6.10 (layout transfer) draws from e1_performance_matrix.csv "
          "and e1_rank_preservation.csv.")


# =====================================================================
# E2 — Permutation control (Table 6.6)
# =====================================================================

def load_e2_results():
    e2_dir = RUNNING_MODELS / "results" / "e2_permutation"
    rows = []
    for p in sorted(e2_dir.glob("*.json")):
        with open(p) as f:
            d = json.load(f)
        if d.get("status") == "ok":
            rows.append(d)
    return pd.DataFrame(rows)


def load_ordered_value(dataset, base_layout, metric="roc_auc"):
    """The 'ordered' comparator: the main benchmark's already-computed
    AG-T2I-{base_layout} aggregate score for this dataset, from the same
    *_raw.csv files statistical_tests.py reads."""
    raw_path = RUNNING_MODELS / "results" / f"{dataset}_raw.csv"
    if not raw_path.exists():
        return None
    df = pd.read_csv(raw_path)
    df = df[(df["model"] == f"AG-T2I-{base_layout}") & (df["subset"] == "test")]
    if df.empty or metric not in df.columns:
        return None
    return float(pd.to_numeric(df[metric], errors="coerce").mean())


def analyse_e2():
    print("\n" + "=" * 70)
    print("E2 — PERMUTATION CONTROL (Table 6.6)")
    print("=" * 70)

    df = load_e2_results()
    if df.empty:
        print("No E2 results found yet in "
              f"{RUNNING_MODELS / 'results' / 'e2_permutation'}")
        return

    rows = []
    for (dataset, base_layout), grp in df.groupby(["dataset", "base_layout"]):
        permuted_mean = grp["roc_auc"].mean()
        permuted_std = grp["roc_auc"].std()
        n_perms = len(grp)
        ordered = load_ordered_value(dataset, base_layout)
        delta = (ordered - permuted_mean) if ordered is not None else None
        rows.append({
            "dataset": dataset, "layout": base_layout,
            "ordered": ordered, "permuted_mean": permuted_mean,
            "permuted_std": permuted_std, "n_permutations": n_perms,
            "delta": delta,
        })

    summary = pd.DataFrame(rows)
    print("\nPer (dataset, layout):")
    print(summary.round(4).to_string(index=False))

    print("\nPer-layout aggregate (mean delta = ordered - permuted, across datasets):")
    for layout, grp in summary.groupby("layout"):
        valid = grp.dropna(subset=["delta"])
        if len(valid) < 3:
            print(f"  {layout:14s}: only {len(valid)} dataset(s) with both "
                  f"values -- too few for a paired test")
            continue
        mean_delta = valid["delta"].mean()
        try:
            stat, p = wilcoxon(valid["ordered"], valid["permuted_mean"])
        except ValueError:
            p = float("nan")
        print(f"  {layout:14s}: mean delta = {mean_delta:+.4f}  "
              f"(N={len(valid)}, Wilcoxon p={p:.4f})")
        if len(valid) < 6:
            floor = 2 / 2 ** len(valid)
            print(f"                  attainable floor at N={len(valid)}: "
                  f"{floor:.4f} -- treat this p-value as descriptive only.")

    tex_path = RUNNING_MODELS / "results" / "table_6_6_permutation.tex"
    with open(tex_path, "w") as f:
        f.write(summary.round(4).to_latex(index=False, na_rep="--"))
    print(f"\nLaTeX table written to: {tex_path}")


# =====================================================================
# E3 — AM decomposition (Table 6.7)
# =====================================================================

def load_e3_result(dataset, variant, seed=0):
    """variant in {'full','flat','1row','nonorm'}."""
    tag = ("attention_map_seed" if variant == "full"
           else f"attention_map-{variant}_seed")
    results_path = (
        PROJECT_ROOT / "data" / "processed" / dataset / f"{tag}{seed}"
        / "arch_tabnetcnn" / "cnn_evaluation_results_attention_map.json"
    )
    if not results_path.exists():
        return None
    with open(results_path) as f:
        return json.load(f)


def analyse_e3(datasets):
    print("\n" + "=" * 70)
    print("E3 — AGT2I-AM DECOMPOSITION (Table 6.7)")
    print("=" * 70)

    variants = ["full", "flat", "1row", "nonorm"]
    rows = []
    any_found = False
    for ds in datasets:
        row = {"dataset": ds}
        for v in variants:
            r = load_e3_result(ds, v)
            row[f"{v}_auc"] = r.get("roc_auc") if r else None
            row[f"{v}_f1"] = r.get("f1_macro") if r else None
            if r:
                any_found = True
        rows.append(row)

    if not any_found:
        print("No E3 results found yet. Run each variant with, e.g.:\n"
              "  MOL_LAYOUT=attention_map AM_VARIANT=flat python "
              "image_builder/tabnet_image_builder.py  (then train_cnn.py, "
              "evaluate_cnn.py)")
        return

    summary = pd.DataFrame(rows)
    print("\nMacro-AUC by variant:")
    print(summary[["dataset"] + [f"{v}_auc" for v in variants]].round(4).to_string(index=False))
    print("\nMacro-F1 by variant:")
    print(summary[["dataset"] + [f"{v}_f1" for v in variants]].round(4).to_string(index=False))

    print("\nMean deltas vs full (positive = full is better):")
    for v in variants[1:]:
        auc_delta = (summary["full_auc"] - summary[f"{v}_auc"]).mean()
        f1_delta = (summary["full_f1"] - summary[f"{v}_f1"]).mean()
        print(f"  full - {v:8s}: AUC delta = {auc_delta:+.4f}   F1 delta = {f1_delta:+.4f}")

    tex_path = RUNNING_MODELS / "results" / "table_6_7_am_decomposition.tex"
    with open(tex_path, "w") as f:
        f.write(summary.round(4).to_latex(index=False, na_rep="--"))
    print(f"\nLaTeX table written to: {tex_path}")


# =====================================================================
# E4 — Threshold sensitivity (Table 6.threshold)
# =====================================================================

def load_e4_results():
    e4_dir = RUNNING_MODELS / "results" / "e4_threshold_sensitivity"
    rows = []
    for p in sorted(e4_dir.glob("*.json")):
        with open(p) as f:
            d = json.load(f)
        for theta_str, result in d.get("thetas", {}).items():
            if result.get("status") == "ok":
                rows.append({
                    "dataset": d["dataset"], "seed": d["seed"],
                    "theta": float(theta_str),
                    "n_features_retained": result.get("n_features_retained"),
                    "roc_auc": result.get("roc_auc"),
                })
    return pd.DataFrame(rows)


def analyse_e4():
    print("\n" + "=" * 70)
    print("E4 — THRESHOLD SENSITIVITY (Table 6.threshold)")
    print("=" * 70)

    df = load_e4_results()
    if df.empty:
        print(f"No E4 results found yet in "
              f"{RUNNING_MODELS / 'results' / 'e4_threshold_sensitivity'}")
        return

    pivot_auc = df.pivot_table(index="dataset", columns="theta", values="roc_auc")
    pivot_f = df.pivot_table(index="dataset", columns="theta", values="n_features_retained")

    print("\nMacro-AUC by theta:")
    print(pivot_auc.round(4).to_string())
    print("\n|F'| (retained features) by theta:")
    print(pivot_f.to_string())

    tex_path = RUNNING_MODELS / "results" / "table_threshold_sensitivity.tex"
    combined = pivot_auc.round(4).astype(str) + " | " + pivot_f.astype("Int64").astype(str)
    with open(tex_path, "w") as f:
        f.write(combined.to_latex(na_rep="--"))
    print(f"\nLaTeX table written to: {tex_path}")


# =====================================================================
# E6 — Shared backbone (proposed Table 6.11)
# =====================================================================

def load_e6_results():
    e6_dir = RUNNING_MODELS / "results" / "e6_shared_backbone"
    rows = []
    for p in sorted(e6_dir.glob("*.json")):
        with open(p) as f:
            d = json.load(f)
        for layout, result in d.get("layouts", {}).items():
            if result.get("status") == "ok":
                rows.append({
                    "dataset": d["dataset"], "seed": d["seed"], "fold": d["fold"],
                    "layout": layout, "roc_auc": result.get("roc_auc"),
                })
    return pd.DataFrame(rows)


def load_independent_backbone_value(dataset, layout, metric="roc_auc"):
    """The main benchmark's per-layout score, where each layout's backbone
    was independently tuned (the condition E6 is contrasted against)."""
    return load_ordered_value(dataset, layout, metric)  # same lookup as E2's "ordered"


def analyse_e6():
    print("\n" + "=" * 70)
    print("E6 — SHARED TABNET BACKBONE ABLATION")
    print("=" * 70)

    df = load_e6_results()
    if df.empty:
        print(f"No E6 results found yet in "
              f"{RUNNING_MODELS / 'results' / 'e6_shared_backbone'}")
        return

    print("\nPer (dataset, seed, fold): spread across the 5 layouts, "
          "shared backbone vs independently-tuned backbone")
    rows = []
    for (dataset, seed, fold), grp in df.groupby(["dataset", "seed", "fold"]):
        shared_spread = grp["roc_auc"].max() - grp["roc_auc"].min()
        independent_vals = [
            load_independent_backbone_value(dataset, layout)
            for layout in grp["layout"]
        ]
        independent_vals = [v for v in independent_vals if v is not None]
        independent_spread = (
            max(independent_vals) - min(independent_vals)
            if len(independent_vals) >= 2 else None
        )
        rows.append({
            "dataset": dataset, "seed": seed, "fold": fold,
            "shared_spread": shared_spread,
            "independent_spread": independent_spread,
        })

    summary = pd.DataFrame(rows)
    print(summary.round(4).to_string(index=False))

    valid = summary.dropna(subset=["independent_spread"])
    if len(valid) >= 1:
        narrower = (valid["shared_spread"] < valid["independent_spread"]).sum()
        print(f"\nShared-backbone spread narrower than independently-tuned "
              f"spread in {narrower}/{len(valid)} (dataset, seed, fold) "
              f"combinations.")
        print("A consistently narrower spread under a shared backbone "
              "supports attributing part of the layouts' apparent "
              "differences to backbone variation rather than geometry alone.")

    tex_path = RUNNING_MODELS / "results" / "table_e6_shared_backbone.tex"
    with open(tex_path, "w") as f:
        f.write(summary.round(4).to_latex(index=False, na_rep="--"))
    print(f"\nLaTeX table written to: {tex_path}")


# =====================================================================
# Main
# =====================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", choices=["e1", "e2", "e3", "e4", "e6"], default=None)
    parser.add_argument("--metric", default="roc_auc", help="Used by E1")
    parser.add_argument("--datasets", nargs="+",
                        default=["Cancer", "Card", "Gene", "Heart", "Soybean", "Thyroid"],
                        help="Used by E3, which has no results directory of its own to glob")
    args = parser.parse_args()

    if args.only in (None, "e1"):
        analyse_e1(metric=args.metric)
    if args.only in (None, "e2"):
        analyse_e2()
    if args.only in (None, "e3"):
        analyse_e3(args.datasets)
    if args.only in (None, "e4"):
        analyse_e4()
    if args.only in (None, "e6"):
        analyse_e6()


if __name__ == "__main__":
    main()