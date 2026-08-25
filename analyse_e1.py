#!/usr/bin/env python3
"""
analyse_e1.py — turn the E1 runs into the numbers the thesis needs.

The reusability claim is not "the layout still works". Any layout still works
in the trivial sense that a network trained on it produces predictions. The
claim that actually distinguishes AGT2I from HACNet is stronger and testable:

    A layout produced without reference to the downstream model retains its
    RELATIVE MERIT when the downstream model changes.

That is a statement about rank preservation, so it is measured with rank
correlation, not with an accuracy delta. This script computes:

  1. The layout x architecture performance matrix (mean +/- std over seeds).
  2. Spearman and Kendall rank correlation between the layout ordering under
     the reference architecture and under each alternative. This is the
     headline reusability number.
  3. Absolute degradation per architecture, so a preserved ranking at
     uniformly collapsed performance is not mistaken for success.
  4. A two-way ANOVA-style variance decomposition: how much of the total
     variation is layout, how much is architecture, how much is interaction.
     A large layout main effect with a small interaction is the pattern that
     supports reusability; a large interaction is the pattern that refutes it.
  5. The pixel_mlp control. PixelMLP is blind to spatial arrangement, so any
     layout difference that survives there is not spatial. This converts E1
     into a partial test of the central hypothesis at no extra compute cost.

Usage:
    python analyse_e1.py --root /path/to/project [--metric roc_auc]
"""

import argparse
import json
import warnings
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from scipy.stats import spearmanr, kendalltau, wilcoxon
    SCIPY = True
except ImportError:  # pragma: no cover
    SCIPY = False
    warnings.warn("scipy not available; rank correlations will be skipped")


REFERENCE_ARCH = "tabnetcnn"

LAYOUT_LABELS = {
    "step_row": "AGT2I-SR",
    "step_sparse": "AGT2I-SS",
    "packed": "AGT2I-PR",
    "packed_T": "AGT2I-PC",
    "attention_map": "AGT2I-AM",
}

ARCH_LABELS = {
    "tabnetcnn": "Reference CNN",
    "deep_cnn": "Deep CNN",
    "small_resnet": "Small ResNet",
    "pixel_mlp": "Pixel MLP (spatially blind)",
}


# ---------------------------------------------------------------------------
# Collection
# ---------------------------------------------------------------------------

def collect(root: Path, metric: str) -> pd.DataFrame:
    """Walk data/processed/*/<layout>_seed<n>/arch_<arch>/ and gather results."""
    rows = []
    processed = root / "data" / "processed"
    if not processed.exists():
        raise FileNotFoundError(f"{processed} not found; is --root correct?")

    for result_file in processed.glob("*/*_seed*/arch_*/cnn_evaluation_results_*.json"):
        try:
            payload = json.loads(result_file.read_text())
        except (json.JSONDecodeError, OSError) as exc:
            warnings.warn(f"skipping unreadable {result_file}: {exc}")
            continue

        arch_dir = result_file.parent.name              # arch_deep_cnn
        layout_seed = result_file.parent.parent.name     # packed_seed1
        dataset = result_file.parent.parent.parent.name

        arch = payload.get("architecture") or arch_dir.replace("arch_", "")
        layout = payload.get("layout") or layout_seed.rsplit("_seed", 1)[0]
        seed = payload.get("seed")
        if seed is None:
            seed = int(layout_seed.rsplit("_seed", 1)[1])

        value = payload.get(metric)
        if value is None:
            # Fall back through the aliases the pipeline may have written.
            for alt in ("roc_auc_macro", "roc_auc", "auroc", "f1_macro", "balanced_accuracy"):
                if payload.get(alt) is not None:
                    value = payload[alt]
                    warnings.warn(
                        f"{result_file.name}: '{metric}' absent, using '{alt}'"
                    )
                    break
        if value is None:
            warnings.warn(f"{result_file.name}: no usable metric, skipped")
            continue

        geometry_file = result_file.parent.parent / f"tabnet_layout_{layout_seed}.json"
        geometry_raw = json.loads(geometry_file.read_text()) if geometry_file.exists() else {}
        image_shape = geometry_raw.get("image_shape", {})
        geometry = {
            "height": image_shape.get("height"),
            "width": image_shape.get("width"),
            "degenerate_1d": geometry_raw.get("degenerate_1d"),
            "sparsity": geometry_raw.get("sparsity"),
        }

        rows.append({
            "dataset": dataset,
            "layout": layout,
            "layout_label": LAYOUT_LABELS.get(layout, layout),
            "arch": arch,
            "arch_label": ARCH_LABELS.get(arch, arch),
            "seed": int(seed),
            "score": float(value),
            "n_parameters": payload.get("n_parameters"),
            "height": geometry.get("height"),
            "width": geometry.get("width"),
            "degenerate_1d": geometry.get("degenerate_1d"),
            "sparsity": geometry.get("sparsity"),
        })

    if not rows:
        raise RuntimeError(
            "No E1 results found. Expected files matching\n"
            "  data/processed/<DS>/<layout>_seed<n>/arch_<arch>/cnn_evaluation_results_*.json"
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Analyses
# ---------------------------------------------------------------------------

def performance_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Layout x architecture, averaged over seeds then over datasets."""
    per_dataset = (df.groupby(["dataset", "layout_label", "arch_label"])["score"]
                     .mean().reset_index())
    return per_dataset.pivot_table(index="layout_label", columns="arch_label",
                                   values="score", aggfunc="mean")


def rank_preservation(df: pd.DataFrame) -> pd.DataFrame:
    """Does the layout ordering survive the change of architecture?

    Computed per dataset, then summarised, so that a single dataset with a
    wide performance spread cannot dominate the correlation.
    """
    if not SCIPY:
        return pd.DataFrame()

    out = []
    for dataset, chunk in df.groupby("dataset"):
        means = (chunk.groupby(["layout_label", "arch_label"])["score"]
                      .mean().unstack())
        ref_label = ARCH_LABELS[REFERENCE_ARCH]
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
                "dataset": dataset,
                "architecture": arch_label,
                "n_layouts": len(paired),
                "spearman_rho": rho,
                "spearman_p": p_rho,
                "kendall_tau": tau,
                "kendall_p": p_tau,
                "mean_delta": float(other.mean() - reference.mean()),
                "best_layout_reference": reference.idxmax(),
                "best_layout_here": other.idxmax(),
                "best_layout_preserved": reference.idxmax() == other.idxmax(),
            })
    return pd.DataFrame(out)


def variance_decomposition(df: pd.DataFrame) -> pd.DataFrame:
    """Share of variance attributable to layout, architecture, interaction.

    Computed within dataset (datasets differ enormously in difficulty, so
    pooling them would swamp everything else), then averaged.

    Interpretation:
      large layout effect, small interaction -> layout merit transfers
      small layout effect, large interaction -> layout merit is architecture-
                                                specific, i.e. not reusable
    """
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


def spatial_blindness_check(df: pd.DataFrame) -> pd.DataFrame:
    """How much of each layout's advantage survives a spatially blind model?

    PixelMLP cannot use adjacency. If the spread between layouts under
    PixelMLP is comparable to the spread under the CNNs, then what separates
    the layouts is the feature set and pixel scaling, not the geometry.
    """
    out = []
    for dataset, chunk in df.groupby("dataset"):
        means = (chunk.groupby(["layout_label", "arch_label"])["score"]
                      .mean().unstack())
        blind_label = ARCH_LABELS["pixel_mlp"]
        ref_label = ARCH_LABELS[REFERENCE_ARCH]
        if blind_label not in means.columns or ref_label not in means.columns:
            continue
        spread_cnn = means[ref_label].max() - means[ref_label].min()
        spread_blind = means[blind_label].max() - means[blind_label].min()
        out.append({
            "dataset": dataset,
            "layout_spread_reference_cnn": spread_cnn,
            "layout_spread_pixel_mlp": spread_blind,
            "spatial_share": (
                1 - spread_blind / spread_cnn if spread_cnn > 1e-9 else np.nan
            ),
        })
    return pd.DataFrame(out)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."),
                        help="project root containing data/processed")
    parser.add_argument("--metric", default="roc_auc")
    parser.add_argument("--out", type=Path, default=None,
                        help="directory for CSV output")
    args = parser.parse_args()

    df = collect(args.root, args.metric)
    out_dir = args.out or (args.root / "experiments" / "e1_layout_transfer")
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "e1_raw.csv", index=False)

    pd.set_option("display.width", 140)
    pd.set_option("display.float_format", lambda v: f"{v:.4f}")

    print("=" * 78)
    print(f"E1 — LAYOUT REUSE ACROSS ARCHITECTURES   (metric: {args.metric})")
    print("=" * 78)
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

    print("\n\n1. PERFORMANCE MATRIX (mean over seeds, then over datasets)")
    print("-" * 78)
    matrix = performance_matrix(df)
    print(matrix.to_string())
    matrix.to_csv(out_dir / "e1_performance_matrix.csv")

    print("\n\n2. RANK PRESERVATION vs reference architecture  [HEADLINE RESULT]")
    print("-" * 78)
    print("   rho near +1 -> the layout ordering transfers")
    print("   rho near  0 -> the ordering is architecture-specific, i.e. not reusable")
    ranks = rank_preservation(df)
    if len(ranks):
        print()
        print(ranks.to_string(index=False))
        ranks.to_csv(out_dir / "e1_rank_preservation.csv", index=False)
        print(f"\n   Mean Spearman rho across datasets and architectures: "
              f"{ranks.spearman_rho.mean():.3f}")
        print(f"   Best layout preserved in "
              f"{ranks.best_layout_preserved.sum()}/{len(ranks)} comparisons")
    else:
        print("   (scipy unavailable or too few layouts)")

    print("\n\n3. VARIANCE DECOMPOSITION (within dataset)")
    print("-" * 78)
    print("   layout large + interaction small -> merit transfers")
    print("   interaction large               -> merit is architecture-specific")
    var = variance_decomposition(df)
    if len(var):
        print()
        print(var.to_string(index=False))
        var.to_csv(out_dir / "e1_variance_decomposition.csv", index=False)
        print(f"\n   Mean: layout {var.pct_layout.mean():.1f}% | "
              f"architecture {var.pct_architecture.mean():.1f}% | "
              f"interaction {var.pct_interaction.mean():.1f}%")

    print("\n\n4. SPATIAL BLINDNESS CONTROL (PixelMLP)")
    print("-" * 78)
    print("   spatial_share near 1 -> layout differences are genuinely spatial")
    print("   spatial_share near 0 -> layout differences survive without geometry,")
    print("                           so they come from feature selection / scaling")
    blind = spatial_blindness_check(df)
    if len(blind):
        print()
        print(blind.to_string(index=False))
        blind.to_csv(out_dir / "e1_spatial_blindness.csv", index=False)
        print(f"\n   Mean spatial share: {blind.spatial_share.mean():.3f}")

    print("\n\n5. PAIRED TESTS: reference vs each alternative architecture")
    print("-" * 78)
    if SCIPY:
        paired = (df.groupby(["dataset", "layout_label", "arch_label"])["score"]
                    .mean().unstack())
        ref_label = ARCH_LABELS[REFERENCE_ARCH]
        for arch_label in paired.columns:
            if arch_label == ref_label:
                continue
            pair = paired[[ref_label, arch_label]].dropna()
            if len(pair) < 6:
                print(f"   {arch_label:32s}  n={len(pair)} too few for a test")
                continue
            try:
                stat, p = wilcoxon(pair[ref_label], pair[arch_label])
                delta = (pair[arch_label] - pair[ref_label]).mean()
                print(f"   {arch_label:32s}  n={len(pair):3d}  "
                      f"mean delta={delta:+.4f}  Wilcoxon p={p:.4f}")
            except ValueError as exc:
                print(f"   {arch_label:32s}  test failed: {exc}")

    print("\n" + "=" * 78)
    print(f"CSV output written to {out_dir}")
    print("=" * 78)
    print("""
HOW TO READ THIS FOR THE THESIS

  Reusability is SUPPORTED if section 2 shows high mean Spearman rho (say
  above 0.7) and section 3 shows a small interaction term. Write: "the layout
  ordering is preserved across architectures (mean rho = [X]), supporting the
  claim that the coordinate map retains its relative merit independently of
  the downstream classifier."

  Reusability is NOT SUPPORTED if rho is near zero or the interaction term
  dominates. That is still a publishable result and a more interesting one.
  Write: "layout merit did not transfer (mean rho = [X]); the best layout
  under the reference architecture was best under an alternative in only
  [N]/[M] comparisons. Decoupling the layout from the classifier therefore
  does not, on this evidence, yield a layout that is useful independently of
  it." Then remove the reusability claim from the Abstract and Table 3.3 and
  report this as a limitation with evidence, which is far stronger than the
  current position of claiming it with none.

  Section 4 is the one to read most carefully. If spatial_share is near zero,
  the layouts differ for reasons that have nothing to do with geometry, and
  Section 6.2 of the thesis must be rewritten accordingly.
""")


if __name__ == "__main__":
    main()