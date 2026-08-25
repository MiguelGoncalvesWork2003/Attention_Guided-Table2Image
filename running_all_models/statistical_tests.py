"""
Statistical significance analysis for benchmark experiments.

Implements the protocol described in Section 5.7 of the thesis, and produces
exactly the tables reported in Section 6.8 (Table 6.8: average ranks; Table
6.9: Wilcoxon vs a single control, Holm-corrected):

1. Load all *_raw.csv benchmark result files (columns: model, seed, fold,
   subset, accuracy, balanced_accuracy, precision_macro, recall_macro,
   f1_macro, precision_weighted, recall_weighted, f1_weighted, roc_auc,
   time_sec — see benchmark_parallel.py's run_dataset_benchmark).
2. Aggregate fold/seed results (subset == "test" only) into one mean score
   per model per dataset.
3. Run sanity checks that catch the two errors that reached an earlier
   draft:
     - average ranks must sum to n(n+1)/2
     - the aggregate table must equal the column means of the dataset-level
       table (true by construction here, but checked anyway)
4. Run a Friedman omnibus test. Post-hoc testing happens ONLY if it rejects
   (Section 5.7: "Only if this omnibus test rejects are post-hoc comparisons
   performed").
5. Post-hoc: Wilcoxon signed-rank against a SINGLE control method, with
   Holm-Bonferroni across that reduced family (Section 5.7: "post-hoc
   testing uses Wilcoxon signed-rank tests against a single control method,
   with the Holm-Bonferroni correction applied across that reduced family" —
   NOT all-pairs, which the previous version of this script did).
6. Report the attainable p-value floor (Section 5.7 / 6.8): with N paired
   datasets the smallest attainable two-sided exact Wilcoxon p is 2/2^N; the
   smallest attainable Holm-adjusted p across m comparisons is m * 2/2^N.

Two comparison families are reported, matching Section 6.8 exactly:
  - "complete": every dataset, only methods present on every one of them
    (drops IGTD/DeepInsight where they have gaps — "the twelve methods that
    completed on all ten benchmarks, excluding IGTD and DeepInsight")
  - "common": every method, only datasets where every method completed
    ("all fourteen methods on the eight benchmarks where every method
    completed")
Separating them avoids silently dropping either baselines or benchmarks
without saying so.

Run: python statistical_tests.py
Output: running_all_models/results/statistics/*.csv, *.tex, *.json
"""

import json
import warnings
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import friedmanchisquare, wilcoxon, studentized_range

try:
    from statsmodels.stats.multitest import multipletests
    _HAVE_STATSMODELS = True
except ImportError:
    _HAVE_STATSMODELS = False

# ============================================================
# Configuration
# ============================================================

RESULTS_DIR = Path(__file__).resolve().parent / "results"

ALPHA = 0.05
METRICS = ["roc_auc", "f1_macro"]  # both are reported: Tables 6.3/6.4, 6.8/6.9

# The control for the post-hoc family. Section 6.8 uses the best-ranked
# proposed variant (AGT2I-SS in the current results); if it is absent from
# a given family, the script falls back to whichever method ranks best.
CONTROL_PREFERENCE = [
    "AG-T2I-step_sparse",
    "AG-T2I-attention_map",
    "AG-T2I-step_row",
]


# ============================================================
# Holm correction (self-contained fallback if statsmodels is unavailable)
# ============================================================

def _holm(pvals):
    """Return Holm-Bonferroni adjusted p-values, same order as input."""
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    order = np.argsort(p)
    adj = np.empty(n)
    prev = 0.0
    for i, idx in enumerate(order):
        prev = min(1.0, max(prev, (n - i) * p[idx]))
        adj[idx] = prev
    return adj


def holm_adjust(pvals, alpha=ALPHA):
    if _HAVE_STATSMODELS:
        reject, p_corr, _, _ = multipletests(pvals, alpha=alpha, method="holm")
        return np.asarray(p_corr), np.asarray(reject)
    p_corr = _holm(pvals)
    return p_corr, p_corr < alpha


# ============================================================
# 1. Load benchmark results
# ============================================================

def load_all_results(results_dir: Path, metric: str) -> pd.DataFrame:
    """
    Load all *_raw.csv files and aggregate to one value per (dataset, model).

    Returns a DataFrame: rows = datasets, columns = models. Missing
    (dataset, model) combinations stay as NaN rather than being silently
    dropped, so coverage gaps are visible rather than hidden.
    """
    frames = []
    csv_files = sorted(results_dir.glob("*_raw.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No *_raw.csv files found in: {results_dir}")

    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        if "subset" in df.columns:
            df = df[df["subset"] == "test"]

        if metric not in df.columns:
            print(f"[WARNING] Metric '{metric}' not in {csv_file.name}")
            continue

        df = df.copy()
        df[metric] = pd.to_numeric(df[metric], errors="coerce")
        n_before = len(df)
        df = df.dropna(subset=[metric])
        if len(df) < n_before:
            print(f"[WARNING] {csv_file.name}: dropped {n_before - len(df)} "
                  f"row(s) with non-numeric/missing '{metric}'")

        if df.empty:
            continue

        agg = df.groupby("model")[metric].mean().reset_index()
        # Dataset name is the raw-file stem with any "_raw" / "_raw_<suffix>"
        # tail removed, since benchmark_parallel.py also writes filtered
        # files like "<dataset>_raw_<model>.csv" for --model-filtered runs.
        stem = csv_file.stem
        dataset_name = stem.split("_raw")[0]
        agg["dataset"] = dataset_name
        frames.append(agg)

    if not frames:
        raise ValueError("No valid benchmark files loaded.")

    all_data = pd.concat(frames, ignore_index=True)
    # If multiple raw files map to the same (dataset, model) — e.g. a filtered
    # rerun of one model — the most recently-loaded file wins.
    all_data = all_data.drop_duplicates(subset=["dataset", "model"], keep="last")
    return all_data.pivot(index="dataset", columns="model", values=metric)


def report_coverage(matrix: pd.DataFrame) -> None:
    """Print exactly which (dataset, model) cells are missing."""
    print("\n" + "=" * 70)
    print("COVERAGE")
    print("=" * 70)
    total = matrix.shape[0] * matrix.shape[1]
    missing = int(matrix.isna().sum().sum())
    print(f"{matrix.shape[0]} datasets x {matrix.shape[1]} models = {total} cells, "
          f"{missing} missing\n")

    for model in matrix.columns:
        absent = matrix.index[matrix[model].isna()].tolist()
        flag = "" if not absent else f"   MISSING: {', '.join(absent)}"
        print(f"  {model:26s} {matrix[model].notna().sum():2d}/{matrix.shape[0]}{flag}")


def build_families(matrix: pd.DataFrame):
    """
    Split into the two comparison families of Section 6.8.

    complete : all datasets, only methods present on every one (Table 6.8/6.9)
    common   : all methods, only datasets where every method completed
    """
    complete_models = [m for m in matrix.columns if matrix[m].notna().all()]
    complete = matrix[complete_models]
    common = matrix.dropna(axis=0, how="any")
    return {"complete": complete, "common": common}


# ============================================================
# 2. Sanity checks
# ============================================================

def sanity_checks(matrix: pd.DataFrame) -> bool:
    """
    Checks that would have caught the average-rank-sum error and an
    aggregate-vs-dataset-level mismatch, if either recurred. Returns True iff
    every check passes; prints a report either way.
    """
    print("\n" + "=" * 70)
    print("SANITY CHECKS")
    print("=" * 70)

    ok = True
    pivot = matrix.dropna(axis=0, how="any")
    n_methods, n_datasets = pivot.shape[1], pivot.shape[0]

    if n_datasets == 0 or n_methods == 0:
        print("  SKIPPED: no complete rows.")
        return False

    # Check 1: average ranks must sum to n(n+1)/2
    ranked = pivot.rank(axis=1, method="average", ascending=False)
    avg_ranks = ranked.mean(axis=0)
    expected = n_methods * (n_methods + 1) / 2
    observed = float(avg_ranks.sum())
    passed = abs(observed - expected) < 0.05
    ok &= passed
    print(f"  [{'PASS' if passed else 'FAIL'}] rank sum: observed {observed:.2f}, "
          f"expected {expected:.2f}  ({n_methods} methods)")

    # Check 2: aggregate equals the column mean of the dataset-level table
    # (true by construction, since Table 6.2-style aggregates should always
    # be derived as AVERAGE() over the dataset-level sheet, never re-typed)
    bad = []
    for m in pivot.columns:
        aggregate = float(pivot[m].mean())
        recomputed = float(np.mean([pivot.loc[d, m] for d in pivot.index]))
        if abs(aggregate - recomputed) >= 1e-9:
            bad.append(m)
    passed = not bad
    ok &= passed
    print(f"  [{'PASS' if passed else 'FAIL'}] aggregate equals column mean "
          f"for all {n_methods} methods" + ("" if passed else f"  -> {bad}"))

    # Check 3: values in [0, 1]
    vals = pivot.values
    out_of_range = int(((vals < 0) | (vals > 1)).sum())
    passed = out_of_range == 0
    ok &= passed
    print(f"  [{'PASS' if passed else 'FAIL'}] all values in [0, 1]: "
          f"{out_of_range} outside")

    return ok


# ============================================================
# 3. Friedman + Holm against a single control
# ============================================================

def pick_control(matrix: pd.DataFrame) -> str:
    for c in CONTROL_PREFERENCE:
        if c in matrix.columns:
            return c
    ranked = matrix.rank(axis=1, method="average", ascending=False)
    return ranked.mean(axis=0).idxmin()


def friedman_holm_vs_control(matrix: pd.DataFrame, control: str,
                              alpha: float = ALPHA) -> dict:
    """
    Friedman omnibus, then (only if it rejects) Wilcoxon signed-rank against
    one control with Holm correction over that reduced family. Also reports
    the attainable p-value floor unconditionally, since it is a property of
    the design (N, family size), not of whether the omnibus rejected.
    """
    pivot = matrix.dropna(axis=0, how="any")
    methods = pivot.columns.tolist()
    datasets = pivot.index.tolist()
    n, k = len(datasets), len(methods)

    out = {"n_datasets": n, "n_methods": k, "control": control,
           "datasets": datasets, "methods": methods}

    if n < 3:
        out["error"] = "Friedman needs at least 3 datasets."
        return out
    if control not in methods:
        out["error"] = f"Control '{control}' absent from this family."
        return out

    chi2, p_om = friedmanchisquare(*[pivot[m].values for m in methods])
    out["friedman"] = {"chi2": float(chi2), "df": k - 1, "p": float(p_om),
                        "rejects": bool(p_om < alpha)}

    ranked = pivot.rank(axis=1, method="average", ascending=False)
    out["avg_ranks"] = ranked.mean(axis=0).sort_values().to_dict()

    m_tests = k - 1
    out["floor"] = {
        "n": n, "m": m_tests,
        "min_raw": 2 / 2 ** n,
        "min_adjusted": m_tests * 2 / 2 ** n,
        "detectable": bool(m_tests * 2 / 2 ** n < alpha),
    }

    if p_om >= alpha:
        out["posthoc"] = None
        return out

    a = pivot[control].values
    raw_p, comparisons = [], []
    for m in methods:
        if m == control:
            continue
        b = pivot[m].values
        try:
            _, p = wilcoxon(a, b)
        except ValueError:  # all differences zero
            p = 1.0
        wins = int(np.sum(a > b))
        losses = int(np.sum(a < b))
        r = abs(wins - losses) / len(a)  # matched-pairs rank-biserial correlation
        comparisons.append({"vs": m, "p_raw": float(p), "effect_r": float(r),
                             "control_wins": wins, "control_losses": losses,
                             "ties": len(a) - wins - losses,
                             "mean_delta": float(np.mean(a - b))})
        raw_p.append(p)

    p_holm, sig = holm_adjust(raw_p, alpha)
    for i in range(len(comparisons)):
        comparisons[i]["p_holm"] = float(p_holm[i])
        comparisons[i]["significant"] = bool(sig[i])
    comparisons.sort(key=lambda r: r["p_raw"])

    out["posthoc"] = comparisons
    return out


# ============================================================
# 4. All-pairs (reference only — Section 5.7 explains why the
#    control-based family above is the one actually reported)
# ============================================================

def pairwise_all(matrix: pd.DataFrame, alpha: float = ALPHA) -> pd.DataFrame:
    models = matrix.columns.tolist()
    results = []
    for m1, m2 in combinations(models, 2):
        valid = matrix[[m1, m2]].dropna()
        if len(valid) < 2:
            results.append({"Model_A": m1, "Model_B": m2, "n_datasets": len(valid),
                             "wilcox_pvalue": np.nan, "mean_delta": np.nan})
            continue
        x1, x2 = valid[m1].values, valid[m2].values
        try:
            _, w_p = stats.wilcoxon(x1, x2, zero_method="wilcox")
        except ValueError:
            w_p = np.nan
        results.append({"Model_A": m1, "Model_B": m2, "n_datasets": len(valid),
                         "wilcox_pvalue": w_p, "mean_delta": float(np.mean(x1 - x2))})

    df_pairs = pd.DataFrame(results)
    mask = df_pairs["wilcox_pvalue"].notna()
    if mask.any():
        p_corr, sig = holm_adjust(df_pairs.loc[mask, "wilcox_pvalue"].values, alpha)
        df_pairs.loc[mask, "wilcox_pvalue_holm"] = p_corr
        df_pairs.loc[mask, "wilcox_significant"] = sig
    else:
        df_pairs["wilcox_pvalue_holm"] = np.nan
        df_pairs["wilcox_significant"] = False
    return df_pairs


# ============================================================
# 5. Nemenyi critical difference (for the CD diagram, Figure 6.2)
# ============================================================

def nemenyi_cd(matrix: pd.DataFrame, alpha: float = ALPHA):
    pivot = matrix.dropna(axis=0, how="any")
    k, N = pivot.shape[1], pivot.shape[0]
    if N < 3 or k < 2:
        return None
    ranked = pivot.rank(axis=1, method="average", ascending=False)
    avg_ranks = ranked.mean(axis=0).sort_values()
    q_alpha = studentized_range.ppf(1 - alpha, k, np.inf) / np.sqrt(2)
    cd = q_alpha * np.sqrt(k * (k + 1) / (6 * N))
    return {"cd": float(cd), "k": k, "N": N, "avg_ranks": avg_ranks.to_dict()}


# ============================================================
# 6. Reporting
# ============================================================

def print_family(name: str, result: dict, cd, metric: str) -> None:
    print("\n" + "=" * 70)
    print(f"FAMILY '{name}'  —  {metric}")
    print("=" * 70)
    print(f"{result['n_datasets']} datasets x {result['n_methods']} methods")
    print(f"datasets: {', '.join(result['datasets'])}")

    if "error" in result:
        print(f"\n  {result['error']}")
        return

    f = result["friedman"]
    print(f"\nFriedman: chi2 = {f['chi2']:.3f}, df = {f['df']}, "
          f"p = {f['p']:.3e}  ->  "
          f"{'REJECTS' if f['rejects'] else 'does not reject'}")

    print("\nAverage ranks (lower is better):")
    total = 0.0
    for m, r in result["avg_ranks"].items():
        print(f"  {m:28s} {r:5.2f}")
        total += r
    k = result["n_methods"]
    print(f"  {'SUM':28s} {total:5.2f}   (must equal {k*(k+1)/2:.2f})")

    if cd:
        print(f"\nNemenyi critical difference at alpha={ALPHA}: {cd['cd']:.3f}")

    fl = result["floor"]
    print(f"\nAttainable p-value floor (N={fl['n']}, m={fl['m']} comparisons):")
    print(f"  smallest raw exact Wilcoxon p : {fl['min_raw']:.6f}")
    print(f"  smallest Holm-adjusted p      : {fl['min_adjusted']:.6f}")
    if not fl["detectable"]:
        print(f"  ** No comparison in this family can reach alpha={ALPHA}, "
              f"regardless of the data. **")

    if result["posthoc"] is None:
        print("\nOmnibus did not reject; no post-hoc performed.")
        return

    print(f"\nWilcoxon vs {result['control']} (Holm over {fl['m']} comparisons):")
    print(f"  {'vs':28s} {'raw p':>9s} {'Holm p':>9s} {'r':>6s} "
          f"{'W/L/T':>10s} {'delta':>8s}  sig")
    for r in result["posthoc"]:
        wlt = f"{r['control_wins']}/{r['control_losses']}/{r['ties']}"
        print(f"  {r['vs']:28s} {r['p_raw']:9.4f} {r['p_holm']:9.4f} "
              f"{r['effect_r']:6.2f} {wlt:>10s} {r['mean_delta']:+8.4f}"
              f"  {'YES' if r['significant'] else ''}")


def save_family(name: str, matrix: pd.DataFrame, result: dict,
                 df_pairs: pd.DataFrame, cd, metric: str, output_dir: Path) -> None:
    output_dir.mkdir(exist_ok=True, parents=True)
    stem = f"{metric}_{name}"

    matrix.to_csv(output_dir / f"performance_matrix_{stem}.csv")
    with open(output_dir / f"performance_matrix_{stem}.tex", "w") as fh:
        fh.write(matrix.to_latex(float_format="%.4f", index=True, na_rep="--"))

    with open(output_dir / f"friedman_holm_{stem}.json", "w") as fh:
        json.dump(result, fh, indent=2, default=str)

    if result.get("posthoc"):
        ph = pd.DataFrame(result["posthoc"])
        ph.insert(0, "control", result["control"])
        ph.to_csv(output_dir / f"posthoc_{stem}.csv", index=False)
        with open(output_dir / f"posthoc_{stem}.tex", "w") as fh:
            fh.write(ph[["control", "vs", "p_raw", "p_holm", "effect_r", "significant"]]
                      .to_latex(index=False, float_format="%.4f"))

    if "avg_ranks" in result:
        pd.Series(result["avg_ranks"], name="avg_rank").to_csv(
            output_dir / f"avg_ranks_{stem}.csv")

    df_pairs.to_csv(output_dir / f"pairwise_all_{stem}.csv", index=False)

    if cd:
        with open(output_dir / f"nemenyi_cd_{stem}.json", "w") as fh:
            json.dump(cd, fh, indent=2)


# ============================================================
# Main
# ============================================================

def main():
    output_dir = RESULTS_DIR / "statistics"
    output_dir.mkdir(exist_ok=True, parents=True)

    print("=" * 70)
    print("STATISTICAL SIGNIFICANCE ANALYSIS")
    print("=" * 70)
    print(f"Results from: {RESULTS_DIR}")
    if not _HAVE_STATSMODELS:
        print("[NOTE] statsmodels not found; using a built-in Holm implementation.")

    all_passed = True

    for metric in METRICS:
        print("\n\n" + "#" * 70)
        print(f"# METRIC: {metric}")
        print("#" * 70)

        try:
            matrix = load_all_results(RESULTS_DIR, metric)
        except (FileNotFoundError, ValueError) as exc:
            print(f"[SKIP] {exc}")
            continue

        report_coverage(matrix)
        print("\nPerformance matrix:")
        print(matrix.round(4).to_string(na_rep="--"))

        all_passed &= sanity_checks(matrix)

        if matrix.shape[1] < 2:
            print("\nNeed at least two models.")
            continue

        for name, fam in build_families(matrix).items():
            if fam.shape[0] < 3 or fam.shape[1] < 2:
                print(f"\n[SKIP] family '{name}': "
                      f"{fam.shape[0]} datasets x {fam.shape[1]} methods")
                continue

            control = pick_control(fam)
            result = friedman_holm_vs_control(fam, control, ALPHA)
            cd = nemenyi_cd(fam, ALPHA)
            df_pairs = pairwise_all(fam, ALPHA)

            print_family(name, result, cd, metric)
            save_family(name, fam, result, df_pairs, cd, metric, output_dir)

    print("\n" + "=" * 70)
    if all_passed:
        print("All sanity checks passed. Safe to copy numbers into the thesis.")
    else:
        print("SANITY CHECKS FAILED. Do not copy these numbers into the thesis "
              "until the failures above are resolved.")
    print(f"Output written to: {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()
