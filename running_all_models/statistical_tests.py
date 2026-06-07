"""
Statistical significance analysis for benchmark experiments.

This script:

1. Loads all *_raw.csv benchmark result files.
2. Aggregates fold/seed results into one mean score per model per dataset.
3. Performs:
   - Paired t-tests
   - Wilcoxon signed-rank tests
   with Holm-Bonferroni correction.
4. Performs:
   - Friedman test across all models
   - Nemenyi post-hoc analysis (if Friedman is significant)
5. Saves all outputs to CSV and LaTeX.

Expected raw CSV format:
-------------------------------------------------
model | seed | fold | accuracy | macro_f1 | ...
-------------------------------------------------
"""

import warnings
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

# ============================================================
# Configuration
# ============================================================

RESULTS_DIR = Path(__file__).resolve().parent / "results"

ALPHA = 0.05
METRIC = "accuracy"  # or "macro_f1"

# ============================================================
# 1. Load benchmark results
# ============================================================

def load_all_results(results_dir: Path, metric: str) -> pd.DataFrame:
    """
    Load all *_raw.csv files and aggregate results.

    Returns
    -------
    matrix : pd.DataFrame
        Rows = datasets
        Columns = models
        Values = mean metric over folds/seeds
    """

    frames = []

    csv_files = sorted(results_dir.glob("*_raw.csv"))

    if not csv_files:
        raise FileNotFoundError(
            f"No *_raw.csv files found in: {results_dir}"
        )

    for csv_file in csv_files:

        df = pd.read_csv(csv_file)
        df = df[df["subset"] == "test"]

        if metric not in df.columns:
            print(f"[WARNING] Metric '{metric}' not found in {csv_file.name}")
            continue

        dataset_name = csv_file.stem.replace("_raw", "")

        agg = (
            df.groupby("model")[metric]
            .mean()
            .reset_index()
        )

        agg["dataset"] = dataset_name

        frames.append(agg)

    if not frames:
        raise ValueError("No valid benchmark files loaded.")

    all_data = pd.concat(frames, ignore_index=True)

    matrix = all_data.pivot(
        index="dataset",
        columns="model",
        values=metric
    )

    return matrix


# ============================================================
# 2. Pairwise statistical tests
# ============================================================

def pairwise_tests(matrix: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
    """
    Run paired t-tests and Wilcoxon signed-rank tests
    for all model pairs.

    Holm-Bonferroni correction is applied separately
    to each family of tests.
    """

    models = matrix.columns.tolist()

    results = []

    for m1, m2 in combinations(models, 2):

        valid = matrix[[m1, m2]].dropna()

        if len(valid) < 2:
            results.append({
                "Model_A": m1,
                "Model_B": m2,
                "n_datasets": len(valid),
                "t_stat": np.nan,
                "t_pvalue": np.nan,
                "wilcox_stat": np.nan,
                "wilcox_pvalue": np.nan
            })
            continue

        x1 = valid[m1].values
        x2 = valid[m2].values

        # ----------------------------------------------------
        # Paired t-test
        # ----------------------------------------------------

        t_stat, t_p = stats.ttest_rel(x1, x2)

        # ----------------------------------------------------
        # Wilcoxon signed-rank test
        # ----------------------------------------------------

        try:
            w_stat, w_p = stats.wilcoxon(
                x1,
                x2,
                zero_method="wilcox"
            )
        except ValueError:
            # Happens when all differences are zero
            w_stat, w_p = np.nan, np.nan

        results.append({
            "Model_A": m1,
            "Model_B": m2,
            "n_datasets": len(valid),
            "t_stat": t_stat,
            "t_pvalue": t_p,
            "wilcox_stat": w_stat,
            "wilcox_pvalue": w_p
        })

    df_pairs = pd.DataFrame(results)

    # ========================================================
    # Holm correction
    # ========================================================

    for col in ["t_pvalue", "wilcox_pvalue"]:

        mask = df_pairs[col].notna()

        pvals = df_pairs.loc[mask, col].values

        if len(pvals) == 0:
            df_pairs[f"{col}_holm"] = np.nan
            df_pairs[f"{col}_significant"] = False
            continue

        reject, p_corr, _, _ = multipletests(
            pvals,
            alpha=alpha,
            method="holm"
        )

        df_pairs.loc[mask, f"{col}_holm"] = p_corr
        df_pairs.loc[mask, f"{col}_significant"] = reject

    return df_pairs


# ============================================================
# 3. Friedman + Nemenyi
# ============================================================

def friedman_nemenyi(matrix: pd.DataFrame, alpha: float = 0.05):
    """
    Perform Friedman test and Nemenyi post-hoc test.
    """

    try:
        import scikit_posthocs as sp
    except ImportError:
        print(
            "\n[WARNING] scikit-posthocs not installed.\n"
            "Install with:\n"
            "pip install scikit-posthocs"
        )
        return None

    # Keep only datasets with complete model results
    pivot = matrix.dropna()

    if pivot.shape[0] < 3:
        print("\n[WARNING] Friedman test requires at least 3 datasets.")
        return None

    # --------------------------------------------------------
    # Friedman test
    # --------------------------------------------------------

    friedman_stat, friedman_p = stats.friedmanchisquare(
        *[pivot[col].values for col in pivot.columns]
    )

    print("\n" + "=" * 60)
    print("FRIEDMAN TEST")
    print("=" * 60)

    print(f"Statistic : {friedman_stat:.6f}")
    print(f"P-value   : {friedman_p:.6f}")

    if friedman_p >= alpha:
        print("\nNo statistically significant difference detected.")
        return None

    print("\nSignificant difference detected.")
    print("Running Nemenyi post-hoc analysis...")

    # --------------------------------------------------------
    # Nemenyi post-hoc
    # --------------------------------------------------------

    melted = (
        pivot.reset_index()
        .melt(
            id_vars="dataset",
            var_name="model",
            value_name="value"
        )
    )

    nemenyi = sp.posthoc_nemenyi_friedman(
        melted,
        y_col="value",
        group_col="model",
        block_col="dataset"
    )

    return nemenyi


# ============================================================
# 4. Pretty printing
# ============================================================

def display_pairwise(df_pairs: pd.DataFrame, metric_name: str):

    print("\n" + "=" * 60)
    print(f"PAIRWISE TESTS ({metric_name})")
    print("=" * 60)

    cols = [
        "Model_A",
        "Model_B",
        "t_pvalue_holm",
        "t_pvalue_significant",
        "wilcox_pvalue_holm",
        "wilcox_pvalue_significant"
    ]

    display_df = df_pairs[cols].copy()

    display_df.columns = [
        "Model A",
        "Model B",
        "t-test p",
        "t-test sig",
        "Wilcoxon p",
        "Wilcoxon sig"
    ]

    print(display_df.to_string(index=False))


# ============================================================
# 5. Save outputs
# ============================================================

def save_results(
    df_pairs: pd.DataFrame,
    nemenyi_df,
    metric_name: str,
    output_dir: Path
):

    output_dir.mkdir(exist_ok=True, parents=True)

    # --------------------------------------------------------
    # Pairwise results
    # --------------------------------------------------------

    pairwise_csv = output_dir / f"pairwise_tests_{metric_name}.csv"
    pairwise_tex = output_dir / f"pairwise_tests_{metric_name}.tex"

    df_pairs.to_csv(pairwise_csv, index=False)

    with open(pairwise_tex, "w") as f:
        f.write(
            df_pairs.to_latex(
                index=False,
                float_format="%.2f"
            )
        )

    # --------------------------------------------------------
    # Nemenyi
    # --------------------------------------------------------

    if nemenyi_df is not None:

        nemenyi_csv = output_dir / f"nemenyi_{metric_name}.csv"
        nemenyi_tex = output_dir / f"nemenyi_{metric_name}.tex"

        nemenyi_df.to_csv(nemenyi_csv)

        with open(nemenyi_tex, "w") as f:
            f.write(
                nemenyi_df.to_latex(
                    float_format="%.2f"
                )
            )

    print(f"\nSaved statistical results to:\n{output_dir}")


# ============================================================
# Main
# ============================================================

def main():
    output_dir = RESULTS_DIR / "statistics"

    print("=" * 60)
    print("STATISTICAL SIGNIFICANCE ANALYSIS")
    print("=" * 60)

    print(f"\nLoading results from:\n{RESULTS_DIR}")

    matrix = load_all_results(RESULTS_DIR, METRIC)

    print("\nDatasets:")
    print(matrix.index.tolist())

    print("\nModels:")
    print(matrix.columns.tolist())

    print("\nPerformance matrix:")
    print(matrix.round(4))

    performance_matrix_csv = output_dir / f"performance_matrix.csv"
    performance_matrix_tex = output_dir / f"performance_matrix.tex"

    matrix.to_csv(performance_matrix_csv)
    matrix.to_latex(performance_matrix_tex, float_format="%.4f",index=True)

    if matrix.shape[1] < 2:
        print("\nNeed at least two models.")
        return

    # ========================================================
    # Pairwise tests
    # ========================================================

    df_pairs = pairwise_tests(matrix, ALPHA)

    display_pairwise(df_pairs, METRIC)

    # ========================================================
    # Friedman + Nemenyi
    # ========================================================

    nemenyi_df = friedman_nemenyi(matrix, ALPHA)

    if nemenyi_df is not None:

        print("\n" + "=" * 60)
        print("NEMENYI POST-HOC MATRIX")
        print("=" * 60)

        print(nemenyi_df.round(4))

    # ========================================================
    # Save
    # ========================================================

    save_results(
        df_pairs=df_pairs,
        nemenyi_df=nemenyi_df,
        metric_name=METRIC,
        output_dir=output_dir
    )


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()