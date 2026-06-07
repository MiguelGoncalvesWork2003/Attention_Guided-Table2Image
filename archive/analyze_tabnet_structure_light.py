# analyze_tabnet_structure_light.py
"""
Compact, read‑only structural analyser for TabNet artefacts.

This script provides a lightweight alternative to `analyser_of_tabnet_structure.py`
for quick inspection of a trained TabNet model. It computes basic statistics,
feature‑importance rankings, and step‑assignment distributions without any
external visualisation dependencies (except matplotlib for optional plots).

The `TabNetStructureAnalyzer` class:
  - Loads the same artefacts (importance, step assignments, configuration,
    optional masks) as the full interpreter.
  - Reports importance statistics (mean, std, Gini coefficient) and step
    utilisation rates.
  - Generates a comprehensive JSON report that can be consumed by downstream
    notebooks or the Streamlit UI.

This module is intended for programmatic use, e.g., within automated
evaluation pipelines, or when only a numerical summary is required.

Relation to the paper:
  It complements the interactive application by providing a lightweight,
  script‑friendly entry point for inspecting TabNet’s attention structure,
  reinforcing the claim that the layout’s spatial semantics are grounded in
  quantifiable, task‑aware feature relevance.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any
from pathlib import Path
import matplotlib.pyplot as plt
import json
from datetime import datetime

class TabNetStructureAnalyzer:
    """
    READ-ONLY analyzer of TabNet structure.
    Uses ONLY saved artifacts (no recomputation).
    """

    def __init__(self, dataset_name: str):
        self.dataset_name = dataset_name
        self.base_dir = Path(__file__).resolve().parents[1]
        self.output_dir = self.base_dir / "tabnet_fs" / "outputs" / f"output_{dataset_name}"

        self.masks_3d = None
        self.feature_names = None
        self.importances = None
        self.step_assignments = None
        self.config = None

        self.load_artifacts()

    # =========================================================
    # LOAD
    # =========================================================
    def load_artifacts(self):
        print(f"Loading artifacts for {self.dataset_name}...")

        importance_path = self.output_dir / "tabnet_feature_importance.csv"
        step_path = self.output_dir / "tabnet_step_assignment.csv"
        config_path = self.output_dir / "tabnet_config.json"
        masks_path = self.output_dir / "tabnet_masks.npy"

        if not importance_path.exists():
            raise FileNotFoundError(f"Missing: {importance_path}")
        if not step_path.exists():
            raise FileNotFoundError(f"Missing: {step_path}")

        # feature importance
        imp_df = pd.read_csv(importance_path)
        self.feature_names = imp_df["feature"].tolist()
        self.importances = imp_df["importance"].values

        # step assignments
        self.step_assignments = pd.read_csv(step_path)

        # config
        if config_path.exists():
            with open(config_path, "r", encoding="utf-8") as f:
                self.config = json.load(f)

        # masks optional
        if masks_path.exists():
            self.masks_3d = np.load(masks_path)

        print(f"Loaded {len(self.feature_names)} features")

    # =========================================================
    # BASIC STATS
    # =========================================================
    def analyze_basic_statistics(self) -> Dict[str, Any]:

        stats = {
            "dataset": self.dataset_name,
            "n_features": len(self.feature_names),
            "has_masks": self.masks_3d is not None,
        }

        if self.importances is not None:
            v = np.abs(self.importances)

            stats["importance_statistics"] = {
                "mean": float(np.mean(v)),
                "std": float(np.std(v)),
                "min": float(np.min(v)),
                "max": float(np.max(v)),
                "median": float(np.median(v)),
                "gini": self._gini(v)
            }

        valid = self.step_assignments[self.step_assignments["dominant_step"] >= 0]

        if len(valid) > 0 and self.config:
            steps_used = valid["dominant_step"].nunique()
            n_steps = self.config.get("n_steps", 0)

            stats["step_statistics"] = {
                "steps_used": int(steps_used),
                "n_steps": int(n_steps),
                "utilization": steps_used / max(n_steps, 1),
                "distribution": valid["dominant_step"].value_counts().to_dict()
            }

        return stats

    # =========================================================
    # GINI SAFE
    # =========================================================
    def _gini(self, x: np.ndarray) -> float:
        x = np.abs(x)
        if len(x) == 0:
            return 0.0

        x = np.sort(x)
        total = np.sum(x)

        if total == 0:
            return 0.0

        n = len(x)
        cum = np.cumsum(x)

        return (n + 1 - 2 * np.sum(cum) / cum[-1]) / n

    # =========================================================
    # FEATURE IMPORTANCE TABLE
    # =========================================================
    def analyze_feature_importance_distribution(self) -> pd.DataFrame:
        df = pd.DataFrame({
            "feature": self.feature_names,
            "importance": self.importances
        })

        df["rank"] = df["importance"].rank(ascending=False)

        if self.step_assignments is not None:
            df = df.merge(
                self.step_assignments[["feature", "dominant_step"]],
                on="feature",
                how="left"
            )

        return df.sort_values("importance", ascending=False)

    # =========================================================
    # STEP ANALYSIS
    # =========================================================
    def analyze_step_structure(self) -> Dict[str, Any]:

        valid = self.step_assignments[self.step_assignments["dominant_step"] >= 0]

        if len(valid) == 0:
            return {"available": False}

        step_dist = valid["dominant_step"].value_counts().sort_index()

        out = {
            "available": True,
            "step_distribution": step_dist.to_dict()
        }

        step_info = {}

        for step, group in valid.groupby("dominant_step"):

            imp = group["global_importance"] if "global_importance" in group else None

            step_info[int(step)] = {
                "n_features": len(group),
                "avg_importance": float(imp.mean()) if imp is not None else 0.0,
            }

        out["step_info"] = step_info

        if self.masks_3d is not None:
            out["coherence"] = self._step_coherence(valid)

        return out

    # =========================================================
    # SAFE COHERENCE (FIXED)
    # =========================================================
    def _step_coherence(self, valid_df: pd.DataFrame) -> Dict[int, float]:

        coherence = {}

        if self.masks_3d is None:
            return coherence

        feature_to_idx = {f: i for i, f in enumerate(self.feature_names)}

        for step, group in valid_df.groupby("dominant_step"):

            features = group["feature"].tolist()
            idxs = [feature_to_idx[f] for f in features if f in feature_to_idx]

            # safety
            if len(idxs) < 2:
                continue

            # SAFE INDEX CHECKS
            if step >= self.masks_3d.shape[0]:
                continue
            if max(idxs) >= self.masks_3d.shape[2]:
                continue

            step_masks = self.masks_3d[step][:, idxs]

            corr = np.corrcoef(step_masks.T)

            if corr.size == 0:
                continue

            upper = corr[np.triu_indices_from(corr, k=1)]
            val = np.nanmean(upper) if len(upper) > 0 else 0.0

            coherence[int(step)] = float(val)

        return coherence

    # =========================================================
    # REPORT
    # =========================================================
    def create_comprehensive_report(self) -> Dict[str, Any]:

        stats = self.analyze_basic_statistics()

        report = {
            "dataset": self.dataset_name,
            "timestamp": datetime.now().isoformat(),
            "basic_statistics": stats,
            "feature_importance": self.analyze_feature_importance_distribution().to_dict("records"),
            "step_analysis": self.analyze_step_structure()
        }

        return report

def analyze_tabnet_structure(dataset_name: str):

    print("=" * 80)
    print(f"TABNET STRUCTURE ANALYSIS: {dataset_name}")
    print("=" * 80)

    analyzer = TabNetStructureAnalyzer(dataset_name)
    report = analyzer.create_comprehensive_report()

    print("\nSUMMARY")
    print("-" * 40)
    print("Features:", report["basic_statistics"]["n_features"])

    if "step_analysis" in report and report["step_analysis"].get("available"):
        print("Steps:", len(report["step_analysis"]["step_distribution"]))

    return report

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    args = parser.parse_args()

    analyze_tabnet_structure(args.dataset)