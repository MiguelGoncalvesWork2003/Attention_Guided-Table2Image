#layout_builder.py
"""
Deterministic spatial layout builder for attention‑guided tabular‑to‑image
representations.

This module implements the **Map** stage of the MOL paradigm: it converts the
frozen TabNet step assignments and feature importance scores into concrete
two‑dimensional grid layouts. Once produced, these layouts serve as the fixed
canvas onto which individual tabular samples are projected to form
CNN‑compatible images.

The `LayoutBuilder` class is designed with a strict “no learning, no
interpretation, no CNN constraints” philosophy:
  - It loads the **authentic** step assignments and importance from saved CSVs
    (produced by `train_tabnet.py`).
  - All layout functions are purely deterministic – given the same assignments,
    the same pixel coordinates are always produced.
  - No feedback from the CNN stage is used, and no hyperparameter search is
    performed on the layout itself.

Provided layout strategies include:
  - `importance_grid` – square grid sorted by global importance.
  - `packed` – row‑major packing, optionally grouped by step.
  - `step_rows` – each TabNet step becomes one image row.
  - `step_sparse` – step‑separated columns with empty slots preserved.

For every layout, the builder exports a CSV mapping feature → (row, col),
a NumPy matrix of the same shape filled with importance scores, and a summary
JSON.

Relation to the paper:
  The deterministic mapping from learned attention to image coordinates is
  the core contribution of the framework. This module is the exact
  implementation of the attention‑to‑grid mapping described in Section 4,
  ensuring that the spatial organisation of the CNN input is directly
  interpretable through its connection to TabNet’s supervised feature masks.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import json
from datetime import datetime

class LayoutBuilder:
    """
    Deterministic layout builder for TabNet representations.
    No fallback learning logic. Assumes valid TabNet artifacts.
    """

    def __init__(self, dataset_name: str):
        self.dataset_name = dataset_name
        self.base_dir = Path(__file__).resolve().parents[1]
        self.output_dir = self.base_dir / "tabnet_fs" / "outputs" / f"output_{dataset_name}"

        self.feature_names = None
        self.feature_importance = None
        self.step_assignments = None
        self.assignment_source = None

        self._load_artifacts()

    def _load_artifacts(self):
        """Load required TabNet outputs (strict mode)."""

        importance_path = self.output_dir / "tabnet_feature_importance.csv"
        step_path = self.output_dir / "tabnet_step_assignment.csv"

        if not importance_path.exists():
            raise FileNotFoundError(f"Missing feature importance: {importance_path}")

        if not step_path.exists():
            raise FileNotFoundError(f"Missing step assignments: {step_path}")

        importance_df = pd.read_csv(importance_path)
        self.feature_names = importance_df["feature"].tolist()
        self.feature_importance = dict(
            zip(importance_df["feature"], importance_df["importance"])
        )

        self.step_assignments = pd.read_csv(step_path)

        if "assignment_source" in self.step_assignments.columns:
            self.assignment_source = self.step_assignments["assignment_source"].iloc[0]
        else:
            self.assignment_source = "unknown"

        # STRICT VALIDATION (important for thesis correctness)
        if "dominant_step" not in self.step_assignments.columns:
            raise ValueError("Missing dominant_step column in assignments")

        if len(self.feature_names) != len(self.step_assignments):
            raise ValueError(
                "Feature count mismatch between importance and assignments"
            )

        print(f"Loaded {len(self.feature_names)} features")
        print(f"Assignment source: {self.assignment_source}")

    # ------------------------------------------------------------
    # CORE LAYOUTS
    # ------------------------------------------------------------

    def create_step_rows_layout(self) -> Dict[str, Tuple[int, int]]:
        """
        Each TabNet step = one row.
        Features ordered by importance within step.
        """

        df = self.step_assignments.copy()

        step_groups = {}

        for _, row in df.iterrows():
            step = int(row["dominant_step"])
            if step < 0:
                continue

            step_groups.setdefault(step, []).append(
                (row["feature"], row["global_importance"])
            )

        # sort within each step
        for s in step_groups:
            step_groups[s].sort(key=lambda x: x[1], reverse=True)

        layout = {}
        for row_idx, step in enumerate(sorted(step_groups.keys())):
            for col_idx, (feat, _) in enumerate(step_groups[step]):
                layout[feat] = (row_idx, col_idx)

        return layout

    def create_importance_grid_layout(self) -> Dict[str, Tuple[int, int]]:
        """
        Pure importance-based square grid.
        """

        sorted_feats = sorted(
            self.feature_importance.items(),
            key=lambda x: x[1],
            reverse=True
        )

        n = len(sorted_feats)
        grid = int(np.ceil(np.sqrt(n)))

        layout = {}

        for i, (feat, _) in enumerate(sorted_feats):
            layout[feat] = (i // grid, i % grid)

        return layout

    def create_packed_layout(self, grid_cols: Optional[int] = None) -> Dict[str, Tuple[int, int]]:
        """
        Row-major packing, optionally grouped by step.
        """

        df = self.step_assignments.copy()

        if grid_cols is None:
            grid_cols = int(np.ceil(np.sqrt(len(df))))

        if "dominant_step" in df.columns:
            df = df.sort_values(["dominant_step", "global_importance"], ascending=[True, False])
        else:
            df = df.sort_values(["global_importance"], ascending=False)

        layout = {}

        for i, row in enumerate(df.itertuples()):
            layout[row.feature] = (i // grid_cols, i % grid_cols)

        return layout

    def create_step_sparse_layout(self, grid_cols: int = 8) -> Dict[str, Tuple[int, int]]:
        """
        Step-separated sparse column layout.
        Each step occupies a column block.
        """

        df = self.step_assignments.copy()
        df = df[df["dominant_step"] >= 0]

        step_groups = {}
        for _, row in df.iterrows():
            step_groups.setdefault(int(row["dominant_step"]), []).append(row["feature"])

        if not step_groups:
            raise ValueError("No valid step structure for sparse layout")

        n_steps = len(step_groups)
        cols_per_step = max(1, grid_cols // n_steps)

        layout = {}

        for step_idx, step in enumerate(sorted(step_groups.keys())):
            start_col = step_idx * cols_per_step
            features = step_groups[step]

            for i, feat in enumerate(features[:cols_per_step]):
                layout[feat] = (step_idx, start_col + i)

        return layout

    # ------------------------------------------------------------
    # MATRIX UTILITIES
    # ------------------------------------------------------------

    def get_grid_dims(self, layout: Dict[str, Tuple[int, int]]) -> Tuple[int, int]:
        if not layout:
            return (0, 0)

        rows = max(v[0] for v in layout.values()) + 1
        cols = max(v[1] for v in layout.values()) + 1

        return rows, cols

    def layout_to_matrix(self, layout: Dict[str, Tuple[int, int]]) -> np.ndarray:
        rows, cols = self.get_grid_dims(layout)

        if rows == 0 or cols == 0:
            return np.array([])

        matrix = np.zeros((rows, cols), dtype=float)

        for feat, (r, c) in layout.items():
            matrix[r, c] = self.feature_importance.get(feat, 0.0)

        return matrix

    # ------------------------------------------------------------
    # SAVE
    # ------------------------------------------------------------

    def save_layouts(self, layouts: Dict[str, Dict[str, Tuple[int, int]]]):
        out_dir = self.output_dir / "layouts"
        out_dir.mkdir(parents=True, exist_ok=True)

        for name, layout in layouts.items():

            df = pd.DataFrame([
                {
                    "feature": f,
                    "row": r,
                    "col": c,
                    "importance": self.feature_importance.get(f, 0.0)
                }
                for f, (r, c) in layout.items()
            ])

            df.to_csv(out_dir / f"{name}.csv", index=False)

            np.save(out_dir / f"{name}.npy", self.layout_to_matrix(layout))

        summary = {
            "dataset": self.dataset_name,
            "assignment_source": self.assignment_source,
            "n_features": len(self.feature_names),
            "layouts": list(layouts.keys()),
            "timestamp": datetime.now().isoformat()
        }

        with open(out_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        print(f"Saved layouts to {out_dir}")
        return summary

def build_layouts(dataset_name: str):
    print(f"Building layouts for {dataset_name}")

    builder = LayoutBuilder(dataset_name)

    layouts = {
        "importance_grid": builder.create_importance_grid_layout(),
        "packed": builder.create_packed_layout(),
        "step_rows": builder.create_step_rows_layout(),
        "step_sparse": builder.create_step_sparse_layout()
    }

    return builder.save_layouts(layouts)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    args = parser.parse_args()

    build_layouts(args.dataset)