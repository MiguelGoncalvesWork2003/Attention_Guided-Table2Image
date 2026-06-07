#unified_layouts.py
"""
Unified, object‑oriented design of layout strategies for attention‑guided
tabular‑to‑image generation.

This module provides a clean, extensible hierarchy for defining how TabNet’s
step assignments and importance scores are translated into 2D image coordinates.
It abstracts the layout logic away from the builder, making it easy to
experiment with new spatial organisations without modifying the core pipeline.

Classes:
  - `BaseLayoutStrategy` – groups features by dominant step, optionally
    collapses empty rows, and defines the interface (`compute_image_shape`,
    `map_feature`).
  - `StepRowLayout` – each TabNet decision step occupies one row; features
    within a step are placed left‑to‑right in descending importance.
  - `PackedLayout` – flattens all features into a dense grid sorted by global
    importance, maximising spatial locality for CNNs.
  - `StepSparseLayout` – allocates a fixed number of columns per step,
    preserving empty slots for interpretability.
  - `AttentionMapLayout` – preserves the full soft attention distribution,
    creating an image where rows are steps and columns are features.

Factory functions (`create_layout`, `create_layout_from_config`) enable
instantiation by name, facilitating integration with the Streamlit interface
and configuration files.

All strategies are **deterministic**: given the same step assignment DataFrame,
they always produce the same coordinate mapping. They are explicitly designed
to be “CNN‑agnostic” – they specify only the spatial layout, leaving the actual
image generation and CNN training to separate modules.

Relation to the paper:
  The five layout strategies correspond to the AG‑T2I‑StepRow, AG‑T2I‑StepSparse,
  AG‑T2I‑PackedRow, AG‑T2I‑PackedCol, and AG‑T2I‑AttentionMap variants evaluated
  in the experimental section. This module therefore provides the exact
  implementations used in the empirical comparison, enabling full reproducibility.
"""

import numpy as np
from typing import Dict, List, Tuple, Any
import pandas as pd
import json

class BaseLayoutStrategy:
    """Base class for all layout strategies."""
    
    def __init__(self, step_df: pd.DataFrame, collapse_empty_rows: bool = True):
        """
        Initialize layout with TabNet step assignments.
        
        Args:
            step_df: DataFrame with columns ['feature', 'dominant_step', 'global_importance']
            collapse_empty_rows: If True, remove empty rows (steps with no features)
        """
        self.step_df = step_df.sort_values(
            ["dominant_step", "global_importance"],
            ascending=[True, False]  # Steps ascending, importance descending
        )
        self.collapse_empty_rows = collapse_empty_rows
        self.step_groups, self.step_remapping = self._group_features_by_step()
        
    def _group_features_by_step(self):
        """Group features by their dominant step and create step remapping."""
        step_groups = {}
        
        # First group by original step
        for step in self.step_df['dominant_step'].unique():
            step_features = self.step_df[self.step_df['dominant_step'] == step]['feature'].tolist()
            if step_features:
                step_groups[int(step)] = step_features
        
        # Create step remapping if collapsing empty rows
        step_remapping = {}
        if self.collapse_empty_rows:
            # Create compressed step numbers (0, 1, 2...)
            sorted_steps = sorted(step_groups.keys())
            for new_step, old_step in enumerate(sorted_steps):
                step_remapping[old_step] = new_step
            
            # Create new step_groups with compressed steps
            compressed_step_groups = {}
            for old_step, features in step_groups.items():
                new_step = step_remapping[old_step]
                compressed_step_groups[new_step] = features
            
            return compressed_step_groups, step_remapping
        
        # No remapping needed
        for step in step_groups.keys():
            step_remapping[step] = step
        
        return step_groups, step_remapping
    
    def compute_image_shape(self) -> Tuple[int, int, int]:
        """
        Compute image dimensions (channels, height, width).
        
        Returns:
            Tuple of (channels, height, width)
        """
        raise NotImplementedError
        
    def map_feature(self, step: int, local_rank: int) -> Tuple[int, int]:
        """
        Map a feature to image coordinates.
        
        Args:
            step: TabNet decision step (0-indexed, after compression if collapse_empty_rows=True)
            local_rank: Rank within the step (importance order)
            
        Returns:
            (row, col) coordinates in the image
        """
        raise NotImplementedError
    
    @property
    def name(self):
        return self.__class__.name
    
    @property
    def description(self):
        return self.__class__.description

class StepRowLayout(BaseLayoutStrategy):
    """
    STEP ROW LAYOUT: Each TabNet step = one row.
    
    Best for: Understanding which features belong to which decision step.
    """
    
    name = "step_row"
    description = "Rows = TabNet steps, Columns = features per step. Preserves TabNet's step-wise structure."
    
    def __init__(self, step_df: pd.DataFrame, collapse_empty_rows: bool = True):
        super().__init__(step_df, collapse_empty_rows)
        
    def compute_image_shape(self) -> Tuple[int, int, int]:
        n_steps = len(self.step_groups)
        max_features_per_step = max(len(features) for features in self.step_groups.values()) if self.step_groups else 0
        
        return (1, n_steps, max_features_per_step)
    
    def map_feature(self, step: int, local_rank: int) -> Tuple[int, int]:
        return (step, local_rank)

class PackedLayout(BaseLayoutStrategy):
    name = "packed"
    description = "Dense packing: ..."

    def __init__(self, step_df, target_width=16, transpose=False, collapse_empty_rows=True):
        super().__init__(step_df, collapse_empty_rows)
        self.target_width = target_width
        self.transpose = transpose
        self.global_features = list(self.step_df['feature'])
        self.feature_to_global_idx = {feature: idx for idx, feature in enumerate(self.global_features)}

    def compute_image_shape(self):
        total = len(self.step_df)
        w = min(self.target_width, total) if total > 0 else 1
        h = (total + w - 1) // w
        if self.transpose:
            return (1, w, h)          # swap height & width
        return (1, h, w)

    def map_feature(self, step, local_rank):
        return (0, 0)   # not used directly

    def map_feature_by_name(self, feature_name):
        if feature_name not in self.feature_to_global_idx:
            return (0, 0)
        total = len(self.step_df)
        orig_w = min(self.target_width, total)
        orig_h = (total + orig_w - 1) // orig_w
        idx = self.feature_to_global_idx[feature_name]
        row = idx // orig_w
        col = idx % orig_w
        if self.transpose:
            return (col, row)         # swap for transpose
        return (row, col)

class StepSparseLayout(BaseLayoutStrategy):
    """
    STEP SPARSE LAYOUT: Each step gets fixed columns, empty slots preserved.
    
    Best for: Interpretability and understanding step boundaries.
    """
    
    name = "step_sparse"
    description = "Step-separated: Fixed columns per step, empty slots preserved. Good for interpretability."
    
    def __init__(self, step_df: pd.DataFrame, columns_per_step: int = 10, collapse_empty_rows: bool = True):
        super().__init__(step_df, collapse_empty_rows)
        self.columns_per_step = columns_per_step
        
    def compute_image_shape(self) -> Tuple[int, int, int]:
        n_steps = len(self.step_groups)
        return (1, n_steps, self.columns_per_step)
    
    def map_feature(self, step: int, local_rank: int) -> Tuple[int, int]:
        col = min(local_rank, self.columns_per_step - 1)
        return (step, col)

class AttentionMapLayout(BaseLayoutStrategy):
    name = "attention_map"
    description = "Full attention matrix: rows = steps, columns = features"

    def __init__(self, step_df: pd.DataFrame, collapse_empty_rows: bool = True):
        super().__init__(step_df, collapse_empty_rows=False)
        if 'step_distribution' not in step_df.columns:
            raise ValueError("AttentionMapLayout requires 'step_distribution' column")
        
        first_dist = step_df.iloc[0]['step_distribution']
        if isinstance(first_dist, str):
            first_dist = json.loads(first_dist)
        self.n_steps = len(first_dist)
        self.n_features = len(step_df)
        
        self.global_importance = step_df['global_importance'].values.astype(np.float32)
        self.step_distributions = []
        for dist in step_df['step_distribution']:
            if isinstance(dist, str):
                dist = json.loads(dist)
            self.step_distributions.append(np.array(dist, dtype=np.float32))
        self.step_distributions = np.array(self.step_distributions)  # (n_features, n_steps)

    def compute_image_shape(self):
        return (1, self.n_steps, self.n_features)

    def map_feature(self, step, local_rank):
        return (step, local_rank)  # not used

    def get_weight_matrix(self):
        """Return the weight matrix (n_steps × n_features).

        Each element = soft_step_distribution[feature, step]
                    × global_importance[feature] (normalised to [0,1]).

        No sample values are involved – this matrix is fixed for all samples.
        The image builder will later multiply by the feature values and scale.
        """
        attention_T = self.step_distributions.T.astype(np.float32)  # (steps, features)
        importance = self.global_importance.astype(np.float32)
        importance = importance / (importance.max() + 1e-8)
        weight_matrix = attention_T * importance[None, :]           # broadcast
        return weight_matrix

def create_layout(layout_name: str, step_df: pd.DataFrame, **kwargs) -> BaseLayoutStrategy:
    """
    Create a layout instance by name.
    """
    collapse_empty_rows = kwargs.get('collapse_empty_rows', True)
    
    if layout_name == "step_row":
        return StepRowLayout(step_df, collapse_empty_rows=collapse_empty_rows)
    elif layout_name == "packed":
        target_width = kwargs.get('target_width', 16)
        return PackedLayout(step_df, target_width=target_width, collapse_empty_rows=collapse_empty_rows)
    elif layout_name == "packed_T":
        target_width = kwargs.get('target_width', 16)
        transpose = kwargs.get('transpose', False)
        return PackedLayout(step_df, target_width=target_width,
                            transpose=True,
                            collapse_empty_rows=collapse_empty_rows)
    elif layout_name == "step_sparse":
        columns_per_step = kwargs.get('columns_per_step', 10)
        return StepSparseLayout(step_df, columns_per_step=columns_per_step, collapse_empty_rows=collapse_empty_rows)
    elif layout_name == "attention_map":
        # collapse_empty_rows is ignored; we keep all steps and features
        return AttentionMapLayout(step_df, collapse_empty_rows=False)
    else:
        raise ValueError(f"Unknown layout: {layout_name}. Available: step_row, packed, step_sparse, attention_map")

def get_available_layouts() -> List[str]:
    """Get list of all available layout names."""
    return ["step_row", "packed", "packed_T", "step_sparse", "attention_map"]

def validate_layout_name(layout_name: str) -> bool:
    """Validate that a layout name exists."""
    return layout_name in get_available_layouts()

def create_layout_from_config(layout_name: str, step_df: pd.DataFrame, **kwargs) -> BaseLayoutStrategy:
    """
    Create layout instance from configuration.
    """
    if not validate_layout_name(layout_name):
        available = get_available_layouts()
        raise ValueError(f"Invalid layout '{layout_name}'. Available: {available}")
    
    return create_layout(layout_name, step_df, **kwargs)