#__init__.py
"""
Layout strategies package.

Re‑exports the base layout class and all concrete strategies from
`unified_layouts.py`, together with helper functions for dynamic creation and
validation. This package is the public interface used by the Streamlit
orchestrator and the image generation stage.

Usage:
  from layouts import StepRowLayout, create_layout

  layout = create_layout("step_row", step_df)
  shape = layout.compute_image_shape()   # (channels, height, width)
  row, col = layout.map_feature(step, local_rank)
"""

from .unified_layouts import (
    BaseLayoutStrategy,
    StepRowLayout,
    PackedLayout,
    StepSparseLayout,
    AttentionMapLayout,
    create_layout,
    get_available_layouts,
    validate_layout_name,
    create_layout_from_config
)

__all__ = [
    'BaseLayoutStrategy',
    'StepRowLayout',
    'PackedLayout',
    'StepSparseLayout',
    'AttentionMapLayout',
    'create_layout',
    'get_available_layouts',
    'validate_layout_name',
    'create_layout_from_config'
]