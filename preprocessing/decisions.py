#decisions.py
"""
Preprocessing configuration builder.

This module translates human‑readable choices (from the Streamlit UI or
environment variables) into a **deterministic, serialisable, and hashable
`PreprocessingConfig` object**. The configuration defines *how* every step of
the preprocessing pipeline will be executed, ensuring **full reproducibility**
across runs and allowing the entire transformation to be audited and recomputed
from the raw data.

The `PreprocessingConfig` dataclass captures:
  - Which features to remove (user‑specified),
  - Missing‑value handling strategies for numerical and categorical columns,
  - Encoding and scaling schemes (all chosen to be TabNet‑compatible),
  - Model‑aware settings (e.g., sparse representation, no categorical embedding),
  - Data‑splitting ratio and random seed.

**Role in the Map–Optimize–Learn pipeline:**
- **Map:**  The configuration explicitly defines the transformation map from the
  raw feature space to the normalised tensor that will be consumed by TabNet and,
  later, converted into an image.
- **Optimize:**  The deterministic configuration replaces an external
  metaheuristic search for optimal preprocessing. Instead, decisions are
  grounded in dataset inspection and principled heuristics, making the
  preprocessing part of the “optimize” stage transparent and debuggable.

The module also provides `build_preprocessing_config_from_env()` to bridge
Streamlit inputs and the `PreprocessingConfig` constructor.
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, List
from dataclasses import dataclass, asdict
import hashlib

@dataclass
class PreprocessingConfig:
    """Explicit preprocessing configuration as a data class."""
    # Dataset info
    dataset_name: str
    target_column: str
    features_to_remove: List[str]
    
    # Missing value handling
    missing_threshold: float = 0.5  # Drop columns with ≥ this missing ratio
    numerical_missing_strategy: str = "median"  # median | mean | constant
    categorical_missing_strategy: str = "explicit"  # explicit | drop
    
    # Encoding
    encode_categoricals: bool = True
    encoding_strategy: str = "label"  # label | target (for TabNet compatibility)
    
    # Scaling
    scaling_strategy: str = "standard"  # none | standard | robust | minmax
    
    # Feature engineering (optional)
    create_interactions: bool = False
    polynomial_degree: int = 1  # 1 = no polynomials
    
    # Model-aware preprocessing
    model_type: str = "tabnet_cnn"  # tabnet_cnn | other (for future extensibility)
    tabnet_specific: Dict[str, Any] = None  # TabNet-specific preprocessing
    
    # Reproducibility
    random_seed: int = 42
    split_ratio: float = 0.3  # Test size
    
    def __post_init__(self):
        """Validate configuration."""
        assert 0 <= self.missing_threshold <= 1, "Missing threshold must be between 0 and 1"
        assert self.numerical_missing_strategy in ["median", "mean", "constant"]
        assert self.categorical_missing_strategy in ["explicit", "drop"]
        assert self.encoding_strategy in ["label"]
        assert self.scaling_strategy in ["none", "standard", "robust", "minmax"]
        assert 0 < self.split_ratio < 1, "Split ratio must be between 0 and 1"
        
        # Initialize TabNet-specific config
        if self.tabnet_specific is None:
            self.tabnet_specific = {
                "sparse_representation": True,
                "categorical_embedding": False,  # TabNet prefers label encoding
                "feature_grouping": "auto"  # auto | manual
            }
    
    @property
    def config_hash(self) -> str:
        """Generate deterministic hash for this configuration."""
        config_str = json.dumps(asdict(self), sort_keys=True)
        return hashlib.sha256(config_str.encode()).hexdigest()[:16]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)
    
    def save(self, output_dir: Path) -> Path:
        """Save configuration to disk."""
        config_path = output_dir / "preprocessing_config.json"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        config_dict = self.to_dict()
        config_dict["config_hash"] = self.config_hash
        
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=2, default=str)
        
        return config_path
    
    @classmethod
    def load(cls, config_path: Path) -> "PreprocessingConfig":
        """Load configuration from disk."""
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        
        # Remove hash for reconstruction
        config_dict.pop("config_hash", None)
        
        return cls(**config_dict)

def build_preprocessing_config_from_env() -> PreprocessingConfig:
    """
    Build configuration from environment variables (from Streamlit).
    
    Returns:
        PreprocessingConfig object
    """
    # Get features to remove
    features_to_remove_str = os.environ.get("FEATURES_TO_REMOVE", "")
    features_to_remove = [f.strip() for f in features_to_remove_str.split(",") 
                         if f.strip()] if features_to_remove_str else []
    
    # Map environment variables to configuration
    cat_missing = os.environ.get("CAT_MISSING", "Treat as category")
    categorical_strategy = "explicit" if cat_missing == "Treat as category" else "drop"
    
    # Build configuration
    config = PreprocessingConfig(
        dataset_name=os.environ.get("DATASET", "unknown"),
        target_column=os.environ.get("TARGET_COL", ""),
        features_to_remove=features_to_remove,
        missing_threshold=float(os.environ.get("DROP_THRESHOLD", 0.5)),
        numerical_missing_strategy=os.environ.get("NUM_MISSING", "Median").lower(),
        categorical_missing_strategy=categorical_strategy,
        scaling_strategy=os.environ.get("SCALING", "standard").lower(),
        encode_categoricals=True,  # Always true for TabNet
        model_type="tabnet_cnn",  # Fixed for this pipeline
    )
    
    return config

def create_config_summary(config: PreprocessingConfig) -> Dict[str, Any]:
    """
    Create human-readable summary of configuration.
    
    Returns:
        Summary dictionary for UI display
    """
    return {
        "dataset": config.dataset_name,
        "target_column": config.target_column,
        "features_removed": len(config.features_to_remove),
        "missing_handling": {
            "threshold": f"{config.missing_threshold:.0%}",
            "numerical": config.numerical_missing_strategy,
            "categorical": config.categorical_missing_strategy
        },
        "encoding": {
            "strategy": config.encoding_strategy,
            "encode_categoricals": config.encode_categoricals
        },
        "scaling": config.scaling_strategy,
        "model_aware": config.model_type,
        "reproducibility": {
            "random_seed": config.random_seed,
            "config_hash": config.config_hash
        }
    }