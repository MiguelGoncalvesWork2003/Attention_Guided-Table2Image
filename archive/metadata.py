#metadata.py
"""
Legacy structured logging and configuration module for TabNet experiments.
Originally designed to capture experiment metadata, configuration hashes,
and training histories in a reproducible format.

In the current Map–Optimize–Learn framework, logging and artifact
persistence are handled directly by `train_tabnet.py` (which saves configs,
metrics, and summaries as part of its output directory). The dataclasses
`TabNetConfig` and `TabNetResults`, along with `TabNetExperimentLogger`,
are not imported or instantiated anywhere in the execution pipeline.

Retained for reference and possible standalone experiments; safe to archive.
"""

import json
import hashlib
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from datetime import datetime

@dataclass
class TabNetConfig:
    """Complete TabNet configuration for reproducible experiments."""
    
    # Dataset info
    dataset_name: str
    target_column: str
    n_features: int
    n_classes: int
    sample_size: int
    
    # TabNet architecture
    n_steps: int = 5
    step_dim: int = 8  # Maps to n_d in TabNet
    attn_dim: int = 8  # Maps to n_a in TabNet
    gamma: float = 1.5
    lambda_sparse: float = 1e-4
    cat_idxs: List[int] = field(default_factory=list)
    cat_dims: List[int] = field(default_factory=list)
    
    # Training parameters
    max_epochs: int = 100
    patience: int = 20
    batch_size: int = 32
    virtual_batch_size: int = 16
    learning_rate: float = 2e-2
    momentum: float = 0.02
    
    # Representation learning specific
    n_steps_a: int = 3
    n_steps_d: int = 3
    epsilon: float = 1e-15
    mask_type: str = "sparsemax"  # sparsemax | entmax
    
    # Feature selection
    feature_threshold: float = 0.0
    min_class_samples: int = 2  # For rare class detection (but not removal)
    
    # Reproducibility
    random_seed: int = 42
    split_ratio: float = 0.2
    
    # Post-processing
    categorical_detection_threshold: Optional[int] = None  # Disabled by default
    retain_top_k_features: Optional[int] = None
    
    def __post_init__(self):
        """Validate configuration."""
        assert self.n_steps > 0, "n_steps must be positive"
        assert self.feature_threshold >= 0, "feature_threshold must be non-negative"
        assert self.min_class_samples >= 2, "min_class_samples must be at least 2"
        assert 0 < self.split_ratio < 1, "split_ratio must be between 0 and 1"
    
    @property
    def config_hash(self) -> str:
        """Generate deterministic hash for this configuration."""
        config_dict = asdict(self)
        # Remove non-deterministic fields
        config_dict.pop('cat_idxs', None)
        config_dict.pop('cat_dims', None)
        
        config_str = json.dumps(config_dict, sort_keys=True)
        return hashlib.sha256(config_str.encode()).hexdigest()[:16]
    
    def save(self, output_dir: Path) -> Path:
        """Save configuration to disk."""
        output_dir.mkdir(parents=True, exist_ok=True)
        config_path = output_dir / "tabnet_config.json"
        
        config_dict = asdict(self)
        config_dict["config_hash"] = self.config_hash
        config_dict["timestamp"] = datetime.now().isoformat()
        
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=2, default=str)
        
        return config_path
    
    @classmethod
    def load(cls, config_path: Path) -> "TabNetConfig":
        """Load configuration from disk."""
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        
        # Remove metadata fields
        metadata_fields = ["config_hash", "timestamp"]
        for field in metadata_fields:
            config_dict.pop(field, None)
        
        return cls(**config_dict)

@dataclass
class TabNetResults:
    """Structured results from TabNet representation learning."""
    
    # Model performance
    train_accuracy: float
    test_accuracy: float
    train_loss: float
    test_loss: float
    convergence_epoch: int
    
    # Representation statistics
    n_assigned_features: int
    n_steps_used: int
    feature_utilization_ratio: float
    step_distribution: Dict[int, int]  # step -> n_features
    
    # Feature importance
    top_features: List[str]
    mean_feature_importance: float
    importance_std: float
    
    # Structural properties
    stepwise_mask_sparsity: Dict[int, float]  # step -> sparsity
    attention_concentration: Dict[int, float]  # step -> concentration
    
    # Layout information (just the name, not config)
    layout_name: str = "step_row"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)

class TabNetExperimentLogger:
    """Logger for TabNet experiments with structured outputs."""
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.log_file = output_dir / "experiment_log.txt"
        self.metrics_history = []
        
    def log_message(self, message: str, level: str = "INFO"):
        """Log a message with timestamp."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] [{level}] {message}"
        
        print(log_entry)
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(log_entry + "\n")
    
    def log_metrics(self, epoch: int, metrics: Dict[str, float]):
        """Log training metrics."""
        metrics['epoch'] = epoch
        metrics['timestamp'] = datetime.now().isoformat()
        self.metrics_history.append(metrics)
    
    def save_results(self, results: TabNetResults, config: TabNetConfig):
        """Save complete experiment results."""
        # Save metrics history
        metrics_df = pd.DataFrame(self.metrics_history)
        metrics_path = self.output_dir / "training_metrics.csv"
        metrics_df.to_csv(metrics_path, index=False)
        
        # Save results
        results_dict = results.to_dict()
        results_dict['config_hash'] = config.config_hash
        
        results_path = self.output_dir / "experiment_results.json"
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results_dict, f, indent=2, default=str)
        
        # Create experiment summary
        summary = self._create_summary(results, config, metrics_df)
        summary_path = self.output_dir / "experiment_summary.txt"
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write(summary)
    
    def _create_summary(self, results: TabNetResults, config: TabNetConfig, 
                       metrics_df: pd.DataFrame) -> str:
        """Create human-readable experiment summary."""
        summary_lines = [
            "=" * 80,
            "TABNET REPRESENTATION LEARNING EXPERIMENT SUMMARY",
            "=" * 80,
            f"Dataset: {config.dataset_name}",
            f"Target: {config.target_column}",
            f"Configuration Hash: {config.config_hash}",
            f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "DATABASE CHARACTERISTICS",
            "-" * 40,
            f"Original Features: {config.n_features}",
            f"Assigned Features: {results.n_assigned_features}",
            f"Feature Utilization Ratio: {results.feature_utilization_ratio:.2%}",
            f"Classes: {config.n_classes}",
            f"Samples: {config.sample_size}",
            "",
            "REPRESENTATION STRUCTURE",
            "-" * 40,
            f"Decision Steps: {config.n_steps}",
            f"Steps Used: {results.n_steps_used}",
            f"Layout Used: {results.layout_name}",  # Simple layout name only
        ]
        
        # Add step distribution
        summary_lines.append("Step Distribution:")
        for step, n_features in sorted(results.step_distribution.items()):
            summary_lines.append(f"  Step {step}: {n_features} features")
        
        # Add performance
        summary_lines.extend([
            "",
            "PERFORMANCE METRICS",
            "-" * 40,
            f"Training Accuracy: {results.train_accuracy:.2%}",
            f"Test Accuracy: {results.test_accuracy:.2%}",
            f"Training Loss: {results.train_loss:.4f}",
            f"Test Loss: {results.test_loss:.4f}",
            f"Convergence Epoch: {results.convergence_epoch}",
        ])
        
        # Add feature importance summary
        summary_lines.extend([
            "",
            "FEATURE IMPORTANCE",
            "-" * 40,
            f"Mean Importance: {results.mean_feature_importance:.4f}",
            f"Importance STD: {results.importance_std:.4f}",
            f"Top Features: {', '.join(results.top_features[:5])}",
        ])
        
        # Add structural properties
        summary_lines.extend([
            "",
            "STRUCTURAL PROPERTIES",
            "-" * 40,
        ])
        for step, sparsity in results.stepwise_mask_sparsity.items():
            concentration = results.attention_concentration.get(step, 0.0)
            summary_lines.append(f"Step {step}: Sparsity={sparsity:.2%}, Concentration={concentration:.3f}")
        
        summary_lines.append("=" * 80)
        
        return "\n".join(summary_lines)