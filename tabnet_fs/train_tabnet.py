# train_tabnet.py
"""
TabNet training script for the attention-guided tabular-to-image framework.

This script implements the **supervised attention learning stage** of the proposed
attention-guided tabular-to-image transformation: it trains an interpretable
TabNet model to learn supervised, task-specific feature attention. The resulting
sparse masks serve as the single source of truth for all subsequent deterministic
spatial layouts.

Key design choices:
  - Only authentic, gradient-based feature attention is used – no synthetic or
    fallback assignments are ever created.
  - A dedicated validation split ensures reliable early stopping.
  - The training procedure is fully reproducible (fixed seed, deterministic
    CUDA settings) and all hyperparameters are recorded.
  - All output artefacts (feature importance, stepwise masks, step assignments,
    trained model, configuration, and metrics) are saved in a structured
    directory, enabling read‑only interpretation and deterministic layout
    construction in later stages.

When executed, it:
  1. Loads the preprocessed tabular data (numpy arrays produced by the
     preprocessing pipeline).
  2. Creates and trains a `TabNetClassifier` with the configured number of
     decision steps `K`.
  3. Extracts the 3D attention masks of shape (K, N, F) and the global
     feature importance vector.
  4. Computes per‑feature step assignments by averaging masks across samples
     and taking the argmax step.
  5. Evaluates train / test accuracy and persists all artefacts.

Relation to the paper:
  The step assignments and importance scores are used by the **Layout Builder**
  to construct the CNN‑compatible image representations. The training
  process embodies the "implicit optimisation" described in Section 4:
  instead of a population‑based metaheuristic, the layout geometry emerges
  directly from supervised learning of feature attention.

Requires: `pytorch-tabnet`, preprocessed data in `data/processed/<dataset>/`.
"""

import numpy as np
import pandas as pd
import torch
import warnings
from pathlib import Path
import os
import sys
import json
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import traceback

from sklearn.model_selection import train_test_split

sys.path.append(str(Path(__file__).resolve().parents[1]))

try:
    from pytorch_tabnet.tab_model import TabNetClassifier
    TABNET_AVAILABLE = True
except ImportError:
    warnings.warn(
        "pytorch-tabnet not installed. Install with: pip install pytorch-tabnet"
    )
    TABNET_AVAILABLE = False

class TabNetTrainingConfig:
    """Configuration for TabNet training."""

    def __init__(self, dataset_name, target_column):
        self.dataset_name = dataset_name
        self.target_column = target_column

        # TabNet architecture
        self.n_steps = int(os.environ.get("TABNET_N_STEPS", "6"))
        self.n_d = int(os.environ.get("TABNET_STEP_DIM", "8"))
        self.n_a = int(os.environ.get("TABNET_ATTN_DIM", "8"))
        self.gamma = float(os.environ.get("TABNET_GAMMA", "1.5"))
        self.lambda_sparse = float(
            os.environ.get("TABNET_LAMBDA_SPARSE", "1e-4")
        )
        self.mask_type = os.environ.get("TABNET_MASK_TYPE", "sparsemax")

        # Training
        self.max_epochs = int(os.environ.get("TABNET_MAX_EPOCHS", "100"))
        self.patience = int(os.environ.get("TABNET_PATIENCE", "20"))
        self.batch_size = int(os.environ.get("TABNET_BATCH_SIZE", "32"))
        self.virtual_batch_size = int(
            os.environ.get("TABNET_VIRTUAL_BATCH_SIZE", "16")
        )
        self.learning_rate = float(
            os.environ.get("TABNET_LEARNING_RATE", "2e-2")
        )

        self.random_seed = int(os.environ.get("SEED", "42"))

class TabNetLearner:
    """TabNet learner for MOL pipeline."""

    def __init__(self, config: TabNetTrainingConfig):
        self.config = config
        self.model = None
        self.output_dir = None
        self.feature_names = None

    def setup_output_directory(self) -> Path:
        """Create output directory."""

        base_dir = Path(__file__).resolve().parents[1]

        self.output_dir = (
            base_dir
            / "tabnet_fs"
            / "outputs"
            / f"output_{self.config.dataset_name}"
        )

        self.output_dir.mkdir(parents=True, exist_ok=True)

        return self.output_dir

    def load_data(
        self
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
        """Load preprocessed data."""

        base_dir = Path(__file__).resolve().parents[1]

        processed_dir = (
            base_dir
            / "data"
            / "processed"
            / self.config.dataset_name
        )

        X_train = np.load(processed_dir / "X_train.npy")
        X_test = np.load(processed_dir / "X_test.npy")

        y_train = np.load(processed_dir / "y_train.npy")
        y_test = np.load(processed_dir / "y_test.npy")

        feature_names = np.load(
            processed_dir / "feature_names.npy",
            allow_pickle=True
        ).tolist()

        self.feature_names = feature_names

        print("=" * 80)
        print(f"Dataset: {self.config.dataset_name}")
        print(f"Train samples: {X_train.shape[0]}")
        print(f"Test samples: {X_test.shape[0]}")
        print(f"Features: {X_train.shape[1]}")
        print(f"Classes: {len(np.unique(y_train))}")
        print("=" * 80)

        return X_train, X_test, y_train, y_test, feature_names

    def create_model(self) -> TabNetClassifier:
        """Create TabNet classifier."""

        if not TABNET_AVAILABLE:
            raise ImportError("pytorch-tabnet is not installed")

        model = TabNetClassifier(
            n_d=self.config.n_d,
            n_a=self.config.n_a,
            n_steps=self.config.n_steps,
            gamma=self.config.gamma,
            lambda_sparse=self.config.lambda_sparse,
            optimizer_fn=torch.optim.Adam,
            optimizer_params=dict(lr=self.config.learning_rate),
            mask_type=self.config.mask_type,
            n_shared=2,
            n_independent=2,
            verbose=1,
            seed=self.config.random_seed
        )

        return model

    def train_model(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray
    ) -> Tuple[TabNetClassifier, np.ndarray]:
        """Train TabNet with proper validation split."""

        print("\n[1/5] Training TabNet model...")

        # Reproducibility
        np.random.seed(self.config.random_seed)
        torch.manual_seed(self.config.random_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.config.random_seed)

        # Validation split
        X_train_fit, X_val, y_train_fit, y_val = train_test_split(
            X_train,
            y_train,
            test_size=0.2,
            random_state=self.config.random_seed,
            stratify=y_train
        )

        print(f"Training split: {X_train_fit.shape[0]}")
        print(f"Validation split: {X_val.shape[0]}")

        model = self.create_model()

        warnings.filterwarnings("ignore")

        model.fit(
            X_train=X_train_fit,
            y_train=y_train_fit,
            eval_set=[(X_val, y_val)],
            eval_name=['validation'],
            eval_metric=['accuracy'],
            max_epochs=self.config.max_epochs,
            patience=self.config.patience,
            batch_size=self.config.batch_size,
            virtual_batch_size=self.config.virtual_batch_size,
            drop_last=False
        )

        print("Training completed successfully")

        return model, X_train_fit

    def extract_explanations(
        self,
        model: TabNetClassifier,
        X_train: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Extract authentic TabNet masks.

        Returns:
            masks_3d: (n_steps, n_samples, n_features)
            explanations: feature importances
        """

        print("\n[2/5] Extracting TabNet explanations...")

        # Feature importance
        if hasattr(model, 'feature_importances_'):
            explanations = model.feature_importances_
        else:
            raise RuntimeError(
                "TabNet did not produce feature importances"
            )

        try:
            explain_matrix, masks = model.explain(X_train)

            if not isinstance(masks, dict):
                raise RuntimeError(
                    "TabNet explain() did not return a valid masks dictionary"
                )

            if len(masks) == 0:
                raise RuntimeError(
                    "TabNet explain() returned empty masks"
                )

            steps = sorted(masks.keys())

            masks_3d = np.stack(
                [masks[s] for s in steps],
                axis=0
            )

            # Validation
            if masks_3d.ndim != 3:
                raise ValueError(
                    f"Expected 3D masks tensor, got shape {masks_3d.shape}"
                )

            if masks_3d.shape[0] != self.config.n_steps:
                print(
                    f"WARNING: configured n_steps={self.config.n_steps} "
                    f"but extracted masks have {masks_3d.shape[0]} steps"
                )

            print(f"Masks extracted successfully: {masks_3d.shape}")

            return explain_matrix, masks_3d, explanations

        except Exception as e:
            raise RuntimeError(
                f"TabNet explain() failed. "
                f"Authentic step masks are required for MOL generation. "
                f"Error: {e}"
            )

    def compute_step_assignments(
        self,
        masks: np.ndarray,
        explanations: np.ndarray
    ) -> pd.DataFrame:
        """
        Compute REAL feature-step assignments from TabNet masks.
        """

        print("\n[3/5] Computing step assignments...")

        if masks is None:
            raise RuntimeError(
                "Masks are required for step assignments"
            )

        if masks.ndim != 3:
            raise RuntimeError(
                f"Expected masks with 3 dimensions, got {masks.shape}"
            )

        if len(self.feature_names) != masks.shape[2]:
            raise RuntimeError(
                "Feature count mismatch between masks and feature names"
            )
        
        # Average across samples
        avg_masks = np.mean(masks, axis=1)

        # Dominant step per feature
        dominant_steps = np.argmax(avg_masks, axis=0)
        mask_strengths = np.max(avg_masks, axis=0)

        assignments = []

        for idx, (
            feature,
            step,
            strength
        ) in enumerate(
            zip(
                self.feature_names,
                dominant_steps,
                mask_strengths
            )
        ):

            step_distribution = avg_masks[:, idx].tolist()

            assignments.append({
                'feature': str(feature),
                'dominant_step': int(step),
                'mask_strength': float(strength),
                'global_importance': float(explanations[idx]),
                'step_distribution': json.dumps(step_distribution),
                'assignment_source': 'mask_based'
            })

        df = pd.DataFrame(assignments)

        df = df.sort_values(
            'global_importance',
            ascending=False
        ).reset_index(drop=True)

        print(f"Features assigned: {len(df)}")
        print(f"Steps used: {df['dominant_step'].nunique()}")

        return df

    def compute_performance_metrics(
        self,
        model: TabNetClassifier,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> Dict[str, float]:
        """Compute metrics."""

        print("\n[4/5] Computing metrics...")

        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        train_acc = float(np.mean(y_train_pred == y_train))
        test_acc = float(np.mean(y_test_pred == y_test))

        metrics = {
            'train_accuracy': train_acc,
            'test_accuracy': test_acc,
            'accuracy_gap': train_acc - test_acc
        }

        print(f"Train Accuracy: {train_acc:.2%}")
        print(f"Test Accuracy: {test_acc:.2%}")

        return metrics

    def save_artifacts(
        self,
        model: TabNetClassifier,
        explain_matrix: np.ndarray,
        masks: np.ndarray,
        assignments: pd.DataFrame,
        explanations: np.ndarray,
        metrics: Dict[str, float]
    ):
        """Save all artifacts."""

        print("\n[5/5] Saving artifacts...")

        # Save feature importance
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': explanations
        })

        importance_df.to_csv(
            self.output_dir / "tabnet_feature_importance.csv",
            index=False
        )

        # Save assignments
        assignments.to_csv(
            self.output_dir / "tabnet_step_assignment.csv",
            index=False
        )

        # Save masks
        np.save(
            self.output_dir / "tabnet_masks.npy",
            masks
        )

        # Save normalized masks
        avg_masks = np.mean(masks, axis=1)

        col_sums = avg_masks.sum(axis=0, keepdims=True)

        avg_masks_norm = avg_masks / np.where(
            col_sums == 0,
            1,
            col_sums 
        )

        mask_df = pd.DataFrame(
            avg_masks_norm.T,
            index=self.feature_names,
            columns=[
                f'step_{i}'
                for i in range(masks.shape[0])
            ]
        )

        mask_df.to_csv(
            self.output_dir / "tabnet_stepwise_masks_normalized.csv"
        )

        # Save trained model
        model.save_model(
            str(self.output_dir / "tabnet_model")
        )

        # Save config
        config_dict = {
            'dataset': self.config.dataset_name,
            'target_column': self.config.target_column,
            'n_steps': self.config.n_steps,
            'n_d': self.config.n_d,
            'n_a': self.config.n_a,
            'gamma': self.config.gamma,
            'lambda_sparse': self.config.lambda_sparse,
            'mask_type': self.config.mask_type,
            'learning_rate': self.config.learning_rate,
            'batch_size': self.config.batch_size,
            'max_epochs': self.config.max_epochs,
            'patience': self.config.patience,
            'assignment_source': 'mask_based',
            'timestamp': datetime.now().isoformat()
        }

        with open(
            self.output_dir / "tabnet_config.json",
            'w',
            encoding='utf-8'
        ) as f:
            json.dump(config_dict, f, indent=2)

        # Save metrics
        metrics_df = pd.DataFrame([metrics])

        metrics_df.to_csv(
            self.output_dir / "tabnet_performance.csv",
            index=False
        )

        # Save Explain_matrix
        np.save(
            self.output_dir / "tabnet_explain_matrix.npy",
            explain_matrix
        )

        # Save summary
        summary = {
            'dataset': self.config.dataset_name,
            'target': self.config.target_column,
            'n_features': len(self.feature_names),
            'steps_used': assignments['dominant_step'].nunique(),
            'n_steps_configured': self.config.n_steps,
            'test_accuracy': metrics['test_accuracy'],
            'train_accuracy': metrics['train_accuracy'],
            'has_step_masks': True,
            'step_assignment_source': 'mask_based',
            'timestamp': datetime.now().isoformat()
        }

        with open(
            self.output_dir / "summary.json",
            'w',
            encoding='utf-8'
        ) as f:
            json.dump(summary, f, indent=2)

        print(f"Artifacts saved to: {self.output_dir}")

    def run(self) -> bool:
        """Main execution method."""

        print("=" * 80)
        print("TABNET TRAINING")
        print("=" * 80)

        try:
            # Step 1
            self.setup_output_directory()

            # Step 2
            (
                X_train,
                X_test,
                y_train,
                y_test,
                feature_names
            ) = self.load_data()

            # Step 3
            model, X_train_fit = self.train_model(
                X_train,
                y_train
            )

            # Step 4
            explain_matrix, masks, explanations = self.extract_explanations(
                model,
                X_train_fit
            )

            # Step 5
            assignments = self.compute_step_assignments(
                masks,
                explanations
            )

            # Step 6
            metrics = self.compute_performance_metrics(
                model,
                X_train,
                y_train,
                X_test,
                y_test
            )

            # Step 7
            self.save_artifacts(
                model,
                explain_matrix,
                masks,
                assignments,
                explanations,
                metrics
            )

            print("\n" + "=" * 80)
            print("TABNET TRAINING COMPLETE")
            print("=" * 80)

            print(f"Dataset: {self.config.dataset_name}")
            print(f"Test Accuracy: {metrics['test_accuracy']:.2%}")
            print(f"Train Accuracy: {metrics['train_accuracy']:.2%}")
            print(f"Features: {len(feature_names)}")

            print(
                f"Steps used: "
                f"{assignments['dominant_step'].nunique()}"
                f"/{self.config.n_steps}"
            )

            print(f"Masks shape: {masks.shape}")

            print("\nGenerated artifacts:")
            print("  - tabnet_step_assignment.csv")
            print("  - tabnet_feature_importance.csv")
            print("  - tabnet_masks.npy")
            print("  - tabnet_stepwise_masks_normalized.csv")
            print("  - tabnet_performance.csv")
            print("  - tabnet_config.json")
            print("  - summary.json")

            print("=" * 80)

            return True

        except Exception as e:

            print("\n TABNET TRAINING FAILED")
            print(str(e))

            traceback.print_exc()

            return False

def main():
    """Main entry point."""

    dataset_name = os.environ.get("DATASET")

    if not dataset_name:
        print("Error: DATASET environment variable not set")
        sys.exit(1)

    target_column = os.environ.get("TARGET_COL")

    if not target_column:
        print("Error: TARGET_COL environment variable not set")
        sys.exit(1)

    if not TABNET_AVAILABLE:
        print("Error: pytorch-tabnet is not installed")
        print("Install with:")
        print("pip install pytorch-tabnet")
        sys.exit(1)

    print("Starting TabNet training...")
    print(f"Dataset: {dataset_name}")
    print(f"Target column: {target_column}")

    config = TabNetTrainingConfig(
        dataset_name,
        target_column
    )

    learner = TabNetLearner(config)

    success = learner.run()

    if success:
        print("\n TabNet training completed successfully")
        sys.exit(0)

    else:
        print("\n TabNet training failed")
        sys.exit(1)

if __name__ == "__main__":
    main()