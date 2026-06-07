#analyzer_of_tabnet_structure.py
"""
Read‑only interpreter for TabNet’s learned internal structure.

This module performs **post‑hoc analysis** of the TabNet artefacts produced
by `train_tabnet.py`. It never recomputes assignments, trains a model, or
generates synthetic data. Its sole purpose is to extract structural insights
that support the interpretability analysis in the paper (Section 5).

The `TabNetStructureInterpreter` class:
  - Loads the **step assignment CSV** (the single source of truth) together
    with feature importance, configuration, and optional 3D masks.
  - Computes structural metrics: feature utilisation ratio, step‑balance Gini,
    normalised step entropy, and the number of active steps.
  - Interprets step roles (Specialist, Focused, High‑impact, Balanced) based on
    feature count and average importance.
  - Optionally identifies *critical features* – those whose attention value is
    consistently the maximum within their assigned step – using the raw 3D masks.
  - Generates publication‑ready visualisations: step distribution, importance by
    step, structural metric bar charts, and step hierarchy plots.

All outputs are strictly derived from the saved artefacts, guaranteeing
reproducibility and full traceability.

Usage:
  python build_tabnet_structure.py <dataset> --visualize

Relation to the paper:
  The structural metrics and visualisations serve as qualitative evidence that
  TabNet’s attention masks induce meaningful feature groupings, which are then
  transferred to the image layout. The interpreter thus bridges the
  “Optimize” (TabNet training) and “Map” (layout construction) stages by
  quantifying the quality of the learned attention structure.
"""

from datetime import datetime
import pandas as pd
import numpy as np
import json
from pathlib import Path
import os
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List, Tuple
import sys

class TabNetStructureInterpreter:
    """
    Interprets TabNet's learned structure from SAVED artifacts.
    READ-ONLY - never recomputes assignments.
    """
    
    def __init__(self, dataset_name: str):
        self.dataset_name = dataset_name
        self.base_dir = Path(__file__).resolve().parents[1]
        self.output_dir = self.base_dir / "tabnet_fs" / "outputs" / f"output_{dataset_name}"
        
        # Load outputs - READ ONLY
        self.masks_3d = None  # Raw 3D masks if available
        self.masks_2d = None  # Normalized 2D masks (features x steps)
        self.feature_names = None
        self.importances = None
        self.step_assignment = None  # SINGLE SOURCE OF TRUTH
        self.config = None
        
        self.load_outputs()
    
    def load_outputs(self):
        """Load all TabNet outputs - READ ONLY."""
        print(f"Loading TabNet outputs from: {self.output_dir}")
        
        # CRITICAL: Check for single source of truth
        required_files = [
            "tabnet_step_assignment.csv",  # SINGLE SOURCE OF TRUTH
            "tabnet_feature_importance.csv",
            "tabnet_config.json"
        ]
        
        missing_files = []
        for file in required_files:
            if not (self.output_dir / file).exists():
                missing_files.append(file)
        
        if missing_files:
            raise FileNotFoundError(
                f"Required files not found: {missing_files}\n"
                f"Run train_tabnet.py first to generate artifacts."
            )
        
        # 1. Load step assignment - SINGLE SOURCE OF TRUTH
        self.step_assignment = pd.read_csv(self.output_dir / "tabnet_step_assignment.csv")
        print(f"  Loaded step assignments: {len(self.step_assignment)} features")
        
        # 2. Load feature importance
        importance_df = pd.read_csv(self.output_dir / "tabnet_feature_importance.csv")
        self.importances = importance_df['importance'].values
        self.feature_names = importance_df['feature'].tolist()
        print(f"  Loaded feature importance: {len(self.feature_names)} features")
        
        # 3. Load configuration
        with open(self.output_dir / "tabnet_config.json", 'r', encoding='utf-8') as f:
            self.config = json.load(f)
        print(f"  Loaded configuration: {self.config['n_steps']} steps")
        
        # 4. Try to load masks (optional for some analyses)
        mask_2d_path = self.output_dir / "tabnet_stepwise_masks_normalized.csv"
        if mask_2d_path.exists():
            mask_df = pd.read_csv(mask_2d_path, index_col=0)
            self.masks_2d = mask_df.values  # (n_features, n_steps)
            print(f"  Loaded 2D masks: {self.masks_2d.shape}")
        
        mask_3d_path = self.output_dir / "tabnet_masks.npy"
        if mask_3d_path.exists():
            self.masks_3d = np.load(mask_3d_path)  # (n_steps, n_samples, n_features)
            print(f"  Loaded 3D masks: {self.masks_3d.shape}")
        
        print(f"[SUCCESS] Loaded {len(self.feature_names)} features")
    
    def interpret_step_hierarchy(self) -> Dict[int, Dict[str, Any]]:
        """
        Interpret step hierarchy BASED ON ASSIGNMENTS.
        NO recomputation of assignments.
        """
        step_info = {}
        
        # Group features by step FROM ASSIGNMENTS
        step_groups = self.step_assignment[
            self.step_assignment['dominant_step'] >= 0
        ].groupby('dominant_step')
        
        for step, group in step_groups:
            features = group['feature'].tolist()
            importances = group['global_importance'].values
            
            # Basic statistics
            n_features = len(features)
            avg_importance = np.mean(importances) if len(importances) > 0 else 0
            importance_std = np.std(importances) if len(importances) > 1 else 0
            
            # Determine role based on statistics
            if n_features == 0:
                role = "Unused"
            elif n_features == 1:
                role = "Specialist"
            elif n_features <= 3:
                role = "Focused"
            elif avg_importance > 0.1:
                role = "High-impact"
            elif importance_std < 0.05:
                role = "Balanced"
            else:
                role = "General"
            
            step_info[step] = {
                'role': role,
                'n_features': n_features,
                'avg_importance': float(avg_importance),
                'top_features': features[:3] if features else []
            }
        
        return step_info
    
    def compute_structure_metrics(self) -> Dict[str, Any]:
        """
        Compute structural metrics WITHOUT recomputing assignments.
        """
        if self.step_assignment is None:
            raise ValueError("Step assignments not loaded")
        
        n_features = len(self.feature_names)
        n_assigned = len(self.step_assignment)
        steps_used = self.step_assignment['dominant_step'].nunique()
        n_steps = self.config.get('n_steps', 0)
        
        # Calculate feature distribution
        feature_distribution = self.step_assignment['dominant_step'].value_counts().to_dict()
        
        # Calculate concentration (Gini coefficient)
        if len(feature_distribution) > 0:
            values = list(feature_distribution.values())
            sorted_values = np.sort(values)
            n = len(sorted_values)
            cumulative = np.cumsum(sorted_values)
            gini = (n + 1 - 2 * np.sum(cumulative) / cumulative[-1]) / n if cumulative[-1] > 0 else 0
        else:
            gini = 0
        
        # Calculate step usage entropy
        feature_distribution = self.step_assignment['dominant_step'].value_counts().to_dict()

        if len(feature_distribution) <= 1:
            normalized_entropy = 0.0
        else:
            total = sum(feature_distribution.values())
            probs = [count / total for count in feature_distribution.values()]

            entropy = -sum(p * np.log(p + 1e-10) for p in probs)
            max_entropy = np.log(len(probs))

            normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0
        
        metrics = {
            'n_features': n_features,
            'n_assigned': n_assigned,
            'utilization_ratio': n_assigned / n_features if n_features > 0 else 0,
            'steps_used': steps_used,
            'total_steps': n_steps,
            'avg_features_per_step': n_assigned / steps_used if steps_used > 0 else 0,
            'step_balance_gini': float(gini),
            'step_entropy': float(normalized_entropy),
            'feature_distribution': feature_distribution
        }
        
        return metrics
    
    def identify_critical_features(self, loyalty_threshold: float = 0.8) -> pd.DataFrame:
        """
        Identify critical features using 3D masks IF AVAILABLE.
        Returns empty DataFrame if masks not available.
        """
        if self.masks_3d is None:
            print("  Warning: 3D masks not available for loyalty analysis")
            return pd.DataFrame()
        
        critical_features = []
        
        for _, row in self.step_assignment.iterrows():
            feature = row['feature']
            step = row['dominant_step']
            
            if feature in self.feature_names:
                feature_idx = self.feature_names.index(feature)
                
                assert step < self.masks_3d.shape[0], f"Invalid step {step}"
                assert feature_idx < self.masks_3d.shape[2], f"Invalid feature index {feature_idx}"
                
                # Get masks for this feature at this step
                feature_masks = self.masks_3d[step, :, feature_idx]  # (n_samples,)
                
                # Get all masks for this step
                step_masks = self.masks_3d[step]  # (n_samples, n_features)
                
                # Find max mask for each sample
                max_mask_per_sample = np.max(step_masks, axis=1)  # (n_samples,)
                
                # Calculate loyalty (how often this feature has the max mask)
                loyalty = np.mean(feature_masks == max_mask_per_sample)
                
                if loyalty >= loyalty_threshold:
                    critical_features.append({
                        'feature': feature,
                        'step': step,
                        'loyalty': float(loyalty),
                        'global_importance': row['global_importance'],
                        'role': 'Critical' if loyalty > 0.9 else 'Dominant'
                    })
        
        return pd.DataFrame(critical_features)
    
    def create_interpretation_report(self) -> Dict[str, Any]:
        """
        Create comprehensive interpretation report.
        READ-ONLY - uses saved artifacts only.
        """
        print("\nCreating interpretation report (read-only)...")
        
        # Get step hierarchy
        step_hierarchy = self.interpret_step_hierarchy()
        
        # Get structure metrics
        structure_metrics = self.compute_structure_metrics()
        
        # Get critical features (if masks available)
        critical_features = self.identify_critical_features()
        
        report = {
            'dataset': self.dataset_name,
            'config_summary': {
                'n_steps': self.config.get('n_steps', 'Unknown'),
                'n_features': len(self.feature_names),
                'target': self.config.get('target_column', 'Unknown'),
                'timestamp': self.config.get('timestamp', 'Unknown')
            },
            'step_hierarchy': step_hierarchy,
            'structural_metrics': structure_metrics,
            'critical_features': critical_features.to_dict('records') if not critical_features.empty else [],
            'analysis_timestamp': datetime.now().isoformat(),
            'analysis_method': 'read_only_interpretation'
        }
        
        return report
    
    def visualize_structure(self, output_dir: Path):
        """
        Create visualizations BASED ON SAVED ARTIFACTS.
        """
        print("\nCreating visualizations...")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Step distribution plot
        self._plot_step_distribution(output_dir)
        
        # 2. Feature importance by step
        self._plot_feature_importance_by_step(output_dir)
        
        # 3. Structure metrics summary
        self._plot_structure_metrics(output_dir)
        
        # 4. Step hierarchy visualization
        self._plot_step_hierarchy(output_dir)
    
    def _plot_step_distribution(self, output_dir: Path):
        """Plot feature distribution across steps."""
        step_counts = self.step_assignment['dominant_step'].value_counts().sort_index()
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(range(len(step_counts)), step_counts.values)
        
        plt.xlabel('Decision Step')
        plt.ylabel('Number of Features')
        plt.title('Feature Distribution Across TabNet Steps')
        plt.xticks(range(len(step_counts)), [f'Step {i}' for i in step_counts.index])
        
        # Add count labels
        for bar, count in zip(bars, step_counts.values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    str(count), ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(output_dir / "step_distribution.png", dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved step distribution plot")
    
    def _plot_feature_importance_by_step(self, output_dir: Path):
        """Plot feature importance aggregated by step."""
        # Group by step and sum importance
        step_importance = self.step_assignment.groupby('dominant_step')['global_importance'].sum()
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(range(len(step_importance)), step_importance.values)
        
        plt.xlabel('Decision Step')
        plt.ylabel('Total Feature Importance')
        plt.title('Feature Importance Aggregated by TabNet Step')
        plt.xticks(range(len(step_importance)), [f'Step {i}' for i in step_importance.index])
        
        # Add importance labels
        for bar, importance in zip(bars, step_importance.values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{importance:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(output_dir / "step_importance_aggregation.png", dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved step importance plot")
    
    def _plot_structure_metrics(self, output_dir: Path):
        """Plot key structure metrics."""
        metrics = self.compute_structure_metrics()
        
        # Select key metrics to visualize
        key_metrics = {
            'Utilization Ratio': metrics['utilization_ratio'],
            'Step Balance (1-Gini)': 1 - metrics['step_balance_gini'],
            'Step Entropy': metrics['step_entropy']
        }
        
        plt.figure(figsize=(8, 6))
        bars = plt.bar(range(len(key_metrics)), list(key_metrics.values()),
                      color=['skyblue', 'lightgreen', 'lightcoral'])
        
        plt.xlabel('Structural Metric')
        plt.ylabel('Value')
        plt.title('TabNet Structure Metrics')
        plt.xticks(range(len(key_metrics)), list(key_metrics.keys()), rotation=45, ha='right')
        plt.ylim(0, 1.1)
        
        # Add value labels
        for bar, value in zip(bars, key_metrics.values()):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{value:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(output_dir / "structure_metrics.png", dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved structure metrics plot")
    
    def _plot_step_hierarchy(self, output_dir: Path):
        """Visualize step hierarchy."""
        step_hierarchy = self.interpret_step_hierarchy()
        
        if not step_hierarchy:
            return
        
        steps = sorted(step_hierarchy.keys())
        roles = [step_hierarchy[step]['role'] for step in steps]
        n_features = [step_hierarchy[step]['n_features'] for step in steps]
        
        fig, ax1 = plt.subplots(figsize=(12, 6))
        
        # Bar plot for number of features
        bars = ax1.bar(range(len(steps)), n_features, alpha=0.7, color='skyblue')
        ax1.set_xlabel('Decision Step')
        ax1.set_ylabel('Number of Features', color='skyblue')
        ax1.tick_params(axis='y', labelcolor='skyblue')
        
        ax2 = ax1.twinx()
        
        # Role annotations
        for i, role in enumerate(roles):
            ax2.text(i, max(n_features) * 0.5, role, ha='center', va='center',
                    rotation=90, color='black', fontweight='bold')
        
        ax2.set_ylabel('Step Role', color='black')
        ax2.set_yticks([])
        
        plt.title('TabNet Step Hierarchy and Feature Distribution')
        plt.xticks(range(len(steps)), [f'Step {step}' for step in steps])
        
        plt.tight_layout()
        plt.savefig(output_dir / "step_hierarchy.png", dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved step hierarchy plot")

def main():
    """Main entry point for TabNet structure interpretation."""
    # Get dataset from environment or command line
    import argparse
    
    parser = argparse.ArgumentParser(description='Interpret TabNet structure from saved artifacts')
    parser.add_argument('dataset', help='Dataset name')
    parser.add_argument('--visualize', action='store_true', help='Generate visualizations')
    
    args = parser.parse_args()
    dataset_name = args.dataset
    
    print("=" * 80)
    print("TABNET STRUCTURE INTERPRETATION (READ-ONLY)")
    print("=" * 80)
    print(f"Dataset: {dataset_name}")
    print("=" * 80)
    
    try:
        # Create interpreter
        interpreter = TabNetStructureInterpreter(dataset_name)
        
        # Create interpretation report
        report = interpreter.create_interpretation_report()
        
        # Print summary
        print("\nSTRUCTURE INTERPRETATION SUMMARY:")
        print("-" * 40)
        print(f"Dataset: {report['dataset']}")
        print(f"Steps: {report['config_summary']['n_steps']}")
        print(f"Features: {report['config_summary']['n_features']}")
        print(f"Steps Used: {report['structural_metrics']['steps_used']}")
        print(f"Utilization Ratio: {report['structural_metrics']['utilization_ratio']:.1%}")
        print(f"Step Balance (Gini): {report['structural_metrics']['step_balance_gini']:.3f}")
        
        print("\nSTEP HIERARCHY:")
        print("-" * 40)
        for step, info in report['step_hierarchy'].items():
            print(f"Step {step}: {info['role']} ({info['n_features']} features)")
            if info['top_features']:
                print(f"      Top: {', '.join(info['top_features'][:2])}")
        
        critical_features = report['critical_features']
        if critical_features:
            print(f"\nCRITICAL FEATURES (Loyalty > 0.8):")
            print("-" * 40)
            for feature in critical_features[:5]:
                print(f"  {feature['feature']} (Step {feature['step']}): "
                      f"Loyalty={feature['loyalty']:.2f}")
        
        # Create visualizations if requested
        if args.visualize:
            visualization_dir = interpreter.output_dir / "structure_visualizations"
            interpreter.visualize_structure(visualization_dir)
            print(f"\nVisualizations saved to: {visualization_dir}")
        
        # Save report
        report_path = interpreter.output_dir / "structure_interpretation_report.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"\n✅ Interpretation complete!")
        print(f"  Report saved to: {report_path}")
        
    except FileNotFoundError as e:
        print(f"\n❌ ERROR: {e}")
        print("\nPlease run train_tabnet.py first to generate artifacts.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Interpretation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()