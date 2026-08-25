import argparse
import optuna
from optuna.visualization.matplotlib import plot_optimization_history
import matplotlib.pyplot as plt
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OPTUNA_DB_DIR = PROJECT_ROOT / "experiments" / "hyperparameter_search"
OUTPUT_DIR = PROJECT_ROOT / "Figures" / "Results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", required=True)
parser.add_argument("--layout", required=True,
                    choices=["step_row", "packed", "packed_T", "step_sparse", "attention_map"])
args = parser.parse_args()

# Updated database and study name for ROC-AUC optimisation
db_path = OPTUNA_DB_DIR / f"optuna_study_{args.layout}.db"
storage = f"sqlite:///{db_path}"
study_name = f"{args.dataset}_{args.layout}_bayesian_v2"

if not db_path.exists():
    print(f"Database file not found: {db_path}")
    print("Available databases:")
    for f in OPTUNA_DB_DIR.glob("optuna_study_*.db"):
        print(f"  - {f.name}")
    exit(1)

try:
    study = optuna.load_study(study_name=study_name, storage=storage)
except KeyError:
    print(f"Study '{study_name}' not found in {db_path.name}.")
    print("Available studies in this database:")
    summaries = optuna.get_all_study_summaries(storage=storage)
    for s in summaries:
        print(f"  - {s.study_name}")
    exit(1)

print(f"Loaded study '{study_name}' with {len(study.trials)} trials.")
print(f"Best value: {study.best_value:.4f}")

fig = plot_optimization_history(study)
plt.xlabel("Trial")
plt.ylabel("Validation Macro ROC-AUC")
plt.tight_layout()

out_pdf = OUTPUT_DIR / f"hpo_history_{args.dataset}_{args.layout}.pdf"
plt.savefig(out_pdf, bbox_inches="tight")
plt.close()
print(f"Plot saved to: {out_pdf}")