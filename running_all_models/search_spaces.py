"""search_spaces.py – Define only what Optuna can explore."""
import optuna


def suggest_xgboost(trial: optuna.Trial) -> dict:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=50),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "gamma": trial.suggest_float("gamma", 0, 2),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
    }


def suggest_lightgbm(trial: optuna.Trial) -> dict:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=50),
        "num_leaves": trial.suggest_int("num_leaves", 15, 127),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "max_depth": trial.suggest_int("max_depth", -1, 10),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 1.0),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 40),
    }


def suggest_catboost(trial: optuna.Trial) -> dict:
    return {
        "iterations": trial.suggest_int("iterations", 100, 500, step=50),
        "depth": trial.suggest_int("depth", 4, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1, 10),
    }


def suggest_tabnet(trial: optuna.Trial) -> dict:
    return {
        "n_d": trial.suggest_int("n_d", 8, 32),
        "n_a": trial.suggest_int("n_a", 8, 32),
        "n_steps": trial.suggest_int("n_steps", 3, 6),
        "gamma": trial.suggest_float("gamma", 1.0, 2.0),
        "lambda_sparse": trial.suggest_float("lambda_sparse", 1e-6, 1e-3, log=True),
        "lr": trial.suggest_float("lr", 1e-3, 2e-2, log=True),
    }


def suggest_fttransformer(trial: optuna.Trial) -> dict:
    return {
        "d_token": trial.suggest_categorical("d_token", [16, 32, 64]),
        "n_heads": trial.suggest_categorical("n_heads", [2, 4, 8]),
        "n_blocks": trial.suggest_categorical("n_blocks", [2, 3, 4]),
        "dropout": trial.suggest_float("dropout", 0.0, 0.3),
        "lr": trial.suggest_float("lr", 1e-4, 3e-3, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64]),
        "epochs": trial.suggest_int("epochs", 50, 100, step=10),
    }


def suggest_cnn(trial: optuna.Trial) -> dict:
    """Shared for IGTD‑inspired and Naive Reshape."""
    return {
        "epochs": trial.suggest_int("epochs", 50, 150, step=10),
        "lr": trial.suggest_float("lr", 1e-4, 1e-3, log=True),
        "dropout": trial.suggest_float("dropout", 0.1, 0.5),
    }


SEARCH_FUNCTIONS = {
    "XGBoost": suggest_xgboost,
    "LightGBM": suggest_lightgbm,
    "CatBoost": suggest_catboost,
    "TabNet": suggest_tabnet,
    "FT-Transformer (lite)": suggest_fttransformer,
    "IGTD-inspired": suggest_cnn,
    "Naive Reshape": suggest_cnn,
}