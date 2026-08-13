"""
Factory that returns all baseline classifiers for the tabular‑to‑image
comparison. Includes tree ensembles, TabNet, FT‑Transformer, and CNN‑based
tabular‑to‑image baselines (IGTD, IGTD‑inspired, DeepInsight, naive reshape).
Extended with get_model_from_params for hyperparameter tuning.
"""

import json
import torch
import torch.nn as nn
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.manifold import MDS
from sklearn.metrics import pairwise_distances

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.manifold import TSNE
import sys
import os
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from cnn.cnn_model import TabNetCNN

import tempfile
import shutil

# ---------- IGTD imports ----------
IGTD_PATH = PROJECT_ROOT / "external" / "IGTD" / "Scripts"
sys.path.append(str(IGTD_PATH))

igtf_path = IGTD_PATH / "IGTD_Functions.py"
if not igtf_path.exists():
    raise FileNotFoundError(f"IGTD_Functions.py not found at {igtf_path}")

from IGTD_Functions import table_to_image

# ---------- DeepInsight import ----------
from pyDeepInsight import ImageTransformer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------------------------------------------------------
# FT-Transformer
# -------------------------------------------------------------------
class FTTransformerNative(nn.Module):
    """Lightweight FT‑Transformer for tabular classification."""
    def __init__(self, n_features, n_classes, d_token=32, n_heads=4, n_blocks=2, dropout=0.1):
        super().__init__()
        self.n_features = n_features
        self.d_token = d_token

        self.feature_embeddings = nn.Parameter(torch.randn(n_features, d_token))
        self.feature_biases = nn.Parameter(torch.zeros(n_features, d_token))
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_token))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_token, nhead=n_heads, dim_feedforward=d_token * 4,
            dropout=dropout, batch_first=True, activation="gelu"
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_blocks)
        self.norm = nn.LayerNorm(d_token)
        self.head = nn.Sequential(
            nn.Linear(d_token, d_token),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_token, n_classes)
        )

    def forward(self, x):
        batch_size = x.shape[0]
        tokens = x.unsqueeze(-1) * self.feature_embeddings + self.feature_biases
        cls = self.cls_token.expand(batch_size, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)
        tokens = self.transformer(tokens)
        cls_repr = tokens[:, 0]
        cls_repr = self.norm(cls_repr)
        return self.head(cls_repr)


class FTTransformerWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, n_features, n_classes, epochs=50, batch_size=32, lr=1e-3,
                 d_token=32, n_heads=4, n_blocks=2, dropout=0.1):
        self.n_features = n_features
        self.n_classes = n_classes
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.d_token = d_token
        self.n_heads = n_heads
        self.n_blocks = n_blocks
        self.dropout = dropout
        self.model = None

    def fit(self, X, y):
        self.model = FTTransformerNative(
            self.n_features, self.n_classes,
            d_token=self.d_token, n_heads=self.n_heads,
            n_blocks=self.n_blocks, dropout=self.dropout
        ).to(DEVICE)
        X_t = torch.tensor(X, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.long)
        dataset = torch.utils.data.TensorDataset(X_t, y_t)
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=1e-5)
        criterion = nn.CrossEntropyLoss()
        self.model.train()
        for _ in range(self.epochs):
            for xb, yb in loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                optimizer.zero_grad()
                logits = self.model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
        return self

    def predict(self, X):
        self.model.eval()
        X_t = torch.tensor(X, dtype=torch.float32).to(DEVICE)
        with torch.no_grad():
            logits = self.model(X_t)
            preds = logits.argmax(dim=1)
        return preds.cpu().numpy()

    def predict_proba(self, X):
        self.model.eval()
        X_t = torch.tensor(X, dtype=torch.float32).to(DEVICE)
        with torch.no_grad():
            logits = self.model(X_t)
            probs = torch.softmax(logits, dim=1)
        return probs.cpu().numpy()

# -------------------------------------------------------------------
# DeepInsight mapper
# -------------------------------------------------------------------
class DeepInsightMapper:
    """
    Uses pyDeepInsight's ImageTransformer with t‑SNE to create a 2D feature
    layout. Outputs 1‑channel grayscale images.
    """
    def __init__(self, n_features, perplexity=None, random_state=42):
        self.n_features = n_features
        self.side = int(np.ceil(np.sqrt(n_features)))
        # t‑SNE perplexity must be < n_features
        if perplexity is None:
            perplexity = min(30, self.n_features - 1)
        self.perplexity = max(1, perplexity)
        self.random_state = random_state
        self.transformer = None

    def fit(self, X):
        X = np.asarray(X, dtype=np.float32)
        tsne = TSNE(
            perplexity=self.perplexity,
            random_state=self.random_state
        )
        # NO cmap – we'll convert to grayscale manually
        self.transformer = ImageTransformer(
            feature_extractor=tsne,
            pixels=(self.side, self.side)
        )
        self.transformer.fit(X)
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=np.float32)
        images = self.transformer.transform(X)          # likely (N, H, W, 3)

        # Convert RGB to grayscale by averaging colour channels
        if images.ndim == 4 and images.shape[-1] == 3:  # (N, H, W, 3)
            images = np.mean(images, axis=-1)           # → (N, H, W)
        # Now add channel dimension → (N, 1, H, W)
        return images[:, None, :, :]

# -------------------------------------------------------------------
# IGTD-inspired mapper (MDS)
# -------------------------------------------------------------------
class IGTD_Mapper:
    """MDS-based IGTD-inspired feature mapper."""
    def __init__(self, n_features):
        self.n_features = n_features
        self.side = int(np.ceil(np.sqrt(n_features)))
        self.positions = None

    def fit(self, X):
        X = np.asarray(X, dtype=np.float32)
        corr = np.corrcoef(X.T)
        corr = np.nan_to_num(corr)
        dist = 1 - np.abs(corr)

        mds = MDS(n_components=2, dissimilarity="precomputed", random_state=0,
                  n_init=4, max_iter=300)
        coords = mds.fit_transform(dist)

        coords = (coords - coords.min(axis=0)) / (coords.max(axis=0) - coords.min(axis=0) + 1e-8)
        order = np.lexsort((coords[:, 1], coords[:, 0]))

        grid = -np.ones((self.side, self.side), dtype=int)
        idx = 0
        for i in range(self.side):
            for j in range(self.side):
                if idx < len(order):
                    grid[i, j] = order[idx]
                    idx += 1
        self.positions = grid
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=np.float32)
        images = np.zeros((X.shape[0], self.side, self.side), dtype=np.float32)
        for i in range(self.side):
            for j in range(self.side):
                feature_idx = self.positions[i, j]
                if feature_idx != -1:
                    images[:, i, j] = X[:, feature_idx]
        return images[:, None, :, :]

# -------------------------------------------------------------------
# Real IGTD mapper (original algorithm, silent)
# -------------------------------------------------------------------
class RealIGTDMapper:
    """
    Original IGTD feature mapper that learns a 2D feature ordering using the
    IGTD algorithm, then maps tabular data to 1‑channel images.
    Prints are suppressed during fitting.
    """
    def __init__(self, n_features):
        self.n_features = n_features
        self.side = int(np.ceil(np.sqrt(n_features)))
        self.index = None          # feature ordering (length n_features)
        self.feat_min = None       # per‑feature min from training
        self.feat_max = None       # per‑feature max from training

    def fit(self, X):
        X = np.asarray(X, dtype=np.float32)

        # store normalisation parameters
        self.feat_min = np.min(X, axis=0)
        self.feat_max = np.max(X, axis=0)
        norm = (X - self.feat_min) / (self.feat_max - self.feat_min + 1e-8)

        # run IGTD silently
        tmp = tempfile.mkdtemp()
        # Suppress all print output from IGTD
        old_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')
        try:
            table_to_image(
                norm_d=norm,
                scale=[self.side, self.side],
                fea_dist_method="Pearson",
                image_dist_method="Euclidean",
                save_image_size=2,
                max_step=3000,
                val_step=200,
                normDir=tmp,
                error="abs"
            )
        finally:
            sys.stdout.close()
            sys.stdout = old_stdout

        self.index = np.load(tmp + "/feature_order.npy")
        shutil.rmtree(tmp)
        return self

    def transform(self, X):
        if self.index is None:
            raise RuntimeError("RealIGTDMapper must be fitted before transform()")

        X = np.asarray(X, dtype=np.float32)

        images = np.zeros((X.shape[0], self.side, self.side), dtype=np.float32)

        for pos, feature in enumerate(self.index):
            r = pos // self.side
            c = pos % self.side
            if feature < self.n_features:
                images[:, r, c] = X[:, feature]

        return images[:, None]   # (N, 1, H, W)

# -------------------------------------------------------------------
# Generic CNN wrapper for tabular-to-image models
# -------------------------------------------------------------------
class T2I_CNN(BaseEstimator, ClassifierMixin):
    def __init__(self, n_features, n_classes, mode="naive", epochs=100, lr=1e-3, dropout=0.3):
        self.n_features = n_features
        self.n_classes = n_classes
        self.side = int(np.ceil(np.sqrt(n_features)))
        self.mode = mode
        self.epochs = epochs
        self.lr = lr
        self.dropout = dropout
        self.mapper = None
        self.permutation = None
        self.model = None
        self.device = DEVICE

    def _to_image(self, X):
        if self.mode == "naive":
            if self.permutation is None:
                rng = np.random.default_rng(42)
                self.permutation = rng.permutation(self.n_features)
            X = X[:, self.permutation]
            X_padded = np.pad(X, ((0, 0), (0, self.side**2 - self.n_features)))
            return X_padded.reshape(-1, 1, self.side, self.side)

        elif self.mode == "real_igtd":
            if self.mapper is None:
                self.mapper = RealIGTDMapper(self.n_features).fit(X)
            return self.mapper.transform(X)

        elif self.mode == "igtd":
            if self.mapper is None:
                self.mapper = IGTD_Mapper(self.n_features).fit(X)
            return self.mapper.transform(X)

        elif self.mode == "deepinsight":
            if self.mapper is None:
                self.mapper = DeepInsightMapper(self.n_features)
                self.mapper.fit(X)
            return self.mapper.transform(X)

        return X.reshape(-1, 1, self.side, self.side)

    def fit(self, X, y):
        X_img = self._to_image(X)
        self.model = TabNetCNN(
            n_classes=self.n_classes,
            input_channels=1,
            image_height=self.side,
            image_width=self.side,
            dropout=self.dropout
        ).to(self.device)

        X_t = torch.tensor(X_img, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.long)
        dataset = torch.utils.data.TensorDataset(X_t, y_t)
        loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss()
        self.model.train()
        for _ in range(self.epochs):
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                optimizer.zero_grad()
                loss = criterion(self.model(xb), yb)
                loss.backward()
                optimizer.step()
        return self

    def predict(self, X):
        self.model.eval()
        X_img = self._to_image(X)
        X_t = torch.tensor(X_img, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            return self.model(X_t).argmax(dim=1).cpu().numpy()

    def predict_proba(self, X):
        self.model.eval()
        X_img = self._to_image(X)
        X_t = torch.tensor(X_img, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            logits = self.model(X_t)
            probs = torch.softmax(logits, dim=1)
        return probs.cpu().numpy()

# -------------------------------------------------------------------
# Model factory functions
# -------------------------------------------------------------------
def get_models(input_dim, n_classes):
    """Return default baseline classifiers (backward compatible)."""
    return {
        "XGBoost": XGBClassifier(n_estimators=100, random_state=42, eval_metric="mlogloss"),
        "LightGBM": LGBMClassifier(n_estimators=100, random_state=42, verbose=-1),
        "CatBoost": CatBoostClassifier(n_estimators=100, random_state=42, verbose=0),
        "TabNet": TabNetClassifier(
            optimizer_fn=torch.optim.Adam,
            optimizer_params=dict(lr=2e-2),
            verbose=0
        ),
        "FT-Transformer (lite)": FTTransformerWrapper(n_features=input_dim, n_classes=n_classes),
        "IGTD": T2I_CNN(n_features=input_dim, n_classes=n_classes, mode="real_igtd"),
        "IGTD-inspired": T2I_CNN(n_features=input_dim, n_classes=n_classes, mode="igtd"),
        "Naive Reshape": T2I_CNN(n_features=input_dim, n_classes=n_classes, mode="naive"),
        "DeepInsight": T2I_CNN(n_features=input_dim, n_classes=n_classes, mode="deepinsight"),
    }

def get_tuned_models(dataset_name: str, n_features: int, n_classes: int):
    """Load tuned hyperparameters from best_params/<dataset>.json."""
    best_params_path = Path(__file__).parent / "best_params" / f"{dataset_name}.json"
    if best_params_path.exists():
        with open(best_params_path, "r") as f:
            all_params = json.load(f)
    else:
        all_params = {}

    models = {}
    for model_name in [
        "XGBoost", "LightGBM", "CatBoost", "TabNet",
        "FT-Transformer (lite)", "IGTD-inspired", "IGTD",
        "Naive Reshape", "DeepInsight"
    ]:
        params = all_params.get(model_name, {})
        models[model_name] = get_model_from_params(model_name, n_features, n_classes, params)
    return models

def get_model_from_params(model_name, n_features, n_classes, params=None):
    """Instantiate a model with given hyperparameters."""
    if params is None:
        params = {}

    if model_name == "XGBoost":
        return XGBClassifier(
            n_estimators=params.get('n_estimators', 100),
            max_depth=params.get('max_depth', 6),
            learning_rate=params.get('learning_rate', 0.3),
            subsample=params.get('subsample', 1.0),
            colsample_bytree=params.get('colsample_bytree', 1.0),
            random_state=42,
            eval_metric="mlogloss"
        )
    elif model_name == "LightGBM":
        return LGBMClassifier(
            n_estimators=params.get('n_estimators', 100),
            num_leaves=params.get('num_leaves', 31),
            learning_rate=params.get('learning_rate', 0.1),
            subsample=params.get('subsample', 1.0),
            colsample_bytree=params.get('colsample_bytree', 1.0),
            random_state=42,
            verbose=-1
        )
    elif model_name == "CatBoost":
        return CatBoostClassifier(
            iterations=params.get('iterations', 100),
            depth=params.get('depth', 6),
            learning_rate=params.get('learning_rate', 0.1),
            l2_leaf_reg=params.get('l2_leaf_reg', 3.0),
            random_state=42,
            verbose=0
        )
    elif model_name == "TabNet":
        return TabNetClassifier(
            n_d=params.get('n_d', 8),
            n_a=params.get('n_a', 8),
            n_steps=params.get('n_steps', 3),
            gamma=params.get('gamma', 1.5),
            lambda_sparse=params.get('lambda_sparse', 1e-4),
            optimizer_fn=torch.optim.Adam,
            optimizer_params=dict(lr=params.get('lr', 2e-2)),
            verbose=0
        )
    elif model_name == "FT-Transformer (lite)":
        return FTTransformerWrapper(
            n_features=n_features,
            n_classes=n_classes,
            epochs=params.get('epochs', 50),
            batch_size=params.get('batch_size', 32),
            lr=params.get('lr', 1e-3),
            d_token=params.get('d_token', 32),
            n_heads=params.get('n_heads', 4),
            n_blocks=params.get('n_blocks', 2),
            dropout=params.get('dropout', 0.1)
        )
    elif model_name in ["IGTD", "IGTD-inspired", "Naive Reshape", "DeepInsight"]:
        if model_name == "IGTD":
            mode = "real_igtd"
        elif model_name == "IGTD-inspired":
            mode = "igtd"
        elif model_name == "DeepInsight":
            mode = "deepinsight"
        else:
            mode = "naive"

        return T2I_CNN(
            n_features=n_features,
            n_classes=n_classes,
            mode=mode,
            epochs=params.get("epochs", 100),
            lr=params.get("lr", 1e-3),
            dropout=params.get("dropout", 0.3)
        )
    else:
        raise ValueError(f"Unknown model name: {model_name}")