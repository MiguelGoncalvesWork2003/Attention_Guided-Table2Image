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

from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from pytorch_tabnet.tab_model import TabNetClassifier as _TabNetClassifierBase


class TabNetClassifier(ClassifierMixin, _TabNetClassifierBase):
    """
    pytorch-tabnet's TabNetClassifier does not inherit ClassifierMixin at all
    (its real MRO is [TabNetClassifier, TabModel, BaseEstimator, ...]),
    despite being functionally a classifier. Recent sklearn's is_classifier()
    returns False for it as a result, and response_method="predict_proba"
    scoring (used by hyperparameter_search.py's RandomizedSearchCV) rejects
    it with "Got a regressor". This subclass adds only the missing mixin;
    no other behaviour changes -- predict_proba, classes_, and every
    constructor argument are exactly as pytorch-tabnet defines them.
    """
    pass
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


class FTTransformerWrapper(ClassifierMixin, BaseEstimator):
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
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import roc_auc_score
        import copy

        # See the identical comment in T2I_CNN.fit() -- required for
        # sklearn's classifier response-value validation, fixed width by
        # construction rather than derived from np.unique(y).
        self.classes_ = np.arange(self.n_classes)

        self.model = FTTransformerNative(
            self.n_features, self.n_classes,
            d_token=self.d_token, n_heads=self.n_heads,
            n_blocks=self.n_blocks, dropout=self.dropout
        ).to(DEVICE)

        # Internal 80/20 split for early stopping (Section 5.5, item 4).
        # random_state fixed at 42 to match the convention already used by
        # T2I_CNN.fit() elsewhere in this module.
        idx = np.arange(len(X))
        y_arr = np.asarray(y)
        try:
            tr, va = train_test_split(idx, test_size=0.2, stratify=y_arr, random_state=42)
        except ValueError:
            tr, va = train_test_split(idx, test_size=0.2, random_state=42)

        X_tr = torch.tensor(np.asarray(X)[tr], dtype=torch.float32)
        y_tr = torch.tensor(y_arr[tr], dtype=torch.long)
        X_va = torch.tensor(np.asarray(X)[va], dtype=torch.float32).to(DEVICE)
        y_va = y_arr[va]

        dataset = torch.utils.data.TensorDataset(X_tr, y_tr)
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=1e-5)
        criterion = nn.CrossEntropyLoss()

        best, best_state, stale, PATIENCE = -np.inf, None, 0, 20
        for _ in range(self.epochs):
            self.model.train()
            for xb, yb in loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                optimizer.zero_grad()
                logits = self.model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()

            self.model.eval()
            with torch.no_grad():
                probs = torch.softmax(self.model(X_va), dim=1).cpu().numpy()
            try:
                score = (roc_auc_score(y_va, probs[:, 1]) if probs.shape[1] == 2
                         else roc_auc_score(y_va, probs, multi_class="ovr",
                                            average="macro",
                                            labels=list(range(self.n_classes))))
            except ValueError:
                score = float((probs.argmax(1) == y_va).mean())

            if score > best:
                best, stale = score, 0
                best_state = copy.deepcopy(self.model.state_dict())
            else:
                stale += 1
                if stale >= PATIENCE:
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state)
        self.best_val_score_ = best
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
# MDS-layout mapper 
# -------------------------------------------------------------------
class IGTD_Mapper:
    """MDS-layout feature mapper."""
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
class T2I_CNN(ClassifierMixin, BaseEstimator):
    def __init__(self, n_features, n_classes, mode="naive", epochs=100, lr=1e-3,
                 dropout=0.3, arch="tabnetcnn"):
        self.n_features = n_features
        self.n_classes = n_classes
        self.side = int(np.ceil(np.sqrt(n_features)))
        self.mode = mode
        self.epochs = epochs
        self.lr = lr
        self.dropout = dropout
        self.arch = arch
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
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import roc_auc_score
        import copy

        # Required for sklearn's classifier response-value validation
        # (response_method="predict_proba") to recognise this estimator and
        # correctly align predict_proba's columns. The output layer has a
        # fixed width of self.n_classes regardless of which classes this
        # particular fold's y contains, so this is np.arange(self.n_classes),
        # not np.unique(y).
        self.classes_ = np.arange(self.n_classes)

        X_img = self._to_image(X)
        h, w = X_img.shape[2], X_img.shape[3]

        from cnn.cnn_architectures import build_model
        self.model = build_model(
            self.arch,
            n_classes=self.n_classes, input_channels=1,
            image_height=h, image_width=w,      # actual shape, not assumed square
            dropout=self.dropout,
        ).to(self.device)

        idx = np.arange(len(X_img))
        try:
            tr, va = train_test_split(idx, test_size=0.2, stratify=y, random_state=42)
        except ValueError:
            tr, va = train_test_split(idx, test_size=0.2, random_state=42)

        X_tr = torch.tensor(X_img[tr], dtype=torch.float32)
        y_tr = torch.tensor(np.asarray(y)[tr], dtype=torch.long)
        X_va = torch.tensor(X_img[va], dtype=torch.float32).to(self.device)
        y_va = np.asarray(y)[va]

        loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(X_tr, y_tr), batch_size=32, shuffle=True)
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        crit = nn.CrossEntropyLoss()

        best, best_state, best_ep, stale, PATIENCE = -np.inf, None, 0, 0, 20
        for epoch in range(self.epochs):
            self.model.train()
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                opt.zero_grad(); crit(self.model(xb), yb).backward(); opt.step()

            self.model.eval()
            with torch.no_grad():
                probs = torch.softmax(self.model(X_va), dim=1).cpu().numpy()
            try:
                score = (roc_auc_score(y_va, probs[:, 1]) if probs.shape[1] == 2
                         else roc_auc_score(y_va, probs, multi_class="ovr",
                                            average="macro",
                                            labels=list(range(self.n_classes))))
            except ValueError:
                score = float((probs.argmax(1) == y_va).mean())

            if score > best:
                best, best_ep, stale = score, epoch + 1, 0
                best_state = copy.deepcopy(self.model.state_dict())
            else:
                stale += 1
                if stale >= PATIENCE:
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state)
        self.best_val_score_, self.best_epoch_ = best, best_ep
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
                "CatBoost": CatBoostClassifier(iterations=100, random_state=42, verbose=0),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, random_state=42, n_jobs=1
        ),
        "MLP": MLPClassifier(
            hidden_layer_sizes=(128, 64), max_iter=500,
            early_stopping=True, validation_fraction=0.2, n_iter_no_change=20,
            random_state=42
        ),
        "TabNet": TabNetClassifier(
            optimizer_fn=torch.optim.Adam,
            optimizer_params=dict(lr=2e-2),
            verbose=0
        ),
        "FT-Transformer (lite)": FTTransformerWrapper(n_features=input_dim, n_classes=n_classes),
        "IGTD": T2I_CNN(n_features=input_dim, n_classes=n_classes, mode="real_igtd"),
        "MDS-layout": T2I_CNN(n_features=input_dim, n_classes=n_classes, mode="igtd"),
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
        "XGBoost", "LightGBM", "CatBoost", "Random Forest",
        "MLP", "TabNet", "FT-Transformer (lite)",
        "MDS-layout", "IGTD", "Naive Reshape", "DeepInsight"
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
    elif model_name == "Random Forest":
        return RandomForestClassifier(
            n_estimators=params.get('n_estimators', 300),
            max_depth=params.get('max_depth', None),
            min_samples_leaf=params.get('min_samples_leaf', 1),
            max_features=params.get('max_features', 'sqrt'),
            random_state=42,
            n_jobs=1
        )
    elif model_name == "MLP":
        return MLPClassifier(
            hidden_layer_sizes=params.get('hidden_layer_sizes', (128, 64)),
            alpha=params.get('alpha', 1e-4),
            learning_rate_init=params.get('learning_rate_init', 1e-3),
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.2,
            n_iter_no_change=20,
            random_state=42
        )
    elif model_name == "TabNet":
        # hyperparameter_search.py's search space for TabNet samples a
        # nested "optimizer_params": {"lr": ...} dict (matching
        # TabNetClassifier's real constructor), not a flat "lr" key.
        # params.get('lr', ...) below would therefore never find the tuned
        # value; read it from the nested dict, with the same fallback.
        tuned_lr = params.get('optimizer_params', {}).get('lr', params.get('lr', 2e-2))
        return TabNetClassifier(
            n_d=params.get('n_d', 8),
            n_a=params.get('n_a', 8),
            n_steps=params.get('n_steps', 3),
            gamma=params.get('gamma', 1.5),
            lambda_sparse=params.get('lambda_sparse', 1e-4),
            optimizer_fn=torch.optim.Adam,
            optimizer_params=dict(lr=tuned_lr),
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
    elif model_name in ["IGTD", "MDS-layout", "Naive Reshape", "DeepInsight"]:
        if model_name == "IGTD":
            mode = "real_igtd"
        elif model_name == "MDS-layout":
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