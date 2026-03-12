"""
Options Price Prediction — PyTorch + Optuna  [GPU Optimized]
=============================================================
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import optuna
from optuna.samplers import TPESampler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.compose import ColumnTransformer
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings, math, copy, time
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────
# 0. GPU SETUP
# ─────────────────────────────────────────────

def configure_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        props  = torch.cuda.get_device_properties(0)
        print(f"GPU: {props.name}")
        print(f"VRAM: {props.total_memory / 1e9:.1f} GB")
        print(f"CUDA: {torch.version.cuda}")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32       = True
        torch.backends.cudnn.benchmark        = True
    else:
        device = torch.device("cpu")
        print("No GPU detected — running on CPU.")
    print(f"Device: {device}\n")
    return device


DEVICE = configure_device()


# ─────────────────────────────────────────────
# 1. COLUMNS
# ─────────────────────────────────────────────
DROP_COLS  = ["simulation", "day", "days_till_expiry"]
TARGET_COL = "target_price"


# ─────────────────────────────────────────────
# 2. CUSTOM ACTIVATION FUNCTIONS
# ─────────────────────────────────────────────

class ParametricRectifiedExponentialUnit(nn.Module):
    def __init__(self, n_features: int):
        super().__init__()
        self.alpha = nn.Parameter(torch.full((n_features,), 0.1))

    def forward(self, x):
        return torch.where(x >= 0, x, self.alpha * (torch.exp(x) - 1.0))


class ElasticReLU(nn.Module):
    def __init__(self, n_features: int):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(n_features))
        self.beta  = nn.Parameter(torch.zeros(n_features))

    def forward(self, x):
        return torch.maximum(self.alpha * x, self.beta * x)


class DPReLU(nn.Module):
    """Dual Parametric ReLU with learned threshold and bias.
    f(x) = beta  * (x - threshold) + bias  if x >= threshold
    f(x) = alpha * (x - threshold) + bias  if x <  threshold
    """
    def __init__(self, n_features: int):
        super().__init__()
        self.alpha     = nn.Parameter(torch.full((n_features,), 0.1))
        self.beta      = nn.Parameter(torch.ones(n_features))
        self.threshold = nn.Parameter(torch.zeros(n_features))
        self.bias      = nn.Parameter(torch.zeros(n_features))

    def forward(self, x):
        shifted = x - self.threshold
        return torch.where(x >= self.threshold, self.beta * shifted, self.alpha * shifted) + self.bias


class BrownianReLU(nn.Module):
    """Brownian ReLU activation.
    f(x; α) = x                                    for x > 0
    f(x; α) = -α · (1/M) Σ B^k(|x|), k=1..M      for x <= 0
    where B^k(|x|) ~ N(0, |x|) are independent Monte Carlo draws.

    Gradient w.r.t. α:
        ∂f/∂α = 0       for x > 0
        ∂f/∂α = -b_i    for x <= 0  (b_i = Monte Carlo average)
    """
    def __init__(self, n_features: int, M: int = 1000):
        super().__init__()
        #self.alpha = nn.Parameter(torch.full((n_features,), 0.5))  # per-feature
        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.M     = M

    def forward(self, x):
        # x shape: (batch, n_features)
        neg_mask = (x <= 0)
        abs_x    = x.abs()                                      # (batch, n_features)

        # Sample M draws from N(0, |x|): std = sqrt(|x|), shape (M, batch, n_features)
        std      = abs_x.unsqueeze(0).expand(self.M, -1, -1)   # (M, batch, n_features)
        draws    = torch.randn_like(std) * std                  # B^k(|x|) ~ N(0, |x|)
        b        = draws.mean(dim=0)                            # (batch, n_features)

        out_neg  = -self.alpha * b
        return torch.where(neg_mask, out_neg, x)


class NLReLU(nn.Module):
    def __init__(self, n_features: int):
        super().__init__()
        self.beta = nn.Parameter(torch.ones(n_features))

    def forward(self, x):
        return torch.log1p(torch.abs(self.beta) * torch.relu(x))


def get_activation(name: str, n_features: int) -> nn.Module:
    mapping = {
        "preu":         ParametricRectifiedExponentialUnit(n_features),
        "swish":        nn.SiLU(),
        "leaky_relu":   nn.LeakyReLU(negative_slope=0.2),
        "relu":         nn.ReLU(),
        "elastic_relu": ElasticReLU(n_features),
        "prelu":        nn.PReLU(num_parameters=n_features),
        "nlrelu":       NLReLU(n_features),
        "dprelu":       DPReLU(n_features),
        "brownian_relu": BrownianReLU(n_features),
    }
    return mapping[name]


# ─────────────────────────────────────────────
# 3. LOSS FUNCTIONS & METRICS
# ─────────────────────────────────────────────

def logcosh_loss(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    diff = y_pred - y_true
    return torch.mean(diff + torch.nn.functional.softplus(-2.0 * diff) - math.log(2))


def wmape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sum(np.abs(y_true - y_pred)) / (np.sum(np.abs(y_true)) + 1e-8))


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mae  = float(np.mean(np.abs(y_true - y_pred)))
    mse  = float(np.mean((y_true - y_pred) ** 2))
    rmse = float(np.sqrt(mse))
    mask = np.abs(y_true) > 1e-8
    mape = float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)
    return {"wmape": wmape(y_true, y_pred), "mae": mae, "mse": mse, "rmse": rmse, "mape": mape}


# ─────────────────────────────────────────────
# 4. DATASET
# ─────────────────────────────────────────────

class PriceDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def make_dataloader(X, y, batch_size, shuffle=False):
    ds = PriceDataset(X, y)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=(DEVICE.type == "cuda"),
        num_workers=0,
        persistent_workers=False,
        drop_last=True,
    )


# ─────────────────────────────────────────────
# 5. MODEL
# ─────────────────────────────────────────────

class ResBlock(nn.Module):
    def __init__(self, in_dim, out_dim, dropout, use_bn, activation):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, out_dim)
        self.fc2 = nn.Linear(out_dim, out_dim)
        self.bn1 = nn.BatchNorm1d(out_dim) if use_bn else nn.Identity()
        self.bn2 = nn.BatchNorm1d(out_dim) if use_bn else nn.Identity()
        self.act1    = get_activation(activation, out_dim)
        self.act2    = get_activation(activation, out_dim)
        self.dropout = nn.Dropout(dropout)
        self.shortcut = (
            nn.Linear(in_dim, out_dim, bias=False)
            if in_dim != out_dim else nn.Identity()
        )
        nn.init.kaiming_normal_(self.fc1.weight, nonlinearity="relu")
        nn.init.kaiming_normal_(self.fc2.weight, nonlinearity="relu")

    def forward(self, x):
        skip = self.shortcut(x)
        h = self.act1(self.bn1(self.fc1(x)))
        h = self.dropout(h)
        h = self.bn2(self.fc2(h))
        return self.act2(h + skip)


class OptionsPriceNet(nn.Module):
    def __init__(self, n_features, n_blocks, base_units, unit_decay,
                 dropout, use_bn, activation):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(n_features, base_units),
            nn.BatchNorm1d(base_units) if use_bn else nn.Identity(),
            get_activation(activation, base_units),
        )
        blocks, cur = [], base_units
        for _ in range(n_blocks):
            nxt = max(32, int(cur * unit_decay))
            blocks.append(ResBlock(cur, nxt, dropout, use_bn, activation))
            cur = nxt
        self.blocks = nn.ModuleList(blocks)
        self.output = nn.Linear(cur, 1)

    def forward(self, x):
        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x)
        return self.output(x)


# ─────────────────────────────────────────────
# 6. CHECKPOINT UTILITIES
# ─────────────────────────────────────────────

def save_checkpoint(model, optimizer, scheduler, epoch, val_wm, path):
    torch.save({
        "epoch":                epoch,
        "model_state_dict":     model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "val_wmape":            val_wm,
    }, path)


def load_checkpoint(path, model, optimizer=None, scheduler=None):
    ckpt = torch.load(path, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scheduler:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    print(f"Loaded checkpoint: {path}  (epoch {ckpt['epoch']}, val_wmape {ckpt['val_wmape']:.6f})")
    return ckpt["epoch"], ckpt["val_wmape"]


# ─────────────────────────────────────────────
# 7. TRAINING UTILITIES
# ─────────────────────────────────────────────

class EarlyStopping:
    def __init__(self, patience=10, min_delta=1e-5):
        self.patience   = patience
        self.min_delta  = min_delta
        self.best       = float("inf")
        self.counter    = 0
        self.best_state = None

    def step(self, val_wmape, model):
        if val_wmape < self.best - self.min_delta:
            self.best       = val_wmape
            self.counter    = 0
            self.best_state = copy.deepcopy(model.state_dict())
        else:
            self.counter += 1
        return self.counter >= self.patience

    def restore(self, model):
        if self.best_state:
            model.load_state_dict(self.best_state)


def train_one_epoch(model, loader, optimizer, scaler):
    model.train()
    total_loss = 0.0
    for X_batch, y_batch in loader:
        X_batch = X_batch.to(DEVICE, non_blocking=True)
        y_batch = y_batch.to(DEVICE, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=DEVICE.type, enabled=DEVICE.type == "cuda"):
            pred = model(X_batch)
            loss = logcosh_loss(pred, y_batch)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    preds, targets = [], []
    for X_batch, y_batch in loader:
        X_batch = X_batch.to(DEVICE, non_blocking=True)
        with torch.autocast(device_type=DEVICE.type, enabled=DEVICE.type == "cuda"):
            pred = model(X_batch)
        preds.append(pred.cpu().float().numpy())
        targets.append(y_batch.numpy())
    y_pred = np.concatenate(preds).ravel()
    y_true = np.concatenate(targets).ravel()
    return wmape(y_true, y_pred), y_true, y_pred


# ─────────────────────────────────────────────
# 8. DATA LOADING
# ─────────────────────────────────────────────

def load_and_prepare(df, n_holdout_sims=5, random_seed=None):
    rng = np.random.default_rng(random_seed)
    all_sim_ids     = df["simulation"].unique()
    holdout_sim_ids = rng.choice(
        all_sim_ids, size=min(n_holdout_sims, len(all_sim_ids)), replace=False
    )
    print(f"Reserved holdout simulations ({len(holdout_sim_ids)}): "
          f"{sorted(holdout_sim_ids.tolist())}")

    holdout_mask = df["simulation"].isin(holdout_sim_ids)
    holdout_df   = df[holdout_mask].copy()
    train_df     = df[~holdout_mask].copy()

    feature_cols     = [c for c in df.columns if c not in DROP_COLS + [TARGET_COL]]
    wavelet_cols     = [c for c in feature_cols if c.startswith("wavelet_bin_")]
    non_wavelet_cols = [c for c in feature_cols if not c.startswith("wavelet_bin_")]
    wavelet_idx      = [feature_cols.index(c) for c in wavelet_cols]
    non_wavelet_idx  = [feature_cols.index(c) for c in non_wavelet_cols]

    print(f"Scaling: RobustScaler on {len(non_wavelet_cols)} features | "
          f"passthrough on {len(wavelet_cols)} wavelet bins")

    scaler_X = ColumnTransformer(
        transformers=[
            ("robust",      RobustScaler(), non_wavelet_idx),
            ("passthrough", "passthrough",  wavelet_idx),
        ],
        remainder="drop",
    )
    scaler_y = RobustScaler()

    X_all = train_df[feature_cols].values.astype(np.float32)
    y_all = np.log1p(train_df[TARGET_COL].values).astype(np.float32).reshape(-1, 1)

    X_train, X_temp, y_train, y_temp = train_test_split(
        X_all, y_all, test_size=0.30, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42
    )

    X_train = scaler_X.fit_transform(X_train).astype(np.float32)
    X_val   = scaler_X.transform(X_val).astype(np.float32)
    X_test  = scaler_X.transform(X_test).astype(np.float32)
    y_train = scaler_y.fit_transform(y_train).astype(np.float32)
    y_val   = scaler_y.transform(y_val).astype(np.float32)
    y_test  = scaler_y.transform(y_test).astype(np.float32)

    return (
        (X_train, y_train),
        (X_val,   y_val),
        (X_test,  y_test),
        holdout_df,
        scaler_X, scaler_y,
        feature_cols,
        holdout_sim_ids,
    )


# ─────────────────────────────────────────────
# 9. OPTUNA OBJECTIVE
# ─────────────────────────────────────────────

def make_objective(X_train, y_train, X_val, y_val, n_features, epochs=50):
    rng = np.random.default_rng(42)
    idx = rng.choice(len(X_train), size=min(500_000, len(X_train)), replace=False)
    X_sub, y_sub = X_train[idx], y_train[idx]

    def objective(trial):
        n_blocks     = trial.suggest_int("n_blocks", 2,8)
        base_units   = trial.suggest_categorical("base_units", [128, 256, 512])
        unit_decay   = trial.suggest_float("unit_decay", 0.4, 0.95)
        dropout      = trial.suggest_float("dropout_rate", 0.05, 0.50)
        use_bn       = trial.suggest_categorical("use_bn", [True, False])
        activation   = trial.suggest_categorical("activation", [
            "prelu", "leaky_relu", "elastic_relu", "dprelu", "brownian_relu", "relu"
        ])
        lr           = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
        batch_size   = trial.suggest_categorical("batch_size", [256, 512])

        print(f"\n>>> Trial {trial.number} starting | "
              f"batch: {batch_size} | blocks: {n_blocks} | "
              f"units: {base_units} | act: {activation} | "
              f"lr: {lr:.2e} | dropout: {dropout:.3f} | bn: {use_bn}",
              flush=True)

        model = OptionsPriceNet(
            n_features, n_blocks, base_units, unit_decay, dropout, use_bn, activation
        ).to(DEVICE)

        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-6
        )
        scaler  = torch.amp.GradScaler("cuda", enabled=DEVICE.type == "cuda")
        stopper = EarlyStopping(patience=10)

        train_dl = make_dataloader(X_sub, y_sub, batch_size, shuffle=True)
        val_dl   = make_dataloader(X_val, y_val, batch_size, shuffle=False)

        t_start = time.time()
        for epoch in range(epochs):
            t0 = time.time()
            train_one_epoch(model, train_dl, optimizer, scaler)
            val_wm, _, _ = evaluate(model, val_dl)
            scheduler.step(val_wm)

            print(f"    Epoch {epoch+1:3d}/{epochs} | "
                  f"val_wmape: {val_wm:.5f} | "
                  f"best: {stopper.best:.5f} | "
                  f"{time.time()-t0:.1f}s",
                  flush=True)

            if stopper.step(val_wm, model):
                print(f"    Early stop at epoch {epoch+1}", flush=True)
                break

        stopper.restore(model)
        elapsed = time.time() - t_start
        print(f"<<< Trial {trial.number} done | "
              f"best_wmape: {stopper.best:.5f} | "
              f"total: {elapsed:.1f}s",
              flush=True)
        return stopper.best

    return objective


# ─────────────────────────────────────────────
# 10. PLOTLY CHARTS
# ─────────────────────────────────────────────

DARK = "plotly_dark"


def plot_actual_vs_predicted(y_actual: np.ndarray, y_pred: np.ndarray, split_name="Test"):
    y_actual  = y_actual.ravel()
    y_pred    = y_pred.ravel()
    abs_err   = np.abs(y_actual - y_pred)
    wmape_val = wmape(y_actual, y_pred)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=y_actual, y=y_pred, mode="markers",
        marker=dict(size=5, opacity=0.55, color=abs_err, colorscale="Viridis",
                    showscale=True,
                    colorbar=dict(
                        title=dict(text="Abs Error", side="right"),
                        x=1.02,
                        thickness=15,
                    )),
        name="Predictions",
        hovertemplate="Actual: %{x:.4f}<br>Predicted: %{y:.4f}<br>Error: %{marker.color:.4f}<extra></extra>",
    ))
    lo, hi = float(y_actual.min()), float(y_actual.max())
    fig.add_trace(go.Scatter(
        x=[lo, hi], y=[lo, hi], mode="lines",
        line=dict(color="red", dash="dash", width=2), name="Perfect prediction",
    ))
    fig.update_layout(
        title=f"{split_name} Set — Actual vs Predicted  |  wMAPE: {wmape_val:.4f}",
        xaxis_title="Actual Price", yaxis_title="Predicted Price",
        template=DARK, width=900, height=620,
        legend=dict(
            x=0.01, y=0.99,
            xanchor="left", yanchor="top",
            bgcolor="rgba(0,0,0,0.4)",
        ),
        margin=dict(r=120),
    )
    fig.show()
    return fig


def plot_training_history(train_wmapes, val_wmapes, train_losses, val_losses):
    epochs = list(range(1, len(train_losses) + 1))
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=("LogCosh Loss", "wMAPE (primary metric)"),
                        horizontal_spacing=0.12)

    fig.add_trace(go.Scatter(x=epochs, y=train_losses, name="Train Loss",
                             line=dict(color="#636EFA", width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=epochs, y=val_losses, name="Val Loss",
                             line=dict(color="#EF553B", dash="dash", width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=epochs, y=train_wmapes, name="Train wMAPE",
                             line=dict(color="#00CC96", width=2)), row=1, col=2)
    fig.add_trace(go.Scatter(x=epochs, y=val_wmapes, name="Val wMAPE",
                             line=dict(color="#AB63FA", dash="dash", width=2)), row=1, col=2)

    best_ep  = int(np.argmin(val_wmapes)) + 1
    best_val = val_wmapes[best_ep - 1]
    fig.add_trace(go.Scatter(
        x=[best_ep], y=[best_val], mode="markers",
        marker=dict(symbol="star", size=16, color="yellow",
                    line=dict(color="black", width=1)),
        name=f"Best val wMAPE (epoch {best_ep}, {best_val:.4f})",
    ), row=1, col=2)

    fig.update_xaxes(title_text="Epoch")
    fig.update_yaxes(title_text="Loss",  row=1, col=1)
    fig.update_yaxes(title_text="wMAPE", row=1, col=2)
    fig.update_layout(title="Training History", template=DARK, width=1100, height=500,
                      legend=dict(orientation="h", yanchor="bottom", y=-0.28))
    fig.show()
    return fig


def plot_holdout_simulations(
    holdout_df, model, scaler_X, scaler_y, feature_cols, holdout_sim_ids, time_col="day"
):
    sim_ids = sorted(holdout_sim_ids.tolist())
    n_sims  = len(sim_ids)

    fig = make_subplots(
        rows=n_sims, cols=1,
        subplot_titles=[f"Simulation {s}" for s in sim_ids],
        shared_xaxes=False,
        vertical_spacing=max(0.04, 0.10 - 0.005 * n_sims),
    )

    model.eval()
    for row_idx, sim_id in enumerate(sim_ids, start=1):
        sim_data = holdout_df[holdout_df["simulation"] == sim_id].sort_values(time_col)
        X_sim    = scaler_X.transform(sim_data[feature_cols].values.astype(np.float32))
        days     = sim_data[time_col].values

        with torch.no_grad():
            X_t           = torch.from_numpy(X_sim).float().to(DEVICE)
            y_pred_scaled = model(X_t).cpu().float().numpy()

        y_pred   = np.expm1(scaler_y.inverse_transform(y_pred_scaled)).ravel()
        y_actual = sim_data[TARGET_COL].values

        if not np.isfinite(y_pred).all():
            print(f"WARNING: Sim {sim_id} has non-finite predictions, skipping.")
            continue

        wmape_sim = wmape(y_actual, y_pred)

        fig.add_trace(go.Scatter(
            x=days, y=y_actual, mode="lines",
            name="Actual",
            line=dict(color="#1f77b4", width=1.5),
            legendgroup="actual",
            showlegend=(row_idx == 1),
            hovertemplate="Day %{x}<br>Actual: %{y:.2f}<extra></extra>",
        ), row=row_idx, col=1)

        fig.add_trace(go.Scatter(
            x=days, y=y_pred, mode="lines",
            name="MLP Predicted",
            line=dict(color="#ff7f0e", width=1.5, dash="dash"),
            legendgroup="predicted",
            showlegend=(row_idx == 1),
            hovertemplate="Day %{x}<br>Predicted: %{y:.2f}<extra></extra>",
        ), row=row_idx, col=1)

        fig.update_yaxes(
            title_text="BS Price", row=row_idx, col=1,
            showgrid=True, gridwidth=1, zeroline=False,
        )
        fig.update_xaxes(
            title_text="Day", row=row_idx, col=1,
            showgrid=True, gridwidth=1,
        )

        fig.add_annotation(
            text=f"wMAPE: {wmape_sim:.4f}",
            xref="paper", yref="paper",
            x=1.0, y=1.02 - (row_idx - 1) / n_sims,
            showarrow=False,
            font=dict(size=11, color="grey"),
            xanchor="right",
        )

    fig.update_layout(
        title=dict(text="Holdout Simulations — Actual vs Predicted", font=dict(size=14)),
        template=DARK,
        height=max(420, 380 * n_sims),
        width=1050,
        showlegend=True,
        legend=dict(orientation="v", x=1.02, y=1),
    )
    fig.show()
    return fig


# ─────────────────────────────────────────────
# 11. MAIN
# ─────────────────────────────────────────────

def run_study(df, n_trials=60, epochs=50, n_holdout_sims=5, random_seed=None):
    (
        (X_train, y_train),
        (X_val,   y_val),
        (X_test,  y_test),
        holdout_df,
        scaler_X, scaler_y,
        feature_cols,
        holdout_sim_ids,
    ) = load_and_prepare(df, n_holdout_sims=n_holdout_sims, random_seed=random_seed)

    n_features = len(feature_cols)
    print(f"Features: {n_features} | Train: {len(X_train)} | Val: {len(X_val)} | "
          f"Test: {len(X_test)} | Holdout rows: {len(holdout_df)}")

    objective = make_objective(X_train, y_train, X_val, y_val, n_features, epochs)
    sampler   = TPESampler(n_startup_trials=15, multivariate=True, seed=42)
    study     = optuna.create_study(
        direction="minimize",
        sampler=sampler,
        study_name="options_price_nn",
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=10),
    )
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    print("\n=== Best Trial ===")
    print(f"  val_wmape : {study.best_value:.6f}")
    for k, v in study.best_params.items():
        print(f"  {k:20s}: {v}")
    print()

    p = study.best_params
    best_model = OptionsPriceNet(
        n_features,
        n_blocks   = p["n_blocks"],
        base_units = p["base_units"],
        unit_decay = p["unit_decay"],
        dropout    = p["dropout_rate"],
        use_bn     = p.get("use_bn", False),
        activation = p["activation"],
    ).to(DEVICE)

    optimizer = optim.AdamW(best_model.parameters(), lr=p["lr"], weight_decay=p["weight_decay"])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=7, min_lr=1e-6
    )
    scaler  = torch.amp.GradScaler("cuda", enabled=DEVICE.type == "cuda")
    stopper = EarlyStopping(patience=20)

    train_dl = make_dataloader(X_train, y_train, p.get("batch_size", 256), shuffle=True)
    val_dl   = make_dataloader(X_val,   y_val,   p.get("batch_size", 256), shuffle=False)
    test_dl  = make_dataloader(X_test,  y_test,  p.get("batch_size", 256), shuffle=False)

    train_wmapes, val_wmapes = [], []
    train_losses, val_losses = [], []

    print(f"\n>>> Final retrain on full {len(X_train):,} rows | "
          f"batch: {p.get('batch_size', 256)} | epochs: {epochs*2}", flush=True)
    t_retrain = time.time()
    for epoch in range(epochs * 2):
        t0     = time.time()
        t_loss = train_one_epoch(best_model, train_dl, optimizer, scaler)
        t_wm, _, _ = evaluate(best_model, train_dl)
        v_wm, _, _ = evaluate(best_model, val_dl)
        scheduler.step(v_wm)

        train_losses.append(t_loss)
        val_losses.append(v_wm)
        train_wmapes.append(t_wm)
        val_wmapes.append(v_wm)

        print(f"    Epoch {epoch+1:3d}/{epochs*2} | "
              f"loss: {t_loss:.5f} | "
              f"train_wmape: {t_wm:.5f} | "
              f"val_wmape: {v_wm:.5f} | "
              f"best: {stopper.best:.5f} | "
              f"{time.time()-t0:.1f}s",
              flush=True)

        if stopper.step(v_wm, best_model):
            print(f"    Early stop at epoch {epoch+1}", flush=True)
            break

    stopper.restore(best_model)
    print(f"<<< Retrain done | total: {time.time()-t_retrain:.1f}s", flush=True)

    print("\n=== Test Set Evaluation ===")
    _, y_test_true_scaled, y_test_pred_scaled = evaluate(best_model, test_dl)
    y_test_actual = np.expm1(scaler_y.inverse_transform(y_test_true_scaled.reshape(-1, 1))).ravel()
    y_test_pred   = np.expm1(scaler_y.inverse_transform(y_test_pred_scaled.reshape(-1, 1))).ravel()

    metrics = compute_metrics(y_test_actual, y_test_pred)
    for k, v in metrics.items():
        print(f"  {k}: {v:.6f}")

    print("\nGenerating Plotly charts …")
    fig1 = plot_actual_vs_predicted(y_test_actual, y_test_pred, split_name="Test")
    fig2 = plot_training_history(train_wmapes, val_wmapes, train_losses, val_losses)
    fig3 = plot_holdout_simulations(
        holdout_df, best_model, scaler_X, scaler_y,
        feature_cols, holdout_sim_ids, time_col="day",
    )

    return best_model, study, scaler_y, (fig1, fig2, fig3)