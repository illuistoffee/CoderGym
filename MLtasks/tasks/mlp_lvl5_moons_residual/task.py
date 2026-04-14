import json
import os
import random
import sys
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
from sklearn.datasets import make_moons #new dataset 
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


@dataclass
class TaskConfig:
    task_id: str = "mlp_lvl5_moons_residual" #mlp, multilayer perceptron
    seed: int = 42
    batch_size: int = 128
    epochs: int = 40
    lr: float = 1e-2
    weight_decay: float = 1e-4
    hidden_dim: int = 32
    dropout: float = 0.05
    val_size: float = 0.2
    n_samples: int = 1000
    noise: float = 0.18
    label_smoothing: float = 0.03
    min_val_accuracy: float = 0.90
    min_val_r2: float = 0.55
    max_val_mse: float = 0.10
    output_dir: str = os.path.join("tasks", "mlp_lvl5_moons_residual", "artifacts")


def get_task_metadata() -> Dict:
    return {
        "task_id": "mlp_lvl5_moons_residual",
        "series": "Neural Networks (MLP)",
        "level": 5,
        "algorithm": "Residual MLP Classifier (AdamW + Label Smoothing)",
        "description": "Residual MLP on two-moons with a self-verifying main block.",
        "interface_protocol": "pytorch_task_v1",
    }


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_dataloaders(cfg: TaskConfig) -> Tuple[DataLoader, DataLoader, Dict[str, torch.Tensor]]:
    X, y = make_moons(n_samples=cfg.n_samples, noise=cfg.noise, random_state=cfg.seed)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=cfg.val_size, random_state=cfg.seed, stratify=y
    )
    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_val = torch.tensor(X_val, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    y_val = torch.tensor(y_val, dtype=torch.long)

    mean = X_train.mean(dim=0, keepdim=True)
    std = X_train.std(dim=0, keepdim=True).clamp_min(1e-6)
    X_train = (X_train - mean) / std
    X_val = (X_val - mean) / std

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=cfg.batch_size, shuffle=False)
    extras = {"X_train": X_train, "y_train": y_train, "X_val": X_val, "y_val": y_val}
    return train_loader, val_loader, extras


class ResidualMLP(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.in_proj = nn.Linear(2, hidden_dim)
        self.block = nn.Sequential(
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.out_proj = nn.Linear(hidden_dim, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.in_proj(x)
        h = torch.relu(h + self.block(h))
        return self.out_proj(h)


def build_model(cfg: TaskConfig) -> nn.Module:
    return ResidualMLP(hidden_dim=cfg.hidden_dim, dropout=cfg.dropout)


def _metrics(y_true: torch.Tensor, y_pred: torch.Tensor, prob1: torch.Tensor) -> Dict[str, float]:
    yt = y_true.cpu().numpy().astype(float)
    yp = y_pred.cpu().numpy().astype(float)
    mse = float(mean_squared_error(yt, yp))
    r2 = float(r2_score(yt, yp))
    accuracy = float((y_true == y_pred).float().mean().item())
    tp = int(((y_true == 1) & (y_pred == 1)).sum().item())
    tn = int(((y_true == 0) & (y_pred == 0)).sum().item())
    fp = int(((y_true == 0) & (y_pred == 1)).sum().item())
    fn = int(((y_true == 1) & (y_pred == 0)).sum().item())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    prob_mse = float(mean_squared_error(yt, prob1.cpu().numpy()))
    return {
        "mse": mse,
        "r2": r2,
        "accuracy": accuracy,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "prob_mse": prob_mse,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    preds, ys, probs = [], [], []
    with torch.no_grad():
        for xb, yb in loader:
            logits = model(xb.to(device))
            p = torch.softmax(logits, dim=1)
            preds.append(torch.argmax(logits, dim=1).cpu())
            ys.append(yb.cpu())
            probs.append(p[:, 1].cpu())
    y_true = torch.cat(ys)
    y_pred = torch.cat(preds)
    prob1 = torch.cat(probs)
    return _metrics(y_true, y_pred, prob1)


def train(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, cfg: TaskConfig, device: torch.device) -> Dict[str, List[float]]:
    model.to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    history = {"train_loss": [], "val_accuracy": [], "val_mse": []}
    best_acc = -1.0
    best_state = None

    for _ in range(cfg.epochs):
        model.train()
        total_loss = 0.0
        total_n = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += loss.item() * xb.size(0)
            total_n += xb.size(0)
        scheduler.step()
        val_metrics = evaluate(model, val_loader, device)
        history["train_loss"].append(total_loss / max(total_n, 1))
        history["val_accuracy"].append(val_metrics["accuracy"])
        history["val_mse"].append(val_metrics["mse"])
        if val_metrics["accuracy"] > best_acc:
            best_acc = val_metrics["accuracy"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    return history


def predict(model: nn.Module, X: torch.Tensor, device: torch.device) -> torch.Tensor:
    model.eval()
    with torch.no_grad():
        logits = model(X.to(device))
        return torch.argmax(logits, dim=1).cpu()


def save_artifacts(model: nn.Module, history: Dict[str, List[float]], metrics: Dict, cfg: TaskConfig) -> Dict[str, str]:
    os.makedirs(cfg.output_dir, exist_ok=True)
    model_path = os.path.join(cfg.output_dir, "model.pt")
    metrics_path = os.path.join(cfg.output_dir, "metrics.json")
    torch.save(model.state_dict(), model_path)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump({"history": history, "metrics": metrics, "metadata": get_task_metadata()}, f, indent=2)
    return {"model_path": model_path, "metrics_path": metrics_path}


if __name__ == "__main__":
    cfg = TaskConfig()
    set_seed(cfg.seed)
    device = get_device()
    train_loader, val_loader, _extras = make_dataloaders(cfg)
    model = build_model(cfg)
    history = train(model, train_loader, val_loader, cfg, device)
    train_metrics = evaluate(model, train_loader, device)
    val_metrics = evaluate(model, val_loader, device)
    artifacts = save_artifacts(model, history, {"train": train_metrics, "val": val_metrics}, cfg)
    outputs = {
        "loss_history": history["train_loss"],
        "val_metric_history": {"accuracy": history["val_accuracy"], "mse": history["val_mse"]},
        "final_metrics": {"train": train_metrics, "val": val_metrics},
        "artifacts": artifacts,
    }
    print(json.dumps(outputs, indent=2))
    exit_code = 0
    try:
        assert val_metrics["accuracy"] >= cfg.min_val_accuracy
        assert val_metrics["r2"] >= cfg.min_val_r2
        assert val_metrics["mse"] <= cfg.max_val_mse
    except AssertionError:
        exit_code = 1
    sys.exit(exit_code) #exit
