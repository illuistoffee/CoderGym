import json
import os
import math
import random
import sys
from typing import Dict, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.datasets import make_blobs
from sklearn.metrics import f1_score, mean_squared_error, precision_score, r2_score, recall_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


torch.set_num_threads(1)

TASK_ID = 'mlp_lvl7_make_blobs_simple'
ARTIFACT_DIR = os.path.join('tasks', TASK_ID, 'artifacts')


def get_task_metadata() -> Dict:
    return {
        'task_id': TASK_ID,
        'series': 'Neural Networks (MLP)',
        'algorithm': 'Simple MLP on make_blobs',
        'metric': 'accuracy',
    }


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def make_dataloaders(batch_size: int = 64, seed: int = 42) -> Tuple[DataLoader, DataLoader]:
    X, y = make_blobs(
        n_samples=900,
        centers=[(-2.2, -1.0), (2.0, 1.5)],
        cluster_std=[1.35, 1.25],
        random_state=seed,
    )

    X = X.astype(np.float32)
    y = y.astype(np.int64)

    idx = np.arange(len(X))
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)
    split = int(0.8 * len(X))
    train_idx, val_idx = idx[:split], idx[split:]

    X_train = torch.tensor(X[train_idx], dtype=torch.float32)
    y_train = torch.tensor(y[train_idx], dtype=torch.long)
    X_val = torch.tensor(X[val_idx], dtype=torch.float32)
    y_val = torch.tensor(y[val_idx], dtype=torch.long)

    mean = X_train.mean(dim=0, keepdim=True)
    std = X_train.std(dim=0, keepdim=True).clamp_min(1e-6)
    X_train = (X_train - mean) / std
    X_val = (X_val - mean) / std

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size, shuffle=False)
    return train_loader, val_loader

def build_model() -> nn.Module:
    return nn.Sequential(
        nn.Linear(2, 16),
        nn.ReLU(),
        nn.Linear(16, 16),
        nn.ReLU(),
        nn.Linear(16, 2),
    )


def train(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, device: torch.device, epochs: int = 18, lr: float = 1e-2) -> Dict:
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history = {'train_loss': [], 'val_loss': []}

    model.to(device)
    for _ in range(epochs):
        model.train()
        total = 0.0
        n = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            total += loss.item() * xb.size(0)
            n += xb.size(0)
        history['train_loss'].append(total / max(n, 1))

        model.eval()
        total = 0.0
        n = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                loss = criterion(logits, yb)
                total += loss.item() * xb.size(0)
                n += xb.size(0)
        history['val_loss'].append(total / max(n, 1))

    return history

def predict(model: nn.Module, data_loader: DataLoader, device: torch.device):
    model.eval()
    probs_all, preds_all = [], []
    with torch.no_grad():
        for xb, _ in data_loader:
            xb = xb.to(device)
            logits = model(xb)
            probs = torch.softmax(logits, dim=1)[:, 1]
            preds = torch.argmax(logits, dim=1)
            probs_all.append(probs.cpu())
            preds_all.append(preds.cpu())
    return torch.cat(probs_all).numpy(), torch.cat(preds_all).numpy()

def evaluate(model: nn.Module, data_loader: DataLoader, device: torch.device) -> Dict:
    X, y_true_t = data_loader.dataset.tensors
    eval_loader = DataLoader(TensorDataset(X, y_true_t), batch_size=data_loader.batch_size or 64, shuffle=False)
    y_true = y_true_t.numpy()
    y_prob, y_pred = predict(model, eval_loader, device)

    return {
        'mse': float(mean_squared_error(y_true, y_prob)),
        'r2': float(r2_score(y_true, y_prob)),
        'accuracy': float((y_true == y_pred).mean()),
        'precision': float(precision_score(y_true, y_pred, zero_division=0)),
        'recall': float(recall_score(y_true, y_pred, zero_division=0)),
        'f1': float(f1_score(y_true, y_pred, zero_division=0)),
    }


def save_artifacts(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, history: Dict, device: torch.device) -> Dict:
    os.makedirs(ARTIFACT_DIR, exist_ok=True)

    X_train = train_loader.dataset.tensors[0].numpy()
    y_train = train_loader.dataset.tensors[1].numpy()
    X_val = val_loader.dataset.tensors[0].numpy()
    y_val = val_loader.dataset.tensors[1].numpy()

    X_all = np.vstack([X_train, X_val])
    y_all = np.concatenate([y_train, y_val])

    x_min, x_max = X_all[:, 0].min() - 0.5, X_all[:, 0].max() + 0.5
    y_min, y_max = X_all[:, 1].min() - 0.5, X_all[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
    grid = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32).to(device)

    model.eval()
    with torch.no_grad():
        zz = torch.argmax(model(grid), dim=1).cpu().numpy().reshape(xx.shape)

    boundary_path = os.path.join(ARTIFACT_DIR, 'decision_boundary.png')
    plt.figure(figsize=(6, 5))
    plt.contourf(xx, yy, zz, alpha=0.3)
    plt.scatter(X_all[:, 0], X_all[:, 1], c=y_all, s=12)
    plt.title('Decision Boundary')
    plt.tight_layout()
    plt.savefig(boundary_path, dpi=150)
    plt.close()

    loss_path = os.path.join(ARTIFACT_DIR, 'loss_curve.png')
    plt.figure(figsize=(6, 4))
    plt.plot(history['train_loss'], label='train')
    plt.plot(history['val_loss'], label='val')
    plt.title('Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Cross Entropy')
    plt.legend()
    plt.tight_layout()
    plt.savefig(loss_path, dpi=150)
    plt.close()

    metrics_path = os.path.join(ARTIFACT_DIR, 'metrics.json')
    return {
        'decision_boundary': boundary_path,
        'loss_curve': loss_path,
        'metrics_json': metrics_path,
    }

def main() -> int:
    set_seed(42)
    device = get_device()
    train_loader, val_loader = make_dataloaders()
    model = build_model()
    history = train(model, train_loader, val_loader, device)

    train_metrics = evaluate(model, train_loader, device)
    val_metrics = evaluate(model, val_loader, device)
    artifacts = save_artifacts(model, train_loader, val_loader, history, device)

    outputs = {
        'loss_history': history,
        'train_metrics': train_metrics,
        'val_metrics': val_metrics,
        'artifacts': artifacts,
    }

    with open(artifacts['metrics_json'], 'w', encoding='utf-8') as f:
        json.dump(outputs, f, indent=2)

    print(json.dumps(outputs, indent=2))

    success = (
        val_metrics['accuracy'] > 0.90
        and val_metrics['f1'] > 0.90
        and history['train_loss'][-1] < history['train_loss'][0]
    )
    return 0 if success else 1


if __name__ == '__main__':
    try:
        sys.exit(main())
    except Exception as exc:
        print(f'Task failed with exception: {exc}', file=sys.stderr)
        sys.exit(2)
