import json
import math
import os
import random
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


@dataclass
class TaskConfig:
    task_id: str = 'linreg_lvl7_mlp_nonlinear_regression'
    seed: int = 42 # 42 
    n_samples: int = 600
    batch_size: int = 128
    train_ratio: float = 0.8
    linear_epochs: int = 35
    mlp_epochs: int = 60
    linear_lr: float = 0.03
    mlp_lr: float = 0.01
    weight_decay: float = 1e-4
    hidden_dims: Tuple[int, int] = (64, 32)
    noise_std: float = 0.15
    r2_threshold: float = 0.90
    improvement_threshold: float = 0.10
    output_dir: str = os.path.join('tasks', 'linreg_lvl7_mlp_nonlinear_regression', 'artifacts')


def get_task_metadata() -> Dict[str, Any]:
    return {
        'task_id': 'linreg_lvl7_mlp_nonlinear_regression',
        'series': 'Linear Regression',
        'level': 7,
        'algorithm': 'Neural Network Regression (MLP vs Linear Baseline)',
        'description': (
            'Train a linear regression baseline and an MLP regressor on a synthetic nonlinear dataset, '
            'compare validation metrics, save plots and metrics, and exit non-zero on failure.'
        ),
        'interface_protocol': 'pytorch_task_v1',
        'required_functions': [
            'get_task_metadata',
            'set_seed',
            'get_device',
            'make_dataloaders',
            'build_model',
            'train',
            'evaluate',
            'predict',
            'save_artifacts',
        ],
    }

# set the seed
def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_num_threads(1)


def get_device() -> torch.device:
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def _r2_score(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    y_true = y_true.view(-1)
    y_pred = y_pred.view(-1)
    ss_res = torch.sum((y_true - y_pred) ** 2)
    ss_tot = torch.sum((y_true - torch.mean(y_true)) ** 2)
    if torch.isclose(ss_tot, torch.tensor(0.0, device=y_true.device)):
        return 0.0
    return float((1.0 - (ss_res / ss_tot)).item())


def _make_dataset(cfg: TaskConfig) -> Tuple[torch.Tensor, torch.Tensor]:
    x = torch.linspace(-2.5, 2.5, cfg.n_samples).unsqueeze(1)
    interaction = torch.sin(2.0 * x)
    x2 = x ** 2
    x3 = x ** 3
    X = torch.cat([x, x2, interaction], dim=1)
    noise = cfg.noise_std * torch.randn_like(x)
    y = x3 - 0.5 * x2 + 2.0 * x + 0.35 * interaction + noise
    return X.float(), y.float()


def _standardize_features(X_train: torch.Tensor, X_val: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
    mean = X_train.mean(dim=0, keepdim=True)
    std = X_train.std(dim=0, keepdim=True).clamp_min(1e-6)
    X_train_std = (X_train - mean) / std
    X_val_std = (X_val - mean) / std
    return X_train_std, X_val_std, {'mean': mean, 'std': std}


def make_dataloaders(batch_size: int = 128, seed: int = 42) -> Dict[str, Any]:
    cfg = TaskConfig(batch_size=batch_size, seed=seed)
    X, y = _make_dataset(cfg)
    n_train = int(len(X) * cfg.train_ratio)

    generator = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(X), generator=generator)
    X = X[perm]
    y = y[perm]

    X_train, X_val = X[:n_train], X[n_train:]
    y_train, y_val = y[:n_train], y[n_train:]
    X_train, X_val, feature_stats = _standardize_features(X_train, X_val)

    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=generator)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return {
        'train_loader': train_loader,
        'val_loader': val_loader,
        'train_dataset': train_dataset,
        'val_dataset': val_dataset,
        'feature_stats': feature_stats,
        'input_dim': X_train.shape[1],
    }


def build_model(input_dim: int, model_type: str = 'mlp') -> nn.Module:
    if model_type == 'linear':
        return nn.Linear(input_dim, 1)
    if model_type == 'mlp':
        return nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )
    raise ValueError(f'Unsupported model_type: {model_type}')


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    weight_decay: float = 0.0,
) -> Dict[str, List[float]]:
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    history: Dict[str, List[float]] = {'train_loss': [], 'val_loss': []}

    model.to(device)
    for _ in range(epochs):
        model.train()
        running_loss = 0.0
        n_samples = 0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
            batch_size = xb.size(0)
            running_loss += loss.item() * batch_size
            n_samples += batch_size

        train_loss = running_loss / max(1, n_samples)
        history['train_loss'].append(train_loss)

        model.eval()
        val_running_loss = 0.0
        val_samples = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                preds = model(xb)
                loss = criterion(preds, yb)
                batch_size = xb.size(0)
                val_running_loss += loss.item() * batch_size
                val_samples += batch_size
        history['val_loss'].append(val_running_loss / max(1, val_samples))

    return history


def predict(model: nn.Module, data_loader: DataLoader, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    preds: List[torch.Tensor] = []
    targets: List[torch.Tensor] = []
    with torch.no_grad():
        for xb, yb in data_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            batch_preds = model(xb)
            preds.append(batch_preds.cpu())
            targets.append(yb.cpu())
    return torch.cat(preds, dim=0), torch.cat(targets, dim=0)


def evaluate(model: nn.Module, data_loader: DataLoader, device: torch.device) -> Dict[str, float]:
    preds, targets = predict(model, data_loader, device)
    mse = float(torch.mean((preds - targets) ** 2).item())
    r2 = _r2_score(targets, preds)
    rmse = math.sqrt(max(mse, 0.0))
    return {
        'mse': mse,
        'rmse': rmse,
        'r2': r2,
    }


def _plot_loss_curves(linear_history: Dict[str, List[float]], mlp_history: Dict[str, List[float]], output_dir: str) -> str:
    path = os.path.join(output_dir, 'loss_curves.png')
    plt.figure(figsize=(8, 5))
    plt.plot(linear_history['train_loss'], label='Linear Train')
    plt.plot(linear_history['val_loss'], label='Linear Val')
    plt.plot(mlp_history['train_loss'], label='MLP Train')
    plt.plot(mlp_history['val_loss'], label='MLP Val')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Training and Validation Loss Curves')
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    return path


def _plot_predictions(
    linear_model: nn.Module,
    mlp_model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    output_dir: str,
) -> str:
    path = os.path.join(output_dir, 'prediction_vs_target.png')
    linear_preds, targets = predict(linear_model, val_loader, device)
    mlp_preds, _ = predict(mlp_model, val_loader, device)

    plt.figure(figsize=(8, 6))
    plt.scatter(targets.numpy(), linear_preds.numpy(), alpha=0.6, label='Linear', s=18)
    plt.scatter(targets.numpy(), mlp_preds.numpy(), alpha=0.6, label='MLP', s=18)
    y_min = min(targets.min().item(), linear_preds.min().item(), mlp_preds.min().item())
    y_max = max(targets.max().item(), linear_preds.max().item(), mlp_preds.max().item())
    plt.plot([y_min, y_max], [y_min, y_max], linestyle='--', linewidth=1.5, label='Ideal')
    plt.xlabel('Target')
    plt.ylabel('Prediction')
    plt.title('Validation Predictions vs Targets')
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    return path


def save_artifacts(outputs: Dict[str, Any], output_dir: str) -> Dict[str, str]:
    os.makedirs(output_dir, exist_ok=True)
    metrics_path = os.path.join(output_dir, 'metrics.json')
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(outputs, f, indent=2)
    return {'metrics_json': metrics_path}


def _print_metrics(name: str, metrics: Dict[str, float]) -> None:
    print(f'{name}: MSE={metrics["mse"]:.6f} RMSE={metrics["rmse"]:.6f} R2={metrics["r2"]:.6f}')


def main() -> int:
    cfg = TaskConfig()
    set_seed(cfg.seed)
    device = get_device()
    os.makedirs(cfg.output_dir, exist_ok=True)

    print('Task metadata:')
    print(json.dumps(get_task_metadata(), indent=2))
    print(f'Using device: {device}')

    data = make_dataloaders(batch_size=cfg.batch_size, seed=cfg.seed)
    train_loader = data['train_loader']
    val_loader = data['val_loader']
    input_dim = data['input_dim']

    linear_model = build_model(input_dim=input_dim, model_type='linear')
    mlp_model = build_model(input_dim=input_dim, model_type='mlp')

    linear_history = train(
        linear_model,
        train_loader,
        val_loader,
        device,
        epochs=cfg.linear_epochs,
        lr=cfg.linear_lr,
        weight_decay=cfg.weight_decay,
    )
    mlp_history = train(
        mlp_model,
        train_loader,
        val_loader,
        device,
        epochs=cfg.mlp_epochs,
        lr=cfg.mlp_lr,
        weight_decay=cfg.weight_decay,
    )

    linear_train_metrics = evaluate(linear_model, train_loader, device)
    linear_val_metrics = evaluate(linear_model, val_loader, device)
    mlp_train_metrics = evaluate(mlp_model, train_loader, device)
    mlp_val_metrics = evaluate(mlp_model, val_loader, device)

    comparison = {
        'val_r2_improvement': mlp_val_metrics['r2'] - linear_val_metrics['r2'],
        'val_mse_reduction': linear_val_metrics['mse'] - mlp_val_metrics['mse'],
    } 
    # returns r2, mse, rmse

    print('\nTrain metrics')
    _print_metrics('Linear train', linear_train_metrics)
    _print_metrics('MLP train', mlp_train_metrics)

    print('\nValidation metrics')
    _print_metrics('Linear val', linear_val_metrics)
    _print_metrics('MLP val', mlp_val_metrics)
    print(f"Validation R2 improvement (MLP - Linear): {comparison['val_r2_improvement']:.6f}")
    print(f"Validation MSE reduction (Linear - MLP): {comparison['val_mse_reduction']:.6f}")

    loss_plot = _plot_loss_curves(linear_history, mlp_history, cfg.output_dir)
    pred_plot = _plot_predictions(linear_model, mlp_model, val_loader, device, cfg.output_dir)

    outputs = {
        'metadata': get_task_metadata(),
        'linear_metrics': {
            'train': linear_train_metrics,
            'validation': linear_val_metrics,
        },
        'mlp_metrics': {
            'train': mlp_train_metrics,
            'validation': mlp_val_metrics,
        },
        'loss_history': {
            'linear': linear_history,
            'mlp': mlp_history,
        },
        'comparison_summary': comparison,
        'artifacts': {
            'loss_curves': loss_plot,
            'prediction_vs_target': pred_plot,
        },
    }
    outputs.update(save_artifacts(outputs, cfg.output_dir))

    exit_code = 0
    failure_reasons: List[str] = []

    if mlp_val_metrics['r2'] <= cfg.r2_threshold:
        failure_reasons.append(
            f"MLP validation R2 {mlp_val_metrics['r2']:.4f} did not exceed threshold {cfg.r2_threshold:.2f}."
        )
    if comparison['val_r2_improvement'] <= cfg.improvement_threshold:
        failure_reasons.append(
            'MLP did not outperform the linear baseline by the required R2 margin: '
            f"{comparison['val_r2_improvement']:.4f} <= {cfg.improvement_threshold:.2f}."
        )

    if failure_reasons:
        exit_code = 1
        print('\nASSERTIONS FAILED:')
        for reason in failure_reasons:
            print(f'- {reason}')
    else: #exit
        print('\nAll assertions passed.')

    print(f'Artifacts saved to: {cfg.output_dir}')
    print(f'Exiting with code {exit_code}')
    sys.exit(exit_code)


if __name__ == '__main__':
    main()
