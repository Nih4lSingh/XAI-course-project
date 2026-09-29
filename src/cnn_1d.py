"""
1D Convolutional Neural Network (1D-CNN) Implementation
Sharma et al. (2024) Intrusion Detection Architecture

Architecture:
- Input: [Batch, 1, 36] (NSL-KDD) or [Batch, 1, 38] (UNSW-NB15)
- Conv1D(64, kernel_size=3, padding=1, ReLU)
- MaxPool1D(kernel_size=2, stride=2)
- Conv1D(32, kernel_size=3, padding=1, ReLU)
- MaxPool1D(kernel_size=2, stride=2)
- Conv1D(32, kernel_size=3, padding=1, ReLU)
- Flatten
- Dense(5, Softmax)
- Dropout: 0.0 (paper canonical)
- Optimizer: AdamW (lr=0.001, weight_decay=0.0001)
- Epochs: 20, Batch Size: 128
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_preprocessing import get_dataset


class CNN1D(nn.Module):
    """3-stage 1D-CNN matching Sharma et al. (2024)."""
    def __init__(self, input_dim: int, num_classes: int = 5):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=32, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        
        # Calculate flattened dimension
        dummy = torch.zeros(1, 1, input_dim)
        with torch.no_grad():
            feat = self.conv3(self.pool2(self.conv2(self.pool1(self.conv1(dummy)))))
            flattened_dim = feat.view(1, -1).shape[1]
            
        self.fc = nn.Linear(flattened_dim, num_classes)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input shape: [Batch, 1, Sequence_Length]
        if x.dim() == 2:
            x = x.unsqueeze(1)
        elif x.dim() == 3 and x.shape[-1] == 1:
            x = x.permute(0, 2, 1)
            
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        return self.fc(x)


def train_cnn1d(
    dataset_name: str = "nsl_kdd",
    epochs: int = 20,
    batch_size: int = 128,
    lr: float = 0.001,
    weight_decay: float = 0.0001,
    device: str | None = None,
    save_model: bool = True,
) -> Dict[str, any]:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)
    
    data = get_dataset(dataset_name, mode="flat")
    X_train, y_train = data["X_train"], data["y_train"]
    X_val, y_val = data["X_val"], data["y_val"]
    X_test, y_test = data["X_test"], data["y_test"]
    classes = data["classes"]
    
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)),
        batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val)),
        batch_size=batch_size, shuffle=False
    )
    test_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test)),
        batch_size=batch_size, shuffle=False
    )
    
    model = CNN1D(input_dim=X_train.shape[1], num_classes=len(classes)).to(dev)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, eps=1e-7)
    criterion = nn.CrossEntropyLoss()
    
    print(f"\nTraining 1D-CNN on {data['dataset']} (seq len {X_train.shape[1]} -> 5 classes) on {device}...")
    start_time = time.time()
    
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(dev), yb.to(dev)
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * len(yb)
            correct += (out.argmax(dim=-1) == yb).sum().item()
            total += len(yb)
            
        train_loss = total_loss / total
        train_acc = correct / total
        
        # Validation
        model.eval()
        v_loss, v_corr, v_tot = 0.0, 0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(dev), yb.to(dev)
                out = model(xb)
                loss = criterion(out, yb)
                v_loss += loss.item() * len(yb)
                v_corr += (out.argmax(dim=-1) == yb).sum().item()
                v_tot += len(yb)
        val_loss = v_loss / v_tot
        val_acc = v_corr / v_tot
        
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        
        if epoch % 5 == 0 or epoch == epochs:
            print(f"Epoch {epoch:02d}/{epochs:02d} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")
            
    train_time = time.time() - start_time
    
    # Test evaluation
    model.eval()
    all_preds, all_probs = [], []
    with torch.no_grad():
        for xb, _ in test_loader:
            xb = xb.to(dev)
            out = model(xb)
            prob = F.softmax(out, dim=-1)
            pred = out.argmax(dim=-1)
            all_preds.append(pred.cpu().numpy())
            all_probs.append(prob.cpu().numpy())
            
    y_pred = np.concatenate(all_preds)
    y_prob = np.concatenate(all_probs)
    test_acc = float((y_pred == y_test).mean())
    print(f"Test Accuracy: {test_acc:.4f} (Wall Time: {train_time:.2f}s)")
    
    if save_model:
        model_dir = PROJECT_ROOT / "models" / "1d_cnn"
        model_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), model_dir / f"cnn1d_{dataset_name}.pt")
        
    return {
        "model": model,
        "test_accuracy": test_acc,
        "y_pred": y_pred,
        "y_prob": y_prob,
        "y_test": y_test,
        "history": history,
        "classes": classes,
        "train_time": train_time,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Sharma et al. (2024) 1D-CNN")
    parser.add_argument("--dataset", choices=["nsl_kdd", "unsw_nb15", "both"], default="both")
    parser.add_argument("--epochs", type=int, default=20)
    args = parser.parse_args()
    
    targets = ["nsl_kdd", "unsw_nb15"] if args.dataset == "both" else [args.dataset]
    for ds in targets:
        train_cnn1d(dataset_name=ds, epochs=args.epochs)
