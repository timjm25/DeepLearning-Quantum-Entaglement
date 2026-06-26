import json
import os
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from models.entanglement_net import EntanglementNet
from training.dataset import QuantumStateDataset

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "trained_models")


def train(
    n_samples: int = 8000,
    epochs: int = 40,
    batch_size: int = 128,
    n_qubits: int = 2,
) -> dict:
    """Train EntanglementNet on synthetic quantum states. Returns metrics dict."""
    os.makedirs(MODEL_DIR, exist_ok=True)

    print(f"  Generating {n_samples} quantum state samples...")
    dataset = QuantumStateDataset(n_samples=n_samples, n_qubits=n_qubits, seed=42)

    n_val = max(1, int(0.2 * len(dataset)))
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(
        dataset, [n_train, n_val], generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size)

    model = EntanglementNet(n_qubits=n_qubits)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    bce = nn.BCELoss()
    mse = nn.MSELoss()

    best_val_acc = 0.0
    history: dict = {"train_loss": [], "val_loss": [], "val_accuracy": []}

    print(f"  Training for {epochs} epochs...")
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for x, y_cls, y_reg in train_loader:
            optimizer.zero_grad()
            prob, measures = model(x)
            loss = bce(prob, y_cls) + 0.5 * mse(measures, y_reg)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0.0
        correct = 0
        with torch.no_grad():
            for x, y_cls, y_reg in val_loader:
                prob, measures = model(x)
                val_loss += (bce(prob, y_cls) + 0.5 * mse(measures, y_reg)).item()
                correct += int(((prob > 0.5).float() == y_cls).sum().item())

        acc = correct / len(val_set)
        history["train_loss"].append(train_loss / len(train_loader))
        history["val_loss"].append(val_loss / len(val_loader))
        history["val_accuracy"].append(acc)

        if acc > best_val_acc:
            best_val_acc = acc
            torch.save(
                model.state_dict(),
                os.path.join(MODEL_DIR, "entanglement_net.pt"),
            )

        if (epoch + 1) % 10 == 0:
            print(
                f"    Epoch {epoch + 1}/{epochs}"
                f" — train_loss: {train_loss / len(train_loader):.4f}"
                f", val_acc: {acc:.4f}"
            )

        scheduler.step()

    with open(os.path.join(MODEL_DIR, "training_history.json"), "w") as f:
        json.dump(history, f)

    print(f"  Training complete — best val accuracy: {best_val_acc:.4f}")
    return {"best_val_accuracy": best_val_acc, "epochs": epochs, "history": history}
