import warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn as nn


class EntanglementNet(nn.Module):
    """
    Multi-task neural network for quantum entanglement analysis.

    Classification head  : P(entangled) in [0, 1]
    Regression head      : [concurrence, entanglement_entropy, negativity] in [0, 1]

    Input: flattened real + imaginary parts of density matrix.
    For 2-qubit (4x4 DM): 2 * 16 = 32 input features.
    """

    def __init__(self, n_qubits: int = 2):
        super().__init__()
        self.n_qubits = n_qubits
        dim = 2 ** n_qubits
        input_size = 2 * dim * dim

        self.backbone = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
        )

        self.classifier = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

        self.regressor = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 3),
            nn.Sigmoid(),
        )

    def forward(self, x):
        features = self.backbone(x)
        entanglement_prob = self.classifier(features).squeeze(-1)
        measures = self.regressor(features)
        return entanglement_prob, measures

    @classmethod
    def from_checkpoint(cls, path: str, n_qubits: int = 2) -> "EntanglementNet":
        model = cls(n_qubits=n_qubits)
        state = torch.load(path, map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        model.eval()
        return model

    def predict_density_matrix(self, rho: np.ndarray) -> dict:
        """Predict entanglement from a single numpy density matrix."""
        from quantum.states import density_matrix_to_features
        self.eval()
        features = density_matrix_to_features(rho)
        x = torch.tensor(list(features), dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            prob, measures = self(x)
        prob_val = float(prob.item())
        m = measures[0].tolist()
        conf_gap = abs(prob_val - 0.5)
        confidence = "high" if conf_gap > 0.3 else "medium" if conf_gap > 0.15 else "low"
        return {
            "entanglement_probability": round(prob_val, 4),
            "is_entangled": prob_val > 0.5,
            "predicted_concurrence": round(float(m[0]), 4),
            "predicted_entropy": round(float(m[1]), 4),
            "predicted_negativity": round(float(m[2]), 4),
            "confidence": confidence,
        }
