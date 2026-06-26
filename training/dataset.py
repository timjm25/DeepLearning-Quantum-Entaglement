import warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
from torch.utils.data import Dataset

from quantum.states import (
    density_matrix,
    random_pure_state,
    random_mixed_state,
    product_state,
    werner_state,
    bell_phi_plus,
    bell_phi_minus,
    bell_psi_plus,
    bell_psi_minus,
    density_matrix_to_features,
)
from quantum.measures import concurrence, entanglement_entropy, negativity


class QuantumStateDataset(Dataset):
    """
    Synthetically generated, labelled quantum state dataset.

    Each sample: (feature_vector, entangled_label, measures_vector)
    where measures_vector = [concurrence, entropy, negativity].
    """

    def __init__(self, n_samples: int = 8000, n_qubits: int = 2, seed: int = 42):
        self.n_samples = n_samples
        self.n_qubits = n_qubits
        rng = np.random.default_rng(seed)
        self.samples: list = []
        self._generate(rng)

    def _generate(self, rng):
        per_type = self.n_samples // 8

        generators = [
            lambda: density_matrix(random_pure_state(self.n_qubits, rng)),
            lambda: density_matrix(bell_phi_plus()),
            lambda: density_matrix(bell_phi_minus()),
            lambda: density_matrix(bell_psi_plus()),
            lambda: density_matrix(product_state(self.n_qubits, rng)),
            lambda: werner_state(float(rng.uniform(0.0, 0.33))),
            lambda: random_mixed_state(self.n_qubits, rng=rng),
            lambda: werner_state(float(rng.uniform(0.0, 1.0))),
        ]

        for gen in generators:
            for _ in range(per_type):
                try:
                    rho = gen()
                    self._add(rho)
                except Exception:
                    pass

        while len(self.samples) < self.n_samples:
            try:
                rho = density_matrix(random_pure_state(self.n_qubits, rng))
                self._add(rho)
            except Exception:
                pass

    def _add(self, rho):
        c = concurrence(rho) if self.n_qubits == 2 else 0.0
        ee = min(1.0, entanglement_entropy(rho, self.n_qubits))
        neg = min(1.0, negativity(rho) * 2.0) if self.n_qubits == 2 else 0.0
        label = 1.0 if c > 0.01 else 0.0
        features = density_matrix_to_features(rho)
        self.samples.append(
            (features, label, np.array([c, ee, neg], dtype=np.float32))
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        features, label, measures = self.samples[idx]
        return (
            torch.tensor(features, dtype=torch.float32),
            torch.tensor(label, dtype=torch.float32),
            torch.tensor(measures, dtype=torch.float32),
        )
