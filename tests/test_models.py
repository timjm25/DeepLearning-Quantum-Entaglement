import os
import sys
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest
import torch

from models.entanglement_net import EntanglementNet
from training.dataset import QuantumStateDataset

MODEL_PATH = Path(__file__).parent.parent / "trained_models" / "entanglement_net.pt"
MODEL_TRAINED = MODEL_PATH.exists()


def test_model_instantiates():
    model = EntanglementNet(n_qubits=2)
    assert model is not None


def test_model_input_size():
    model = EntanglementNet(n_qubits=2)
    dim = 2 ** 2
    expected_input = 2 * dim * dim
    assert expected_input == 32


def test_model_forward_shapes():
    model = EntanglementNet(n_qubits=2)
    model.eval()
    x = torch.randn(4, 32)
    prob, measures = model(x)
    assert prob.shape == (4,)
    assert measures.shape == (4, 3)


def test_model_classifier_output_range():
    model = EntanglementNet(n_qubits=2)
    model.eval()
    x = torch.randn(8, 32)
    with torch.no_grad():
        prob, _ = model(x)
    assert torch.all(prob >= 0.0) and torch.all(prob <= 1.0)


def test_model_regressor_output_range():
    model = EntanglementNet(n_qubits=2)
    model.eval()
    x = torch.randn(8, 32)
    with torch.no_grad():
        _, measures = model(x)
    assert torch.all(measures >= 0.0) and torch.all(measures <= 1.0)


def test_predict_density_matrix_keys():
    model = EntanglementNet(n_qubits=2)
    model.eval()
    from quantum.states import bell_phi_plus, density_matrix
    rho = density_matrix(bell_phi_plus())
    result = model.predict_density_matrix(rho)
    expected = {"entanglement_probability", "is_entangled", "predicted_concurrence",
                "predicted_entropy", "predicted_negativity", "confidence"}
    assert expected.issubset(result.keys())


def test_predict_density_matrix_prob_range():
    model = EntanglementNet(n_qubits=2)
    model.eval()
    from quantum.states import bell_phi_plus, density_matrix
    rho = density_matrix(bell_phi_plus())
    result = model.predict_density_matrix(rho)
    assert 0.0 <= result["entanglement_probability"] <= 1.0


def test_predict_density_matrix_confidence_valid():
    model = EntanglementNet(n_qubits=2)
    model.eval()
    from quantum.states import bell_phi_plus, density_matrix
    rho = density_matrix(bell_phi_plus())
    result = model.predict_density_matrix(rho)
    assert result["confidence"] in ("high", "medium", "low")


@pytest.mark.skipif(not MODEL_TRAINED, reason="model not trained")
def test_trained_model_bell_entangled():
    model = EntanglementNet.from_checkpoint(str(MODEL_PATH))
    from quantum.states import bell_phi_plus, density_matrix
    rho = density_matrix(bell_phi_plus())
    result = model.predict_density_matrix(rho)
    assert result["is_entangled"] is True


@pytest.mark.skipif(not MODEL_TRAINED, reason="model not trained")
def test_trained_model_product_separable():
    model = EntanglementNet.from_checkpoint(str(MODEL_PATH))
    from quantum.states import density_matrix, product_state
    rng = np.random.default_rng(99)
    rho = density_matrix(product_state(2, rng))
    result = model.predict_density_matrix(rho)
    assert result["is_entangled"] is False


def test_dataset_length():
    ds = QuantumStateDataset(n_samples=200, n_qubits=2, seed=0)
    assert len(ds) >= 180


def test_dataset_item_types():
    ds = QuantumStateDataset(n_samples=100, n_qubits=2, seed=1)
    x, y, m = ds[0]
    assert x.dtype == torch.float32
    assert y.dtype == torch.float32
    assert m.dtype == torch.float32


def test_dataset_feature_shape():
    ds = QuantumStateDataset(n_samples=100, n_qubits=2, seed=2)
    x, y, m = ds[0]
    assert x.shape == (32,)
    assert m.shape == (3,)


def test_dataset_label_binary():
    ds = QuantumStateDataset(n_samples=200, n_qubits=2, seed=3)
    for i in range(min(20, len(ds))):
        _, y, _ = ds[i]
        assert y.item() in (0.0, 1.0)


def test_trainer_smoke():
    from training.trainer import train
    result = train(n_samples=200, epochs=3, batch_size=32)
    assert "best_val_accuracy" in result
    assert 0.0 <= result["best_val_accuracy"] <= 1.0
