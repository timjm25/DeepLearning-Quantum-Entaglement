import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest

from quantum.states import (
    bell_phi_minus,
    bell_phi_plus,
    bell_psi_minus,
    bell_psi_plus,
    density_matrix,
    density_matrix_to_features,
    ghz_state,
    ket_one,
    ket_zero,
    partial_trace,
    product_state,
    random_mixed_state,
    random_pure_state,
    w_state,
    werner_state,
)


def test_ket_zero_normalised():
    v = ket_zero()
    assert abs(np.linalg.norm(v) - 1.0) < 1e-10


def test_ket_one_normalised():
    v = ket_one()
    assert abs(np.linalg.norm(v) - 1.0) < 1e-10


def test_bell_phi_plus_dimension():
    assert bell_phi_plus().shape == (4,)


def test_bell_phi_plus_normalised():
    assert abs(np.linalg.norm(bell_phi_plus()) - 1.0) < 1e-10


def test_bell_phi_minus_normalised():
    assert abs(np.linalg.norm(bell_phi_minus()) - 1.0) < 1e-10


def test_bell_psi_plus_normalised():
    assert abs(np.linalg.norm(bell_psi_plus()) - 1.0) < 1e-10


def test_bell_psi_minus_normalised():
    assert abs(np.linalg.norm(bell_psi_minus()) - 1.0) < 1e-10


def test_bell_states_orthogonal():
    states = [bell_phi_plus(), bell_phi_minus(), bell_psi_plus(), bell_psi_minus()]
    for i in range(4):
        for j in range(4):
            ip = abs(np.dot(states[i].conj(), states[j]))
            if i == j:
                assert abs(ip - 1.0) < 1e-10
            else:
                assert ip < 1e-10


def test_density_matrix_shape():
    rho = density_matrix(bell_phi_plus())
    assert rho.shape == (4, 4)


def test_density_matrix_hermitian():
    rho = density_matrix(bell_phi_plus())
    assert np.allclose(rho, rho.conj().T, atol=1e-10)


def test_density_matrix_trace_one():
    rho = density_matrix(bell_phi_plus())
    assert abs(np.trace(rho) - 1.0) < 1e-10


def test_density_matrix_positive_semidefinite():
    rho = density_matrix(bell_psi_minus())
    eigvals = np.linalg.eigvalsh(rho)
    assert np.all(eigvals >= -1e-10)


def test_ghz_state_dimension():
    assert ghz_state(3).shape == (8,)


def test_ghz_state_normalised():
    assert abs(np.linalg.norm(ghz_state(3)) - 1.0) < 1e-10


def test_ghz_state_two_nonzero():
    g = ghz_state(3)
    nonzero = np.count_nonzero(np.abs(g) > 1e-10)
    assert nonzero == 2


def test_w_state_dimension():
    assert w_state(3).shape == (8,)


def test_w_state_normalised():
    assert abs(np.linalg.norm(w_state(3)) - 1.0) < 1e-10


def test_w_state_nonzero_entries():
    w = w_state(3)
    nonzero = np.count_nonzero(np.abs(w) > 1e-10)
    assert nonzero == 3


def test_werner_state_trace_one():
    rho = werner_state(0.5)
    assert abs(np.trace(rho) - 1.0) < 1e-10


def test_werner_state_hermitian():
    rho = werner_state(0.8)
    assert np.allclose(rho, rho.conj().T, atol=1e-10)


def test_random_pure_state_normalised():
    rng = np.random.default_rng(0)
    psi = random_pure_state(2, rng)
    assert abs(np.linalg.norm(psi) - 1.0) < 1e-10


def test_random_mixed_state_trace_one():
    rng = np.random.default_rng(1)
    rho = random_mixed_state(2, rng=rng)
    assert abs(np.trace(rho) - 1.0) < 1e-10


def test_product_state_normalised():
    rng = np.random.default_rng(2)
    psi = product_state(2, rng)
    assert abs(np.linalg.norm(psi) - 1.0) < 1e-10


def test_partial_trace_preserves_trace():
    rng = np.random.default_rng(3)
    rho = density_matrix(random_pure_state(2, rng))
    rho_A = partial_trace(rho, keep=[0], dims=[2, 2])
    assert abs(np.trace(rho_A) - 1.0) < 1e-10


def test_partial_trace_shape():
    rho = density_matrix(bell_phi_plus())
    rho_A = partial_trace(rho, keep=[0], dims=[2, 2])
    assert rho_A.shape == (2, 2)


def test_density_matrix_to_features_shape():
    rho = density_matrix(bell_phi_plus())
    feats = density_matrix_to_features(rho)
    assert feats.shape == (32,)


def test_density_matrix_to_features_dtype():
    rho = density_matrix(bell_phi_plus())
    feats = density_matrix_to_features(rho)
    assert feats.dtype == np.float32
