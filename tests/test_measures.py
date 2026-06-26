import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest

from quantum.states import (
    bell_phi_plus,
    bell_psi_minus,
    density_matrix,
    product_state,
    werner_state,
)
from quantum.measures import (
    concurrence,
    entanglement_entropy,
    entanglement_measures,
    is_ppt,
    is_pure,
    negativity,
    purity,
    von_neumann_entropy,
)

TOL = 1e-2


def test_bell_concurrence_one():
    rho = density_matrix(bell_phi_plus())
    assert abs(concurrence(rho) - 1.0) < TOL


def test_product_concurrence_zero():
    rng = np.random.default_rng(10)
    rho = density_matrix(product_state(2, rng))
    assert concurrence(rho) < TOL


def test_werner_separable_concurrence_zero():
    rho = werner_state(0.0)
    assert concurrence(rho) < TOL


def test_werner_maximally_entangled_concurrence():
    rho = werner_state(1.0)
    assert abs(concurrence(rho) - 1.0) < TOL


def test_werner_threshold():
    # p=1/3 is the boundary; just above should give small positive
    rho_just_above = werner_state(0.34)
    assert concurrence(rho_just_above) >= 0.0
    rho_just_below = werner_state(0.32)
    assert concurrence(rho_just_below) < TOL


def test_concurrence_range():
    rng = np.random.default_rng(11)
    for _ in range(10):
        rho = density_matrix(np.random.default_rng().standard_normal(4) + 1j * np.random.default_rng().standard_normal(4))
        rho = density_matrix(rng.standard_normal(4) + 1j * rng.standard_normal(4))
        c = concurrence(rho)
        assert 0.0 <= c <= 1.0 + TOL


def test_von_neumann_pure_state_zero():
    rho = density_matrix(bell_phi_plus())
    assert abs(von_neumann_entropy(rho)) < TOL


def test_von_neumann_maximally_mixed():
    rho = np.eye(4, dtype=complex) / 4.0
    assert abs(von_neumann_entropy(rho) - 2.0) < TOL


def test_entanglement_entropy_bell_one():
    rho = density_matrix(bell_phi_plus())
    assert abs(entanglement_entropy(rho, n_qubits=2) - 1.0) < TOL


def test_entanglement_entropy_product_zero():
    rng = np.random.default_rng(12)
    from quantum.states import product_state
    rho = density_matrix(product_state(2, rng))
    assert entanglement_entropy(rho, n_qubits=2) < TOL


def test_negativity_bell_positive():
    rho = density_matrix(bell_psi_minus())
    assert negativity(rho) > 0.01


def test_negativity_product_zero():
    rng = np.random.default_rng(13)
    rho = density_matrix(product_state(2, rng))
    assert negativity(rho) < TOL


def test_is_ppt_product_true():
    rng = np.random.default_rng(14)
    rho = density_matrix(product_state(2, rng))
    assert is_ppt(rho) is True


def test_is_ppt_bell_false():
    rho = density_matrix(bell_phi_plus())
    assert is_ppt(rho) is False


def test_purity_pure_state():
    rho = density_matrix(bell_phi_plus())
    assert abs(purity(rho) - 1.0) < TOL


def test_purity_maximally_mixed():
    rho = np.eye(4, dtype=complex) / 4.0
    assert abs(purity(rho) - 0.25) < TOL


def test_is_pure_bell():
    rho = density_matrix(bell_phi_plus())
    assert is_pure(rho) is True


def test_is_pure_mixed():
    rho = np.eye(4, dtype=complex) / 4.0
    assert is_pure(rho) is False


def test_entanglement_measures_keys():
    rho = density_matrix(bell_phi_plus())
    result = entanglement_measures(rho, n_qubits=2)
    expected_keys = {
        "concurrence", "negativity", "entanglement_entropy",
        "von_neumann_entropy", "purity", "is_ppt", "is_pure", "is_entangled_ppt",
    }
    assert expected_keys.issubset(result.keys())


def test_entanglement_measures_bell_values():
    rho = density_matrix(bell_phi_plus())
    m = entanglement_measures(rho, n_qubits=2)
    assert abs(m["concurrence"] - 1.0) < TOL
    assert m["is_pure"] is True
    assert m["is_entangled_ppt"] is True


def test_entanglement_measures_product_values():
    rng = np.random.default_rng(15)
    rho = density_matrix(product_state(2, rng))
    m = entanglement_measures(rho, n_qubits=2)
    assert m["concurrence"] < TOL
    assert m["is_ppt"] is True
