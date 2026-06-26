"""
Example: compute exact entanglement measures for common quantum states.
Run from the repo root: python3 examples/compute_entanglement.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
from quantum.states import (
    density_matrix, bell_phi_plus, bell_psi_minus,
    werner_state, product_state, ghz_state, w_state,
)
from quantum.measures import entanglement_measures

STATES = [
    ("Bell |Φ+⟩  (max entangled)",  density_matrix(bell_phi_plus())),
    ("Bell |Ψ-⟩  (singlet)",        density_matrix(bell_psi_minus())),
    ("Werner p=0.9 (entangled)",     werner_state(0.9)),
    ("Werner p=0.2 (separable)",     werner_state(0.2)),
    ("Product state (separable)",    density_matrix(product_state(2))),
    ("Random pure state",            density_matrix(__import__('quantum.states', fromlist=['random_pure_state']).random_pure_state(2))),
]

def fmt(v):
    if isinstance(v, bool): return "Yes" if v else "No"
    return f"{v:.4f}"

print(f"\n{'State':<35} {'Conc':>6} {'Entropy':>8} {'Neg':>6} {'Purity':>7} {'PPT':>5} {'Pure':>5}")
print("-" * 80)
for name, rho in STATES:
    m = entanglement_measures(rho, n_qubits=2)
    print(
        f"{name:<35} "
        f"{fmt(m['concurrence']):>6} "
        f"{fmt(m['entanglement_entropy']):>8} "
        f"{fmt(m['negativity']):>6} "
        f"{fmt(m['purity']):>7} "
        f"{fmt(m['is_ppt']):>5} "
        f"{fmt(m['is_pure']):>5}"
    )

print("\nGHZ(3) and W(3) entanglement entropy:")
for name, state in [("GHZ(3)", ghz_state(3)), ("W(3)", w_state(3))]:
    rho = density_matrix(state)
    m = entanglement_measures(rho, n_qubits=3)
    print(f"  {name}: entropy={fmt(m['entanglement_entropy'])}, purity={fmt(m['purity'])}")
