import numpy as np

SIGMA_X = np.array([[0, 1], [1, 0]], dtype=complex)
SIGMA_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
SIGMA_Z = np.array([[1, 0], [0, -1]], dtype=complex)


def ket_zero():
    return np.array([1.0, 0.0], dtype=complex)


def ket_one():
    return np.array([0.0, 1.0], dtype=complex)


def bell_phi_plus():
    """(|00⟩ + |11⟩) / √2"""
    state = np.zeros(4, dtype=complex)
    state[0] = state[3] = 1 / np.sqrt(2)
    return state


def bell_phi_minus():
    """(|00⟩ - |11⟩) / √2"""
    state = np.zeros(4, dtype=complex)
    state[0] = 1 / np.sqrt(2)
    state[3] = -1 / np.sqrt(2)
    return state


def bell_psi_plus():
    """(|01⟩ + |10⟩) / √2"""
    state = np.zeros(4, dtype=complex)
    state[1] = state[2] = 1 / np.sqrt(2)
    return state


def bell_psi_minus():
    """(|01⟩ - |10⟩) / √2 — singlet state"""
    state = np.zeros(4, dtype=complex)
    state[1] = 1 / np.sqrt(2)
    state[2] = -1 / np.sqrt(2)
    return state


def density_matrix(state_vector: np.ndarray) -> np.ndarray:
    """Pure state density matrix ρ = |ψ⟩⟨ψ|"""
    psi = state_vector / np.linalg.norm(state_vector)
    return np.outer(psi, psi.conj())


def ghz_state(n_qubits: int = 3) -> np.ndarray:
    """GHZ state (|00...0⟩ + |11...1⟩) / √2"""
    dim = 2 ** n_qubits
    state = np.zeros(dim, dtype=complex)
    state[0] = state[-1] = 1 / np.sqrt(2)
    return state


def w_state(n_qubits: int = 3) -> np.ndarray:
    """W state: uniform superposition of single-excitation states"""
    dim = 2 ** n_qubits
    state = np.zeros(dim, dtype=complex)
    for k in range(n_qubits):
        idx = 2 ** (n_qubits - 1 - k)
        state[idx] = 1 / np.sqrt(n_qubits)
    return state


def werner_state(p: float) -> np.ndarray:
    """Werner state: p|Ψ-⟩⟨Ψ-| + (1-p)I/4. Entangled iff p > 1/3."""
    psi_minus = density_matrix(bell_psi_minus())
    identity = np.eye(4, dtype=complex) / 4.0
    return p * psi_minus + (1 - p) * identity


def random_pure_state(n_qubits: int = 2, rng=None) -> np.ndarray:
    """Haar-random pure state."""
    if rng is None:
        rng = np.random.default_rng()
    dim = 2 ** n_qubits
    real = rng.standard_normal(dim)
    imag = rng.standard_normal(dim)
    state = real + 1j * imag
    return state / np.linalg.norm(state)


def random_mixed_state(n_qubits: int = 2, rank: int = None, rng=None) -> np.ndarray:
    """Random mixed state as mixture of Haar-random pure states."""
    if rng is None:
        rng = np.random.default_rng()
    dim = 2 ** n_qubits
    if rank is None:
        rank = int(rng.integers(1, dim + 1))
    states = [random_pure_state(n_qubits, rng) for _ in range(rank)]
    weights = rng.dirichlet(np.ones(rank))
    rho = sum(w * density_matrix(s) for w, s in zip(weights, states))
    return rho


def product_state(n_qubits: int = 2, rng=None) -> np.ndarray:
    """Separable product state: tensor product of random single-qubit states."""
    if rng is None:
        rng = np.random.default_rng()
    state = np.array([1.0], dtype=complex)
    for _ in range(n_qubits):
        single = random_pure_state(1, rng)
        state = np.kron(state, single)
    return state


def partial_trace(rho: np.ndarray, keep: list, dims: list) -> np.ndarray:
    """Partial trace: keep subsystems in `keep`, trace out the rest."""
    n = len(dims)
    rho_tensor = rho.reshape(dims * 2)
    trace_out = [i for i in range(n) if i not in keep]
    for idx in sorted(trace_out, reverse=True):
        rho_tensor = np.trace(rho_tensor, axis1=idx, axis2=idx + n)
        n -= 1
    keep_dims = [dims[i] for i in keep]
    d = int(np.prod(keep_dims))
    return rho_tensor.reshape(d, d)


def density_matrix_to_features(rho: np.ndarray) -> np.ndarray:
    """Flatten real and imaginary parts into a float32 feature vector."""
    return np.concatenate([rho.real.flatten(), rho.imag.flatten()]).astype(np.float32)
