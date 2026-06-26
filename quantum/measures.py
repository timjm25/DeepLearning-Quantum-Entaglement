import numpy as np
from scipy.linalg import sqrtm
from .states import partial_trace


def von_neumann_entropy(rho: np.ndarray, base: int = 2) -> float:
    """S(ρ) = -Tr(ρ log ρ). Zero for pure states, max log(d) for maximally mixed."""
    eigenvalues = np.linalg.eigvalsh(rho)
    eigenvalues = eigenvalues[eigenvalues > 1e-12]
    if base == 2:
        return float(-np.sum(eigenvalues * np.log2(eigenvalues)))
    return float(-np.sum(eigenvalues * np.log(eigenvalues)))


def entanglement_entropy(rho: np.ndarray, n_qubits: int = 2) -> float:
    """Entanglement entropy = von Neumann entropy of reduced density matrix of subsystem A."""
    nA = n_qubits // 2
    dims = [2] * n_qubits
    rho_A = partial_trace(rho, keep=list(range(nA)), dims=dims)
    return von_neumann_entropy(rho_A)


def concurrence(rho: np.ndarray) -> float:
    """Wootters concurrence for 2-qubit density matrix. Range [0, 1]."""
    sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sysy = np.kron(sy, sy)
    rho_tilde = sysy @ rho.conj() @ sysy
    sqrt_rho = np.array(sqrtm(rho), dtype=np.complex128)
    M = np.array(sqrt_rho @ rho_tilde @ sqrt_rho, dtype=np.complex128)
    eigvals = np.sort(np.real(np.sqrt(np.maximum(0, np.linalg.eigvals(M)))))[::-1]
    return float(max(0.0, eigvals[0] - eigvals[1] - eigvals[2] - eigvals[3]))


def partial_transpose(rho: np.ndarray, subsystem: int = 0, dims: tuple = (2, 2)) -> np.ndarray:
    """Partial transpose with respect to given subsystem."""
    d0, d1 = dims
    rho_reshaped = rho.reshape(d0, d1, d0, d1)
    if subsystem == 0:
        rho_pt = rho_reshaped.transpose(2, 1, 0, 3)
    else:
        rho_pt = rho_reshaped.transpose(0, 3, 2, 1)
    return rho_pt.reshape(d0 * d1, d0 * d1)


def negativity(rho: np.ndarray, dims: tuple = (2, 2)) -> float:
    """Negativity: sum of absolute negative eigenvalues of partial transpose."""
    rho_pt = partial_transpose(rho, subsystem=0, dims=dims)
    eigenvalues = np.linalg.eigvalsh(rho_pt)
    negative = eigenvalues[eigenvalues < 0]
    return float(abs(np.sum(negative)))


def is_ppt(rho: np.ndarray, dims: tuple = (2, 2), tol: float = 1e-8) -> bool:
    """Peres-Horodecki PPT criterion. Separable iff partial transpose is positive semi-definite."""
    rho_pt = partial_transpose(rho, subsystem=0, dims=dims)
    eigenvalues = np.linalg.eigvalsh(rho_pt)
    return bool(np.all(eigenvalues >= -tol))


def is_pure(rho: np.ndarray, tol: float = 1e-6) -> bool:
    """Check if state is pure: Tr(ρ²) ≈ 1."""
    return bool(abs(np.real(np.trace(rho @ rho)) - 1.0) < tol)


def purity(rho: np.ndarray) -> float:
    """Purity Tr(ρ²). = 1 for pure, 1/d for maximally mixed."""
    return float(np.real(np.trace(rho @ rho)))


def entanglement_measures(rho: np.ndarray, n_qubits: int = 2) -> dict:
    """Compute all entanglement measures for a density matrix."""
    c = concurrence(rho) if n_qubits == 2 else 0.0
    neg = negativity(rho) if n_qubits == 2 else 0.0
    ee = entanglement_entropy(rho, n_qubits)
    vne = von_neumann_entropy(rho)
    p = purity(rho)
    ppt = is_ppt(rho) if n_qubits == 2 else True
    return {
        "concurrence": round(c, 6),
        "negativity": round(neg, 6),
        "entanglement_entropy": round(ee, 6),
        "von_neumann_entropy": round(vne, 6),
        "purity": round(p, 6),
        "is_ppt": ppt,
        "is_pure": is_pure(rho),
        "is_entangled_ppt": not ppt,
    }
