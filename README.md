# Deep Learning for Quantum Entanglement

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Quantum](https://img.shields.io/badge/Domain-Quantum%20Information-blueviolet.svg)
![Deep Learning](https://img.shields.io/badge/Deep%20Learning-Multi--task%20NN-orange.svg)
![Flask](https://img.shields.io/badge/Flask-3.0-green.svg)

An open-source research toolkit that applies deep learning to the detection and quantification of quantum entanglement. The system combines exact analytic computation of entanglement measures with a trained neural network for fast, approximate classification — giving researchers an interactive tool to explore quantum states across the separability boundary.

---

## Overview

**Who is this for?**
- Quantum information researchers studying entanglement detection algorithms
- Graduate students learning quantum computing and quantum optics
- Quantum computing engineers benchmarking NISQ device outputs
- Educators teaching quantum mechanics interactively

**What does it do?**
1. Generates or accepts quantum density matrices (Bell states, GHZ, W, Werner, random pure/mixed states)
2. Computes exact entanglement measures analytically: Wootters concurrence, von Neumann entropy, negativity, and the Peres–Horodecki PPT criterion
3. Runs a trained multi-task neural network (`EntanglementNet`) to predict entanglement probability and estimate all three measures simultaneously
4. Compares neural network predictions against exact values in a Chart.js bar chart
5. Visualises the density matrix as an interactive heatmap
6. Exposes all functionality via a REST API for programmatic integration

---

## Scientific Background

### The Entanglement Problem

Quantum entanglement is a correlation between quantum systems that has no classical analogue. For a bipartite system with subsystems A and B, a quantum state ρ is **separable** if it can be written as:

```
ρ = Σᵢ pᵢ ρᵢ^A ⊗ ρᵢ^B     (pᵢ ≥ 0, Σpᵢ = 1)
```

Otherwise it is **entangled**. Determining separability is:
- **Easy for pure states**: ρ = |ψ⟩⟨ψ| is entangled iff the Schmidt rank > 1
- **NP-hard for mixed states in general** (Gurvits 2003)
- **Decidable for 2×2 and 2×3 systems** via the PPT criterion (Peres 1996; Horodecki 1996)

### Why Deep Learning?

For larger systems or noisy experimental data, exact entanglement detection becomes computationally intractable. Neural networks trained on labelled quantum states can learn to approximate the separability boundary — trading provable guarantees for speed. Recent work (Gao et al. 2018; Carleo et al. 2019) has shown that physics-informed ML models achieve high accuracy on state classification while being orders of magnitude faster than convex optimisation approaches.

This tool implements that idea with:
- **Synthetic dataset generation** from analytically tractable state families
- **Multi-task learning**: classification (entangled/separable) and regression (concurrence, entropy, negativity) share a backbone
- **Exact validation**: every neural network prediction is shown alongside the analytic ground truth

---

## Supported Quantum States

| State | Formula | Properties |
|---|---|---|
| Bell \|Φ+⟩ | `(|00⟩ + |11⟩) / √2` | Maximally entangled, C=1, used in quantum teleportation |
| Bell \|Φ-⟩ | `(|00⟩ - |11⟩) / √2` | Maximally entangled, C=1 |
| Bell \|Ψ+⟩ | `(|01⟩ + |10⟩) / √2` | Maximally entangled, C=1 |
| Bell \|Ψ-⟩ | `(|01⟩ - |10⟩) / √2` | Singlet state; ground state of isotropic Heisenberg model |
| GHZ (n-qubit) | `(|00...0⟩ + |11...1⟩) / √2` | Maximally multipartite entangled; fragile to particle loss |
| W State (n-qubit) | `(|100⟩ + |010⟩ + |001⟩) / √3` | Robust to particle loss; different entanglement class from GHZ |
| Werner State | `p\|Ψ-⟩⟨Ψ-\| + (1-p)I/4` | Entangled iff p > 1/3 (PPT boundary); parameterised by p |
| Product State | `\|ψ⟩_A ⊗ \|φ⟩_B` | Separable — no entanglement |
| Haar-random Pure | Gaussian random vector, normalised | Typically entangled for 2-qubit systems |

---

## Entanglement Measures

### Concurrence (Wootters 1998)

The Wootters concurrence C(ρ) ∈ [0,1] is the most widely used entanglement measure for 2-qubit mixed states. C=0 iff separable; C=1 iff maximally entangled.

```
C(ρ) = max(0, λ₁ - λ₂ - λ₃ - λ₄)

where λᵢ are eigenvalues in decreasing order of:
  R = √( √ρ · ρ̃ · √ρ )

and ρ̃ = (σy ⊗ σy) ρ* (σy ⊗ σy)   [spin-flip operation]
```

### Von Neumann Entropy

```
S(ρ) = -Tr(ρ log₂ ρ) = -Σᵢ λᵢ log₂ λᵢ

Entanglement entropy of a pure bipartite state |ψ⟩:
  E(|ψ⟩) = S(ρ_A)  where  ρ_A = Tr_B(|ψ⟩⟨ψ|)
```

### Negativity and PPT Criterion

```
N(ρ) = ( ‖ρ^{Γ_A}‖₁ - 1 ) / 2

ρ^{Γ_A} is the partial transpose of ρ w.r.t. subsystem A.
N > 0  ⟺  NPT state  ⟹  entangled
For 2×2 and 2×3 systems:  PPT  ⟺  separable  (Peres-Horodecki theorem)
```

---

## Deep Learning Architecture — EntanglementNet

### Model Design

```
Input: density matrix ρ (4×4 complex → 32 real features)
       [Re(ρ₀₀), ..., Re(ρ₃₃), Im(ρ₀₀), ..., Im(ρ₃₃)]

Backbone:
  Linear(32, 256) → BatchNorm1d → ReLU → Dropout(0.3)
  Linear(256, 128) → BatchNorm1d → ReLU → Dropout(0.2)
  Linear(128, 64)  → ReLU

         ┌─────────────────────┬──────────────────────────┐
         │  Classification     │  Regression              │
         │  Linear(64, 32)     │  Linear(64, 32)          │
         │  ReLU               │  ReLU                    │
         │  Linear(32, 1)      │  Linear(32, 3)           │
         │  Sigmoid            │  Sigmoid                 │
         │  P(entangled)∈[0,1] │  [C, S, N] ∈ [0,1]³    │
         └─────────────────────┴──────────────────────────┘

Multi-task loss:
  L = BCE(ŷ_cls, y_cls) + 0.5 × MSE(ŷ_reg, y_reg)
```

### Training Data Generation

The dataset (`QuantumStateDataset`) generates 8,000 labelled quantum states from 8 state families:
1. Haar-random pure 2-qubit states (typically entangled)
2. Bell |Φ+⟩ copies (maximally entangled)
3. Bell |Φ-⟩ copies
4. Bell |Ψ+⟩ copies
5. Random 2-qubit product states (separable)
6. Werner states with p ∈ [0, 0.33] (always separable)
7. Random mixed states (varied entanglement)
8. Werner states with p ∈ [0, 1] (full range)

Ground-truth labels are computed analytically using the Wootters concurrence formula. A state is labelled entangled if C > 0.01.

### Training Configuration

- Optimiser: Adam (lr=1e-3, weight_decay=1e-4)
- LR schedule: Cosine annealing over 40 epochs
- Train/val split: 80/20
- Batch size: 128
- Best model saved by validation accuracy

---

## Features

| Feature | Details |
|---|---|
| Interactive web UI | Select state type, tune Werner parameter p, click Predict |
| Neural network prediction | Entanglement probability + 3 continuous measure estimates |
| Exact analytic computation | Concurrence, entropy, negativity, purity, PPT check |
| Side-by-side comparison | Chart.js bar chart comparing predicted vs. exact measures |
| Density matrix heatmap | CSS grid heatmap of the real part of ρ |
| Theory page | Full background with formulas and clickable paper links |
| REST API | 3 JSON endpoints for programmatic access |
| MIT licence | Open for academic and commercial use |

---

## Directory Structure

```
DeepLearning-Quantum-Entaglement/
├── quantum/
│   ├── states.py          Bell/GHZ/W/Werner/random state generators, partial trace
│   └── measures.py        Concurrence, von Neumann entropy, negativity, PPT
├── models/
│   └── entanglement_net.py  EntanglementNet (PyTorch multi-task NN)
├── training/
│   ├── dataset.py         QuantumStateDataset — synthetic labelled states
│   └── trainer.py         Training loop, checkpoint saving, history export
├── app/
│   ├── web.py             Flask factory — browser routes + REST API
│   └── templates/
│       ├── base.html          Dark theme layout, Chart.js
│       ├── index.html         State selector, prediction UI, heatmap
│       └── about.html         Theory, formulas, references
├── tests/
│   ├── test_states.py     25 tests — state generation, partial trace
│   ├── test_measures.py   22 tests — all entanglement measures
│   └── test_models.py     15 tests — model, dataset, trainer smoke test
├── trained_models/        Model checkpoint saved here after training
├── LICENSE                MIT
├── requirements.txt
└── run.py                 Entry point: train if needed, start Flask on :5090
```

---

## Installation

```bash
# Clone
git clone https://github.com/timjm25/DeepLearning-Quantum-Entaglement.git
cd DeepLearning-Quantum-Entaglement

# Install dependencies
pip install -r requirements.txt
```

**Dependencies:**
| Package | Version | Purpose |
|---|---|---|
| PyTorch | ≥2.0 | Neural network training and inference |
| NumPy | ≥1.24 | Quantum state linear algebra |
| SciPy | ≥1.10 | Matrix square root (`sqrtm`) for concurrence |
| Flask | ≥3.0 | Web UI and REST API |
| pytest | ≥8.0 | Test runner |

Python ≥3.9 required.

---

## Running the Application

```bash
python3 run.py
```

On first run, the model is automatically trained (~60 seconds on CPU for 8,000 states, 40 epochs). Subsequent runs load the saved checkpoint. The app starts on **http://localhost:5090**.

### Train separately (optional)

```bash
python3 -c "
import warnings; warnings.filterwarnings('ignore')
from training.trainer import train
metrics = train(n_samples=8000, epochs=40)
print('Best val accuracy:', metrics['best_val_accuracy'])
"
```

### Run tests

```bash
python3 -m pytest tests/ -v
```

---

## Usage

### Web UI

1. Open `http://localhost:5090`
2. Select a quantum state from the dropdown (Bell states, Werner, product, random)
3. For Werner states, drag the slider to set parameter p (entangled iff p > 0.33)
4. Click **Predict Entanglement**
5. View:
   - Entanglement probability (colour-coded: purple=entangled, red=separable)
   - Neural net predictions for concurrence, entropy, negativity
   - Bar chart comparing neural net vs. exact analytic values
   - Density matrix real-part heatmap
   - Full exact measures table with interpretations

### REST API

```bash
# Predict entanglement for a Bell state
curl -s -X POST http://localhost:5090/api/predict \
  -H "Content-Type: application/json" \
  -d '{"state_type": "bell_phi_plus"}' | python3 -m json.tool

# Predict for a Werner state with p=0.7
curl -s -X POST http://localhost:5090/api/predict \
  -H "Content-Type: application/json" \
  -d '{"state_type": "werner", "werner_p": 0.7}' | python3 -m json.tool

# List all available state types
curl -s http://localhost:5090/api/states | python3 -m json.tool

# Check model status and training history
curl -s http://localhost:5090/api/model/status | python3 -m json.tool
```

### Python API

```python
import warnings; warnings.filterwarnings("ignore")
import numpy as np
from quantum.states import density_matrix, bell_phi_plus, werner_state, product_state
from quantum.measures import entanglement_measures, concurrence, negativity
from models.entanglement_net import EntanglementNet

# Compute exact measures for any state
rho = density_matrix(bell_phi_plus())
measures = entanglement_measures(rho, n_qubits=2)
print(measures)
# {'concurrence': 1.0, 'negativity': 0.5, 'entanglement_entropy': 1.0, ...}

# Werner state at PPT boundary
rho_w = werner_state(p=0.34)
print("Entangled:", concurrence(rho_w) > 0)   # True

# Neural network prediction
model = EntanglementNet.from_checkpoint("trained_models/entanglement_net.pt")
result = model.predict_density_matrix(rho)
print(result)
# {'entanglement_probability': 0.9987, 'is_entangled': True, 'confidence': 'high', ...}
```

---

## API Reference

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/predict` | Predict entanglement + measures for a named state type |
| `GET` | `/api/states` | List all available quantum state types |
| `GET` | `/api/model/status` | Model info, training history, best validation accuracy |

### POST /api/predict

**Request body:**
```json
{
  "state_type": "bell_phi_plus",
  "werner_p": 0.8
}
```

**Response:**
```json
{
  "state_type": "bell_phi_plus",
  "n_qubits": 2,
  "prediction": {
    "entanglement_probability": 0.9987,
    "is_entangled": true,
    "predicted_concurrence": 0.9812,
    "predicted_entropy": 0.9741,
    "predicted_negativity": 0.4893,
    "confidence": "high"
  },
  "exact_measures": {
    "concurrence": 1.0,
    "negativity": 0.5,
    "entanglement_entropy": 1.0,
    "von_neumann_entropy": 0.0,
    "purity": 1.0,
    "is_ppt": false,
    "is_pure": true,
    "is_entangled_ppt": true
  },
  "density_matrix_real": [[...], ...],
  "density_matrix_imag": [[...], ...]
}
```

---

## Extending the System

### Add a new quantum state

1. Add a generator function to `quantum/states.py`:
```python
def cluster_state(n_qubits: int = 4) -> np.ndarray:
    """Linear cluster state for measurement-based QC."""
    ...
```

2. Add to the `state_map` dict in `app/web.py`
3. Add an `<option>` to the select in `app/templates/index.html`

### Add a new entanglement measure

1. Implement in `quantum/measures.py`:
```python
def entanglement_of_formation(rho: np.ndarray) -> float:
    """E_F = h((1 + sqrt(1 - C²)) / 2) where h is binary entropy."""
    c = concurrence(rho)
    x = (1 + np.sqrt(max(0, 1 - c**2))) / 2
    return float(-x * np.log2(x) - (1-x) * np.log2(1-x)) if 0 < x < 1 else 0.0
```

2. Add to `entanglement_measures()` return dict
3. Add to the `renderExactTable` function in `index.html`

### Re-train with a custom dataset

```python
from training.trainer import train

metrics = train(
    n_samples=20000,   # more data
    epochs=100,        # longer training
    batch_size=256,
)
print(f"Final accuracy: {metrics['best_val_accuracy']:.4f}")
```

---

## Research References

1. Einstein, Podolsky, Rosen — [Can Quantum-Mechanical Description of Physical Reality Be Considered Complete?](https://journals.aps.org/pr/abstract/10.1103/PhysRev.47.777) — *Phys. Rev.* (1935)

2. J.S. Bell — [On the Einstein-Podolsky-Rosen Paradox](https://cds.cern.ch/record/111654/files/vol1p195-200_001.pdf) — *Physics* (1964)

3. Aspect, Grangier, Roger — [Experimental Tests of Bell's Inequalities Using Time-Varying Analyzers](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.49.91) — *PRL* (1982)

4. Peres — [Separability Criterion for Density Matrices](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.77.1413) — *PRL* (1996)

5. Horodecki, Horodecki, Horodecki — [Separability of Mixed States: Necessary and Sufficient Conditions](https://www.sciencedirect.com/science/article/abs/pii/0375960196007062) — *Phys. Lett. A* (1996)

6. Wootters — [Entanglement of Formation of an Arbitrary State of Two Qubits](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.80.2245) — *PRL* (1998)

7. Nielsen & Chuang — [Quantum Computation and Quantum Information](https://www.cambridge.org/highereducation/books/quantum-computation-and-quantum-information/01E10196D0A682A6AEFFEA52D53BE9AE) — Cambridge University Press (2000)

8. Horodecki et al. — [Quantum Entanglement](https://journals.aps.org/rmp/abstract/10.1103/RevModPhys.81.865) — *Rev. Mod. Phys.* (2009) — Comprehensive review

9. Gao, Duan et al. — [Experimental Machine Learning of Quantum States](https://arxiv.org/abs/1712.09912) — *PRL* (2018)

10. Carleo, Cirac, et al. — [Machine Learning and the Physical Sciences](https://arxiv.org/abs/1910.08918) — *Rev. Mod. Phys.* (2019)

---

## Licence

MIT — see [LICENSE](LICENSE). Free for academic and commercial use with attribution.
