import json
import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
from flask import Flask, jsonify, render_template, request

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "trained_models", "entanglement_net.pt")
HISTORY_PATH = os.path.join(os.path.dirname(__file__), "..", "trained_models", "training_history.json")


def _load_or_train():
    from models.entanglement_net import EntanglementNet

    if not os.path.exists(MODEL_PATH):
        print("No trained model found — training now (takes ~60 seconds)...")
        from training.trainer import train
        train(n_samples=6000, epochs=30)
    return EntanglementNet.from_checkpoint(MODEL_PATH, n_qubits=2)


def create_app() -> Flask:
    app = Flask(__name__)
    model = _load_or_train()

    def _make_prediction(state_type: str, werner_p: float = 0.8) -> dict:
        from quantum.measures import entanglement_measures
        from quantum.states import (
            bell_phi_minus,
            bell_phi_plus,
            bell_psi_minus,
            bell_psi_plus,
            density_matrix,
            ghz_state,
            product_state,
            random_pure_state,
            w_state,
            werner_state,
        )

        rng = np.random.default_rng()
        state_map = {
            "bell_phi_plus": lambda: density_matrix(bell_phi_plus()),
            "bell_phi_minus": lambda: density_matrix(bell_phi_minus()),
            "bell_psi_plus": lambda: density_matrix(bell_psi_plus()),
            "bell_psi_minus": lambda: density_matrix(bell_psi_minus()),
            "ghz": lambda: density_matrix(ghz_state(3)),
            "w": lambda: density_matrix(w_state(3)),
            "werner": lambda: werner_state(max(0.0, min(1.0, werner_p))),
            "product": lambda: density_matrix(product_state(2, rng)),
            "random_pure": lambda: density_matrix(random_pure_state(2, rng)),
        }

        rho = state_map.get(state_type, state_map["random_pure"])()
        nq = int(np.log2(rho.shape[0]))
        prediction = model.predict_density_matrix(rho)
        exact = entanglement_measures(rho, n_qubits=nq)

        return {
            "state_type": state_type,
            "n_qubits": nq,
            "prediction": prediction,
            "exact_measures": exact,
            "density_matrix_real": rho.real.tolist(),
            "density_matrix_imag": rho.imag.tolist(),
        }

    @app.route("/healthz")
    def healthz():
        return "ok", 200

    @app.route("/")
    def index():
        return render_template("index.html")

    @app.route("/about")
    def about():
        return render_template("about.html")

    @app.route("/api/predict", methods=["POST"])
    def api_predict():
        data = request.get_json(force=True) or {}
        state_type = data.get("state_type", "random_pure")
        werner_p = float(data.get("werner_p", 0.8))
        result = _make_prediction(state_type, werner_p)
        return jsonify(result)

    @app.route("/api/states", methods=["GET"])
    def api_states():
        return jsonify(
            [
                {"key": "bell_phi_plus", "name": "Bell |Φ+⟩", "description": "Maximally entangled Bell state (|00⟩+|11⟩)/√2", "expected": "entangled"},
                {"key": "bell_phi_minus", "name": "Bell |Φ-⟩", "description": "(|00⟩-|11⟩)/√2", "expected": "entangled"},
                {"key": "bell_psi_plus", "name": "Bell |Ψ+⟩", "description": "(|01⟩+|10⟩)/√2", "expected": "entangled"},
                {"key": "bell_psi_minus", "name": "Bell |Ψ-⟩", "description": "(|01⟩-|10⟩)/√2 — singlet state", "expected": "entangled"},
                {"key": "werner", "name": "Werner State", "description": "p|Ψ-⟩⟨Ψ-| + (1-p)I/4 — entangled iff p > 1/3", "expected": "depends on p"},
                {"key": "product", "name": "Product State", "description": "Separable tensor product state", "expected": "separable"},
                {"key": "random_pure", "name": "Random Pure State", "description": "Haar-random pure state", "expected": "usually entangled"},
            ]
        )

    @app.route("/api/model/status", methods=["GET"])
    def api_model_status():
        history: dict = {}
        if os.path.exists(HISTORY_PATH):
            with open(HISTORY_PATH) as f:
                history = json.load(f)
        best_acc = max(history.get("val_accuracy", [0.0]))
        return jsonify(
            {
                "model_loaded": True,
                "model_path": MODEL_PATH,
                "best_val_accuracy": round(best_acc, 4),
                "training_history": history,
            }
        )

    return app
