import warnings
warnings.filterwarnings("ignore")

import os


def main():
    model_path = os.path.join("trained_models", "entanglement_net.pt")
    if not os.path.exists(model_path):
        print("Training EntanglementNet on synthetic quantum states...")
        from training.trainer import train
        metrics = train(n_samples=8000, epochs=40)
        print(f"Training complete. Val accuracy: {metrics['best_val_accuracy']:.4f}")

    from app.web import create_app
    app = create_app()
    print("Starting Quantum Entanglement Predictor on http://localhost:5090")
    app.run(debug=False, port=5090)


if __name__ == "__main__":
    main()
