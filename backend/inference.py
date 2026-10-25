"""Prediction adapter for the user's saved v12 from-scratch NumPy model.

Original backend/main.py (FastAPI) is left unchanged and is not imported.
Each prediction uses a fresh NeuralNetwork instance, avoiding cross-user
mutability in Streamlit sessions.
"""
from pathlib import Path
import numpy as np

from backend.neural_network import NeuralNetwork

MODEL_PATH = Path(__file__).resolve().parent / "retire_model_v12.npz"
FEATURE_NAMES = (
    "age", "retirement", "corpus", "monthly_savings",
    "monthly_expense_today", "equity_alloc", "infl",
)


def load_checkpoint(path=MODEL_PATH):
    """Load the exact weights, architecture and feature scaling from training."""
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing model file: {path.name}. Keep retire_model_v12.npz "
            "in your existing backend folder."
        )

    with np.load(path, allow_pickle=False) as saved:
        input_features = int(saved["input_features"])
        hidden_layer = int(saved["hidden_layer"])
        hidden_layers = int(saved["num_hidden_layers"])
        mean = saved["X_mean"].astype(float).reshape(-1).copy()
        std = saved["X_std"].astype(float).reshape(-1).copy()
        if input_features != len(FEATURE_NAMES):
            raise ValueError("Saved model input count does not match the seven form features")
        if hidden_layer < 1 or hidden_layers < 1:
            raise ValueError("Saved model architecture is invalid")
        if mean.shape != (input_features,) or std.shape != (input_features,):
            raise ValueError("Saved scaler does not contain seven mean/std values")
        if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(std)) or np.any(std == 0):
            raise ValueError("Saved feature scaler is invalid")
        weights = []
        biases = []
        layer_dims = [input_features] + [hidden_layer] * hidden_layers + [1]
        for i, (n_in, n_out) in enumerate(zip(layer_dims[:-1], layer_dims[1:])):
            w = saved[f"W{i}"].astype(float).copy()
            b = saved[f"b{i}"].astype(float).copy()
            if w.shape != (n_in, n_out) or b.shape != (n_out,):
                raise ValueError(f"Layer {i} shape in saved model is inconsistent")
            if not np.all(np.isfinite(w)) or not np.all(np.isfinite(b)):
                raise ValueError(f"Layer {i} has invalid weights/biases")
            weights.append(w)
            biases.append(b)

    return (input_features, hidden_layer, hidden_layers, mean, std, weights, biases)


def predict_retirement(payload):
    """Return a value between 0 and 1, in the same scale as FastAPI /predict."""
    if not isinstance(payload, dict):
        raise ValueError("Invalid form input")
    try:
        values = {key: float(payload[key]) for key in FEATURE_NAMES}
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Please enter valid values for all seven fields") from exc

    if not all(np.isfinite(v) for v in values.values()):
        raise ValueError("All inputs must be finite numbers")
    if not 0 <= values["age"] <= 120 or not 0 <= values["retirement"] <= 120:
        raise ValueError("Age and retirement age must be between 0 and 120")
    if values["retirement"] <= values["age"]:
        raise ValueError("Retirement age must be greater than current age")
    for k in ("corpus", "monthly_savings", "monthly_expense_today"):
        if values[k] < 0:
            raise ValueError("Savings and spending inputs cannot be negative")
    for k in ("equity_alloc", "infl"):
        if not 0 <= values[k] <= 1:
            raise ValueError("Equity allocation and inflation must be between 0% and 100%")

    n_input, width, depth, mean, std, weights, biases = load_checkpoint()
    nn = NeuralNetwork(n_input, width, depth)
    for i in range(len(nn.network_layers)):
        nn.network_layers[i][0].weights = weights[i]
        nn.network_layers[i][0].bias = biases[i]

    x = np.array([[values[key] for key in FEATURE_NAMES]], dtype=float)
    x_scaled = (x - mean) / std
    result = float(nn.forward(x_scaled)[0, 0])
    if not np.isfinite(result) or not 0 <= result <= 1:
        raise ValueError("Model returned an invalid probability")
    return result
