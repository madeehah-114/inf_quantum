# app_uber.py  — Quantum Stock Prediction using PennyLane + PyTorch

from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import pandas as pd
import numpy as np
import torch
from torch import nn
import pennylane as qml
import joblib

# ---------------------------------------------------
# Initialize FastAPI
# ---------------------------------------------------
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# ---------------------------------------------------
# Load Dataset and Scaler
# ---------------------------------------------------
df = pd.read_csv("UBER.csv", parse_dates=["Date"])
df = df.sort_values("Date").reset_index(drop=True)
scaler = joblib.load("uber_scaler.pkl")

# ---------------------------------------------------
# Rebuild Quantum Model and Load Weights
# ---------------------------------------------------
n_qubits = 4
n_layers = 3
dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev, interface="torch")
def circuit(inputs, weights):
    """Quantum node defining the variational circuit."""
    qml.templates.AngleEmbedding(inputs, wires=range(n_qubits))
    qml.templates.BasicEntanglerLayers(weights, wires=range(n_qubits))
    return qml.expval(qml.PauliZ(0))

# Define model structure (must match training code)
weight_shapes = {"weights": (n_layers, n_qubits)}
qlayer = qml.qnn.TorchLayer(circuit, weight_shapes)
qnn_model = nn.Sequential(qlayer)

# Load trained weights
qnn_model.load_state_dict(torch.load("uber_qnn_model.pt", map_location=torch.device("cpu")))
qnn_model.eval()

# ---------------------------------------------------
# Forecast Function
# ---------------------------------------------------
def forecast_next_days_uber(df, n_days=5, n_steps=10):
    close_prices = df["Close"].values
    forecasted = []
    last_sequence = close_prices[-n_steps:]

    for day in range(1, n_days + 1):
        # Scale and trim/pad to match qubits
        seq_scaled = scaler.transform(last_sequence.reshape(-1, 1)).flatten()
        if len(seq_scaled) < n_qubits:
            seq_scaled = np.pad(seq_scaled, (0, n_qubits - len(seq_scaled)), 'constant')
        elif len(seq_scaled) > n_qubits:
            seq_scaled = seq_scaled[:n_qubits]

        seq_tensor = torch.tensor([seq_scaled], dtype=torch.float32)
        with torch.no_grad():
            pred_scaled = qnn_model(seq_tensor).numpy().flatten()[0]
        pred_close = scaler.inverse_transform([[pred_scaled]])[0, 0]

        forecast_date = df["Date"].max() + pd.Timedelta(days=day)
        forecasted.append({
            "date": forecast_date.strftime("%Y-%m-%d"),
            "pred_close": float(pred_close)
        })

        # Slide the window
        last_sequence = np.append(last_sequence[1:], pred_close)

    return forecasted

# ---------------------------------------------------
# Routes
# ---------------------------------------------------
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(
        "index_uber.html",
        {"request": request, "forecast": None, "chart_type": "line"}
    )

@app.post("/forecast", response_class=HTMLResponse)
def forecast(request: Request, days: int = Form(...), chart_type: str = Form(...)):
    forecasted = forecast_next_days_uber(df, n_days=days)
    forecast_df = pd.DataFrame(forecasted)

    return templates.TemplateResponse(
        "index_uber.html",
        {
            "request": request,
            "forecast": forecasted,
            "chart_type": chart_type,
            "actual_dates": df["Date"].dt.strftime("%Y-%m-%d").tolist()[-30:],
            "actual_values": df["Close"].tolist()[-30:],
            "pred_dates": forecast_df["date"].tolist(),
            "pred_values": forecast_df["pred_close"].tolist(),
        }
    )
