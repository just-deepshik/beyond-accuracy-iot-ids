from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import torch
import joblib
import numpy as np


# ============================================================
# FASTAPI APP
# ============================================================

app = FastAPI(
    title="IoT Intrusion Detection System API",
    description="""
## IoT Intrusion Detection System

A cloud-deployed Machine Learning API for detecting network traffic
as **Normal** or **Attack**.

### How to use `/predict`

Send exactly **16 numerical values** in the `features` field.

The values must be provided in this exact order:

1. `src_port`
2. `dst_port`
3. `duration`
4. `src_bytes`
5. `dst_bytes`
6. `missed_bytes`
7. `src_pkts`
8. `src_ip_bytes`
9. `dst_pkts`
10. `dst_ip_bytes`
11. `dns_qclass`
12. `dns_qtype`
13. `dns_rcode`
14. `http_request_body_len`
15. `http_response_body_len`
16. `http_status_code`

### Prediction labels

- `0` → Normal
- `1` → Attack
""",
    version="1.0.0"
)


# ============================================================
# FEATURE INFORMATION
# ============================================================

FEATURE_NAMES = [
    "src_port",
    "dst_port",
    "duration",
    "src_bytes",
    "dst_bytes",
    "missed_bytes",
    "src_pkts",
    "src_ip_bytes",
    "dst_pkts",
    "dst_ip_bytes",
    "dns_qclass",
    "dns_qtype",
    "dns_rcode",
    "http_request_body_len",
    "http_response_body_len",
    "http_status_code"
]


# ============================================================
# REQUEST MODEL
# ============================================================

class PredictionRequest(BaseModel):

    features: list[float] = Field(
        ...,
        min_length=16,
        max_length=16,
        description=(
            "Exactly 16 numerical values in this order: "
            "src_port, dst_port, duration, src_bytes, dst_bytes, "
            "missed_bytes, src_pkts, src_ip_bytes, dst_pkts, "
            "dst_ip_bytes, dns_qclass, dns_qtype, dns_rcode, "
            "http_request_body_len, http_response_body_len, "
            "http_status_code"
        ),
        json_schema_extra={
            "example": [
                12345,
                80,
                1.52,
                450,
                1200,
                0,
                10,
                650,
                8,
                1400,
                1,
                28,
                0,
                0,
                0,
                200
            ]
        }
    )


# ============================================================
# MLP MODEL
# ============================================================

class MLP(torch.nn.Module):

    def __init__(self):
        super().__init__()

        self.net = torch.nn.Sequential(
            torch.nn.Linear(16, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 2)
        )

    def forward(self, x):
        return self.net(x)


# ============================================================
# LOAD MODEL
# ============================================================

model = MLP()

model.load_state_dict(
    torch.load(
        "mlp_ton_iot.pt",
        map_location=torch.device("cpu")
    )
)

model.eval()


# ============================================================
# LOAD SCALER
# ============================================================

scaler = joblib.load("scaler.pkl")


# ============================================================
# ROOT ENDPOINT
# ============================================================

@app.get("/")
def home():

    return {
        "message": "IoT IDS Cloud Running",
        "docs": "/docs",
        "prediction_endpoint": "/predict",
        "required_features": 16,
        "feature_order": FEATURE_NAMES,
        "prediction_labels": {
            "0": "Normal",
            "1": "Attack"
        }
    }


# ============================================================
# PREDICTION ENDPOINT
# ============================================================

@app.post(
    "/predict",
    summary="Predict network traffic",
    description="""
Predict whether a network traffic record is **Normal** or an **Attack**.

### Required input

Exactly **16 numerical values** must be provided in the `features` array.

### Feature order

| Position | Feature |
|---|---|
| 1 | src_port |
| 2 | dst_port |
| 3 | duration |
| 4 | src_bytes |
| 5 | dst_bytes |
| 6 | missed_bytes |
| 7 | src_pkts |
| 8 | src_ip_bytes |
| 9 | dst_pkts |
| 10 | dst_ip_bytes |
| 11 | dns_qclass |
| 12 | dns_qtype |
| 13 | dns_rcode |
| 14 | http_request_body_len |
| 15 | http_response_body_len |
| 16 | http_status_code |

### Output

- `0` = Normal
- `1` = Attack

The API also returns the predicted class probabilities and confidence.
"""
)
def predict(request: PredictionRequest):

    features = request.features

    # Extra safety check
    if len(features) != 16:
        raise HTTPException(
            status_code=400,
            detail="Exactly 16 numerical features are required."
        )

    try:

        # Convert input to NumPy
        input_data = np.array(
            features,
            dtype=np.float32
        ).reshape(1, -1)

        # Scale input using trained scaler
        scaled_data = scaler.transform(input_data)

        # Convert to PyTorch tensor
        x = torch.tensor(
            scaled_data,
            dtype=torch.float32
        )

        # Model prediction
        with torch.no_grad():

            output = model(x)

            probabilities = torch.softmax(
                output,
                dim=1
            )

            prediction = torch.argmax(
                probabilities,
                dim=1
            ).item()

            confidence = probabilities[
                0,
                prediction
            ].item()

        # Extract probabilities
        normal_probability = probabilities[0, 0].item()
        attack_probability = probabilities[0, 1].item()

        # Label
        label = (
            "Normal"
            if prediction == 0
            else "Attack"
        )

        return {
            "prediction": prediction,
            "label": label,
            "confidence": round(confidence, 4),
            "probabilities": {
                "normal": round(normal_probability, 4),
                "attack": round(attack_probability, 4)
            }
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )