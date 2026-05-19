from fastapi import FastAPI
import torch
import joblib
import numpy as np

app = FastAPI()

# -------- LOAD MODEL --------
class MLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(16,256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(256,128),
            torch.nn.ReLU(),
            torch.nn.Linear(128,2)
        )

    def forward(self,x):
        return self.net(x)

model = MLP()
model.load_state_dict(torch.load("mlp_ton_iot.pt"))
model.eval()

# -------- LOAD SCALER --------
scaler = joblib.load("scaler.pkl")

# -------- ROUTES --------

@app.get("/")
def home():
    return {"message":"IoT IDS Cloud Running"}

@app.post("/predict")
def predict(features:list):

    # scale input
    x = scaler.transform([features])

    x = torch.tensor(x).float()

    with torch.no_grad():
        output = model(x)

    pred = torch.argmax(output,dim=1).item()

    return {"prediction":pred}