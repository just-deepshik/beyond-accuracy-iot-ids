# Beyond Accuracy: Robust and Explainable IoT IDS
## 🚀 Live Demo

[**Open IoT IDS API — Interactive Swagger Docs**](https://iot-ids-deploy.onrender.com/docs)

> **Deployment Note:** This API is hosted on Render's free tier. After periods of inactivity, the service may enter a sleep state. The first request after inactivity may take some time to respond while the service starts up. Once active, the API can be used normally.

## Overview

This project proposes a robust and explainable intrusion detection system (IDS) for IoT environments.

The system evaluates:
- Concept Drift
- Adversarial Robustness (FGSM)
- Cross-Dataset Generalization
- Explainability using SHAP

## Technologies Used

- Python
- PyTorch
- FastAPI
- Gradio
- SHAP
- Scikit-learn

## Features

- MLP-based intrusion detection
- Adversarial evaluation
- Drift-aware testing
- Explainable AI
- Cloud deployment

## Results

- Baseline Accuracy: 97.47%
- Adversarial Accuracy Drop: 38.8%

## Deployment

Deployed using Render + FastAPI
