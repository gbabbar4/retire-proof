# Retire-Proof 🌅

### A Neural Network–Powered Retirement Simulator

Retire-Proof is an experimental retirement planning application powered by a fully connected neural network built from scratch using Python and NumPy.

The project demonstrates how neural networks can learn patterns from simulated financial scenarios and estimate retirement survival probability through an interactive web interface.

## Features

- Interactive retirement planning simulator
- Seven financial inputs covering savings, spending, retirement timeline and market assumptions
- Retirement survival probability predictions
- Neural network implemented from scratch with NumPy
- Dedicated "Inside the Model" page explaining architecture, training and evaluation
- Responsive interface built with HTML, CSS and JavaScript
- Deployment-ready Streamlit application

## Neural Network Architecture

| Component | Details |
|---|---|
| Input features | 7 |
| Hidden layers | 2 |
| Neurons per hidden layer | 16 |
| Hidden activation | ReLU |
| Output activation | Sigmoid |
| Loss function | Mean Squared Error |
| Optimization | Gradient Descent |

**Architecture:** `7 → 16 → 16 → 1`

The model uses saved weights from the v12 checkpoint for inference without retraining on each prediction.

## Tech Stack

**Machine Learning:** Python, NumPy

**Frontend:** HTML, CSS, JavaScript

**Application framework:** Streamlit

**Original API implementation:** FastAPI

**Version Control:** Git and GitHub

## Running Locally

Clone the repository:

```bash
git clone https://github.com/gbabbar4/retire-proof.git
cd retire-proof
```

Create a virtual environment:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Launch the application:

```bash
python -m streamlit run app.py
```

## Model Limitations

The model is trained on simulated retirement scenarios and is intended for educational and experimental use.

Predictions are estimates, not guarantees. The model should not be used as the sole basis for financial decisions.

## Project Status

Neural network training, application integration and initial local inference checks have been completed. Cloud deployment and additional validation are in progress.

---

Built from scratch to explore neural networks, financial simulation and practical machine learning deployment.