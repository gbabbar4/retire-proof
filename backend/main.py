from fastapi.middleware.cors import CORSMiddleware

import numpy as np
from pathlib import Path

from fastapi import FastAPI, HTTPException
# FastAPI is a class which creates an API based application
# HTTPException allows us to stop request processing and return an HTTP error response with a status code and message.

from pydantic import BaseModel, Field
# Pydantic is a Python library that helps define and validate structured data.
# if the user submits a invalid input , so then Pydantic helps validate incoming data before our prediction function processes it.

# BaseModel - Used to create a data model, meaning a class that describes which fields are expected and what types they should have.

# Fields - Used to set the rules

from neural_network import NeuralNetwork


# Creating the API application
app = FastAPI(title="Retire-Proof API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5501",
        "http://localhost:5501",
        "http://localhost:5173",
        "http://127.0.0.1:5173"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


# Loading the trained model
model_path = Path(__file__).parent / "retire_model_v12.npz"

with np.load(model_path, allow_pickle=False) as saved:

    nn = NeuralNetwork(
        input_features=int(saved["input_features"]),
        hidden_layer=int(saved["hidden_layer"]),
        num_hidden_layers=int(saved["num_hidden_layers"])

    # at this exact moment, the model has newly initialized weights!
    # as Creating the architecture doesn't automatically load the learned weights.
    )

# Here we used `with` because keeping files open unnecessarily can waste resources.


# here we are loading the Trained weights & Biases values
    for i in range(len(nn.network_layers)):

        # Retrieving the Linear layer object from your network's layer structure.
        linear = nn.network_layers[i][0]

        # Replacing the newly initialized weights & biases with the trained weights and biases values
        linear.weights = saved[f"W{i}"].copy()
        linear.bias = saved[f"b{i}"].copy()

# After the loop finishes, our neural network has the parameters learned during 40 epochs of training.

    X_mean = saved["X_mean"].copy()
    X_std = saved["X_std"].copy()
    # @title Correct Feature Names for NN Inference

# Features in the exact order used during NN training
    feature_names = [
    "age",
    "retirement",
    "corpus",
    "monthly_savings",
    "monthly_expense_today",
    "equity_alloc",
    "infl"
    ]

    # Verify feature count matches saved NN architecture
    assert len(feature_names) == int(saved["input_features"])

    print("Correct feature names loaded:", feature_names)

if np.any(X_std == 0):
    raise ValueError("Invalid model scaler: zero standard deviation")
    # Here it prevents the backend from silently proceeding with an invalid scaler.


# Now the main thing is How does the backend know which inputs to accept from the frontend
# How will Python know what inputs are being passed 

# Defining the user input format
class RetirementInput(BaseModel):
    # Here we are using Pydantic as its primary purpose is to describe and validate the structure of incoming data.
    age: float = Field(ge=0, le=120, allow_inf_nan=False)
    retirement: float = Field(ge=0, le=120, allow_inf_nan=False) ####################
    corpus: float = Field(ge=0, allow_inf_nan=False)
    monthly_savings: float = Field(ge=0, allow_inf_nan=False)
    monthly_expense_today: float = Field(ge=0, allow_inf_nan=False)
    equity_alloc: float = Field(ge=0, le=1, allow_inf_nan=False)
    infl: float = Field(ge=0, le=1, allow_inf_nan=False)


# Creating a health-check endpoint
@app.get("/") # decorator
# basically the line means When someone makes a GET request to / (home), use the function immediately below this decorator.
# where GET is an HTTP method typically used to retrieve information.
def home():
    return {"message": "Retire-Proof API is running!"}


# Creating the prediction endpoint
@app.post("/predict")
# POST is an HTTP method commonly used to submit data for processing.
def predict(data: RetirementInput):

    values = data.model_dump()

    # Match the exact training feature order
    x = np.array(
        [[values[name] for name in feature_names]],
        dtype=float
    )

    # Apply saved training normalization
    x_scaled = (x - X_mean) / X_std

    # Run our from-scratch neural network
    prediction = nn.forward(x_scaled)

    result = float(prediction[0, 0])

    if not np.isfinite(result):
        raise HTTPException(
            status_code=500, # NN returns an invalid numerical result
            detail="Model returned an invalid prediction"
        )

    return {
        "prediction": result,
        "model_version": "v1.0"
    }
