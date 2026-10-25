import numpy as np
from pathlib import Path # this will handle the path of the files and folders


from neural_network import NeuralNetwork 
# Fetching the NN Engine from the neural_network.py file
# Code reusability

# Locating the saved model (weights)
model_path = Path(__file__).parent / "retire_model_v12.npz"

'''
__file__ is a special variable. It contains the path of the currently executing module's file.
like for this file __file__ will store - "/Users/tanishq/Downloads/retirement_sim/backend/test_model.py"

.parent - represents the parent file of the current file we are working on. (in our case it will be "backend/")

Path(__file__).parent - this gives the output:
/Users/yourname/Downloads/retirement_sim/backend/

by adding the - / "retire_model_v1.npz" , it becomes:
/Users/yourname/Downloads/retirement_sim/backend/retire_model_v12.npz

'''

# Loading the saved parameters
saved = np.load(model_path, allow_pickle=False)
'''
.npz is like a suitcase where we have packed the things (archive) and can be accessed later on

p.load() opens that archive.

And saved becomes an object through which we can access the stored arrays by name.

allow_pickle=False --> It prevents from untrusted pickle data which when can execute malicious code.
'''

# Rebuilding the original NN architecture
nn = NeuralNetwork(
    input_features=int(saved["input_features"]),
    hidden_layer=int(saved["hidden_layer"]),
    num_hidden_layers=int(saved["num_hidden_layers"])
)

# Here we are importing the saved values from the .npz file and converting them to integer using int()
# Because values loaded from NumPy can be zero-dimensional NumPy arrays. int() converts their scalar values into normal Python integers.


# Restoring the trained weights and biases
# self.network_layers is a dict which stored the layers information which we used in the forward pass
# like in this format - {0 : [linear_object_0, relu_object_0], 1 : [linear_object_1, relu_object_1], ... , N : [linear_object_n, relu_object_n]}
# So we will run a loop to get those layers for our prediction
for i in range(len(nn.network_layers)):

    linear = nn.network_layers[i][0]

    linear.weights = saved[f"W{i}"].copy()
    linear.bias = saved[f"b{i}"].copy()

print("Model loaded successfully!")
# After the loop: correct NN architecture and the learned weights and biases from Colab.


# Loading the preprocessing information
# Because The network should receive inputs transformed using the same preprocessing rules used during training.
X_mean = saved["X_mean"]
X_std = saved["X_std"]
feature_names = saved["feature_names"]

print("Input features:", feature_names)


# Using a sample input

# For this initial smoke test, use the saved
# training feature means as the input.
sample_input = X_mean.reshape(1, -1)
# -1 = NumPy automatically figures out the remaining dimension

# Applying original normalization (Standardisation)
sample_scaled = (sample_input - X_mean) / X_std

# Generating prediction
prediction = nn.forward(sample_scaled)

print("\nPrediction:", prediction.flatten())

# Performing the Basic checks
assert prediction.shape == (1, 1)
assert np.isfinite(prediction).all()

print("SUCCESS: VS Code inference works!")

# @title Debug VS Code Prediction

import inspect

print("\n========== DEBUGGING ==========")

# Which Python file contains the NN class?
print("NN source file:", inspect.getfile(NeuralNetwork))

# Inspect output activation
last_layer = nn.network_layers[len(nn.network_layers) - 1]

print("Final activation:", type(last_layer[1]).__name__)

# Raw output BEFORE sigmoid
print("Raw final layer output:", nn.final_layer)

# Output calculated directly using sigmoid formula
expected_sigmoid = 1 / (1 + np.exp(-nn.final_layer))

print("Expected sigmoid output:", expected_sigmoid)

# Output from YOUR sigmoid class
actual_sigmoid = last_layer[1].calc(nn.final_layer)

print("Your sigmoid class output:", actual_sigmoid)

# Output returned from forward()
print("NN.forward() output:", prediction)

print("\n========== CHECK ==========")

assert np.all((prediction >= 0) & (prediction <= 1)), \
    "ERROR: NN prediction is outside [0, 1]!"

print("SUCCESS: Prediction is a valid probability!")