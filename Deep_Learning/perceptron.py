"""
Single Neuron with Backpropagation
Write a Python function that simulates a single neuron with sigmoid activation,
    and implements backpropagation to update the neuron's weights and bias.
The function should take a list of feature vectors, associated true binary labels,
 initial weights, initial bias, a learning rate, and the number of epochs.
The function should update the weights and bias using gradient descent
    based on the MSE loss, and return the updated weights, bias,
    and a list of MSE values for each epoch, each rounded to four decimal places.
"""

import numpy as np

def sigmoid(x:float)->float:
	return 1.0/(1.0+np.exp(-x))
def train_neuron(
	features: np.ndarray,
	labels: np.ndarray,
	initial_weights: np.ndarray,
	initial_bias: float,
	learning_rate: float,
	epochs: int
	) -> (np.ndarray, float, list[float]):
	weights = initial_weights
	bias = initial_bias
	mse_values = []
	N = features.shape[0]
	for e in range(epochs):
		# forward pass
		z = bias + np.dot(features, weights)
		pred = sigmoid(z)
		#backward pass
		error = pred - labels
		mix = pred*(1.0-pred)
		mse_values.append(round(np.dot(error, error)/N, 4))

		grad_w = 2 / N* np.dot(features.T, error * mix )
		grad_b = 2/N * np.sum(error * mix)

		#optimize
		weights -= learning_rate * grad_w
		bias -= learning_rate * grad_b

	return np.round(weights,4).tolist(), round(bias,4), mse_values
