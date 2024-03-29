"""
A Python script to implement a simple neural network using NumPy.

author: Fabrizio Musacchio
date: Feb 15, 2024

For reproducibility:

conda create -n numpy_ann python=3.11
conda activate numpy_ann
conda install -y mamba
mamba install -y numpy matplotlib keras ipykernel

If you are using an Apple Silicon chip, follow these instructions to install TensorFlow:
https://www.fabriziomusacchio.com/blog/2022-11-10-apple_silicon_and_tensorflow/
"""
# %% IMPORTS
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
# %% INITIALIZE NETWORK PARAMETERS
def initialize_network(input_size, hidden_size, output_size):
    """
    This function initializes the parameters of the neural network.
    W1 and W2 are weight matrices for the input layer and the hidden layer, respectively.
    b1 and b2 are bias vectors for the hidden layer and the output layer, respectively.
    """
    np.random.seed(42)  # for reproducibility
    network = {
        'W1': np.random.randn(input_size, hidden_size) * 0.01,
        'b1': np.zeros((1, hidden_size)),
        'W2': np.random.randn(hidden_size, output_size) * 0.01,
        'b2': np.zeros((1, output_size))
    }
    return network

# initialize the network:
input_size = 28*28  # MNIST images have 28x28 pixels 
hidden_size = 5  # number of neurons in the hidden layer
output_size = 10  # Number of classes (the digits 0-9)
network = initialize_network(input_size, hidden_size, output_size)
# %% DEFINE FEEDFORWARD FUNCTION
def sigmoid(z):
    """
    This function serves as the activation function for the hidden layer.
    """
    return 1 / (1 + np.exp(-z))

# plot the sigmoid function:
fig = plt.figure(figsize=(4, 3))
z = np.linspace(-10, 10, 100)
plt.plot(z, sigmoid(z))
plt.xlabel('z')
plt.ylabel('sigmoid(z)')
plt.title('Sigmoid Function')
# remove top and right spines
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig('sigmoid_function.png', dpi=300)
plt.show()

def softmax(z):
    """
    This function serves as the activation function for the output layer.
    """
    exp_scores = np.exp(z)
    return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

def feedforward(network, X):
    """
    This function performs the feedforward step of the neural network. It works as follows:
    - It takes the input X and propagates it through the network to compute the output.
    - It returns the activations of the hidden layer (a1) and the output layer (a2).    
    """
    z1 = X.dot(network['W1']) + network['b1']
    a1 = sigmoid(z1)
    z2 = a1.dot(network['W2']) + network['b2']
    a2 = softmax(z2)
    activations = {
        'a1': a1,
        'a2': a2
    }
    return activations
# %% DEFINE LOSS FUNCTION
def compute_cost(a2, Y):
    """
    We use the cross-entropy loss function to compute the loss.
    """
    m = Y.shape[0]  # number of examples
    log_probs = np.multiply(np.log(a2), Y) + np.multiply((1 - Y), np.log(1 - a2))
    cost = -np.sum(log_probs) / m
    return cost
# %% DEFINE BACKPROPAGATION FUNCTION
def backpropagate(network, activations, X, Y):
    """
    This function performs the backpropagation step of the neural network. It works as follows:
    - It takes the input X, the true labels Y, and the activations from the feedforward step.
    - It computes the gradients of the loss with respect to the parameters of the network.
    - It returns the gradients.
    """
    m = X.shape[0]
    
    # output from the feedforward:
    a1, a2 = activations['a1'], activations['a2']
    
    # error during output:
    dz2 = a2 - Y
    dW2 = (1 / m) * np.dot(a1.T, dz2)
    db2 = (1 / m) * np.sum(dz2, axis=0, keepdims=True)
    
    # error in the hidden layer:
    dz1 = np.dot(dz2, network['W2'].T) * a1 * (1 - a1)
    dW1 = (1 / m) * np.dot(X.T, dz1)
    db1 = (1 / m) * np.sum(dz1, axis=0, keepdims=True)
    
    # gradients:
    gradients = {
        'dW1': dW1,
        'db1': db1,
        'dW2': dW2,
        'db2': db2
    }
    
    return gradients
# %% DEFINE TRAINING FUNCTION
def update_parameters(network, gradients, learning_rate):
    """
    This function updates the parameters of the network using the gradients and the learning rate.
    """
    network['W1'] -= learning_rate * gradients['dW1']
    network['b1'] -= learning_rate * gradients['db1']
    network['W2'] -= learning_rate * gradients['dW2']
    network['b2'] -= learning_rate * gradients['db2']
    
def train_network(network, X_train, Y_train, X_val, Y_val, num_iterations=1000, learning_rate=0.1):
    """
    We predict the labels for the input data X using the trained network:
    - We perform the feedforward step to compute the output.
    - We take the class (digits 0-9) with the highest probability as the predicted label.
    """
    train_costs = []
    val_costs = []
    for i in range(num_iterations):
        # Training
        train_activations = feedforward(network, X_train)
        train_cost = compute_cost(train_activations['a2'], Y_train)
        train_costs.append(train_cost)
        gradients = backpropagate(network, train_activations, X_train, Y_train)
        update_parameters(network, gradients, learning_rate)
        
        # Validation
        val_activations = feedforward(network, X_val)
        val_cost = compute_cost(val_activations['a2'], Y_val)
        val_costs.append(val_cost)

        if i % 100 == 0:
            print(f"Costs after iteration {i}: Training {train_cost}, Validation {val_cost}")
    return train_costs, val_costs

# define the predict function:
def predict(network, X):
    """
    We predict the labels for the input data X using the trained network:
    - We perform the feedforward step to compute the output.
    - We take the class (digits 0-9) with the highest probability as the predicted label    
    """
    activations = feedforward(network, X)
    predictions = np.argmax(activations['a2'], axis=1)
    return predictions
# %% LOAD AND PREPROCESSING THE DATA

# load MNIST handwritten digit data:
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# display some images:
fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(10, 10))
for i, ax in enumerate(axes.flat):
	ax.imshow(X_train[i], cmap='gray')
	ax.set_title(f"true label: {y_train[i]}")
	ax.axis('off')
plt.tight_layout()
plt.savefig('mnist_images.png', dpi=300)
plt.show()

# flatten the input data and normalize:
X_train_flat = X_train.reshape(X_train.shape[0], -1) / 255.
X_test_flat  = X_test.reshape(X_test.shape[0], -1) / 255.

# convert the labels to "one-hot" format
def convert_to_one_hot(Y, C):
    """
    "one-hote" encoding of the labels, i.e., the labels are represented as binary vectors.
    E.g., the label 3 is represented as [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]. This is required 
    for the output layer of the network, as it requires a binary vector for each label.
    """
    Y = np.eye(C)[Y.reshape(-1)]
    return Y

num_classes = 10  #  number of classes (digits 0-9) in the MNIST dataset
y_train_one_hot = convert_to_one_hot(y_train, num_classes)
y_test_one_hot  = convert_to_one_hot(y_test, num_classes)
# %% TRAINING AND VALIDATION THE NETWORK
""" # reset the network:
network = initialize_network(input_size, hidden_size, output_size) """

# set the hyperparameters:
epochs = 1000 # here, epochs equals the number of iterations since we use the entire dataset for each iteration (no mini-batch)
learning_rate = 0.5

# train the network:
train_costs, val_costs = train_network(network, X_train_flat, y_train_one_hot, X_test_flat, y_test_one_hot, 
                                       num_iterations=epochs, learning_rate=learning_rate)

# summarize the network parameters, i.e., number of parameters and the architecture:
print(f"Network summary: ")
print(f"  # of input-layers: {input_size}")
print(f"  # of hidden-layers: {hidden_size}")
print(f"  # of output-layers: {output_size}")
print(f"  # of parameters: {input_size * hidden_size + hidden_size * output_size + hidden_size + output_size}")
print(f"  # of training samples: {len(X_train_flat)}")
print(f"  # of test samples: {len(X_test_flat)}")
print(f"  # of epochs: {epochs}")
print(f"  learning rate: {learning_rate}")

fig = plt.figure(figsize=(6, 4))
plt.plot(train_costs, label='Training Loss')
plt.plot(val_costs, label='Validation Loss')
plt.xlabel('iteration')
plt.ylabel('loss')
plt.title('Training and validation loss curves')
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.legend()
plt.tight_layout()
plt.savefig('training_validation_loss.png', dpi=300)
plt.show()

# predict the test data:
predictions = predict(network, X_test_flat)

# calculate the accuracy:
actual_labels = y_test
accuracy = np.mean(predictions == actual_labels)
print(f'Accuracy of the network on the test data: {accuracy * 100:.2f}%')

# plot some test images with the predicted labels:
fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(10, 10))
for i, ax in enumerate(axes.flat):
	ax.imshow(X_test[i], cmap='gray')
	ax.set_title(f'predicted label: {predictions[i]}')
	ax.axis('off')
plt.tight_layout()
plt.savefig('predicted_labels.png', dpi=300)
plt.show()
# %% END
