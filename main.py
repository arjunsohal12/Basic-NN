import numpy as np

from random import seed
from random import random


def initialize_network(n_inputs, n_hidden, n_outputs):
    network = list()
    if n_hidden != 0:
        hidden_layer = [{'weights': [random() for i in range(n_inputs + 1)]} for i in range(n_hidden)]
        network.append(hidden_layer)
    output_layer = [{'weights': [random() for i in range(n_hidden + 1)]} for i in range(n_outputs)]
    network.append(output_layer)
    # print(hidden_layer)
    return network


def activate(weights, inputs):
    activation = weights[-1]  # Bias
    for i in range(len(weights) - 1):  # -1 Since operation is not applied to Bias
        activation += weights[i] * inputs[i]
    return activation


def transfer(activation):
    return 1.0 / (1.0 + np.exp(-activation))  # Sigmoid transfer function


def forward_propagate(network, inputs1):
    inputs = inputs1
    for layer in network:
        new_inputs = []
        for neuron in layer:
            activation = activate(neuron['weights'], inputs)
            neuron['output'] = transfer(activation)
            new_inputs.append(neuron['output'])
        inputs = new_inputs
    return inputs


# Back Propagation

def transfer_derivative(output):
    return output * (1.0 - output)


def backward_propagate_error(network, expected):
    for i in reversed(range(len(network))):
        layer = network[i]
        # print(layer)
        errors = list()
        if i != len(network) - 1:
            for j in range(len(layer)):
                error = 0.0
                for neuron in network[i + 1]:
                    error += (neuron['weights'][j] * neuron['delta'])
                errors.append(error)
        else:
            for j in range(len(layer)):
                neuron = layer[j]
                errors.append(expected[j] - neuron['output'])
        for j in range(len(layer)):  # Loops through backwards for error and then forwards for the delta??
            neuron = layer[j]
            neuron['delta'] = errors[j] * transfer_derivative(neuron['output'])


# Updating weights and biases

# Updating on multilayered networks
def update_weights_ml(network, row, l_rate):
    for i in range(len(network)):
        # print(network)
        # print(row)
        inputs = row[:-1]
        # print(inputs)
        if i != 0:
            inputs = [neuron['output'] for neuron in network[i - 1]]
        for neuron in network[i]:
            for j in range(len(inputs)):
                # Updating weights depending on adjacent layers
                neuron['weights'][j] += l_rate * neuron['delta'] * inputs[j]
            # Updating Biases
            neuron['weights'][-1] += l_rate * neuron['delta']


# Updating on single layered networks
def update_weights_sl(network, l_rate):
    for neuron in network[-1]:
        for i in range(len(neuron['weights'])):
            neuron['weights'][i] += neuron['delta'] * l_rate


def train_network(network, train, l_rate, n_epoch, n_outputs):
    for epoch in range(n_epoch):
        sum_error = 0
        for row in train:
            outputs = forward_propagate(network, row)
            expected = [0 for i in range(n_outputs)]
            expected[row[-1]] = 1
            sum_error += sum([(expected[i] - outputs[i]) ** 2 for i in range(len(expected))])
            backward_propagate_error(network, expected)
            update_weights_ml(network, row, l_rate)
        print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, l_rate, sum_error))


def predict(network, row):
    outputs = forward_propagate(network, row)
    # return outputs
    return outputs.index(max(outputs))

'''
# Test making predictions with the network
dataset = [[0, 1, 0, 0],
           [1, 1, 1, 1],
           [0, 0, 1, 0],
           [1, 0, 1, 1]]

n_inputs = len(dataset[0]) - 1
n_outputs = len(set([row[-1] for row in dataset]))
network = initialize_network(n_inputs, 2, n_outputs)
train_network(network, dataset, 0.5, 10000, n_outputs)

test = [[0, 1, 0], [1, 1, 1], [1, 0, 1], [0, 0, 1]]

for row in test:
    prediction = predict(network, row)
    print(prediction)
    # print('Expected=%d, Got=%d' % (row[-1], prediction))
'''