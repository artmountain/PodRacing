import json
import math
from copy import deepcopy
from random import randint, random

import matplotlib.pyplot as plt
import numpy as np

LEARNING_RATE = 1
NUMBER_OF_DP_FOR_PICKLE = 6

class Neuron:
    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias
        self.num_inputs = len(weights)
        self.inputs = np.zeros(self.num_inputs)
        self.Z = 0
        self.A = 0

    @staticmethod
    def sigmoid(value):
        # print(x)
        if value > 100:
            return 1
        if value < -100:
            return 0
        return 1 / (1 + math.exp(-value))

    def evaluate(self, inputs):
        self.inputs = inputs
        self.Z = np.dot(self.weights, inputs) + self.bias
        self.A = self.sigmoid(self.Z)
        return self.A

    def tune(self, dA):
        dZ = dA * self.A * (1 - self.A)
        dA_prev = LEARNING_RATE * dZ * self.weights
        for i in range(self.num_inputs):
            self.weights[i] = self.weights[i] + LEARNING_RATE * dZ * self.inputs[i]
        self.bias += LEARNING_RATE * dZ
        if abs(dZ) > 1e6:
            breakpoint()
        return dA_prev


class NeuralNetwork:
    def __init__(self, num_inputs, num_outputs, weights, biases):
        self.num_inputs = num_inputs
        self.num_layers = len(weights)
        self.num_outputs = num_outputs
        self.neurons = []
        self.input_sizes = [num_inputs]
        for layer in range(self.num_layers):
            number_of_neurons = len(weights[layer])
            self.neurons.append([Neuron(weights[layer][i], biases[layer][i]) for i in range(number_of_neurons)])
            self.input_sizes.append(number_of_neurons)

    def evaluate(self, inputs):
        network_data = [deepcopy(inputs)]
        for layer in range(self.num_layers):
            layer_outputs = []
            for neuron_idx in range(len(self.neurons[layer])):
                layer_outputs.append(self.neurons[layer][neuron_idx].evaluate(network_data[layer]))
            network_data.append(layer_outputs)
        return network_data[-1]

    @staticmethod
    def create_from_json(json_str):
        nn_data = json.loads(json_str)
        return NeuralNetwork(6, 2, [np.array(nn_data['weights_0']), np.array(nn_data['weights_1'])],
                             [np.array(nn_data['biases_0']), np.array(nn_data['biases_1'])])

    # Fitting below this line

    def get_total_fit_score(self, inputs, outputs):
        score = 0
        for i in range(len(inputs)):
            net_outputs = self.evaluate(inputs[i])
            # score += 0.5 * (np.dot(np.log(A), Y.T) + np.dot(log(1 - A), 1 - Y.T))
            score += sum([(net_outputs[j] - outputs[i][j]) ** 2 for j in range(len(net_outputs))])
        print("Fitting score : " + str(score))
        return score

    def train(self, inputs, outputs, number_of_fitting_runs):
        training_data_size = len(inputs)
        self.get_total_fit_score(inputs, outputs)
        for fit_idx in range(number_of_fitting_runs):
            fit_data_idx = randint(0, training_data_size - 1)
            current_outputs = self.evaluate(inputs[fit_data_idx])
            # score += 0.5 * (np.dot(np.log(A), Y.T) + np.dot(log(1 - A), 1 - Y.T))

            # Update neuron layers in reverse order
            net_data = [[outputs[fit_data_idx][i] - current_outputs[i] for i in range(self.num_outputs)]]
            for neuron_layer in reversed(self.neurons):
                index = 0
                this_layer_net_data = np.zeros(neuron_layer[0].num_inputs)
                for neuron in neuron_layer:
                    this_layer_net_data += neuron.tune(net_data[-1][index])
                    index += 1
                net_data.append(this_layer_net_data)
            if fit_idx % 100 == 0:
                self.get_total_fit_score(inputs, outputs)
                # Plot out target and NN output
                # plot = True
                plot = False
                if plot:
                    xx = [inputs[i][0] - inputs[i][1] for i in range(len(inputs))]
                    yy = [outputs[i][0] for i in range(len(inputs))]
                    zz = [self.evaluate(inputs[i])[0] for i in range(len(inputs))]
                    plt.scatter(xx, yy)
                    plt.scatter(xx, zz)
                    plt.show(block=False)
                    plt.pause(0.5)
                    plt.close()

    def mutate(self, mutation_rate):
        weights = []
        biases = []
        for layer in range(len(self.neurons)):
            weights.append([])
            biases.append([])
            for neuron in self.neurons[layer]:
                weights[-1].append(neuron.weights)
                for i in range(len(weights[-1][-1])):
                    weights[-1][-1][i] += (random() - 0.5) * mutation_rate
                biases[-1].append(neuron.bias + (random() - 0.5) * mutation_rate)
        return NeuralNetwork(self.num_inputs, self.num_outputs, weights, biases)

    def print_neuron_config(self):
        for layer in range(len(self.neurons)):
            print('\nLayer ' + str(layer) + ' weights:')
            for neuron in self.neurons[layer]:
                print(neuron.weights)
            print('\nLayer ' + str(layer) + ' biases:')
            print([neuron.bias for neuron in self.neurons[layer]])

    def pickle_neuron_config(self, filename):
        nn_config = {}
        for layer in range(len(self.neurons)):
            weights = []
            biases = []
            for neuron in self.neurons[layer]:
                weights.append(np.around(neuron.weights, NUMBER_OF_DP_FOR_PICKLE).tolist())
                biases.append(np.around(neuron.bias, NUMBER_OF_DP_FOR_PICKLE).tolist())
            nn_config['weights_' + str(layer)] = weights
            nn_config['biases_' + str(layer)] = biases
        with open(filename, 'a') as outfile:
            outfile.write(json.dumps(nn_config) + '\n')


# shape is [hidden_layer_1_size, ..., hidden_layer_N_size]
def createNeuralNetwork(inputs, outputs, hidden_node_shape, number_of_fitting_runs):
    num_inputs = len(inputs[0])
    num_outputs = len(outputs[0])
    shape = [num_inputs] + hidden_node_shape + [num_outputs]
    weights = []
    biases = []
    for i in range(1, len(shape)):
        weights.append(np.random.random((shape[i], shape[i - 1])) / shape[i - 1])
        biases.append(np.zeros(shape[i]))
    nn = NeuralNetwork(num_inputs, num_outputs, weights, biases)
    nn.train(inputs, outputs, number_of_fitting_runs)
    return nn
