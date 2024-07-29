import unittest
from math import sin, cos

import matplotlib.pyplot as plt
import numpy as np
from NeuralNet import createNeuralNetwork

class TestNeuralNetTrainToSinCos(unittest.TestCase):
    def train_neural_network_with_trig_functions(self):
        #plot = False
        plot = True

        inputs = []
        outputs = []
        for i in range(500):
            x, y = np.random.random(2)
            inputs.append([x, y])
            outputs.append([sin(x + y), cos(x - y)])

        # Build and train network
        neural_network = createNeuralNetwork(inputs, outputs, [4])

        for i in range(len(outputs)):
            net_outputs = neural_network.evaluate((inputs[i]))
            #print(net_outputs, outputs[i])

        # Output node data
        neural_network.print_neuron_config()

        # Plot out training data
        if plot:
            xx = [inputs[i][0] + inputs[i][1] for i in range(len(inputs))]
            yy = [outputs[i][0] for i in range(len(inputs))]
            zz = [neural_network.evaluate(inputs[i])[0] for i in range(len(inputs))]
            xx2 = [inputs[i][0] - inputs[i][1] for i in range(len(inputs))]
            yy2 = [outputs[i][1] for i in range(len(inputs))]
            zz2 = [neural_network.evaluate(inputs[i])[1] for i in range(len(inputs))]
            plt.scatter(xx, yy)
            plt.scatter(xx, zz)
            plt.scatter(xx2, yy2)
            plt.scatter(xx2, zz2)
            plt.show(block=True)
            plt.pause(10)
            plt.close()

    def train_neural_network_with_two_input_trig_functions(self):
        # plot = False
        plot = True

        inputs = []
        outputs = []
        for i in range(500):
            x, y = np.random.random(2)
            inputs.append([x, y])
            outputs.append([cos(x - y) * 2 - 1])

        print(min([inputs[n][0] for n in range(500)]), max([inputs[n][0] for n in range(500)]))
        print(min([outputs[n][0] for n in range(500)]), max([outputs[n][0] for n in range(500)]))

        # Build and train network
        neural_network = createNeuralNetwork(inputs, outputs, [2])

        for i in range(len(outputs)):
            net_outputs = neural_network.evaluate((inputs[i]))
            # print(net_outputs, outputs[i])

        # Output node data
        neural_network.print_neuron_config()

        # Plot out training data
        if plot:
            xx = [inputs[i][0] - inputs[i][1] for i in range(len(inputs))]
            yy = [outputs[i][0] for i in range(len(inputs))]
            zz = [neural_network.evaluate(inputs[i])[0] for i in range(len(inputs))]
            plt.scatter(xx, yy)
            plt.scatter(xx, zz)
            plt.show(block=True)
            plt.pause(3)
            plt.close()

    def train_neural_network_with_linear_function(self):
        # plot = False
        plot = True

        inputs = []
        outputs = []
        for i in range(500):
            x = np.random.random(1)
            inputs.append([x])
            outputs.append([1 - x])

        # Build and train network
        neural_network = createNeuralNetwork(inputs, outputs, [])

        # Output node data
        neural_network.print_neuron_config()

        # Plot out training data
        if plot:
            xx = [inputs[i][0] for i in range(len(inputs))]
            yy = [outputs[i][0] for i in range(len(inputs))]
            zz = [neural_network.evaluate(inputs[i])[0] for i in range(len(inputs))]
            plt.scatter(xx, yy)
            plt.scatter(xx, zz)
            plt.show(block=True)
            plt.pause(2)
            plt.close()
        self.assertTrue(neural_network.get_total_fit_score(inputs, outputs) < 0.4)

    def train_neural_network_with_cos_function(self):
        # plot = False
        plot = True

        inputs = []
        outputs = []
        for i in range(500):
            x = np.random.random(1)
            inputs.append([x])
            outputs.append([cos(x)])

        # Build and train network
        neural_network = createNeuralNetwork(inputs, outputs, [])

        for i in range(len(outputs)):
            net_outputs = neural_network.evaluate((inputs[i]))
            # print(net_outputs, outputs[i])

        # Output node data
        neural_network.print_neuron_config()

        # Plot out training data
        if plot:
            xx = [inputs[i][0] for i in range(len(inputs))]
            yy = [outputs[i][0] for i in range(len(inputs))]
            zz = [neural_network.evaluate(inputs[i])[0] for i in range(len(inputs))]
            plt.scatter(xx, yy)
            plt.scatter(xx, zz)
            plt.show(block=True)
            plt.pause(3)
            plt.close()
        self.assertTrue(neural_network.get_total_fit_score(inputs, outputs) < 0.11)


if __name__ == '__main__':
    unittest.main()
