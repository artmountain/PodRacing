import unittest
from math import sin, cos

import matplotlib.pyplot as plt
import numpy as np
from NeuralNet import createNeuralNetwork


class TestNeuralNetTrainToPrimitiveFunctions(unittest.TestCase):
    def test_train_neural_network_with_trig_functions(self):
        #plot = False
        plot = True

        data_set_size = 500
        inputs = []
        outputs = []
        for i in range(data_set_size):
            x, y = np.random.random(2)
            inputs.append([x, y])
            outputs.append([sin(x + y), cos(1 + x - y)])

        # Build and train network
        neural_network = createNeuralNetwork(inputs, outputs, [4], 20000)

        # Get outputs
        net_outputs = [neural_network.evaluate(i) for i in inputs]
        #print(net_outputs, outputs[i])

        # Output node data
        neural_network.print_neuron_config()

        # Plot out training data
        if plot:
            xx = [inputs[i][0] + inputs[i][1] for i in range(data_set_size)]
            yy = [outputs[i][0] for i in range(data_set_size)]
            zz = [net_outputs[i][0] for i in range(len(inputs))]
            xx2 = [inputs[i][0] - inputs[i][1] for i in range(data_set_size)]
            yy2 = [outputs[i][1] for i in range(data_set_size)]
            zz2 = [net_outputs[i][1] for i in range(data_set_size)]
            plt.scatter(xx, yy, label='Inputs 1')
            plt.scatter(xx, zz, label='Fitted 1')
            plt.scatter(xx2, yy2, label='Inputs 2')
            plt.scatter(xx2, zz2, label='Fitted 2')
            plt.legend()
            plt.show(block=True)
            plt.pause(10)
            plt.close()

    def test_train_neural_network_with_two_input_trig_function(self):
        # plot = False
        plot = True

        inputs = []
        outputs = []
        data_set_size = 500
        for i in range(data_set_size):
            x, y = np.random.random(2)
            inputs.append([x, y])
            outputs.append([(sin(x - y) + 1) / 2])

        print(min([inputs[n][0] for n in range(data_set_size)]), max([inputs[n][0] for n in range(data_set_size)]))
        print(min([outputs[n][0] for n in range(data_set_size)]), max([outputs[n][0] for n in range(data_set_size)]))

        # Build and train network
        neural_network = createNeuralNetwork(inputs, outputs, [2], 20000)

        # Output node data
        neural_network.print_neuron_config()

        # Plot out training data
        if plot:
            xx = [inputs[i][0] - inputs[i][1] for i in range(data_set_size)]
            yy = [outputs[i][0] for i in range(len(inputs))]
            zz = [neural_network.evaluate(i)[0] for i in inputs]
            plt.scatter(xx, yy)
            plt.scatter(xx, zz)
            plt.show(block=True)
            plt.pause(3)
            plt.close()

    def test_train_neural_network_with_linear_function(self):
        # plot = False
        plot = True

        inputs = []
        outputs = []
        data_set_size = 500
        for i in range(data_set_size):
            inputs.append(1 + np.random.random(1))
            outputs.append([2 - inputs[-1]])

        # Build and train network
        neural_network = createNeuralNetwork(inputs, outputs, [2], 20000)

        # Output node data
        neural_network.print_neuron_config()

        # Plot out training data
        if plot:
            xx = [inputs[i][0] for i in range(data_set_size)]
            yy = [outputs[i][0] for i in range(data_set_size)]
            zz = [neural_network.evaluate(inputs[i])[0] for i in range(data_set_size)]
            plt.scatter(xx, yy)
            plt.scatter(xx, zz)
            plt.show(block=True)
            plt.pause(2)
            plt.close()
        self.assertTrue(neural_network.get_total_fit_score(inputs, outputs) < 0.4)

    def test_train_neural_network_with_cos_function(self):
        # plot = False
        plot = True

        inputs = []
        outputs = []
        data_set_size = 500
        for i in range(data_set_size):
            x = np.random.random(1)
            inputs.append([x])
            outputs.append([cos(x)])

        # Build and train network
        neural_network = createNeuralNetwork(inputs, outputs, [2], 20000)

        # Get outputs
        net_outputs = [neural_network.evaluate(i) for i in inputs]
        # print(net_outputs, outputs[i])

        # Output node data
        neural_network.print_neuron_config()

        # Plot out training data
        if plot:
            xx = [inputs[i][0] for i in range(data_set_size)]
            yy = [outputs[i][0] for i in range(data_set_size)]
            zz = [n[0] for n in net_outputs]
            plt.scatter(xx, yy)
            plt.scatter(xx, zz)
            plt.show(block=True)
            plt.pause(3)
            plt.close()
        self.assertTrue(neural_network.get_total_fit_score(inputs, outputs) < 0.11)


if __name__ == '__main__':
    unittest.main()
