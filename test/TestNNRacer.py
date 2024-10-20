import unittest

import numpy as np
from Courses import Course, create_courses
from DisplayRace import plot_pod_race
from NeuralNet import NeuralNetwork
from NeuralNetConfigs import RACER_NN_INPUTS, RACER_NN_OUTPUTS
from PodRacerFunctions import transform_race_data_to_nn_inputs, transform_nn_outputs_to_instructions
from TrainPodRacer import train_pod_racer, PodRacerGeneticAlgorithm


class TestNeuralNetRacer(unittest.TestCase):
    # Test a racer with arbitrary inputs
    def test_racer(self):
        weights = [[[0] * 6] * 6, [[0] * 6] * 2]
        biases = [[0] * 6, [0] * 2]
        racer = NeuralNetwork(RACER_NN_INPUTS, RACER_NN_OUTPUTS, weights, biases)

        # Evaluate some inputs
        racer.evaluate([0.5, 0.12, -0.35, 0.14, -0.06, 0.28])
        racer.evaluate([0.01, 0.12, -0.35, 0.14, -0.06, 0.28])

        # Create course
        checkpoints = [np.array((9000, 1000)), np.array((14000, 5000)), np.array((4000, 3000))]
        start_position = [7000, 4000]
        course = Course(checkpoints, start_position)

        # Run race
        score, next_checkpoint_idx, path, inputs, nn_data = PodRacerGeneticAlgorithm.evaluate_racer(course, racer, True)
        plot_pod_race(course.get_checkpoints(), path, inputs, nn_data)
        self.assertEqual(True, False)  # add assertion here

        # Test a pre-configured racer

    def test_simple_racer_inputs(self):
        weights = [
            [
                [-2.92890854, -0.84827645, -0.15971566, 1.2960568, -1.23610342, 1.6653922],
                [-1.90605494, -1.06705688, 0.15926978, -1.90423215, 0.44585698, 1.7483963],
                [17.64502511, -0.50862944, -0.27463063, -0.28175166, 0.26395527, 0.23928137],
                [3.06235512, 0.88848113, 0.0935362, 0.16079921, -0.54123909, -0.57633882],
                [2.08061651, -1.43889162, 1.33446187, 1.273069, 0.25124628, 1.46380962],
                [1.79860601, -1.09762158, 1.02775479, 0.63165818, 2.20642089, 0.88001477]
            ], [
                [-2.50112018, -2.41892578, 12.77787529, 2.26921266, -0.63000829, 0.08866582],
                [3.41716074, 3.3466966, 0.51777951, 0.43704714, 2.11151665, 1.90996106]
            ]
        ]
        biases = [
            [0.4202446901252953, 0.3778271671611065, 0.2846779801109069, -0.8891363402137249, 0.0030088556489044765,
             -0.47657457434741207],
            [-4.233631943952178, -3.366140338372965]
        ]

        racer = NeuralNetwork(RACER_NN_INPUTS, RACER_NN_OUTPUTS, weights, biases)
        nn_inputs1 = transform_race_data_to_nn_inputs(0.05, [50, 100], 0, 100, 0.5, 200)
        nn_outputs1 = racer.evaluate(nn_inputs1)
        racer_outputs1 = transform_nn_outputs_to_instructions(nn_outputs1)
        print(racer_outputs1)
        nn_inputs2 = transform_race_data_to_nn_inputs(-0.05, [50, 100], 0, 100, 0.5, 200)
        nn_outputs2 = racer.evaluate(nn_inputs2)
        print(nn_outputs2)
        nn_inputs3 = transform_race_data_to_nn_inputs(0.1, [-200, 50], 0.2, 500, -0.5, 300)
        nn_outputs3 = racer.evaluate(nn_inputs3)
        print(nn_outputs3)
        nn_inputs4 = transform_race_data_to_nn_inputs(-0.1, [-200, 50], 0.2, 500, -0.5, 300)
        nn_outputs4 = racer.evaluate(nn_inputs4)
        print(nn_outputs4)
        breakpoint()

    # Test training and running a racer
    def test_racer_training(self):
        racer = train_pod_racer(250)
        print(racer.print_neuron_config())

        course = create_courses(1)[0]
        score, next_checkpoint_idx, path, inputs, nn_data = PodRacerGeneticAlgorithm.evaluate_racer(course, racer, True, True)
        plot_pod_race(course.get_checkpoints(), path, inputs, nn_data)

    # Test a pre-configured racer
    def test_trained_racer(self):
        weights = [
            [
                [-2.92890854, -0.84827645, -0.15971566, 1.2960568, -1.23610342, 1.6653922],
                [-1.90605494, -1.06705688, 0.15926978, -1.90423215, 0.44585698, 1.7483963],
                [17.64502511, -0.50862944, -0.27463063, -0.28175166, 0.26395527, 0.23928137],
                [3.06235512, 0.88848113, 0.0935362, 0.16079921, -0.54123909, -0.57633882],
                [2.08061651, -1.43889162, 1.33446187, 1.273069, 0.25124628, 1.46380962],
                [1.79860601, -1.09762158, 1.02775479, 0.63165818, 2.20642089, 0.88001477]
            ],
            [
                [-2.50112018, -2.41892578, 12.77787529, 2.26921266, -0.63000829, 0.08866582],
                [3.41716074, 3.3466966, 0.51777951, 0.43704714, 2.11151665, 1.90996106]
            ]
        ]
        biases = [
            [0.4202446901252953, 0.3778271671611065, 0.2846779801109069, -0.8891363402137249, 0.0030088556489044765,
             -0.47657457434741207],
            [-4.233631943952178, -3.366140338372965]
        ]

        '''
        Layer 0 weights:
        [-2.92890854 -0.84827645 -0.15971566  1.2960568  -1.23610342  1.6653922 ]
        [-1.90605494 -1.06705688  0.15926978 -1.90423215  0.44585698  1.7483963 ]
        [17.64502511 -0.50862944 -0.27463063 -0.28175166  0.26395527  0.23928137]
        [ 3.06235512  0.88848113  0.0935362   0.16079921 -0.54123909 -0.57633882]
        [ 2.08061651 -1.43889162  1.33446187  1.273069    0.25124628  1.46380962]
        [ 1.79860601 -1.09762158  1.02775479  0.63165818  2.20642089  0.88001477]

        Layer 0 biases:
        [0.4202446901252953, 0.3778271671611065, 0.2846779801109069, -0.8891363402137249, 0.0030088556489044765, -0.47657457434741207]

        Layer 1 weights:
        [-2.50112018 -2.41892578 12.77787529  2.26921266 -0.63000829  0.08866582]
        [3.41716074 3.3466966  0.51777951 0.43704714 2.11151665 1.90996106]

        Layer 1 biases:
        [-4.233631943952178, -3.366140338372965]
        '''

        racer = NeuralNetwork(RACER_NN_INPUTS, RACER_NN_OUTPUTS, weights, biases)
        course = create_courses(1)[0]
        score, next_checkpoint_idx, path, inputs, nn_data = PodRacerGeneticAlgorithm.evaluate_racer(course, racer, True, True)
        plot_pod_race(course.get_checkpoints(), path, inputs, nn_data)


if __name__ == '__main__':
    unittest.main()
