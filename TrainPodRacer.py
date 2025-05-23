import math
from copy import deepcopy

import numpy as np

from Courses import create_courses
from GeneticAlgorithm import GeneticAlgorithm
from NeuralNet import NeuralNetwork
from NeuralNetConfigs import RACER_NN_CONFIG
from Pod import Pod
from PodRaceSimulator import PodRaceSimulator
from PodRacerFunctions import transform_distance_to_input, get_angle, get_distance, get_next_racer_action

# Test flag
TEST = False
# TEST = True

# Training configuration
NUMBER_OF_TRAINING_COURSES = 10
POPULATION_SIZE = 5 if TEST else 50
NUMBER_OF_DRIVE_STEPS = 10 if TEST else 200
NUMBER_OF_RACER_GENERATIONS = 10 if TEST else 1000
NN_MUTATION_RATE = 0.05
RANDOM_VARIATION = 0.2


class PodRacerGeneticAlgorithm(GeneticAlgorithm):
    def __init__(self, nn_config, population_size, mutation_rate, random_variation):
        self.nn_config = nn_config
        self.courses = []

        # Get gene length
        self.gene_length = 0
        for layer in range(1, len(nn_config)):
            self.gene_length += (nn_config[layer - 1] + 1) * nn_config[layer]

        GeneticAlgorithm.__init__(self, self.gene_length, population_size, True, mutation_rate, random_variation, True)

    def score_gene(self, gene):
        racer = self.build_racer_from_gene(self.nn_config, gene)
        return np.mean([self.evaluate_racer(course, racer, False)[0] for course in self.courses])

    def configure_next_generation(self):
        self.courses = create_courses(NUMBER_OF_TRAINING_COURSES)

    @staticmethod
    def evaluate_racer(course, racer_nn, record_path):
        checkpoints = course.get_checkpoints()
        start_position = deepcopy(course.get_start_position())
        path = []
        inputs = []
        next_checkpoints = []
        if record_path:
            path.append(deepcopy(start_position))
            next_checkpoints.append(0)
            inputs.append([0, 0])
        pod = Pod(start_position, np.array((0, 0)), get_angle(checkpoints[0] - start_position), 0)
        simulator = PodRaceSimulator(checkpoints, [pod])
        for step in range(NUMBER_OF_DRIVE_STEPS):
            steer, thrust, command = get_next_racer_action(pod, checkpoints, racer_nn)
            if record_path:
                path.append(deepcopy(pod.position))
                next_checkpoints.append(pod.next_checkpoint_id)
                inputs.append([round(math.degrees(steer)), int(thrust) if command is None else command])
            simulator.single_step([[pod.angle + steer, thrust, command]])

        distance_to_next_checkpoint = get_distance(checkpoints[pod.next_checkpoint_id] - pod.position)
        score = 100 * (pod.checkpoints_passed + transform_distance_to_input(distance_to_next_checkpoint))
        return score, pod.next_checkpoint_id, path, next_checkpoints, inputs

    @staticmethod
    def get_gene_from_racer(nn_config, racer):
        gene = []
        for layer in range(1, len(nn_config)):
            for neuron in range(nn_config[layer]):
                gene += racer.neurons[layer - 1][neuron].weights.tolist()
            for neuron in range(nn_config[layer]):
                gene.append(racer.neurons[layer - 1][neuron].bias)
        return gene

    @staticmethod
    def build_racer_from_gene(nn_config, gene):
        weights = []
        biases = []
        gene_index = 0
        for layer in range(1, len(nn_config)):
            weights_this_layer = []
            for neuron in range(nn_config[layer]):
                weights_this_layer.append(np.array(gene[gene_index:gene_index + nn_config[layer - 1]]))
                gene_index += nn_config[layer - 1]
            weights.append(weights_this_layer)
            biases.append(deepcopy(gene[gene_index:gene_index + nn_config[layer]]))
            gene_index += nn_config[layer]
        racer = NeuralNetwork(nn_config[0], nn_config[-1], weights, biases)
        return racer


def train_pod_racer(output_file, racers_seed_file):
    # Set up genetic algorithm
    racer_training_ga = PodRacerGeneticAlgorithm(RACER_NN_CONFIG, POPULATION_SIZE, NN_MUTATION_RATE, RANDOM_VARIATION)
    if racers_seed_file is not None:
        # Start from pre-configured racers
        with open(racers_seed_file, 'r') as f:
            for line in f.readlines():
                racer = NeuralNetwork.create_from_json(line.rstrip(), RACER_NN_CONFIG)
                gene = racer_training_ga.get_gene_from_racer(RACER_NN_CONFIG, racer)
                racer_training_ga.add_gene_to_pool(gene)

    # Add in random racers to complete the population
    racer_training_ga.complete_population_with_random_genes()

    # Evolve the genetic algorithm
    racer_training_ga.evolve(NUMBER_OF_RACER_GENERATIONS)

    # Output best racers
    population = racer_training_ga.get_population()
    open(output_file, "w").close()
    for gene in population:
        racer = PodRacerGeneticAlgorithm.build_racer_from_gene(RACER_NN_CONFIG, gene[0])
        racer.pickle_neuron_config(output_file)


if __name__ == '__main__':
    # train_pod_racer('nn_data/racer_config.txt', None)
    # train_pod_racer('nn_data/racer_config2.txt', 'nn_data/racer_config.txt')
    # train_pod_racer('nn_data/racer_config3.txt', 'nn_data/racer_config2.txt')
    # train_pod_racer('nn_data/racer_config4.txt', 'nn_data/racer_config3.txt')
    # train_pod_racer('nn_data/racer_config5.txt', 'nn_data/racer_config4.txt')
    # train_pod_racer('nn_data/racer_config6.txt', 'nn_data/racer_config5.txt')
    # train_pod_racer('nn_data/racer_config7.txt', 'nn_data/racer_config6.txt')
    # train_pod_racer('nn_data/racer_config8.txt', 'nn_data/racer_config7.txt')
    # train_pod_racer('nn_data/racer_config9.txt', 'nn_data/racer_config8.txt')
    # train_pod_racer('nn_data/racer_config10.txt', 'nn_data/racer_config9.txt')

    # train_pod_racer('nn_data/legend_racer_config.txt', None)
    # train_pod_racer('nn_data/legend_racer_config2.txt', 'nn_data/legend_racer_config.txt')
    # train_pod_racer('nn_data/legend_racer_config3.txt', 'nn_data/legend_racer_config2.txt')
    # train_pod_racer('nn_data/legend_racer_config4.txt', 'nn_data/legend_racer_config3.txt')
    # train_pod_racer('nn_data/legend_racer_config5.txt', 'nn_data/legend_racer_config4.txt')
    # train_pod_racer('nn_data/legend_racer_config6.txt', 'nn_data/legend_racer_config5.txt')
    # train_pod_racer('nn_data/legend_racer_config7.txt', 'nn_data/legend_racer_config6.txt')
    # train_pod_racer('nn_data/legend_racer_config8.txt', 'nn_data/legend_racer_config7.txt')
    #train_pod_racer('nn_data/legend_racer_config9.txt', 'nn_data/legend_racer_config8.txt')
    train_pod_racer('nn_data/legend_racer_config10.txt', 'nn_data/legend_racer_config9.txt')
