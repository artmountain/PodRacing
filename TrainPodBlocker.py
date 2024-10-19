import math
from copy import deepcopy

import numpy as np

from Courses import create_courses
from GeneticAlgorithm import GeneticAlgorithm
from NeuralNet import NeuralNetwork
from PodRaceSimulator import PodRaceSimulator, Pod
from PodRacerFunctions import transform_distance_to_input, get_angle, get_distance, get_next_racer_action, \
    get_next_blocker_action
from TrainPodRacer import PodRacerGeneticAlgorithm, NUMBER_OF_DRIVE_STEPS, \
    POPULATION_SIZE, NN_MUTATION_RATE, RANDOM_VARIATION, RACER_MID_LAYER_SIZE, RACER_NN_OUTPUTS, \
    NUMBER_OF_TRAINING_COURSES

# Test flag
# TEST = False
TEST = True

POPULATION_SIZE = 5 if TEST else 50
NUMBER_OF_DRIVE_STEPS = 10 if TEST else 200
NUMBER_OF_BLOCKER_GENERATIONS = 10 if TEST else 1000
BLOCKER_NN_INPUTS = 8
BLOCKER_NN_CONFIG = [BLOCKER_NN_INPUTS, BLOCKER_NN_INPUTS, RACER_MID_LAYER_SIZE, RACER_MID_LAYER_SIZE, RACER_NN_OUTPUTS]


class PodBlockerGeneticAlgorithm(GeneticAlgorithm):
    def __init__(self, racer, blocker_nn_config, population_size, mutation_rate, random_variation):
        self.racer = racer
        self.blocker_nn_config = blocker_nn_config
        self.courses = []

        # Get gene length
        self.gene_length = 0
        for layer in range(1, len(blocker_nn_config)):
            self.gene_length += (blocker_nn_config[layer - 1] + 1) * blocker_nn_config[layer]

        GeneticAlgorithm.__init__(self, self.gene_length, population_size, True, mutation_rate, random_variation, False)

    def score_gene(self, gene):
        blocker = PodRacerGeneticAlgorithm.build_racer_from_gene(self.blocker_nn_config, gene)
        return np.mean([self.evaluate_racer_and_blocker(course, self.racer, None, False)[0] - self.evaluate_racer_and_blocker(course, self.racer, blocker, False)[0] for course in self.courses])

    def configure_next_generation(self):
        self.courses = create_courses(NUMBER_OF_TRAINING_COURSES)

    @staticmethod
    def evaluate_racer_and_blocker(course, racer_nn, blocker_nn, record_path):
        checkpoints = course.get_checkpoints()

        # Create opponent racer and blocker
        start_position = course.get_start_position()
        angle = get_angle(checkpoints[0] - start_position)
        racer = Pod(deepcopy(start_position) - np.array((600, 0)), np.array((0, 0)), angle, 0)
        pods = [racer]
        blocker = Pod(deepcopy(start_position) + np.array((600, 0)), np.array((0, 0)), angle, 0) if blocker_nn is not None else None
        if blocker is not None:
            pods.append(blocker)

        # Debug data
        racer_path = []
        blocker_path = []
        paths = [racer_path, blocker_path]
        next_checkpoints = []
        inputs = []
        if record_path:
            racer_path.append(deepcopy(start_position))
            blocker_path.append(deepcopy(start_position))
            inputs.append([0, 0, 0, 0])

        simulator = PodRaceSimulator(checkpoints, pods)
        for step in range(NUMBER_OF_DRIVE_STEPS):
            # Evolve opponent racer
            racer_steer, racer_thrust, racer_command = get_next_racer_action(racer, checkpoints, racer_nn)
            simulator_inputs = [[racer.angle + racer_steer, racer_thrust, racer_command]]

            # Evolve blocker
            if blocker_nn is not None:
                blocker_steer, blocker_thrust, blocker_command = get_next_blocker_action(blocker, racer, checkpoints, blocker_nn)
                simulator_inputs.append([blocker.angle + blocker_steer, blocker_thrust, blocker_command])

                if record_path:
                    racer_path.append(deepcopy(racer.position))
                    inputs_this_step = [round(math.degrees(racer_steer)), int(racer_thrust) if racer_command is None else racer_command]
                    if blocker_nn is not None:
                        blocker_path.append(deepcopy(blocker.position))
                        inputs_this_step.append([round(math.degrees(blocker_steer)), int(blocker_thrust) if blocker_command is None else blocker_command])
                    inputs.append(inputs_this_step)

            simulator.single_step(simulator_inputs)

        # Score is how far the opponent racer gets - we want to minimize this
        distance_to_next_checkpoint = get_distance(checkpoints[racer.next_checkpoint_id] - racer.position)
        score = 100 * (racer.checkpoints_passed + transform_distance_to_input(distance_to_next_checkpoint))
        return score, paths, next_checkpoints, inputs

    def on_generation_complete(self, population):
        # Output best config to temporary file
        output_file = 'nn_data/temp_blocker_config.txt'
        open(output_file, "w").close()
        blocker = PodRacerGeneticAlgorithm.build_racer_from_gene(BLOCKER_NN_CONFIG, population[0][0])
        blocker.pickle_neuron_config(output_file)

        # Score best blocker and output racer score with and without blocker to test effectiveness
        course = self.courses[0]
        score_racer_alone = np.around(self.evaluate_racer_and_blocker(course, self.racer, None, False)[0], 2)
        score_with_blocker = np.around(self.evaluate_racer_and_blocker(course, self.racer, blocker, False)[0], 2)
        print(f'Racer score {score_racer_alone}, With blocker {score_with_blocker}, blocker impact: {score_racer_alone - score_with_blocker}')


def train_pod_blocker(racer_file, output_file, blockers_seed_file):
    # Get racer to train against
    with open(racer_file, 'r') as f:
        racer = NeuralNetwork.create_from_json(f.readlines()[0].rstrip(), BLOCKER_NN_CONFIG)

    # Set up genetic algorithm
    blocker_training_ga = PodBlockerGeneticAlgorithm(racer, BLOCKER_NN_CONFIG, POPULATION_SIZE, NN_MUTATION_RATE, RANDOM_VARIATION)
    if blockers_seed_file is not None:
        # Start from pre-configured blockers
        with open(blockers_seed_file, 'r') as f:
            for line in f.readlines():
                blocker = NeuralNetwork.create_from_json(line.rstrip(), BLOCKER_NN_CONFIG)
                gene = PodRacerGeneticAlgorithm.get_gene_from_racer(BLOCKER_NN_CONFIG, blocker)
                blocker_training_ga.add_gene_to_pool(gene)

    # Add in random blockers to complete the population
    blocker_training_ga.complete_population_with_random_genes()

    # Evolve the genetic algorithm
    blocker_training_ga.evolve(NUMBER_OF_BLOCKER_GENERATIONS)

    # Output best racers
    population = blocker_training_ga.get_population()
    open(output_file, "w").close()
    for gene in population:
        blocker = PodRacerGeneticAlgorithm.build_racer_from_gene(BLOCKER_NN_CONFIG, gene[0])
        blocker.pickle_neuron_config(output_file)


if __name__ == '__main__':
    train_pod_blocker('nn_data/live_racer_nn_config.txt', 'nn_data/blocker_config_test.txt', None)
    # train_pod_blocker('nn_data/live_racer_nn_config.txt', 'nn_data/blocker_config2.txt', 'nn_data/blocker_config.txt')
    # train_pod_blocker('nn_data/live_racer_nn_config.txt', 'nn_data/blocker_config3.txt', 'nn_data/blocker_config2.txt')
    # train_pod_blocker('nn_data/live_racer_nn_config.txt', 'nn_data/blocker_config4.txt', 'nn_data/blocker_config3.txt')
    # train_pod_blocker('nn_data/live_racer_nn_config.txt', 'nn_data/blocker_config5.txt', 'nn_data/blocker_config4.txt')
