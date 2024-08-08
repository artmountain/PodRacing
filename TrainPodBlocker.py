import math
from copy import deepcopy

import numpy as np

from Courses import create_courses
from GeneticAlgorithm import GeneticAlgorithm
from NeuralNet import NeuralNetwork
from PodRaceSimulator import PodRaceSimulator, Pod
from PodRacerFunctions import transform_race_data_to_nn_inputs, transform_nn_outputs_to_instructions, \
    transform_distance_to_input, get_angle, get_relative_angle_and_distance, \
    get_distance
from TrainPodRacer import PodRacerGeneticAlgorithm, RACER_NN_CONFIG, NUMBER_OF_TRAINING_COURSES, NUMBER_OF_DRIVE_STEPS, \
    POPULATION_SIZE, NN_MUTATION_RATE, RANDOM_VARIATION, NUMBER_OF_RACER_GENERATIONS


class PodBlockerGeneticAlgorithm(GeneticAlgorithm):
    def __init__(self, racer, blocker_nn_config, population_size, mutation_rate, random_variation):
        self.racer = racer
        self.blocker_nn_config = blocker_nn_config
        self.courses = []

        # Get gene length
        self.gene_length = 0
        for layer in range(1, len(blocker_nn_config)):
            self.gene_length += (blocker_nn_config[layer - 1] + 1) * blocker_nn_config[layer]

        GeneticAlgorithm.__init__(self, self.gene_length, population_size, True, mutation_rate, random_variation, True, False)

    def score_gene(self, gene):
        blocker = PodRacerGeneticAlgorithm.build_racer_from_gene(self.blocker_nn_config, gene)
        return np.mean([self.evaluate_racer_and_blocker(course, self.racer, blocker, False)[0] for course in self.courses])

    def configure_next_generation(self):
        self.courses = create_courses(NUMBER_OF_TRAINING_COURSES)

    @staticmethod
    def evaluate_racer_and_blocker(course, racer_nn, blocker_nn, record_path):
        checkpoints = course.get_checkpoints()

        # Create opponent racer and blocker
        start_position = course.get_start_position()
        angle = get_angle(checkpoints[0] - start_position)
        racer = Pod(deepcopy(start_position), np.array((0, 0)), angle, 0)
        blocker = Pod(deepcopy(start_position), np.array((0, 0)), angle, 0)

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

        simulator = PodRaceSimulator(checkpoints, [racer, blocker])
        for step in range(NUMBER_OF_DRIVE_STEPS):
            # Evolve opponent racer
            velocity_angle, speed = get_relative_angle_and_distance(racer.velocity, racer.angle)
            checkpoint_position = checkpoints[racer.next_checkpoint_id % len(checkpoints)]
            checkpoint_angle, checkpoint_distance = get_relative_angle_and_distance(checkpoint_position - racer.position, racer.angle)
            next_checkpoint_position = checkpoints[(racer.next_checkpoint_id + 1) % len(checkpoints)]
            next_checkpoint_angle, next_checkpoint_distance = get_relative_angle_and_distance(next_checkpoint_position - racer.position, racer.angle)
            nn_inputs = transform_race_data_to_nn_inputs(velocity_angle, speed, checkpoint_angle, checkpoint_distance, next_checkpoint_angle, next_checkpoint_distance)
            nn_outputs = racer_nn.evaluate(nn_inputs)
            racer_steer, racer_thrust, racer_command = transform_nn_outputs_to_instructions(nn_outputs)

            # Evolve blocker
            velocity_angle, speed = get_relative_angle_and_distance(blocker.velocity, blocker.angle)
            racer_angle, racer_distance = get_relative_angle_and_distance(racer.position - blocker.position, blocker.angle)
            checkpoint_angle, checkpoint_distance = get_relative_angle_and_distance(checkpoint_position - blocker.position, blocker.angle)
            nn_inputs = transform_race_data_to_nn_inputs(velocity_angle, speed, racer_angle, racer_distance, checkpoint_angle, checkpoint_distance)
            nn_outputs = blocker_nn.evaluate(nn_inputs)
            blocker_steer, blocker_thrust, blocker_command = transform_nn_outputs_to_instructions(nn_outputs)

            if record_path:
                racer_path.append(deepcopy(racer.position))
                blocker_path.append(deepcopy(blocker.position))
                inputs.append([[round(math.degrees(racer_steer)), int(racer_thrust) if racer_command is None else racer_command],
                               [round(math.degrees(blocker_steer)), int(blocker_thrust) if blocker_command is None else blocker_command]])
            simulator.single_step([[racer.angle + racer_steer, racer_thrust, racer_command],
                                   [blocker.angle + blocker_steer, blocker_thrust, blocker_command]])

        # Score is how far the opponent racer gets - we want to minimize this
        distance_to_next_checkpoint = get_distance(checkpoints[racer.next_checkpoint_id % len(checkpoints)] - racer.position)
        score = 100 * (racer.checkpoints_passed + transform_distance_to_input(distance_to_next_checkpoint))
        return score, racer.next_checkpoint_id, paths, next_checkpoints, inputs

def train_pod_blocker(racer_file, output_file, blockers_seed_file):
    # Get racer to train against
    with open(racer_file, 'r') as f:
        racer = NeuralNetwork.create_from_json(f.readlines()[0].rstrip(), RACER_NN_CONFIG)

    # Set up genetic algorithm
    blocker_training_ga = PodBlockerGeneticAlgorithm(racer, RACER_NN_CONFIG, POPULATION_SIZE, NN_MUTATION_RATE, RANDOM_VARIATION)
    if blockers_seed_file is not None:
        # Start from pre-configured blockers
        with open(blockers_seed_file, 'r') as f:
            for line in f.readlines():
                blocker = NeuralNetwork.create_from_json(line.rstrip(), RACER_NN_CONFIG)
                gene = PodRacerGeneticAlgorithm.get_gene_from_racer(RACER_NN_CONFIG, blocker)
                blocker_training_ga.add_gene_to_pool(gene)

    # Add in random blockers to complete the population
    blocker_training_ga.complete_population_with_random_genes()

    # Evolve the genetic algorithm
    blocker_training_ga.evolve(NUMBER_OF_RACER_GENERATIONS)

    # Output best racers
    population = blocker_training_ga.get_population()
    open(output_file, "w").close()
    for gene in population:
        blocker = PodRacerGeneticAlgorithm.build_racer_from_gene(RACER_NN_CONFIG, gene[0])
        blocker.pickle_neuron_config(output_file)

if __name__ == '__main__':
    train_pod_blocker('nn_data/live_racer_nn_config.txt', 'nn_data/blocker_config.txt', None)
