import math
import random
from copy import deepcopy

import numpy as np

from Courses import create_courses
from NeuralNet import NeuralNetwork
from PodRacerFunctions import transform_race_data_to_nn_inputs, transform_nn_outputs_to_instructions, \
    evaluate_game_step, transform_distance_to_input, get_angle, get_relative_angle_and_distance, \
    get_distance

# Test flag
TEST = False
#TEST = True

# NN shape data
RACER_NN_INPUTS = 6
RACER_NN_OUTPUTS = 2
RACER_NN_LAYERS = 2
RACER_NN_LAYER_SIZES = [RACER_NN_INPUTS, RACER_NN_OUTPUTS]

# Training configuration
POPULATION_SIZE = 5 if TEST else 50 # todo
NUMBER_OF_DRIVE_STEPS = 20 if TEST else 200
NUMBER_OF_TRAINING_COURSES = 5
NUMBER_OF_RACER_GENERATIONS = 10 if TEST else 100
NUMBER_OF_RACER_MUTATIONS = 10
NN_MUTATION_RATE = 0.05
RANDOM_VARIATION = 0.2


def evaluate_racer(course, racer, record_path):
    checkpoints = course.get_checkpoints()
    position = deepcopy(course.get_start_position())
    next_checkpoint_idx = 0
    path = []
    next_checkpoints = []
    inputs = []
    angles = []  # todo remove when bug fixed
    if record_path:
        path.append(deepcopy(position))
        next_checkpoints.append(next_checkpoint_idx)
        inputs.append([0, 0])
        angles.append(0)
    velocity = [0, 0]
    angle = get_angle(checkpoints[0] - position)
    for step in range(NUMBER_OF_DRIVE_STEPS):
        velocity_angle, speed = get_relative_angle_and_distance(velocity, angle)
        checkpoint_position = checkpoints[next_checkpoint_idx % len(checkpoints)]
        checkpoint_angle, checkpoint_distance = get_relative_angle_and_distance(checkpoint_position - position, angle)
        next_checkpoint_position = checkpoints[(next_checkpoint_idx + 1) % len(checkpoints)]
        next_checkpoint_angle, next_checkpoint_distance = get_relative_angle_and_distance(next_checkpoint_position - position, angle)
        nn_inputs = transform_race_data_to_nn_inputs(velocity_angle, speed, checkpoint_angle, checkpoint_distance, next_checkpoint_angle, next_checkpoint_distance)
        nn_outputs = racer.evaluate(nn_inputs)
        steer, thrust = transform_nn_outputs_to_instructions(nn_outputs)
        if record_path:
            path.append(deepcopy(position))
            next_checkpoints.append(next_checkpoint_idx)
            inputs.append([round(math.degrees(steer)), thrust])
            angles.append(math.degrees(angle))
        position, velocity, angle, hit_checkpoint = evaluate_game_step(position, velocity, angle, checkpoint_position, angle + steer, thrust)
        if hit_checkpoint:
            next_checkpoint_idx += 1

    distance_to_next_checkpoint = get_distance(checkpoints[next_checkpoint_idx % len(checkpoints)] - position)
    score = 100 * (next_checkpoint_idx + transform_distance_to_input(distance_to_next_checkpoint))
    return score, next_checkpoint_idx, path, next_checkpoints, inputs, angles

def score_racer(racer, courses):
    return np.mean([evaluate_racer(course, racer, False)[0] for course in courses])

def get_gene_from_racer(racer):
    gene = []
    for layer in range(RACER_NN_LAYERS):  # TODO : make this work for generic NN shape
        for neuron in range(RACER_NN_LAYER_SIZES[layer]):
            gene += racer.neurons[layer][neuron].weights.tolist()
        for neuron in range(RACER_NN_LAYER_SIZES[layer]):
            gene.append(racer.neurons[layer][neuron].bias)
    return gene

def build_racer_from_gene(gene):
    weights = []
    biases = []
    gene_index = 0
    for layer in range(RACER_NN_LAYERS):  # TODO : make this work for generic NN shape
        weights_this_layer = []
        for neuron in range(RACER_NN_LAYER_SIZES[layer]):
            weights_this_layer.append(np.array(gene[gene_index:gene_index + RACER_NN_INPUTS]))
            gene_index += RACER_NN_INPUTS
        weights.append(weights_this_layer)
        biases.append(deepcopy(gene[gene_index:gene_index + RACER_NN_LAYER_SIZES[layer]]))
        gene_index += RACER_NN_LAYER_SIZES[layer]
    racer = NeuralNetwork(RACER_NN_INPUTS, RACER_NN_OUTPUTS, weights, biases)
    return racer

def train_pod_racer(output_file, racers_seed_file):
    racers = []
    courses = create_courses(NUMBER_OF_TRAINING_COURSES)
    if racers_seed_file is not None:
        # Start from pre-configured racers
        with open(racers_seed_file, 'r') as f:
            for line in f.readlines():
                racer = NeuralNetwork.create_from_json(line.rstrip())
                score = score_racer(racer, courses)
                gene = get_gene_from_racer(racer)
                racers.append([racer, score, gene])

    # Start from a set of random racers
    gene_length = RACER_NN_LAYER_SIZES[0] * (RACER_NN_INPUTS + 1) + RACER_NN_LAYER_SIZES[1] * (RACER_NN_LAYER_SIZES[0] + 1)
    for _i in range(POPULATION_SIZE - len(racers)):
        gene = np.random.rand(gene_length) * 2 - 1
        racer = build_racer_from_gene(gene)
        score = score_racer(racer, courses)
        racers.append([racer, score, gene])

    # Genetic algorithm
    for generation in range(NUMBER_OF_RACER_GENERATIONS):
        # Create new training courses each time to avoid over-fitting
        courses = create_courses(NUMBER_OF_TRAINING_COURSES)

        # Update scores for existing racers
        for racer_idx in range(POPULATION_SIZE):
            score = score_racer(racers[racer_idx][0], courses)
            new_score = 0.8 * racers[racer_idx][1] + 0.2 * score
            racers[racer_idx][1] = new_score

        # Create and evaluate the next generation - first create new racers by mutation and replace parent if better
        for racer_idx in range(POPULATION_SIZE):
            new_racer = racers[racer_idx][0].mutate(NN_MUTATION_RATE)
            score = score_racer(new_racer, courses)
            if score > racers[racer_idx][1]:
                new_racer_gene = get_gene_from_racer(new_racer)
                racers[racer_idx] = [new_racer, score, new_racer_gene]

        # Now breed racers
        new_genes = []
        for idx in range(2, POPULATION_SIZE):
            breeding_pair = random.sample(racers[:idx], 2)
            parent1 = breeding_pair[0][2]
            parent2 = breeding_pair[1][2]
            # Create one child gene where the individual genes are mixed and one which is an interpolation
            child_gene_interleave = [0] * gene_length
            child_gene_scale = [0] * gene_length
            child_gene_splice = [0] * gene_length
            splice_point = random.randint(0, gene_length)
            scale_factor = 1.2 * random.random() - 0.1
            for j in range(gene_length):
                random_variation = (1 if random.random() < NN_MUTATION_RATE else 0) * 2 * RANDOM_VARIATION * random.random() * (random.random() - 0.5)
                child_gene_interleave[j] = (parent1[j] if random.random() > 0.5 else parent2[j]) + random_variation
                random_variation_for_scale = (1 if random.random() < NN_MUTATION_RATE else 0) * 2 * RANDOM_VARIATION * random.random() * (random.random() - 0.5)
                child_gene_scale[j] = (parent1[j] * scale_factor + parent2[j] * (
                            1 - scale_factor)) + random_variation_for_scale
                random_variation_for_splice = (1 if random.random() < NN_MUTATION_RATE else 0) * 2 * RANDOM_VARIATION * random.random() * (random.random() - 0.5)
                child_gene_splice[j] = (parent1[j] if j < splice_point else parent2[j]) + random_variation_for_splice
            new_genes.append(child_gene_interleave)
            new_genes.append(child_gene_scale)
            new_genes.append(child_gene_splice)

        # Score and add new racers
        for gene in new_genes:
            racer = build_racer_from_gene(gene)
            score = score_racer(racer, courses)
            racers.append([racer, score, gene])

        # Filter and select best of population
        racers = sorted(racers, key=lambda x: x[1], reverse=True)[:POPULATION_SIZE]
        print(f'Generation {generation}. Best score: {racers[0][1]}')
        print(f'All scores: {np.around(np.array([r[1] for r in racers]), 2).tolist()}')

        # TODO - remove
        # Output results of best racer
        for i in range(len(courses)):
            score, next_checkpoint_idx, path, next_checkpoints, inputs, angles = evaluate_racer(courses[i], racers[0][0], False)
            print(f'Best racer course {i}. Score {score}, Checkpoint {next_checkpoint_idx}')

    # Output best racers
    best_racer = racers[0][0]
    best_racer.print_neuron_config()
    open(output_file, "w").close()
    for racer in racers:
        racer[0].pickle_neuron_config(output_file)