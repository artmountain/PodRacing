import copy

import numpy as np

from Courses import create_courses
from NeuralNet import NeuralNetwork
from PodRacerFunctions import transform_race_data_to_nn_inputs, transform_nn_outputs_to_instructions, \
    evaluate_game_step, transform_distance_to_input, get_angle_and_distance, get_angle, get_relative_angle_and_distance

# NN shape data
RACER_NN_INPUTS = 6
RACER_NN_OUTPUTS = 2
RACER_NN_LAYERS = 2
RACER_NN_LAYER_SIZES = [RACER_NN_INPUTS, RACER_NN_OUTPUTS]

# Training data
POPULATION_SIZE = 50 # TODO
NUMBER_OF_DRIVE_STEPS = 200 # TODO
NUMBER_OF_TRAINING_COURSES = 5 # TODO
NUMBER_OF_RACER_GENERATIONS = 100 # TODO
NUMBER_OF_RACER_MUTATIONS = 10 # TODO
NN_MUTATION_RATE = 0.05


def evaluate_racer(course, racer, record_path):
    checkpoints = course.get_checkpoints()
    position = copy.deepcopy(course.get_start_position())
    next_checkpoint_idx = 0
    path = []
    if record_path:
        path.append(copy.deepcopy(position))
    velocity = [0, 0]
    angle = get_angle(checkpoints[0] - position)
    for step in range(NUMBER_OF_DRIVE_STEPS):
        velocity_angle, speed = get_relative_angle_and_distance(velocity, angle)
        checkpoint_position = checkpoints[next_checkpoint_idx % len(checkpoints)]
        checkpoint_angle, checkpoint_distance = get_relative_angle_and_distance(checkpoint_position - position, angle)
        next_checkpoint_position = checkpoints[(next_checkpoint_idx + 1) % len(checkpoints)] # todo
        next_checkpoint_angle, next_checkpoint_distance = get_relative_angle_and_distance(next_checkpoint_position - position, angle)
        nn_inputs = transform_race_data_to_nn_inputs(velocity_angle, speed, checkpoint_angle, checkpoint_distance, next_checkpoint_angle, next_checkpoint_distance)
        nn_outputs = racer.evaluate(nn_inputs)
        steer, thrust = transform_nn_outputs_to_instructions(nn_outputs)
        position, velocity, angle, hit_checkpoint = evaluate_game_step(position, velocity, angle, checkpoint_position, angle + steer, thrust)
        path.append(copy.deepcopy(position))
        if hit_checkpoint:
            next_checkpoint_idx += 1

    distance_to_next_checkpoint = get_angle_and_distance(checkpoints[next_checkpoint_idx % len(checkpoints)] - position)[1]
    score = 100 * (next_checkpoint_idx + transform_distance_to_input(distance_to_next_checkpoint))
    return score, next_checkpoint_idx, path

def train_pod_racer(output_file):
    # Create training courses
    courses = create_courses(NUMBER_OF_TRAINING_COURSES)

    # Start from a set of random racers
    racers = []
    for _i in range(POPULATION_SIZE):
        weights = []
        biases = []
        for layer in range(RACER_NN_LAYERS):
            weights.append([np.random.rand(RACER_NN_INPUTS) for _j in range(RACER_NN_LAYER_SIZES[layer])])
            biases.append(np.random.rand(RACER_NN_LAYER_SIZES[layer]))
        racer = NeuralNetwork(RACER_NN_INPUTS, RACER_NN_OUTPUTS, weights, biases)
        score = sum([evaluate_racer(course, racer, False)[0] for course in courses])
        racers.append([racer, score])

    for generation in range(NUMBER_OF_RACER_GENERATIONS):
        # Create and evaluate the next generation
        for racer_idx in range(POPULATION_SIZE):
            # Create new racer
            new_racer = racers[racer_idx][0].mutate(NN_MUTATION_RATE)
            score = sum([evaluate_racer(course, new_racer, False)[0] for course in courses])
            racers.append([new_racer, score])

        # Filter and select best of population
        racers = sorted(racers, key=lambda x: x[1], reverse=True)[:POPULATION_SIZE]
        print(f'Generation {generation}. Best score: {racers[0][1]}')
        print(f'All scores: {[r[1] for r in racers]}')

    # Output best racer
    best_racer = racers[0][0]
    best_racer.print_neuron_config()
    best_racer.pickle_neuron_config(output_file)