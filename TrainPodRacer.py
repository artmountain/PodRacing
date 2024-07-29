import json
import math
import random
from copy import deepcopy

import numpy as np
from Courses import create_courses, Course, create_preset_courses, create_preset_courses2
from DisplayRace import plot_pod_race
from NeuralNet import NeuralNetwork, createNeuralNetwork
from PodRacerFunctions import DISTANCE_SCALING, get_angle_and_distance, transform_distance, \
    transform_race_data_to_nn_inputs, \
    transform_nn_outputs_to_instructions, \
    evaluate_game_step, MAX_STEER_PER_TURN

# NN shape data
RACER_NN_INPUTS = 6
RACER_NN_OUTPUTS = 2
RACER_NN_SHAPE = [RACER_NN_INPUTS, RACER_NN_INPUTS, RACER_NN_OUTPUTS]
RACER_NN_LAYER_SIZES = [RACER_NN_INPUTS, RACER_NN_OUTPUTS]

# Training data
NUMBER_OF_DRIVE_STEPS = 300 # TODO
NUMBER_OF_TRAINING_COURSES = 20 # TODO
POPULATION_SIZE = 25 # TODO
FRACTION_TO_BREED = 1 # TODO
FRACTION_TO_MUTATE = 0.2 # TODO
NN_MUTATION_AMOUNTS = [1, 0.5, 0.1]


# Basic training data
BASIC_TRAINING_INPUTS = [
    [0, np.array((0, 50)), 0, 3000, 0.5, 2000],
    [0, np.array((0, 100)), 0.1, 3000, 0.5, 2000],
    [0, np.array((0, 100)), -0.1, 3000, 0.5, 2000],
    [0, np.array((0, 100)), 1, 3000, 0.5, 2000],
    [0, np.array((0, 100)), -1, 3000, 0.5, 2000],
    [0, np.array((0, 100)), 2, 3000, 0.5, 2000],
    [0, np.array((0, 100)), -2, 3000, 0.5, 2000],
    [0, np.array((0, 100)), 3, 3000, 0.5, 2000],
    [0, np.array((0, 100)), -3, 3000, 0.5, 2000],
    [0, np.array((0, 100)), 1.5, 1000, 0.5, 2000],
    [0, np.array((0, 100)), -1.5, 1000, 0.5, 2000],
    [0, np.array((0, 50)), 0, 3000, -0.5, 2000],
    [0, np.array((0, 100)), 0.1, 3000, -0.5, 2000],
    [0, np.array((0, 100)), -0.1, 3000, -0.5, 2000],
    [0, np.array((0, 100)), 1, 3000, -0.5, 2000],
    [0, np.array((0, 100)), -1, 3000, -0.5, 2000],
    [0, np.array((0, 100)), 2, 3000, -0.5, 2000],
    [0, np.array((0, 100)), -2, 3000, -0.5, 2000],
    [0, np.array((0, 100)), 3, 3000, -0.5, 2000],
    [0, np.array((0, 100)), -3, 3000, -0.5, 2000],
    [0, np.array((0, 100)), 1.5, 1000, -0.5, 2000],
    [0, np.array((0, 100)), -1.5, 1000, -0.5, 2000],
]
BASIC_TRAINING_OUTPUTS = [
    [0, 100],
    [0.1, 100],
    [-0.1, 100],
    [1, 100],
    [-1, 100],
    [1, 100],
    [-1, 100],
    [1, 100],
    [-1, 100],
    [1, 25],
    [-1, 25],
    [0, 100],
    [0.1, 100],
    [-0.1, 100],
    [1, 100],
    [-1, 100],
    [1, 100],
    [-1, 100],
    [1, 100],
    [-1, 100],
    [1, 25],
    [-1, 25],
]


def evaluate_racer(course, racer, record_path, record_nn):
    checkpoints = course.get_checkpoints()
    position = deepcopy(course.get_start_position())
    next_checkpoint_idx = 1
    path = []
    nn_data = []
    inputs = [[0, 0]]
    if record_path:
        path.append(deepcopy(position))
    velocity = np.zeros(2)
    #angle = get_angle(checkpoints[0] - position)
    angle = 0
    closest_distance_to_checkpoint = None
    for step in range(NUMBER_OF_DRIVE_STEPS):
        next_checkpoint_pos = checkpoints[next_checkpoint_idx % len(checkpoints)]
        checkpoint_angle, checkpoint_distance = get_angle_and_distance(next_checkpoint_pos - position)
        closest_distance_to_checkpoint = checkpoint_distance if closest_distance_to_checkpoint is None else min(checkpoint_distance, closest_distance_to_checkpoint)
        if next_checkpoint_idx > 1:
            next_checkpoint_angle, next_checkpoint_distance = get_angle_and_distance(
                checkpoints[(next_checkpoint_idx + 1) % len(checkpoints)] - next_checkpoint_pos)
        else:
            # Simulate the game where we don't know this on the first leg
            next_checkpoint_angle = checkpoint_angle
            next_checkpoint_distance = 2 * checkpoint_distance
        #print(angle, velocity, checkpoint_angle, checkpoint_distance, next_checkpoint_angle, next_checkpoint_distance)
        nn_inputs = transform_race_data_to_nn_inputs(angle, velocity, checkpoint_angle, checkpoint_distance,
                                                     next_checkpoint_angle, next_checkpoint_distance)
        #print(nn_inputs)
        nn_outputs = racer.evaluate(nn_inputs)
        #print(nn_outputs)
        if record_nn:
            nn_data.append([nn_inputs, deepcopy(nn_outputs)])
        steer, thrust = transform_nn_outputs_to_instructions(nn_outputs)
        #print(steer, thrust)
        position, velocity, angle, hit_checkpoint = evaluate_game_step(position, velocity, angle, next_checkpoint_pos, angle + steer, thrust)
        if record_path:
            #path.append([deepcopy(position), angle + steer])
            path.append(deepcopy(position))
            inputs.append([int(math.degrees(steer)), int(thrust)])
        if hit_checkpoint:
            next_checkpoint_idx += 1
            closest_distance_to_checkpoint = next_checkpoint_distance

    # Score this drive
    score = 100 * next_checkpoint_idx
    score += 100 * transform_distance(closest_distance_to_checkpoint, DISTANCE_SCALING)
    return score, next_checkpoint_idx, path, inputs, nn_data


def score_racer(racer, courses):
    score = 0
    for course in courses:
        score += evaluate_racer(course, racer, False, False)[0]
    return score


def create_basic_trained_pod_racer():
    nn_inputs = [transform_race_data_to_nn_inputs(*x) for x in BASIC_TRAINING_INPUTS]
    nn_outputs = [[max(0, min(1, x[0] / (2 * MAX_STEER_PER_TURN) + 0.5)), x[1] / 100] for x in BASIC_TRAINING_OUTPUTS]
    racer = createNeuralNetwork(nn_inputs, nn_outputs, [6])

    for i in range(len(nn_inputs)):
        print(racer.evaluate(nn_inputs[i]), nn_outputs[i])
    return racer


def load_racer_from_file(filename):
    with open(filename, 'r') as f:
        config_text = f.readline()
    nn_data = json.loads(config_text)
    racer = NeuralNetwork(6, 2, [np.array(nn_data[x]) for x in ['weights_0', 'weights_1']],
                          [np.array(nn_data[x]) for x in ['biases_0', 'biases_1']])
    return racer


def train_pod_racer(number_of_generations, use_preset_courses, initial_racer_file):
    # Create training courses
    courses = create_preset_courses2() if use_preset_courses else create_courses(NUMBER_OF_TRAINING_COURSES)

    # Always start with a new trained racer
    initial_racer = create_basic_trained_pod_racer()
    all_racers = [[initial_racer, score_racer(initial_racer, courses)]]
    # return initial_racer

    # Add the pre-configured racer if a data file is supplied
    if initial_racer_file is not None:
        racer = load_racer_from_file(initial_racer_file)
        all_racers.append([racer, score_racer(racer, courses)])

    # Create random variants
    base_config = initial_racer.get_neuron_config()
    for i in range(POPULATION_SIZE - len(all_racers)):
        weights = []
        biases = []
        for layer in range(1, len(RACER_NN_SHAPE)):
            layer_weights = []
            for j in range(RACER_NN_SHAPE[layer]):
                layer_weights.append(np.array(base_config['weights_' + str(layer - 1)][j]) + np.random.randn(RACER_NN_SHAPE[layer - 1]))
            weights.append(layer_weights)
            biases.append(np.array(base_config['biases_' + str(layer - 1)]) + np.random.randn(RACER_NN_SHAPE[layer]))
        racer = NeuralNetwork(RACER_NN_INPUTS, RACER_NN_OUTPUTS, weights, biases)
        all_racers.append([racer, score_racer(racer, courses)])

    for generation in range(number_of_generations):
        # Start with the best 4 performing of the previous round
        #new_racers = [all_racers[0], all_racers[1], all_racers[2], all_racers[3]]
        new_racers = all_racers

        # Create a next generation by breeding
        for i in range(2, POPULATION_SIZE):
            parent1 = random.choice(all_racers[:i // 2])[0].get_neuron_config()
            parent2 = random.choice(all_racers[i // 2:POPULATION_SIZE + 1])[0].get_neuron_config()
            new_weights = []
            new_biases = []
            for k, v in parent1.items():
                w = max(0.0, min(1.0, random.random() * 2 - 0.5))
                if 'weights' in k:
                    #new_weights.append(
                    #    [w * np.array([v[y][x] + (1 - w) * parent2[k][y][x] for x in range(len(v[y]))]) for y in range(len(v))])
                    new_weights.append(np.zeros((len(v), len(v[0]))))
                    for x in range(len(v)):
                        for y in range(len(v[x])):
                            w = max(0.0, min(1.0, random.random() * 2 - 0.5))
                            new_weights[-1][x][y] = w * v[x][y] + (1 - w) * parent2[k][x][y]
                else:
                    #new_biases.append(np.array([w * v[x] + (1 - w) * parent2[k][x] for x in range(len(v))]))
                    new_biases.append(np.zeros(len(v)))
                    for x in range(len(v)):
                        w = max(0.0, min(1.0, random.random() * 2 - 0.5))
                        new_biases[-1][x] = w * v[x] + (1 - w) * parent2[k][x]
            new_racer = NeuralNetwork(RACER_NN_INPUTS, RACER_NN_OUTPUTS, new_weights, new_biases)
            new_racer_score = score_racer(new_racer, courses)
            new_racers.append([new_racer, new_racer_score])

            # Evolve this racer by mutations
            mutated_racer = new_racers[-1][0].mutate(FRACTION_TO_MUTATE, random.choice(NN_MUTATION_AMOUNTS))
            mutated_racer_score = score_racer(mutated_racer, courses)
            new_racers.append([mutated_racer, mutated_racer_score])
        all_racers = new_racers

        # Select best racers
        all_racers_sorted = sorted(all_racers, key=lambda x: x[1], reverse=True)
        print([int(x[1]) for x in all_racers_sorted])
        all_racers = all_racers_sorted[:POPULATION_SIZE]
        racer = all_racers[0][0]
        scores = []
        for course in courses:
            scores.append(evaluate_racer(course, racer, False, False)[0])
        print(int(sum(scores)), [int(score) for score in scores])

        # Generate new courses
        #courses = create_courses(NUMBER_OF_TRAINING_COURSES)
    return racer


def simulate_race(racer_filename):
    racer = load_racer_from_file(racer_filename)
    course = create_courses(1)[0]
    score, next_checkpoint_idx, path, inputs, nn_data = evaluate_racer(course, racer, True, True)
    plot_pod_race(course.get_checkpoints(), path, inputs, nn_data)


def simulate_race_with_checkpoints(racer_filename, checkpoints, start_position):
    racer = load_racer_from_file(racer_filename)
    course = Course(checkpoints, start_position)
    score, next_checkpoint_idx, path, inputs, nn_data = evaluate_racer(course, racer, True, True)
    #for i in range(len(path)):
    #    print(path[i], inputs[i])
    #print(nn_data)
    plot_pod_race(course.get_checkpoints(), path, inputs, nn_data)
