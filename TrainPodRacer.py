from NeuralNet import NeuralNetwork
import PodRacerFunctions
from Courses import create_courses


# NN shape data
RACER_NN_INPUTS = 6
RACER_NN_OUTPUTS = 2
RACER_NN_LAYERS = 2
RACER_NN_LAYER_SIZES = [RACER_NN_INPUTS, RACER_NN_OUTPUTS]

# Training data
NUMBER_OF_DRIVE_STEPS = 200 # TODO
NUMBER_OF_TRAINING_COURSES = 1 # TODO
NUMBER_OF_RACER_MUTATIONS = 10 # TODO
NUMBER_OF_RACER_GENERATIONS = 10 # TODO
NN_MUTATION_RATE = 0.05


def evaluate_racer(course, racer, record_path):
    checkpoints = course.get_checkpoints()
    position = copy.deepcopy(course.get_start_position())
    next_checkpoint_idx = 0
    path = []
    if record_path:
        path.append(copy.deepcopy(position))
    velocity = 0
    angle = get_angle(checkpoints[0] - start_position)
    for step in range(NUMBER_OF_DRIVE_STEPS):
        next_checkpoint_pos = checkpoints[next_checkpoint_idx % len(checkpoints)]
        following_checkpoint_pos = checkpoints[(next_checkpoint_idx + 1) % len(checkpoints)]
        nn_inputs = transform_race_data_to_nn_inputs
        nn_outputs = racer.evaluate(nn_inputs)
        steer, thrust = transform_nn_outputs_to_instructions(nn_outputs)
        position, velocity, angle, hit_checkpoint = evaluate_game_step(position, velocity, angle,
                                                                       next_checkpoint_pos, angle + inputs[i][0],
                                                                       inputs[i][1])
        path.append(copy.deepcopy(position))
        if hit_checkpoint:
            next_checkpoint_idx += 1

    return next_checkpoint_idx, path, score

def train_pod_racer():
    # Create training courses
    courses = create_courses(NUMBER_OF_TRAINING_COURSES)

    # Start from a default racer
    weights = []
    biases = []
    for layer in range(RACER_NN_LAYERS):
        weights.append([[0] * RACER_NN_INPUTS] * RACER_NN_LAYER_SIZES[layer])
        biases.append([0] * RACER_NN_LAYER_SIZES[layer])
    racer = NeuralNetwork(RACER_NN_INPUTS, RACER_NN_OUTPUTS, weights, biases)

    for generation in range(NUMBER_OF_RACER_GENERATIONS):
        # Create the next generation
        all_racers = [racer]
        for mutation in range(NUMBER_OF_RACER_MUTATIONS):
            all_racers.append(racer.mutate(NN_MUTATION_RATE))

        # Evaluate this generation
        scores = []
        for racer in all_racers:
            score = 0
            for course in courses:
                score += evaluate_racer(course, racer, False)
