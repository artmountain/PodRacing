import json
import sys
import math
import numpy as np

MAX_STEER_PER_TURN = math.radians(18)
FULL_CIRCLE = math.radians(360)
HALF_CIRCLE = math.radians(180)


class Neuron:
    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias
        self.num_inputs = len(weights)
        self.inputs = np.zeros(self.num_inputs)
        self.Z = 0
        self.A = 0

    @staticmethod
    def sigmoid(z):
        if z > 100:
            return 1
        if z < -100:
            return 0
        return 1 / (1 + math.exp(-z))

    def evaluate(self, inputs):
        self.inputs = inputs
        self.Z = np.dot(self.weights, inputs) + self.bias
        self.A = self.sigmoid(self.Z)
        return self.A


class NeuralNetwork:
    def __init__(self, num_inputs, num_outputs, weights, bias, input_scaling):
        self.num_inputs = num_inputs
        self.num_layers = len(weights)
        self.num_outputs = num_outputs
        self.neurons = []
        self.input_sizes = [num_inputs]
        self.input_scaling = input_scaling
        for layer in range(self.num_layers):
            number_of_neurons = len(weights[layer])
            self.neurons.append([Neuron(weights[layer][i], bias[layer][i]) for i in range(number_of_neurons)])
            self.input_sizes.append(number_of_neurons)

    def evaluate(self, inputs):
        network_data = [[inputs[i] / self.input_scaling[i] for i in range(self.num_inputs)]]
        for layer in range(self.num_layers):
            layer_outputs = []
            for neuron_idx in range(len(self.neurons[layer])):
                layer_outputs.append(self.neurons[layer][neuron_idx].evaluate(network_data[layer]))
            network_data.append(layer_outputs)
        return network_data[-1]


def get_angle(vector):
    xx, yy = vector[0], vector[1]
    return math.atan2(xx, yy)


def get_angle_and_distance(vector):
    angle_deg = round(math.degrees(get_angle(vector)))
    distance = round(math.sqrt(np.sum(np.square(vector))))
    return np.array((angle_deg, distance))

# Transform distance to a number between 0 and 1
def transform_distance(distance):
    return 1 / (1 + distance / 1000)

def get_relative_angle_and_distance_for_nn_input(position, target, my_angle):
    relative_position = target - position
    distance = round(math.sqrt(np.sum(np.square(relative_position))))
    distance = transform_distance(distance)
    angle = math.atan2(relative_position[0], relative_position[1])
    angle = (angle - my_angle) / (2 * math.pi)
    if angle > 1:
        angle -= 1
    elif angle < -1:
        angle += 1
    return [angle, distance]

# Everything relative to current angle pod is facing
def transform_race_data_to_nn_inputs(velocity_angle, speed, checkpoint_angle, checkpoint_distance, next_checkpoint_angle, next_checkpoint_distance):
    return [(nn_outputs[0] - 0.5) * 2 * MAX_STEER_PER_TURN, nn_outputs[1] * 100]

def transform_nn_outputs_to_instructions(nn_outputs):
    return [(nn_outputs[0] - 0.5) * 2 * MAX_STEER_PER_TURN, nn_outputs[1] * 100]

def update_angle(current_angle, target_angle):
    clockwise = target_angle - current_angle + (FULL_CIRCLE if target_angle < current_angle else 0)
    anticlockwise = current_angle - target_angle + (FULL_CIRCLE if target_angle > current_angle else 0)
    if anticlockwise < clockwise:
        new_angle = current_angle - min(MAX_STEER_PER_TURN, anticlockwise)
        if new_angle < -HALF_CIRCLE:
            new_angle += FULL_CIRCLE
    else:
        new_angle = current_angle + min(MAX_STEER_PER_TURN, clockwise)
        if new_angle > HALF_CIRCLE:
            new_angle -= FULL_CIRCLE
    # print(target_angle * 180 / math.pi, current_angle * 180 / math.pi, new_angle * 180 / math.pi, file=sys.stderr, flush=True)
    return new_angle

# Angles all in radians
def evaluate_game_step(current_position, current_velocity, old_angle, next_checkpoint_pos, input_angle, new_thrust):
    # Calculate new angle
    #new_angle = get_angle(next_checkpoint_pos - current_position) if initial_step else update_angle(old_angle, input_angle)
    new_angle = update_angle(old_angle, input_angle)
    # print('Angle: ' + str(angle * 180 / math.pi), file=sys.stderr, flush=True)

    # Calculate thrust and update speed
    thrust_v = new_thrust * np.array((math.sin(new_angle), math.cos(new_angle)))
    #print(new_angle, file=sys.stderr, flush=True)
    #print(thrust_v, file=sys.stderr, flush=True)
    new_velocity = current_velocity + thrust_v
    #print(new_velocity, file=sys.stderr, flush=True)

    # Move
    new_position = np.round(current_position + new_velocity)

    # Apply Drag
    new_velocity = np.trunc(0.85 * new_velocity)

    # See whether we hit a checkpoint
    touched_checkpoint = np.sum(np.square(new_position - next_checkpoint_pos)) < 360000

    return new_position, new_velocity, new_angle, touched_checkpoint


# Build neural network
nn_data_str = '{"weights_0": [[-0.3987265797799542, -0.17407220204341273, -0.4845179187634501, -0.36977557217291235, -0.4190792174732678, -0.5940213994168384], [-0.4903635201439646, -0.10690330425686506, -0.5292902910192154, -0.20404985454740487, -0.5712182128569357, -0.4841427717053779], [-0.4296185944917154, -0.2529644069102661, -0.5087192813535565, -0.3178591743839243, -0.374772837361659, -0.5889064999096321], [-0.24713175172668256, -0.3596416739056198, -0.41130249882326464, -0.6603107734454461, -0.24579331071768237, -0.7083529708552363], [-0.5481559173493218, -0.21682881506800444, -0.4329994952968969, -0.24669019704535916, -0.505929161553728, -0.4524436767040857], [-0.28505636803194156, -0.18508740263558696, -0.4319453297245097, -0.477585980370779, -0.40158164840485544, -0.5776815796973052], [-0.4542605578493702, -0.2440269856829377, -0.45066448606977827, -0.35809682038981283, -0.3701879447213677, -0.5354173429078161], [-0.42479590789376254, -0.16598596119909353, -0.47191762969324713, -0.3255053390201331, -0.517646282967048, -0.4819998637389056]], "biases_0": [-1.1057236361522982, -1.11206815918043, -1.1216439729478742, -1.0350917196612286, -1.1132240621248963, -1.1377692250311555, -1.1221694865779896, -1.1414101027050065], "weights_1": [[-1.268551014801275, -1.1633826514557668, -1.1763447813085157, -1.203928376304715, -1.260838855695839, -1.3265139606132907, -1.313207188867517, -1.202657173452028], [-0.046654423952866536, 0.15061564069214226, -0.027251661385409548, -0.3550506638178381, 0.121367514629782, -0.14820284781225881, -0.026864071802762103, 0.039281967778453504]], "biases_1": [-4.08216293156321, 0.21407039748160284], "input_scaling": [359, 15529, 357, 13928, 359, 459]}'
nn_data = json.loads(nn_data_str)
neural_network = NeuralNetwork(6, 2, [nn_data['weights_0'], nn_data['weights_1']],
                               [nn_data['biases_0'], nn_data['biases_1']], nn_data['input_scaling'])

# game loop
sim_pos = np.array((0, 0))
velocity = np.zeros(2)
angle = 0
thrust = 0
initialized = 0
while True:
    # next_checkpoint_x: x position of the next check point
    # next_checkpoint_y: y position of the next check point
    # next_checkpoint_dist: distance to the next checkpoint
    # next_checkpoint_angle: angle between your pod orientation and the direction of the next checkpoint
    x, y, next_checkpoint_x, next_checkpoint_y, next_checkpoint_dist, next_checkpoint_angle = [int(i) for i in
                                                                                               input().split()]
    opponent_x, opponent_y = [int(i) for i in input().split()]
    print('My position : ' + str(x) + ' ' + str(y), file=sys.stderr, flush=True)
    print('Checkpoint : ' + str(next_checkpoint_x) + ' ' + str(next_checkpoint_y), file=sys.stderr, flush=True)
    print('Checkpoint angle : ' + str(next_checkpoint_angle) + '   Checkpoint distance : ' + str(next_checkpoint_dist), file=sys.stderr, flush=True)

    next_checkpoint_position = np.array((next_checkpoint_x, next_checkpoint_y))
    position = np.array((x, y))
    if initialized < 5:
        sim_pos = np.array((x, y))
        angle = get_angle(next_checkpoint_position - position)
        target_angle = angle
        thrust = 100
        # print(math.degrees(angle), file=sys.stderr, flush=True)
        initialized += 1
    else:
        # Create neural network inputs
        # [Checkpoint angle, Checkpoint distance, direction, speed, new_angle, thrust]
        # Angles are relative to current heading
        angle_degrees = round(math.degrees(angle))
        direction_and_speed = get_angle_and_distance(velocity)
        checkpoint_angle_and_distance = get_angle_and_distance(next_checkpoint_position - position)
        print(checkpoint_angle_and_distance[0], file=sys.stderr, flush=True)
        checkpoint_angle_and_distance[0] = (checkpoint_angle_and_distance[0] - angle_degrees + 180) % 360
        print(checkpoint_angle_and_distance[0], file=sys.stderr, flush=True)
        next_checkpoint_angle_and_distance = [checkpoint_angle_and_distance[0], checkpoint_angle_and_distance[1] * 2]  # TODO get true value
        velocity_angle = (direction_and_speed[0] - angle_degrees + 180) % 360

        # Get new direction and thrust from neural network
        nn_inputs = checkpoint_angle_and_distance.tolist() + next_checkpoint_angle_and_distance + [velocity_angle, direction_and_speed[1]]
        # print(nn_inputs, file=sys.stderr, flush=True)
        [target_angle, thrust] = neural_network.evaluate(nn_inputs)
        target_angle = (angle_degrees + round(target_angle * 360) - 180) * math.pi / 180
        # print(checkpoint_angle_and_distance[0], round(target_angle * 180 / math.pi), file=sys.stderr, flush=True)
        thrust = round(thrust * 100)
        print(target_angle, thrust, file=sys.stderr, flush=True)

    print(x == sim_pos[0], y == sim_pos[1], file=sys.stderr, flush=True)
    print('x : ' + str(x) + '  sim_x : ' + str(sim_pos[0]) + '  y : ' + str(y) + '  sim_y : ' + str(sim_pos[1]), file=sys.stderr, flush=True)
    sim_pos, velocity, angle, hit_checkpoint = evaluate_game_step(position, velocity, angle, next_checkpoint_position, target_angle, thrust)
    # print('Hit Checkpoint : ' + str(hit_checkpoint), file=sys.stderr, flush=True)

    # Output the target position followed by the power (0 <= thrust <= 100)
    target_position = position + 10000 * np.array((math.sin(angle), math.cos(angle)))
    outputs = map(round, np.append(target_position, thrust))
    print(*outputs, 'Get out of my way')
