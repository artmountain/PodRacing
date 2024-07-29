import math

import numpy as np

FULL_CIRCLE = math.radians(360)
HALF_CIRCLE = math.radians(180)
MAX_STEER_PER_TURN = math.radians(18)
DISTANCE_SCALING = 1000
VELOCITY_SCALING = 100


def get_angle(vector):
    xx, yy = vector[0], vector[1]
    return math.atan2(xx, yy)


def get_angle_and_distance(vector):
    angle = math.atan2(vector[0], vector[1])
    distance = round(math.sqrt(vector.dot(vector)))
    return np.array((angle, distance))


# Transform distance to a number between 0 and 1
def transform_distance(distance, scaling):
    return scaling / (scaling + distance)


# Get angle relative to current direction we are facing and scale for NN input
def get_relative_angle_for_nn_input(angle, my_angle):
    angle = (angle - my_angle) / (2 * math.pi)
    if angle > 1:
        angle -= 1
    elif angle < -1:
        angle += 1
    return angle


# Everything relative to current angle pod is facing
def transform_race_data_to_nn_inputs(angle, velocity, checkpoint_angle, checkpoint_distance,
                                     next_checkpoint_angle, next_checkpoint_distance):
    velocity_angle, speed = get_angle_and_distance(velocity)
    nn_inputs = [
        get_relative_angle_for_nn_input(checkpoint_angle, angle),
        transform_distance(checkpoint_distance, DISTANCE_SCALING),
        get_relative_angle_for_nn_input(next_checkpoint_angle, angle),
        transform_distance(next_checkpoint_distance, DISTANCE_SCALING),
        get_relative_angle_for_nn_input(velocity_angle, angle),
        transform_distance(speed, VELOCITY_SCALING)
    ]
    return nn_inputs


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
def evaluate_game_step(position, velocity, old_angle, next_checkpoint_pos, input_angle, new_thrust):
    # Calculate new angle
    new_angle = update_angle(old_angle, input_angle)

    # Calculate thrust and update speed
    velocity[0] = velocity[0] + new_thrust * math.sin(new_angle)
    velocity[1] = velocity[1] + new_thrust * math.cos(new_angle)

    # Move
    position = np.round(position + velocity)

    # Apply Drag
    velocity[0] = np.trunc(0.85 * velocity[0])
    velocity[1] = np.trunc(0.85 * velocity[1])

    # See whether we hit a checkpoint
    touched_checkpoint = False
    if position[0] - next_checkpoint_pos[0] < 600:
        if position[1] - next_checkpoint_pos[1] < 600:
            touched_checkpoint = np.sum(np.square(position - next_checkpoint_pos)) < 360000

    return position, velocity, new_angle, touched_checkpoint
