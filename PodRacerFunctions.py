import math

import numpy as np

FULL_CIRCLE = math.radians(360)
HALF_CIRCLE = math.radians(180)
MAX_STEER_PER_TURN = math.radians(18)


# All angles in radians

def get_angle(vector):
    xx, yy = vector[0], vector[1]
    return math.atan2(xx, yy)

def get_distance(vector):
    return round(math.sqrt(np.sum(np.square(vector))))

def get_relative_angle_and_distance(vector, reference_angle):
    angle = get_angle(vector) - reference_angle
    if angle > math.pi:
        angle -= 2 * math.pi
    elif angle < -math.pi:
        angle += 2 * math.pi
    distance = get_distance(vector)
    return np.array((angle, distance))

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
    return new_angle

# Transform distance to a number between 0 and 1
def transform_distance_to_input(distance):
    return 1 / (1 + distance / 1000)

def transform_speed_to_input(_speed):
    return 1 / (1 + _speed / 100)

def transform_output_to_thrust(output):
    return output * 100

# Everything relative to current angle pod is facing. Angles in radians
def transform_race_data_to_nn_inputs(velocity_angle, speed, checkpoint_angle, checkpoint_distance, next_checkpoint_angle, next_checkpoint_distance):
    return [
        velocity_angle * math.pi + 0.5,
        transform_speed_to_input(speed),
        checkpoint_angle * math.pi + 0.5,
        transform_distance_to_input(checkpoint_distance),
        next_checkpoint_angle * math.pi + 0.5,
        transform_distance_to_input(next_checkpoint_distance)
        ]

# Output angle in radians
def transform_nn_outputs_to_instructions(nn_outputs):
    command = 'BOOST' if nn_outputs[2] > 0.5 else None
    return [2 * (nn_outputs[0] - 0.5) * MAX_STEER_PER_TURN, transform_output_to_thrust(nn_outputs[1]), command]
