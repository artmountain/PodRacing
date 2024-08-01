import numpy as np
import math

MAX_STEER_PER_TURN = math.radians(18)
FULL_CIRCLE = math.radians(360)
HALF_CIRCLE = math.radians(180)


def get_angle(vector):
    xx, yy = vector[0], vector[1]
    return math.atan2(xx, yy)

def get_angle_and_distance(vector):
    angle_deg = round(math.degrees(get_angle(vector)))
    distance = round(math.sqrt(np.sum(np.square(vector))))
    return np.array((angle_deg, distance))

def get_relative_angle(_angle, reference_angle):
    return (_angle - reference_angle + 180) % 360 - 180

def get_relative_angle_and_distance(vector, reference_angle):
    angle_deg = get_relative_angle(round(math.degrees(get_angle(vector))), reference_angle)
    distance = round(math.sqrt(np.sum(np.square(vector))))
    return np.array((angle_deg, distance))

# Transform distance to a number between 0 and 1
def transform_distance_to_input(distance):
    return 1 / (1 + distance / 1000)

def transform_output_to_distance(output):
    return (1 / output - 1) * 1000

def transform_speed_to_input(_speed):
    return 1 / (1 + _speed / 100)

def transform_output_to_speed(output):
    return (1 / output - 1) * 100

def get_relative_angle_and_distance_for_nn_input(_position, _target, _my_angle):
    relative_position = _target - _position
    distance = round(math.sqrt(np.sum(np.square(relative_position))))
    distance = transform_distance_to_input(distance)
    _angle = math.atan2(relative_position[0], relative_position[1])
    _angle = (_angle - _my_angle) / (2 * math.pi)
    if _angle > 1:
        _angle -= 1
    elif _angle < -1:
        _angle += 1
    return [_angle, distance]

# Everything relative to current angle pod is facing. Angles in degrees
def transform_race_data_to_nn_inputs(velocity_angle, speed, checkpoint_angle, checkpoint_distance, next_checkpoint_angle, next_checkpoint_distance):
    return [
        velocity_angle + 180 / 360,
        transform_speed_to_input(speed),
        checkpoint_angle + 180 / 360,
        transform_distance_to_input(checkpoint_distance),
        next_checkpoint_angle + 180 / 360,
        transform_distance_to_input(next_checkpoint_distance)
        ]

# Output angle in radians
def transform_nn_outputs_to_instructions(nn_outputs):
    return [2 * (nn_outputs[0] - 0.5) * MAX_STEER_PER_TURN, transform_output_to_speed(nn_outputs[1])]

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