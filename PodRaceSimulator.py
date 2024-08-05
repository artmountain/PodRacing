import math

import numpy as np

FULL_CIRCLE = math.radians(360)
HALF_CIRCLE = math.radians(180)
MAX_STEER_PER_TURN = math.radians(18)

class PodRaceSimulator:
    def __init__(self):
        self.boost_used = False

    @staticmethod
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
    def single_step(self, current_position, current_velocity, old_angle, next_checkpoint_pos, input_angle, thrust, command):
        # Process BOOST
        if command == 'BOOST' and not self.boost_used:
            thrust = 650
            self.boost_used = True

        # Calculate new angle
        new_angle = self.update_angle(old_angle, input_angle)

        # Calculate thrust and update speed
        thrust_v = thrust * np.array((math.sin(new_angle), math.cos(new_angle)))
        new_velocity = current_velocity + thrust_v

        # Move
        new_position = np.round(current_position + new_velocity)

        # Apply Drag
        new_velocity = np.trunc(0.85 * new_velocity)

        # See whether we hit a checkpoint
        touched_checkpoint = np.sum(np.square(new_position - next_checkpoint_pos)) < 360000

        return new_position, new_velocity, new_angle, touched_checkpoint
