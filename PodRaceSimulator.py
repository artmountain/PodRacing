import math

import numpy as np

FULL_CIRCLE = math.radians(360)
HALF_CIRCLE = math.radians(180)
MAX_STEER_PER_TURN = math.radians(18)

class Pod:
    def __init__(self, position, velocity, angle, next_checkpoint_id):
        self.position = position
        self.velocity = velocity
        self.angle = angle
        self.next_checkpoint_id = next_checkpoint_id

        self.boost_used = False

# Angles all in radians
class PodRaceSimulator:
    def __init__(self, checkpoints, pods):
        self.checkpoints = checkpoints
        self.pods = pods

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

    def single_step(self, input_angle, thrust, command):
        # Todo - take multiple inputs
        for pod_id in range(len(self.pods)):
            pod = self.pods[pod_id]

            # Process BOOST
            if command == 'BOOST' and not not pod.boost_used:
                thrust = 650
                pod.boost_used = True

            # Calculate new angle
            new_angle = self.update_angle(pod.angle, input_angle)

            # Calculate thrust and update speed
            thrust_v = thrust * np.array((math.sin(new_angle), math.cos(new_angle)))
            new_velocity = pod.velocity + thrust_v

            # Move
            pod.position = np.round(pod.position + new_velocity)

            # Apply Drag
            pod.velocity = np.trunc(0.85 * new_velocity)

            # See whether we hit a checkpoint
            touched_checkpoint = np.sum(np.square(pod.position - self.checkpoints[pod.next_checkpoint_id])) < 360000
            if touched_checkpoint:
                pod.next_checkpoint_id = (pod.next_checkpoint_id + 1) % len(self.checkpoints)

        # TODO - handle collisions

    def get_pod(self, index):
        return self.pods[index]