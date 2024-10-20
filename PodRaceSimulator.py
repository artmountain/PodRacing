import itertools
import math

import numpy as np

from PodRacerFunctions import get_distance, get_angle, update_angle


# Angles all in radians
class PodRaceSimulator:
    def __init__(self, checkpoints, pods):
        self.checkpoints = checkpoints
        self.pods = pods

    # inputs are a vector of [input_angle, thrust, command] for each pod
    def single_step(self, inputs):
        # First move the pods
        for pod_id in range(len(self.pods)):
            pod = self.pods[pod_id]
            input_angle, thrust, command = inputs[pod_id]

            # Process BOOST
            if command == 'BOOST' and not not pod.boost_used:
                thrust = 650
                pod.boost_used = True

            # Calculate new angle
            pod.angle = update_angle(pod.angle, input_angle)

            # Calculate thrust and update speed
            thrust_v = thrust * np.array((math.sin(pod.angle), math.cos(pod.angle)))
            pod.velocity = pod.velocity + thrust_v

            # Move
            pod.position = np.round(pod.position + pod.velocity)

        # Now check for collisions
        for pod1, pod2 in itertools.combinations(self.pods, 2):
            separation = pod2.position - pod1.position
            if get_distance(separation) < 1000:
                # Collision occurred
                angle_of_separation = get_angle(separation)
                line_of_separation = np.array((math.sin(angle_of_separation), math.cos(angle_of_separation)))
                relative_speed_along_line = np.dot(pod1.velocity, line_of_separation) - np.dot(pod2.velocity, line_of_separation)
                # Minimum impulse is 120
                relative_speed_along_line = max(120, relative_speed_along_line)
                pod1.velocity = pod1.velocity - line_of_separation * relative_speed_along_line
                pod2.velocity = pod2.velocity + line_of_separation * relative_speed_along_line
                for pod in [pod1, pod2]:
                    pod.position = pod.position + pod.velocity

        # Finally apply drag and see whether any checkpoints hit
        for pod in self.pods:
            # Apply Drag
            pod.velocity = np.trunc(0.85 * pod.velocity)

            # See whether we hit a checkpoint
            touched_checkpoint = np.sum(np.square(pod.position - self.checkpoints[pod.next_checkpoint_id])) < 360000
            if touched_checkpoint:
                pod.next_checkpoint_id = (pod.next_checkpoint_id + 1) % len(self.checkpoints)
                pod.checkpoints_passed += 1

    def get_pod(self, index):
        return self.pods[index]
