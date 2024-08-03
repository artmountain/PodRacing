# Arena is 16000 wide by 9000 high
import math
from random import randint, sample

import numpy as np


class Course:
    def __init__(self, checkpoints, start_position):
        self.checkpoints = checkpoints
        self.start_position = start_position
        nCheckpoints = len(checkpoints)
        self.distance_between_checkpoints = [math.sqrt((checkpoints[i][0] - checkpoints[(i - 1) % nCheckpoints][0]) ** 2 + (
                checkpoints[i][1] - checkpoints[(i - 1) % nCheckpoints][1]) ** 2) for i in range(nCheckpoints)]

    def get_checkpoints(self):
        return self.checkpoints

    def get_start_position(self):
        return self.start_position

    def get_distance_between_checkpoints(self, target_checkpoint):
        return self.distance_between_checkpoints[target_checkpoint]


def create_courses(number_of_courses):
    number_of_checkpoints = np.random.randint(3, 6)
    courses = []
    for i in range(number_of_courses):
        checkpoints_x = sample(range(1, 16), number_of_checkpoints)
        checkpoints_y = sample(range(1, 8), number_of_checkpoints)
        start_position = np.array((1000 * randint(1, 15), 1000 * randint(1, 8)))
        courses.append(Course([np.array((1000 * checkpoints_x[j], 1000 * checkpoints_y[j])) for j in range(number_of_checkpoints)], start_position))

    return courses


def create_preset_courses():
    checkpoints = [
        [[4021, 1481], [11972, 5519], [10517, 7014]],
        [[13493, 2008], [5483, 7471], [4479, 6026]],
        [[12023, 2011], [13997, 2978], [3985, 7778]],
        [[12023, 5526], [10495, 6984], [3991, 1488]],
    ]
    return [Course([np.array((checkpoints[i][j])) for j in range(3)], np.array((checkpoints[i][0]))) for i in range(len(checkpoints))]


def create_preset_courses2():
    checkpoints = [
        [[13985, 3021], [4009, 7809], [12015, 2005], [10713, 4995]],
        [[5477, 7480], [6473, 5815], [4525, 5995], [13529, 2001]],
        [[6521, 5771], [4520, 5990], [13473, 1984], [5507, 7502]],
        [[4020, 7780], [11979, 1991], [10679, 5018], [14023, 3009]],
        [[5477, 7480], [6473, 5815], [4525, 5995], [13529, 2001]],
    ]
    return [Course([np.array((checkpoints[i][j])) for j in range(4)], np.array((checkpoints[i][0]))) for i in range(len(checkpoints))]
