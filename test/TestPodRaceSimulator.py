import unittest

import numpy as np

from PodRaceSimulator import Pod, PodRaceSimulator


class TestPodRaceSimulator(unittest.TestCase):
    def test_head_on_collision(self):
        checkpoints = [np.array((1000, 4000))]
        pod1 = Pod(np.array((1000, 1000)), np.array((0, 85)), 0, 0)
        pod2 = Pod(np.array((1000, 3000)), np.array((0, -85)), 0, 0)
        simulator = PodRaceSimulator(checkpoints, [pod1, pod2])

        # Should drive towards each other 5 times. Thrust 15 will maintain a constant speed
        inputs = [[0, 15, None], [0, -15, None]]
        for i in range(5):
            simulator.single_step(inputs)

        # Pods have moved 500 units
        self.assertTrue(np.array_equal(np.array((1000, 1500)), pod1.position), 'Check pod 1 position')
        self.assertTrue(np.array_equal(np.array((1000, 2500)), pod2.position), 'Check pod 2 position')

        # Another step sees a collision and rebound to the same place
        simulator.single_step(inputs)
        self.assertTrue(np.array_equal(np.array((1000, 1500)), pod1.position), 'Check pod 1 position')
        self.assertTrue(np.array_equal(np.array((1000, 2500)), pod2.position), 'Check pod 2 position')

        # And they move apart
        simulator.single_step(inputs)
        self.assertTrue(np.array_equal(np.array((1000, 1430)), pod1.position), 'Check pod 1 position')
        self.assertTrue(np.array_equal(np.array((1000, 2570)), pod2.position), 'Check pod 2 position')

    def test_low_speed_collision(self):
        checkpoints = [np.array((1000, 4000))]
        pod1 = Pod(np.array((1000, 1000)), np.array((0, 17)), 0, 0)
        pod2 = Pod(np.array((1000, 2200)), np.array((0, -17)), 0, 0)
        simulator = PodRaceSimulator(checkpoints, [pod1, pod2])

        # Should drive towards each other 5 times. Thrust 3 will maintain a constant speed of 20
        inputs = [[0, 3, None], [0, -3, None]]
        for i in range(5):
            simulator.single_step(inputs)

        # Pods have moved 100 units
        self.assertTrue(np.array_equal(np.array((1000, 1100)), pod1.position), 'Check pod 1 position')
        self.assertTrue(np.array_equal(np.array((1000, 2100)), pod2.position), 'Check pod 2 position')

        # Another step sees a collision and rebound with an impulse of 120 - changes velocity to -100
        simulator.single_step(inputs)
        self.assertTrue(np.array_equal(np.array((1000, 1020)), pod1.position), 'Check pod 1 position')
        self.assertTrue(np.array_equal(np.array((1000, 2180)), pod2.position), 'Check pod 2 position')

        # And they move apart
        simulator.single_step(inputs)
        self.assertTrue(np.array_equal(np.array((1000, 938)), pod1.position), 'Check pod 1 position')
        self.assertTrue(np.array_equal(np.array((1000, 2262)), pod2.position), 'Check pod 2 position')

if __name__ == '__main__':
    unittest.main()
