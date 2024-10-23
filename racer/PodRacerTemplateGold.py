import math
import sys
import itertools

# INSERT NEURAL NETWORK CODE

# INSERT GAME FUNCTIONS

# INSERT POD

# Build neural network
# INSERT NN CONFIGS
racer_nn_data_str = '% INSERT RACER NN PARAMETERS %'
blocker_nn_data_str = '% INSERT BLOCKER NN PARAMETERS %'
racer = NeuralNetwork.create_from_json(racer_nn_data_str, RACER_NN_CONFIG)
blocker = NeuralNetwork.create_from_json(blocker_nn_data_str, BLOCKER_NN_CONFIG)

laps = int(input())
checkpoint_count = int(input())
checkpoints = []
for i in range(checkpoint_count):
    checkpoint_x, checkpoint_y = [int(j) for j in input().split()]
    checkpoints.append(np.array((checkpoint_x, checkpoint_y)))
# print(f'Checkpoints: {checkpoints}', file=sys.stderr, flush=True)

my_next_checkpoints = [[0], [0]]
opponent_next_checkpoints = [[0], [0]]
pod_boost_used = [False, False]

# game loop
while True:
    all_pods = []
    for i in range(4):
        # x: x position of the pod
        # y: y position of the pod
        # vx: x speed of the pod
        # vy: y speed of the pod
        # angle: angle of the pod
        # next_check_point_id: next check point id of the pod
        x, y, vx, vy, angle, next_checkpoint_id = [int(j) for j in input().split()]
        position = np.array((x, y))
        velocity = np.array((vx, vy))
        absolute_checkpoint_angle = get_angle(checkpoints[next_checkpoint_id] - position)
        # Get pod angle allowing for the fact that the first time through the input angle will be given as -1
        pod_angle = absolute_checkpoint_angle if angle == -1 else math.radians((270 - angle) % 360 - 180)
        # print(f'Input angle: {angle}, pod angle: {pod_angle}', file=sys.stderr, flush=True)
        all_pods.append(Pod(position, velocity, pod_angle, next_checkpoint_id))
        if i > 1:
            if next_checkpoint_id != opponent_next_checkpoints[i - 2][-1]:
                opponent_next_checkpoints[i - 2].append(next_checkpoint_id)
        else:
            if next_checkpoint_id != my_next_checkpoints[i][-1]:
                my_next_checkpoints[i].append(next_checkpoint_id)

    pods = all_pods[:2]
    opponent_pods = all_pods[2:]

    outputs = []
    for pod_index in range(2):
        pod = pods[pod_index]

        # Angles are relative to current heading and are all in radians except the input checkpoint angle
        # print(f'velocity : {pod.velocity} angle : {round(math.degrees(pod.angle))}  vel angle : {round(math.degrees(velocity_angle))}', file=sys.stderr, flush=True)

        # Racer and blocker behave differently
        if pod_index == 0:
            # Racer - inputs are [velocity_angle, speed, checkpoint_angle, checkpoint_distance, next_checkpoint_angle, next_checkpoint_distance]
            steer, thrust, command = get_next_racer_action(pod, checkpoints, racer)
            target_angle = pod.angle + steer
            thrust = round(thrust)
            # print(f'Steer: {round(math.degrees(steer))} Thrust: {thrust} Command: {command}', file=sys.stderr, flush=True)
        else:
            # Blocker - inputs are [velocity_angle, speed, velocity_angle, speed, racer_angle, racer_distance, checkpoint_angle, checkpoint_distance]
            opponent = opponent_pods[1 if len(opponent_next_checkpoints[1]) > len(opponent_next_checkpoints[0]) else 0]

            if min(my_next_checkpoints) > max(opponent_next_checkpoints):
                # We are winning - use racer config
                blocker_steer, blocker_thrust, command = get_next_racer_action(pod, checkpoints, racer)
            else:
                # Act as blocker
                lead_opponent_id = 1 if len(opponent_next_checkpoints[1]) > len(opponent_next_checkpoints[0]) else 0
                print(f'Lead opponent pod : {lead_opponent_id} targeting checkpoint {opponent_pods[lead_opponent_id].next_checkpoint_id}', file=sys.stderr, flush=True)
                blocker_steer, blocker_thrust, command = get_next_blocker_action(pod, opponent_pods[lead_opponent_id], checkpoints, blocker)
            target_angle = pod.angle + blocker_steer
            thrust = round(blocker_thrust)

        # Output the target position followed by the power (0 <= thrust <= 100)
        target_position = list(map(round, pod.position + 10000 * np.array((math.sin(target_angle), math.cos(target_angle)))))
        if command == 'BOOST':
            thrust = 100 if pod_boost_used[pod_index] else 'BOOST'
            pod_boost_used[pod_index] = True
        outputs.append(np.append(deepcopy(target_position), thrust))

    # Record state
    #if last_positions is not None:
    #    print(x == simulators[pod_index].pods[0].position[0], y == simulators[pod_index].pods[0].position[1], file=sys.stderr, flush=True)
    #    print('x : ' + str(x) + '  sim_x : ' + str(simulators[pod_index].pods[0].position[0]) + '  y : ' + str(y) + '  sim_y : ' + str(simulators[pod_index].pods[0].position[1]), file=sys.stderr, flush=True)

    print(*outputs[0], 'Beep beep')
    print(*outputs[1], 'Meep meep')
