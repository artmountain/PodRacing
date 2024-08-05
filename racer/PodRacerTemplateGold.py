import math
import sys

# INSERT NEURAL NETWORK CODE

# INSERT GAME FUNCTIONS

# INSERT SIMULATOR

# Build neural network
nn_data_str = '% INSERT RACER NN CONFIG %'
nn_shape = (6, 6, 4, 3)
racer = NeuralNetwork.create_from_json(nn_data_str, nn_shape)

class Pod:
    def __init__(self, position, velocity, angle, next_checkpoint_id):
        self.position = position
        self.velocity = velocity
        self.angle = angle
        self.next_checkpoint_id = next_checkpoint_id

        self.boost_used = False

# game loop
checkpoints = []
last_positions = None
simulators = [PodRaceSimulator(), PodRaceSimulator()]
sim_pos = [None, None]

laps = int(input())
checkpoint_count = int(input())
for i in range(checkpoint_count):
    checkpoint_x, checkpoint_y = [int(j) for j in input().split()]
    checkpoints.append(np.array((checkpoint_x, checkpoint_y)))

while True:
    pods = []
    for i in range(2):
        # x: x position of your pod
        # y: y position of your pod
        # vx: x speed of your pod
        # vy: y speed of your pod
        # angle: angle of your pod
        # next_check_point_id: next check point id of your pod
        x, y, vx, vy, angle, next_checkpoint_id = [int(j) for j in input().split()]
        position = np.array((x, y))
        velocity = np.zeros(2) if last_positions is None else np.trunc((position - last_positions[i]) * 0.85)
        absolute_checkpoint_angle = get_angle(checkpoints[next_checkpoint_id] - position)
        pod_angle = absolute_checkpoint_angle + math.radians(angle)
        pods.append(Pod(position, velocity, pod_angle, next_checkpoint_id))
    for i in range(2):
        # x_2: x position of the opponent's pod
        # y_2: y position of the opponent's pod
        # vx_2: x speed of the opponent's pod
        # vy_2: y speed of the opponent's pod
        # angle_2: angle of the opponent's pod
        # next_check_point_id_2: next check point id of the opponent's pod
        x_2, y_2, vx_2, vy_2, angle_2, next_check_point_id_2 = [int(j) for j in input().split()]

    outputs = []
    for pod_index in range(2):
        pod = pods[pod_index]

        # Create neural network inputs
        # [Checkpoint angle, Checkpoint distance, direction, speed, new_angle, thrust]
        # Angles are relative to current heading and are all in radians except the input checkpoint angle
        #print(f'Last position: {last_position}, position : {position}  velocity: {velocity}', file=sys.stderr, flush=True)
        velocity_angle, speed = get_relative_angle_and_distance(pod.velocity, pod.angle)
        checkpoint_position = checkpoints[pod.next_checkpoint_id]
        checkpoint_angle, checkpoint_distance = get_relative_angle_and_distance(checkpoint_position - pod.position, pod.angle)
        # Check my view of angle to checkpoint vs the game - note game takes things to the R as +ve angles
        print(f'My angle to cp : {-round(math.degrees(checkpoint_angle))}, system angle to cp : {round(math.degrees(pod.angle))}', file=sys.stderr, flush=True)
        #print(f'My angle : {round(math.degrees(pod.angle))}, velocity angle : {round(math.degrees(velocity_angle))}', file=sys.stderr, flush=True)
        next_checkpoint_position = checkpoints[(pod.next_checkpoint_id + 1) % checkpoint_count]
        next_checkpoint_angle, next_checkpoint_distance = get_relative_angle_and_distance(next_checkpoint_position - pod.position, pod.angle)
        nn_inputs = transform_race_data_to_nn_inputs(velocity_angle, speed, checkpoint_angle, checkpoint_distance,
                                                     next_checkpoint_angle, next_checkpoint_distance)
        print(f'Raw inputs: velocity angle {math.degrees(velocity_angle)}, speed : {speed}  checkpointAngle: {math.degrees(checkpoint_angle)}  Checkpoint dist: {checkpoint_distance} Next ch angle: {math.degrees(next_checkpoint_angle)}  Next cp dist: {next_checkpoint_distance}', file=sys.stderr, flush=True)
        nn_outputs = racer.evaluate(nn_inputs)
        #print(f'NN inputs: {nn_inputs}', file=sys.stderr, flush=True)
        #print(f'NN outputs: {nn_outputs}', file=sys.stderr, flush=True)

        steer, thrust, command = transform_nn_outputs_to_instructions(nn_outputs)
        thrust = round(thrust)
        print(f'Steer: {round(math.degrees(steer))} Thrust: {thrust} Command: {command}', file=sys.stderr, flush=True)
        target_angle = pod.angle + steer

        # On the first go through, override the calculated values
        if last_positions is None:
            target_angle = pod.angle
            thrust = 100

        # Record state
        if last_positions is not None:
            print(x == sim_pos[pod_index][0], y == sim_pos[pod_index][1], file=sys.stderr, flush=True)
            print('x : ' + str(x) + '  sim_x : ' + str(sim_pos[pod_index][0]) + '  y : ' + str(y) + '  sim_y : ' + str(sim_pos[pod_index][1]), file=sys.stderr, flush=True)

        # Simulate move - simulator works in radians
        sim_pos[pod_index], velocity, pod_angle, hit_checkpoint = simulators[pod_index].single_step(pod.position, pod.velocity, pod.angle, checkpoint_position, target_angle, thrust, command)

        # Output the target position followed by the power (0 <= thrust <= 100)
        target_position = list(map(round, pod.position + 10000 * np.array((math.sin(target_angle), math.cos(target_angle)))))
        if command == 'BOOST':
            thrust = 100 if pod.boost_used else 'BOOST'
            pod.boost_used = True
        outputs.append(np.append(target_position, thrust))

    # Store current positions
    last_positions = [pod.position for pod in pods]

    print(*outputs[0], 'Beep beep')
    print(*outputs[1], 'Meep meep')
