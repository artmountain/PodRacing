# INSERT NEURAL NETWORK CODE

# INSERT GAME FUNCTIONS

""" # REMOVE THIS lINE
# Build neural network
nn_data_str = '%INSERT RACER NN CONFIG%'
nn_data = json.loads(nn_data_str)
neural_network = NeuralNetwork(6, 2, [nn_data['weights_0'], nn_data['weights_1']],
                               [nn_data['biases_0'], nn_data['biases_1']], nn_data['input_scaling'])

# game loop
sim_pos = np.array((0, 0))
velocity = np.zeros(2)
angle = 0
thrust = 0
initialized = 0
while True:
    # next_checkpoint_x: x position of the next check point
    # next_checkpoint_y: y position of the next check point
    # next_checkpoint_dist: distance to the next checkpoint
    # next_checkpoint_angle: angle between your pod orientation and the direction of the next checkpoint
    x, y, next_checkpoint_x, next_checkpoint_y, next_checkpoint_dist, next_checkpoint_angle = [int(i) for i in
                                                                                               input().split()]
    opponent_x, opponent_y = [int(i) for i in input().split()]
    print('My position : ' + str(x) + ' ' + str(y), file=sys.stderr, flush=True)
    print('Checkpoint : ' + str(next_checkpoint_x) + ' ' + str(next_checkpoint_y), file=sys.stderr, flush=True)
    print('Checkpoint angle : ' + str(next_checkpoint_angle) + '   Checkpoint distance : ' + str(next_checkpoint_dist), file=sys.stderr, flush=True)

    next_checkpoint_position = np.array((next_checkpoint_x, next_checkpoint_y))
    position = np.array((x, y))
    if initialized < 5:
        sim_pos = np.array((x, y))
        angle = get_angle(next_checkpoint_position - position)
        target_angle = angle
        thrust = 100
        # print(math.degrees(angle), file=sys.stderr, flush=True)
        initialized += 1
    else:
        # Create neural network inputs
        # [Checkpoint angle, Checkpoint distance, direction, speed, new_angle, thrust]
        # Angles are relative to current heading
        angle_degrees = round(math.degrees(angle))
        direction_and_speed = get_angle_and_distance(velocity)
        checkpoint_angle_and_distance = get_angle_and_distance(next_checkpoint_position - position)
        print(checkpoint_angle_and_distance[0], file=sys.stderr, flush=True)
        checkpoint_angle_and_distance[0] = (checkpoint_angle_and_distance[0] - angle_degrees + 180) % 360
        print(checkpoint_angle_and_distance[0], file=sys.stderr, flush=True)
        next_checkpoint_angle_and_distance = [checkpoint_angle_and_distance[0], checkpoint_angle_and_distance[1] * 2]  # TODO get true value
        velocity_angle = (direction_and_speed[0] - angle_degrees + 180) % 360

        # Get new direction and thrust from neural network
        nn_inputs = checkpoint_angle_and_distance.tolist() + next_checkpoint_angle_and_distance + [velocity_angle, direction_and_speed[1]]
        # print(nn_inputs, file=sys.stderr, flush=True)
        [target_angle, thrust] = neural_network.evaluate(nn_inputs)
        target_angle = (angle_degrees + round(target_angle * 360) - 180) * math.pi / 180
        # print(checkpoint_angle_and_distance[0], round(target_angle * 180 / math.pi), file=sys.stderr, flush=True)
        thrust = round(thrust * 100)
        print(target_angle, thrust, file=sys.stderr, flush=True)

    print(x == sim_pos[0], y == sim_pos[1], file=sys.stderr, flush=True)
    print('x : ' + str(x) + '  sim_x : ' + str(sim_pos[0]) + '  y : ' + str(y) + '  sim_y : ' + str(sim_pos[1]), file=sys.stderr, flush=True)
    sim_pos, velocity, angle, hit_checkpoint = evaluate_game_step(position, velocity, angle, next_checkpoint_position, target_angle, thrust)
    # print('Hit Checkpoint : ' + str(hit_checkpoint), file=sys.stderr, flush=True)

    # Output the target position followed by the power (0 <= thrust <= 100)
    target_position = position + 10000 * np.array((math.sin(angle), math.cos(angle)))
    outputs = map(round, np.append(target_position, thrust))
    print(*outputs, 'Get out of my way')
"""  # REMOVE THIS lINE

# STOP!
if __name__ == '__main__':
    with open('../racer/codingame_file/PodRacer.py', 'w') as racer:
        with open('../../PodRacing/racer/PodRacerTemplate.py', 'r') as template:
            for line in template:
                if '# INSERT NEURAL NETWORK CODE' in line:
                    with open('../NeuralNet.py', 'r') as nn:
                        for nn_line in nn:
                            if 'Fitting below this line' in nn_line:
                                break
                            if not 'matplotlib.pyplot' in nn_line:
                                racer.write(nn_line)
                elif '# INSERT GAME FUNCTIONS' in line:
                    with open('../PodRacerFunctions.py', 'r') as functions:
                        for functions_line in functions:
                            if not 'import ' in functions_line:
                                racer.write(functions_line)
                elif '# REMOVE THIS lINE' not in line:
                    if '% INSERT RACER NN CONFIG %' in line:
                        with open('../nn_data/live_racer_nn_config.txt', 'r') as nn_config:
                            line = line.replace('% INSERT RACER NN CONFIG %', nn_config.read())
                    racer.write(line)
