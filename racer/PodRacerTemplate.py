import sys

# INSERT NEURAL NETWORK CODE

# INSERT GAME FUNCTIONS


# Build neural network
nn_data_str = '% INSERT RACER NN CONFIG %'
racer = NeuralNetwork.create_from_json(nn_data_str)

# game loop
sim_pos = np.array((0, 0))
velocity = np.zeros(2)
angle = 0
thrust = 0
initialized = 0
checkpoint_index = 0
while True:
    # next_checkpoint_x: x position of the next check point
    # next_checkpoint_y: y position of the next check point
    # next_checkpoint_dist: distance to the next checkpoint
    # next_checkpoint_angle: angle between your pod orientation and the direction of the next checkpoint
    x, y, checkpoint_x, checkpoint_y, checkpoint_dist, checkpoint_angle = [int(i) for i in input().split()]
    opponent_x, opponent_y = [int(i) for i in input().split()]
    print('My position : ' + str(x) + ' ' + str(y), file=sys.stderr, flush=True)
    print('Checkpoint : ' + str(checkpoint_x) + ' ' + str(checkpoint_y), file=sys.stderr, flush=True)
    #print('Checkpoint angle : ' + str(checkpoint_angle) + '   Checkpoint distance : ' + str(checkpoint_dist), file=sys.stderr, flush=True)

    checkpoint_position = np.array((checkpoint_x, checkpoint_y))
    position = np.array((x, y))
    if initialized < 5:  # todo remove this
        sim_pos = np.array((x, y))
        angle = get_angle(checkpoint_position - position)
        target_angle = angle
        thrust = 100
        # print(math.degrees(angle), file=sys.stderr, flush=True)
        initialized += 1
    else:
        # Create neural network inputs
        # [Checkpoint angle, Checkpoint distance, direction, speed, new_angle, thrust]
        # Angles are relative to current heading
        velocity_angle, speed = get_relative_angle_and_distance(velocity, angle)
        checkpoint_angle, checkpoint_distance = get_relative_angle_and_distance(checkpoint_position - position, angle)
        next_checkpoint_angle, next_checkpoint_distance = get_relative_angle_and_distance(
            checkpoint_position - position, angle)  # todo
        nn_inputs = transform_race_data_to_nn_inputs(velocity_angle, speed, checkpoint_angle, checkpoint_distance,
                                                     next_checkpoint_angle, next_checkpoint_distance)
        nn_outputs = racer.evaluate(nn_inputs)
        steer, thrust = transform_nn_outputs_to_instructions(nn_outputs)

        target_angle = angle + steer

    print(x == sim_pos[0], y == sim_pos[1], file=sys.stderr, flush=True)
    print('x : ' + str(x) + '  sim_x : ' + str(sim_pos[0]) + '  y : ' + str(y) + '  sim_y : ' + str(sim_pos[1]),
          file=sys.stderr, flush=True)
    print(np.around(np.array((target_angle, thrust)), 2).tolist(), file=sys.stderr, flush=True)
    sim_pos, velocity, angle, hit_checkpoint = evaluate_game_step(position, velocity, angle, checkpoint_position,
                                                                  target_angle, thrust)
    if hit_checkpoint:
        checkpoint_index += 1
    print(f'Next checkpoint : {checkpoint_index}', file=sys.stderr, flush=True)

    # Output the target position followed by the power (0 <= thrust <= 100)
    target_position = position + 10000 * np.array((math.sin(angle), math.cos(angle)))
    outputs = map(round, np.append(target_position, thrust))
    print(*outputs, 'Get out of my way')
