import sys

# INSERT NEURAL NETWORK CODE

# INSERT GAME FUNCTIONS


# Build neural network
nn_data_str = '% INSERT RACER NN CONFIG %'
racer = NeuralNetwork.create_from_json(nn_data_str)

# game loop
sim_pos = np.array((0, 0))
sim_angle = None
velocity = np.zeros(2)
thrust = 0
target_angle = 0
initialized = False
checkpoint_index = 0
last_checkpoint_position = np.array((-1, -1))
seen_all_checkpoints = False
checkpoints = []
while True:
    # next_checkpoint_x: x position of the next check point
    # next_checkpoint_y: y position of the next check point
    # next_checkpoint_dist: distance to the next checkpoint
    # next_checkpoint_angle: angle between your pod orientation and the direction of the next checkpoint
    x, y, checkpoint_x, checkpoint_y, input_checkpoint_distance, input_checkpoint_angle = [int(i) for i in input().split()]
    opponent_x, opponent_y = [int(i) for i in input().split()]

    # Arrays for my position and position of target checkpoint
    position = np.array((x, y))
    checkpoint_position = np.array((checkpoint_x, checkpoint_y))
    #print('My position : ' + str(x) + ' ' + str(y), file=sys.stderr, flush=True)
    print('Checkpoint : ' + str(checkpoint_x) + ' ' + str(checkpoint_y), file=sys.stderr, flush=True)

    # Update list of checkpoints
    new_checkpoint = True
    for checkpoint_index in range(len(checkpoints)):
        checkpoint = checkpoints[checkpoint_index]
        if checkpoint[0] == checkpoint_x and checkpoint[1] == checkpoint_y:
            new_checkpoint = False
            break
    print(f'Checkpoint index : {checkpoint_index}', file=sys.stderr, flush=True)
    print(f'Checkpoints : {checkpoints}', file=sys.stderr, flush=True)
    if new_checkpoint:
        checkpoints.append(checkpoint_position)
        print(f'New checkpoint', file=sys.stderr, flush=True)

    if not initialized:
        sim_pos[0] = x
        sim_pos[1] = y
        sim_angle = get_angle(checkpoint_position - position)
        print(f'Start angle : {round(math.degrees(sim_angle))}', file=sys.stderr, flush=True)
        steer = 0
        thrust = 100
        target_angle = sim_angle
        initialized = True
    else:
        # Create neural network inputs
        # [Checkpoint angle, Checkpoint distance, direction, speed, new_angle, thrust]
        # Angles are relative to current heading and are all in radians except the input checkpoint angle
        velocity_angle, speed = get_relative_angle_and_distance(velocity, sim_angle)
        checkpoint_angle, checkpoint_distance = get_relative_angle_and_distance(checkpoint_position - position, sim_angle)
        # Check my view of angle to checkpoint vs the game - note game takes things to the R as +ve angles
        print(f'My angle to cp : {-round(math.degrees(checkpoint_angle))}, system angle to cp : {input_checkpoint_angle}', file=sys.stderr, flush=True)
        #print(f'My angle : {round(math.degrees(sim_angle))}, velocity angle : {round(math.degrees(velocity_angle))}', file=sys.stderr, flush=True)
        next_checkpoint_angle = checkpoint_angle
        next_checkpoint_distance = 2 * checkpoint_distance
        if seen_all_checkpoints:
            next_checkpoint_position = checkpoints[(checkpoint_index + 1) % len(checkpoints)]
            next_checkpoint_angle, next_checkpoint_distance = get_relative_angle_and_distance(next_checkpoint_position - position, sim_angle)
        nn_inputs = transform_race_data_to_nn_inputs(velocity_angle, speed, checkpoint_angle, checkpoint_distance,
                                                     next_checkpoint_angle, next_checkpoint_distance)
        print(f'Raw inputs: velocity angle {velocity_angle}, speed : {speed}  chckpointAngle: {checkpoint_angle}  Checkpoint dist: {checkpoint_distance} Next ch angle: {next_checkpoint_angle}  Next cp dist: {next_checkpoint_distance}', file=sys.stderr, flush=True)
        nn_outputs = racer.evaluate(nn_inputs)
        print(f'NN inputs: {nn_inputs}', file=sys.stderr, flush=True)
        print(f'NN outputs: {nn_outputs}', file=sys.stderr, flush=True)

        nn_outputs = racer.evaluate(nn_inputs)
        steer, thrust = transform_nn_outputs_to_instructions(nn_outputs)
        print(f'Steer: {round(math.degrees(steer))} Thrust: {thrust}', file=sys.stderr, flush=True)
        # Grid in Codingame is inverted - need to subtract the steer angle
        target_angle = sim_angle - steer

    # Record state
    print(x == sim_pos[0], y == sim_pos[1], file=sys.stderr, flush=True)
    print('x : ' + str(x) + '  sim_x : ' + str(sim_pos[0]) + '  y : ' + str(y) + '  sim_y : ' + str(sim_pos[1]),
          file=sys.stderr, flush=True)

    # Simulate move - simulator works in radians
    #print(f'Sim angle : {round(math.degrees(sim_angle))}  target angle: {round(math.degrees(target_angle))}', file=sys.stderr, flush=True)
    sim_pos, velocity, sim_angle, hit_checkpoint = evaluate_game_step(position, velocity, sim_angle, checkpoint_position, target_angle, thrust)
    #print(f'Sim angle after : {round(math.degrees(sim_angle))}', file=sys.stderr, flush=True)
    if hit_checkpoint:
        checkpoint_index += 1
    #print(f'Commands : {np.around(np.array((steer, thrust)), 2).tolist()}', file=sys.stderr, flush=True)
    #print(f'Next checkpoint : {checkpoint_index}', file=sys.stderr, flush=True)

    # Output the target position followed by the power (0 <= thrust <= 100)
    #print(f'Target angle : {round(math.degrees(target_angle))}', file=sys.stderr, flush=True)
    target_position = position + 10000 * np.array((math.sin(target_angle), math.cos(target_angle)))
    outputs = map(round, np.append(target_position, thrust))
    print(*outputs, 'Get out of my way')
