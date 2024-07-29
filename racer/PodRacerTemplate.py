import sys

# INSERT NEURAL NETWORK CODE

# INSERT GAME FUNCTIONS


# Build neural network
nn_data_str = '{"weights_0": [[-0.3987265797799542, -0.17407220204341273, -0.4845179187634501, -0.36977557217291235, -0.4190792174732678, -0.5940213994168384], [-0.4903635201439646, -0.10690330425686506, -0.5292902910192154, -0.20404985454740487, -0.5712182128569357, -0.4841427717053779], [-0.4296185944917154, -0.2529644069102661, -0.5087192813535565, -0.3178591743839243, -0.374772837361659, -0.5889064999096321], [-0.24713175172668256, -0.3596416739056198, -0.41130249882326464, -0.6603107734454461, -0.24579331071768237, -0.7083529708552363], [-0.5481559173493218, -0.21682881506800444, -0.4329994952968969, -0.24669019704535916, -0.505929161553728, -0.4524436767040857], [-0.28505636803194156, -0.18508740263558696, -0.4319453297245097, -0.477585980370779, -0.40158164840485544, -0.5776815796973052], [-0.4542605578493702, -0.2440269856829377, -0.45066448606977827, -0.35809682038981283, -0.3701879447213677, -0.5354173429078161], [-0.42479590789376254, -0.16598596119909353, -0.47191762969324713, -0.3255053390201331, -0.517646282967048, -0.4819998637389056]], "biases_0": [-1.1057236361522982, -1.11206815918043, -1.1216439729478742, -1.0350917196612286, -1.1132240621248963, -1.1377692250311555, -1.1221694865779896, -1.1414101027050065], "weights_1": [[-1.268551014801275, -1.1633826514557668, -1.1763447813085157, -1.203928376304715, -1.260838855695839, -1.3265139606132907, -1.313207188867517, -1.202657173452028], [-0.046654423952866536, 0.15061564069214226, -0.027251661385409548, -0.3550506638178381, 0.121367514629782, -0.14820284781225881, -0.026864071802762103, 0.039281967778453504]], "biases_1": [-4.08216293156321, 0.21407039748160284], "input_scaling": [359, 15529, 357, 13928, 359, 459]}'
nn_data = json.loads(nn_data_str)
neural_network = NeuralNetwork(6, 2, [nn_data['weights_0'], nn_data['weights_1']],
                               [nn_data['biases_0'], nn_data['biases_1']])

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
        velocity_angle, speed = get_relative_angle_and_distance(velocity, angle_degrees)
        checkpoint_angle_and_distance = get_relative_angle_and_distance(next_checkpoint_position - position, angle_degrees)
        print(checkpoint_angle_and_distance[0], file=sys.stderr, flush=True)
        print(checkpoint_angle_and_distance[0], file=sys.stderr, flush=True)
        next_checkpoint_angle_and_distance = [checkpoint_angle_and_distance[0], checkpoint_angle_and_distance[1] * 2]  # TODO get true value

        # Get new direction and thrust from neural network
        nn_inputs = checkpoint_angle_and_distance.tolist() + next_checkpoint_angle_and_distance + [velocity_angle, speed]
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
