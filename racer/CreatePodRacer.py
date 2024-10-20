GOLD = True

template_file = '../../PodRacing/racer/' + 'PodRacerTemplateGold.py' if GOLD else 'PodRacerTemplate.py'
output_file = '../racer/codingame_file/' + 'PodRacerGold.py' if GOLD else 'PodRacer.py'


if __name__ == '__main__':
    with open(output_file, 'w') as racer:
        with open(template_file, 'r') as template:
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
                elif '# INSERT SIMULATOR' in line:
                    with open('../PodRaceSimulator.py', 'r') as simulator:
                        for simulator_line in simulator:
                            if not 'import ' in simulator_line:
                                racer.write(simulator_line)
                elif '# INSERT NN CONFIGS' in line:
                    with open('../NeuralNetConfigs.py', 'r') as nn_configs:
                        for nn_configs_line in nn_configs:
                            racer.write(nn_configs_line)
                elif '# REMOVE THIS lINE' not in line:
                    if '% INSERT RACER NN PARAMETERS %' in line:
                        with open('../nn_data/live_racer_nn_config.txt', 'r') as nn_config:
                            line = line.replace('% INSERT RACER NN CONFIG %', nn_config.read())
                    if '% INSERT BLOCKER NN PARAMETERS %' in line:
                        with open('../nn_data/live_blocker_nn_config.txt', 'r') as nn_config:
                            line = line.replace('% INSERT BLOCKER NN CONFIG %', nn_config.read())
                    racer.write(line)
