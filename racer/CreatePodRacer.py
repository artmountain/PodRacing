# INSERT NEURAL NETWORK CODE

# INSERT GAME FUNCTIONS

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
