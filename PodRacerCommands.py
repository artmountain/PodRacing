from DisplayRace import generate_and_display_race
from TrainPodRacer import train_pod_racer

# Initial population
train_pod_racer('../PodRacing/nn_data/racer_config.txt', None)

# Evolve the population
#train_pod_racer('nn_data/racer_config2.txt', 'nn_data/racer_config.txt')
#train_pod_racer('nn_data/racer_config3.txt', 'nn_data/racer_config2.txt')
#train_pod_racer('nn_data/racer_config4.txt', 'nn_data/racer_config3.txt')

#generate_and_display_race(open('nn_data/racer_config4.txt').readlines()[0])

#test_genetic_optimizer(False, True)
#generate_training_set_for_neural_net()
#plot_training_data(True)
#plot_training_data(False)
#train_neural_network_for_pod_racer()
#test_neural_network_outputs()
#test_drive_with_neural_network()
