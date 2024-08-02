from DisplayRace import generate_and_display_race
from TrainPodRacer import train_pod_racer

# Initial population
#train_pod_racer('../PodRacing/nn_data/racer_config.txt', None)

# Evolve the population
#train_pod_racer('nn_data/racer_config2.txt', 'nn_data/racer_config.txt')
train_pod_racer('nn_data/racer_config3.txt', 'nn_data/racer_config2.txt')
#train_pod_racer('nn_data/racer_config4.txt', 'nn_data/racer_config3.txt')

#generate_and_display_race(open('nn_data/racer_config2.txt').readlines()[0])
