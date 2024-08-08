from TrainPodBlocker import train_pod_blocker
from TrainPodRacer import train_pod_racer

# Racer training
#train_pod_racer('nn_data/racer_config.txt', None)
#train_pod_racer('nn_data/racer_config2.txt', 'nn_data/racer_config.txt')
#train_pod_racer('nn_data/racer_config3.txt', 'nn_data/racer_config2.txt')
#train_pod_racer('nn_data/racer_config4.txt', 'nn_data/racer_config3.txt')

# Display race
#generate_and_display_race(open('nn_data/racer_config.txt').readlines()[0])

# Blocker training
train_pod_blocker('nn_data/live_racer_nn_config.txt', 'nn_data/blocker_config.txt', None)
#train_pod_blocker('nn_data/live_racer_nn_config.txt', 'nn_data/blocker_config2.txt', 'nn_data/blocker_config.txt'):

