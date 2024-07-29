from TrainPodRacer import simulate_race, train_pod_racer, simulate_race_with_checkpoints
import numpy as np
import cProfile

# Initial configuration
#train_pod_racer(0, None).pickle_neuron_config('basic_trained_racer.txt')

# Next generation
#train_pod_racer(0, None).pickle_neuron_config('nn_saved.txt')
#train_pod_racer(50, 'nn_saved.txt').pickle_neuron_config('nn_saved2.txt')
#train_pod_racer(50, 'nn_saved2.txt').pickle_neuron_config('nn_saved3.txt')
#train_pod_racer(50, 'nn_saved3.txt').pickle_neuron_config('nn_saved4.txt')
#train_pod_racer(50, 'nn_saved4.txt').pickle_neuron_config('nn_saved5.txt')
#train_pod_racer(50, 'nn_saved5.txt').pickle_neuron_config('nn_saved6.txt')
#train_pod_racer(250, 'nn_saved6.txt').pickle_neuron_config('nn_saved7.txt')
#train_pod_racer(250, 'nn_saved7.txt').pickle_neuron_config('nn_saved8.txt')
#train_pod_racer(250, True, 'nn_saved8.txt').pickle_neuron_config('nn_saved9.txt')
#train_pod_racer(250, True, 'nn_saved10.txt').pickle_neuron_config('nn_saved11.txt')
#train_pod_racer(100, True, 'nn_saved11.txt').pickle_neuron_config('nn_saved12.txt')
#train_pod_racer(25, False, 'nn_saved12.txt').pickle_neuron_config('nn_saved13.txt')
#train_pod_racer(25, True, 'nn_saved13.txt').pickle_neuron_config('nn_saved14.txt')
#train_pod_racer(25, True, 'nn_saved14.txt').pickle_neuron_config('nn_saved15.txt')
#train_pod_racer(25, False, '').pickle_neuron_config('nn_raw.txt')
#train_pod_racer(25, True, 'nn_saved15.txt').pickle_neuron_config('nn_saved16.txt')
train_pod_racer(100, True, 'nn_saved16.txt').pickle_neuron_config('nn_saved17.txt')



#train_pod_racer(100, 'nn_saved8.txt').pickle_neuron_config('nn_saved9.txt')
#train_pod_racer(100, 'nn_saved9.txt').pickle_neuron_config('nn_saved10.txt')
#train_pod_racer(100, 'nn_saved10.txt').pickle_neuron_config('nn_saved11.txt')

# Simulate race
#simulate_race_with_checkpoints('nn_raw.txt', [np.array((13499, 2006)), np.array((5507, 7523)), np.array((4517, 5984))], np.array((13215, 1595)))


# Profile code
#cProfile.run("train_pod_racer(1, 'nn_saved6.txt').pickle_neuron_config('nn_saved7.txt')")