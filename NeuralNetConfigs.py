# NN config for racer and blocker. First entry is size of inputs. Others are number of nodes in each layer

# Racer
RACER_NN_INPUTS = 6
RACER_NN_OUTPUTS = 3
RACER_MID_LAYER_SIZE = 4
RACER_NN_CONFIG = [RACER_NN_INPUTS, RACER_NN_INPUTS, RACER_MID_LAYER_SIZE, RACER_MID_LAYER_SIZE, RACER_NN_OUTPUTS]

# Blocker
BLOCKER_NN_INPUTS = 8
BLOCKER_NN_CONFIG = [BLOCKER_NN_INPUTS, BLOCKER_NN_INPUTS, RACER_MID_LAYER_SIZE, RACER_MID_LAYER_SIZE, RACER_NN_OUTPUTS]