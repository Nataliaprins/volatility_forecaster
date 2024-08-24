import numpy as np


def reshape_parameters(sim_parameters):
    if len(np.shape(sim_parameters)) != 1:
        sim_parameters = np.ravel(sim_parameters)
        print(f"Forma de sim_parameters después de aplanar: {np.shape(sim_parameters)}")

    return sim_parameters
