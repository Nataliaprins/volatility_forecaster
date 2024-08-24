import numpy as np
import pandas as pd
from arch import arch_model

from volatility_forecaster.arch_model.get_garch_param_list import get_garch_param_list


def simulate_data(param_combinations, sim_parameters, nobs):

    garch_param_list = get_garch_param_list(param_combinations)
    sim_model = arch_model(None, p=garch_param_list[0], q=garch_param_list[1])
    sim_data = sim_model.simulate(sim_parameters, nobs=nobs)

    return sim_data
