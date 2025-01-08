def get_garch_param_list(param_combinations):
    list_garch_params = ["p", "o", "q"]
    filtered_param_dict = {
        k: v for k, v in param_combinations.items() if k in list_garch_params
    }

    garch_param_list = [
        filtered_param_dict["p"],
        filtered_param_dict["q"],
    ]

    return garch_param_list
