def generate_parameter_string(param_combinations):
    parameters = "_".join(
        [f"{key}:{value}" for key, value in param_combinations.items()]
    )

    return parameters
