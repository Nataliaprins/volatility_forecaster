def param_combinations_to_string(param_combinations):
    param_combinations_str = {
        key: str(value) for key, value in param_combinations.items()
    }

    return param_combinations_str
