def generate_run_name(param_combinations_str):
    run_name = "_".join(
        [f"{key}:{value}" for key, value in param_combinations_str.items()]
    )

    return run_name
