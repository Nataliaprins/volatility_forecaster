import keras_tuner as kt


def tuning_params(model, num_max_epochs, model_name):
    tuner = kt.Hyperband(
        model,
        objective="val_loss",
        max_epochs=num_max_epochs,
        factor=3,
        project_name=str(model_name),
        overwrite=True,
    )
    return tuner
