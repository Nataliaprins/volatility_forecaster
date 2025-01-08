def _convert_to_3d(xtrain, xtest):
    xtrain = xtrain.reshape(xtrain.shape[0], xtrain.shape[1], 1)
    xtest = xtest.reshape(xtest.shape[0], xtest.shape[1], 1)
    return xtrain, xtest
