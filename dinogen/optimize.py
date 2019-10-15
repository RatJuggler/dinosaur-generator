from .rnn import rnn_forward, rnn_backward


def clip(gradients, maxValue):
    dWaa, dWax, dWya, db, dby = gradients['dWaa'], gradients['dWax'], gradients['dWya'], gradients['db'], gradients['dby']

    # clip to mitigate exploding gradients, loop over [dWax, dWaa, dWya, db, dby]. (≈2 lines)
    for gradient in [dWax, dWaa, dWya, db, dby]:
        index = gradient < -maxValue
        gradient[index] = -maxValue
        index = gradient > maxValue
        gradient[index] = maxValue

    gradients = {"dWaa": dWaa, "dWax": dWax, "dWya": dWya, "db": db, "dby": dby}

    return gradients


def update_parameters(parameters, gradients, lr):
    parameters['Wax'] += -lr * gradients['dWax']
    parameters['Waa'] += -lr * gradients['dWaa']
    parameters['Wya'] += -lr * gradients['dWya']
    parameters['b'] += -lr * gradients['db']
    parameters['by'] += -lr * gradients['dby']
    return parameters


def optimize(X, Y, a_prev, parameters, vocab_size, learning_rate):
    # Forward propagate through time
    loss, cache = rnn_forward(X, Y, a_prev, parameters, vocab_size)
    # Back-propagate through time
    gradients, a = rnn_backward(X, Y, parameters, cache)
    # Clip your gradients between -5 (min) and 5 (max)
    gradients = clip(gradients, 5)
    # Update parameters
    update_parameters(parameters, gradients, learning_rate)
    return loss, gradients, a[len(X) - 1]
