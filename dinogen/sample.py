import numpy as np

from .utils import softmax


def sample(parameters, newline_index, seed):
    # Retrieve parameters and relevant shapes from "parameters" dictionary
    Waa, Wax, Wya, by, b = parameters['Waa'], parameters['Wax'], parameters['Wya'], parameters['by'], parameters['b']
    vocab_size = by.shape[0]
    n_a = Waa.shape[1]
    # Step 1: Create the one-hot vector x for the first character (initializing the sequence generation).
    x = np.zeros((vocab_size, 1))
    # Step 1': Initialize a_prev as zeros
    a_prev = np.zeros((n_a, 1))

    # Create an empty list to contain the indices generated to form a name.
    result = []

    # The selected index of the next character in the name.
    index = -1

    # Loop over time-steps t. At each time-step, sample a character from a probability distribution and append
    # its index to "result". We'll stop if we reach 50 characters (which should be very unlikely with a well
    # trained model), which will prevent us entering an infinite loop.
    counter = 0

    while index != newline_index and counter < 50:
        # Step 2: Forward propagate x using the equations (1), (2) and (3)
        a = np.tanh(np.dot(Wax, x) + np.dot(Waa, a_prev) + b)
        z = np.dot(Wya, a) + by
        y = softmax(z)
        # for grading purposes
#        np.random.seed(counter + seed)
        # Step 3: Sample the index of a character within the vocabulary from the probability distribution y
        index = np.random.choice(list(range(vocab_size)), p=y.ravel())
        # Append the generated index to the result.
        result.append(index)
        # Step 4: Re-create x with the sampled index so it becomes the input for the next iteration.
        x = np.zeros((vocab_size, 1))
        x[index] = 1
        # Update "a_prev" to be "a"
        a_prev = a
        # for grading purposes
        seed += 1
        counter += 1

    if counter == 50:
        result.append(newline_index)

    return result
