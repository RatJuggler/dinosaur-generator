import numpy as np

from .optimize import optimize
from .sample import sample


def initialize_parameters(n_a, n_x, n_y):
    np.random.seed(1)
    Wax = np.random.randn(n_a, n_x) * 0.01  # input to hidden
    Waa = np.random.randn(n_a, n_a) * 0.01  # hidden to hidden
    Wya = np.random.randn(n_y, n_a) * 0.01  # hidden to output
    b = np.zeros((n_a, 1))  # hidden bias
    by = np.zeros((n_y, 1))  # output bias
    parameters = {"Wax": Wax, "Waa": Waa, "Wya": Wya, "b": b, "by": by}
    return parameters


def get_initial_loss(vocab_size, seq_length):
    return -np.log(1.0 / vocab_size) * seq_length


def smooth(loss, new_loss):
    return loss * 0.999 + new_loss * 0.001


def print_samples(to_generate, parameters, vocab_to_index, index_to_vocab):
    # The seed for the sample to print.
    seed = 0
    for name in range(to_generate):
        # Sample indices and print them.
        sampled_indices = sample(parameters, vocab_to_index['\n'], seed)
        sampled_name = ''.join(index_to_vocab[i] for i in sampled_indices)
        print(sampled_name.capitalize(), end='')
        seed += 1  # To get the next sample increment the seed by one.


def model(training_names, vocab_to_index, index_to_vocab, vocab_size, n_a, to_generate, iterations):
    # Set n_x and n_y to the vocab_size.
    n_x, n_y = vocab_size, vocab_size
    # Initialize parameters.
    parameters = initialize_parameters(n_a, n_x, n_y)
    # Initialize loss (this is required because we want to smooth our loss).
    loss = get_initial_loss(vocab_size, to_generate)
    # Shuffle the list of training names.
    np.random.seed(0)
    np.random.shuffle(training_names)
    # Initialize the hidden state of the LSTM.
    a_prev = np.zeros((n_a, 1))
    # Optimization training loop.
    for j in range(iterations):
        # Select the next name to use for training.
        train_on = j % len(training_names)
        # Split the name into a vocabulary vector.
        X = [None] + [vocab_to_index[ch] for ch in training_names[train_on]]
        Y = X[1:] + [vocab_to_index["\n"]]
        # Perform one optimization step: Forward-prop -> Backward-prop -> Clip -> Update parameters
        # Choose a learning rate of 0.01
        new_loss, gradients, a_prev = optimize(X, Y, a_prev, parameters, vocab_size, 0.01)
        # Use a latency trick to keep the loss smooth. It happens here to accelerate the training.
        loss = smooth(loss, new_loss)
        # Every 2000 Iteration, generate "n" samples to check if the model is learning properly.
        if j % 2000 == 0:
            print("Iteration: {0} - Loss: {1}".format(j, loss))
            print_samples(to_generate, parameters, vocab_to_index, index_to_vocab)
            print('\n')
    return parameters
