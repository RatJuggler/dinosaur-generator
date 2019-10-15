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
    return -np.log(1.0/vocab_size)*seq_length


def smooth(loss, cur_loss):
    return loss * 0.999 + cur_loss * 0.001


def print_sample(sample_ix, ix_to_char):
    txt = ''.join(ix_to_char[ix] for ix in sample_ix)
    txt = txt[0].upper() + txt[1:]  # capitalize first character
    print('%s' % (txt, ), end='')


def model(sample_names, ix_to_char, char_to_ix, vocab_size, n_a=50, to_generate=7, iterations=35000):
    # Retrieve n_x and n_y from vocab_size
    n_x, n_y = vocab_size, vocab_size
    # Initialize parameters
    parameters = initialize_parameters(n_a, n_x, n_y)
    # Initialize loss (this is required because we want to smooth our loss, don't worry about it)
    loss = get_initial_loss(vocab_size, to_generate)
    # Shuffle list of sample names
    np.random.seed(0)
    np.random.shuffle(sample_names)
    # Initialize the hidden state of your LSTM
    a_prev = np.zeros((n_a, 1))
    # Optimization loop
    for j in range(iterations):
        # Use the hint above to define one training example (X,Y) (≈ 2 lines)
        index = j % len(sample_names)
        X = [None] + [char_to_ix[ch] for ch in sample_names[index]]
        Y = X[1:] + [char_to_ix["\n"]]
        # Perform one optimization step: Forward-prop -> Backward-prop -> Clip -> Update parameters
        # Choose a learning rate of 0.01
        curr_loss, gradients, a_prev = optimize(X, Y, a_prev, parameters, vocab_size, learning_rate=0.01)
        # Use a latency trick to keep the loss smooth. It happens here to accelerate the training.
        loss = smooth(loss, curr_loss)
        # Every 2000 Iteration, generate "n" characters thanks to sample() to check if the model is learning properly
        if j % 2000 == 0:
            print('Iteration: %d, Loss: %f' % (j, loss) + '\n')
            # The number of dinosaur names to print
            seed = 0
            for name in range(to_generate):
                # Sample indices and print them
                sampled_indices = sample(parameters, char_to_ix, seed)
                print_sample(sampled_indices, ix_to_char)
                seed += 1  # To get the same result for grading purposed, increment the seed by one.
            print('\n')
    return parameters
