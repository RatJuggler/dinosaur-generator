import pathlib

from .model import model


def read_training_file():
    training_file = pathlib.Path(__file__).parent / "dinosaurs.csv"
    with training_file.open('r', encoding='utf-8') as f:
        training_names = f.readlines()
    training_names = [x.lower().strip() for x in training_names]
    return training_names


def generate():
    # Load a file of training names and extract the vocabulary that makes up the names.
    training_names = read_training_file()
    training_text = '\n'.join(training_names)
    # For this example the vocabulary consists of the characters that make up each name.
    training_vocab = set(training_text)
    vocab_size = len(training_vocab)
    print("The training data contains {0} names with {1} total characters and {2} unique characters (the vocabulary).\n"
          .format(len(training_names), len(training_text), vocab_size))
    # Create dictionary's of the extracted vocabulary to code to/from indices.
    vocab_to_index = {ch: i for i, ch in enumerate(sorted(training_vocab))}
    print("Map the vocabulary to indices:\n{0}\n".format(vocab_to_index))
    index_to_vocab = {i: ch for i, ch in enumerate(sorted(training_vocab))}
    print("Map indices back to the vocabulary:\n{0}\n".format(index_to_vocab))
    model(training_names, vocab_to_index, index_to_vocab, vocab_size, 50, 7, 35000)


if __name__ == "__main__":
    generate()
