import pathlib

from .model import model


def read_training_file():
    training_file = pathlib.Path(__file__).parent / "dinosaurs.csv"
    with training_file.open('r', encoding='utf-8') as f:
        training_names = f.readlines()
    training_names = [x.lower().strip() for x in training_names]
    return training_names


def generate():
    training_names = read_training_file()
    training_text = '\n'.join(training_names)
    training_chars = set(training_text)
    vocab_size = len(training_chars)
    print("There are {0} names with {1} total characters and {2} unique characters in the training data."
          .format(len(training_names), len(training_text), vocab_size))
    char_to_index = {ch: i for i, ch in enumerate(sorted(training_chars))}
    index_to_char = {i: ch for i, ch in enumerate(sorted(training_chars))}
    print(char_to_index)
    print(index_to_char)
    model(training_names, index_to_char, char_to_index, vocab_size, 50, 7, 35000)


if __name__ == "__main__":
    generate()
