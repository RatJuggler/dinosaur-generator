import pathlib

from .model import model


def read_sample_file():
    sample_file = pathlib.Path(__file__).parent / "dinosaurs.csv"
    with sample_file.open('r', encoding='utf-8') as f:
        sample_names = f.readlines()
    sample_names = [x.lower().strip() for x in sample_names]
    return sample_names


def generate():
    sample_names = read_sample_file()
    name_chars = set('\n'.join(sample_names))
    text_size, vocab_size = len(sample_names), len(name_chars)
    print("There are {0} total characters and {1} unique characters in the sample name data."
          .format(text_size, vocab_size))
    char_to_index = {ch: i for i, ch in enumerate(sorted(name_chars))}
    index_to_char = {i: ch for i, ch in enumerate(sorted(name_chars))}
    print(char_to_index)
    print(index_to_char)
    model(sample_names, index_to_char, char_to_index, len(name_chars), 50, 7, 40000)


if __name__ == "__main__":
    generate()
