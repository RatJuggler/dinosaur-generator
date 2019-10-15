import pathlib


def read_existing_names():
    existing_names = pathlib.Path(__file__).parent / "dinosaurs.csv"
    with existing_names.open('r', encoding='utf-8') as f:
        raw_text = f.read()
    return raw_text.lower()


def model(name_text, ix_to_char, char_to_ix):
    print("TODO Model...")


def generate():
    name_text = read_existing_names()
    name_chars = list(set(name_text))
    text_size, vocab_size = len(name_text), len(name_chars)
    print('There are %d total characters and %d unique characters in your data.' % (text_size, vocab_size))
    char_to_ix = {ch: i for i, ch in enumerate(sorted(name_chars))}
    ix_to_char = {i: ch for i, ch in enumerate(sorted(name_chars))}
    print(char_to_ix)
    print(ix_to_char)
    model(name_text, ix_to_char, char_to_ix)


if __name__ == "__main__":
    generate()
