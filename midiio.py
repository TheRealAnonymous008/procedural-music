import librosa


def read_file(filename: str):
    return librosa.load(filename)

