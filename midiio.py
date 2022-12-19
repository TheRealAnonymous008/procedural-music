from pydub import AudioSegment
from scipy.io import wavfile
from tempfile import mktemp


def read_file(filename: str):
    mp3_audio = AudioSegment.from_file(filename, format='mp3')
    w_name = mktemp('temp.wav')
    mp3_audio.export(w_name, format='wav')
    return wavfile.read(w_name)

