from utils.constants import *
import librosa

def read_file(path : str): 
    return librosa. load(path, sr= 44100, mono=DEFAULT_SAMPLING_RATE)