import matplotlib.pyplot as plt
from pydub import AudioSegment
from tempfile import mktemp
from scipy.io import wavfile

mp3_audio = AudioSegment.from_file("data/piano.mp3", format='mp3')
w_name = mktemp('temp.wav')
mp3_audio.export(w_name, format='wav')

FS, data = wavfile.read(w_name)

plt.specgram(data[:, 1], Fs=FS, NFFT=pow(2, 20))
plt.show()
