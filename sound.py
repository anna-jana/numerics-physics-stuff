import numpy as np
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt

# load the data
dfile = "test.wav"
sample_rate, channels = wav.read(dfile)

# two channel file -> one signal
signal = (channels[:, 0] + channels[:, 1])/2.0

# the take the dft for a lot of subsignals
parts = 1000
part_length = signal.size // parts

num_freqs = part_length//2
num_time = parts
spectrum = np.zeros((num_freqs, num_time))

for i in range(parts):
    data = signal[i*part_length:(i + 1)*part_length]
    freqs = np.fft.fft(data)
    freqs = np.abs(freqs[:num_freqs])
    spectrum[:, i] = freqs

# plot it
time = part_length*np.arange(parts)/float(sample_rate)
freq = np.fft.fftfreq(part_length, 1.0/sample_rate)
freq = freq[:num_freqs] # remove negative frequencies

interesting = 500
plt.pcolormesh(time, freq[:interesting], spectrum[:interesting,:], cmap="hot")
plt.colorbar()
plt.xlabel("time [s]")
plt.ylabel("freq [Hz]")
plt.show()
