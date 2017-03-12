from numpy import *
from scipy import fft
from scipy.io.wavfile import write
import os

output_file = "test.wav"
samples_per_sec = 44100

time = 1.5 # s
samples_total = samples_per_sec*time
t = linspace(0, time, samples_total)

def tone(f):
    omega = 2*pi*f # RAD/s
    signal = exp(-0.8*t)*sum(exp(0.1*k)*sin(omega*t/k) for k in range(1,4))
    return signal

# hamming(samples_total)
def seq(sounds):
    return hstack(sounds)

def play(signal):
    signal = int16(signal/max(abs(signal)) * 32767)
    write(output_file, samples_per_sec, signal)
    os.system("cvlc " + output_file)

base_freq = 440
step_ratio = 2**(1/12.)
piano = [base_freq*step_ratio**(n - 49) for n in range(1,88)]
play(seq(map(tone, piano)))

# display spectrum
#spectrum = abs(fft(signal))
#spectrum = spectrum[:spectrum.size//2]
#plot(linspace(0, float(samples_total)/time, spectrum.size), spectrum)
#show()

