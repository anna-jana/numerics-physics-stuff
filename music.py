import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io.wavfile import write
from scipy.fftpack import ifft, fft, fftshift, ifftshift, fftfreq

# sampling
rate = 48000 # [SAMPLES/s]
time_step = 1 / rate

# envelope
max_attack_amp = 0.11
attack_duration = 0.05 # [s]
decay_duration = 0.01 # [s]

sustain_amp = 0.2
sustain_duration = 0.3

release_time_scale = 0.1
release_duration = 0.3

t_attack  = 0
t_decay   = attack_duration
t_sustain = t_decay + decay_duration
t_release = t_sustain + sustain_duration
note_duration = attack_duration + decay_duration + sustain_duration + release_duration

note_num_samples = int(note_duration * rate)
i = np.arange(note_num_samples)
t_s = time_step * i

envelope = np.empty(note_num_samples)

for j, t in enumerate(t_s):
    if t_attack <= t < t_decay:
        envelope[j] = max_attack_amp / attack_duration * t
    elif t_decay <= t < t_sustain:
        envelope[j] = max_attack_amp - (max_attack_amp - sustain_amp) / decay_duration * (t - t_decay)
    elif t_sustain <= t < t_release:
        envelope[j] = sustain_amp
    else: # t > t_release
        envelope[j] = sustain_amp * np.exp(- (t - t_release) / release_time_scale)

#plt.plot(t_s, envelope)
#plt.show()

# the scale
N = 12
n = np.arange(N)
freqs = 440 * 2**(n/N) # [Hz]

def sine_wave(f):
    return np.sin(2*np.pi * f * t_s)

# harmonics
num_harmonics = 5
harmonic_decay = 2

def harmonic(k, l):
    return harmonic_decay**(- l) * sine_wave(freqs[k] * l)

def note(k):
    return sum(harmonic(k, l) for l in range(num_harmonics))

# generate signal
num_samples = N * note_num_samples
signal = np.empty(num_samples)
for k in range(N):
    note_signal = note(k) * envelope
    signal[note_num_samples * k : note_num_samples * (k + 1)] = note_signal

# play
filename = "sound.wav"
write(filename, rate, signal)
os.system(f"vlc {filename}")

