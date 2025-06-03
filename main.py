"""
#  Project: CA of signal processing
#  Author: Mahdy Mokhtari
#  SID: 810101515
#
#
"""

import numpy as np
from scipy.io.wavfile import read
from scipy.signal import correlate
from scipy.io.wavfile import write
from scipy.fft import fft, fftshift
import matplotlib.pyplot as plt
import pandas as pd
from scipy.io import wavfile
from notes import noteHarryPotter
from note_frequencies import note_freqs
from my_optional_notes import optional_notes
from scipy.signal import find_peaks


sample_rate = 44100  # freq of getting the samples
silence_duration = 0.025  # in seconds
silence_time = int(sample_rate * silence_duration)

silence_samples = np.zeros(silence_time)

scale_factor = 32767


# Function to generate a sine wave for a given frequency and duration
def generate_sine_wave(frequency, duration, sample_rate, amplitude=1.0):
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    wave = amplitude * np.sin(2 * np.pi * frequency * t)
    return wave


sound_sequence = []
note_names = []


for note in noteHarryPotter:
    note_info = note.split()
    note_name = note_info[0] + " " + note_info[1]
    note_names.append(note_name)
    duration = float(note_info[2])

    frequency = note_freqs.get(note_name)

    if frequency:
        note_wave = generate_sine_wave(frequency, duration, sample_rate)

        sound_sequence.extend(note_wave)

        sound_sequence.extend(silence_samples)

sound_sequence = np.array(sound_sequence) * scale_factor
sound_sequence = sound_sequence.astype(np.int16)

print("note names: ", note_names)

write('noteHarryPotter_mehdy.wav', sample_rate, sound_sequence)


#######################################################

#######################################################
# Faze 2

########## part 2.1
note_files = [
    'A#5.wav', 'A5.wav', 'D5.wav', 'D#5.wav', 'E5.wav', 'F5.wav',
    'F#5.wav', 'G#5.wav', 'G5.wav', 'C5.wav', 'C#5.wav', 'B5.wav'
]

fft_dict = {}
FOLDER_PATH = './piano notes/'

for note_file in note_files:

    sr, audio = wavfile.read(FOLDER_PATH+note_file)

    if len(audio.shape) == 2:
        audio = audio.mean(axis=1)

    fft_result = np.fft.fft(audio)

    freqs = np.fft.fftfreq(len(fft_result), 1/sr)

    magnitude = np.abs(fft_result)

    fft_dict[note_file] = {
        'fft_result': fft_result,
        'freqs': freqs,
        'magnitude': magnitude
    }

    plt.figure(figsize=(10, 6))
    plt.plot(freqs[:len(freqs)//10], magnitude[:len(freqs)//10])
    plt.title(f'Frequency Spectrum of {note_file}')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.show()


for note_file, data in fft_dict.items():
    print(f"Note: {note_file}")
    print(f"Frequencies: {data['freqs'][:10]} ...")
    print(f"Magnitudes: {data['magnitude'][:10]} ...")
    print("-" * 50)


####################################
######## part 2.2
fundamental_frequencies = {
    'A#5.wav': 932.328, 'A5.wav': 880.000, 'D5.wav': 587.330, 'D#5.wav': 622.254,
    'E5.wav': 659.255, 'F5.wav': 698.456, 'F#5.wav': 739.989, 'G#5.wav': 830.609,
    'G5.wav': 783.991, 'B5.wav': 987.767, 'C5.wav': 523.251, 'C#5.wav': 554.365
}

harmonic_coefficients_dict = {}

for note_file, data in fft_dict.items():
    fundamental_freq = fundamental_frequencies.get(note_file)

    harmonic_freqs = [fundamental_freq * i for i in range(1, 7)]

    harmonic_coefficients = []
    for h_freq in harmonic_freqs:
        idx = np.argmin(np.abs(data['freqs'] - h_freq))

        harmonic_coefficients.append(data['magnitude'][idx] / data['magnitude'][0])

    harmonic_coefficients_dict[note_file] = harmonic_coefficients

    print(f"Harmonic Coefficients for {note_file}:")
    print(f"Fundamental: {fundamental_freq} Hz")
    print(f"Harmonics: {harmonic_coefficients}")
    print("-" * 50)


#########################
###### part 2.3
# now lets save to excell

harmonics_data = []

for note_file, harmonic_coeffs in harmonic_coefficients_dict.items():
    harmonics_data.append([note_file] + harmonic_coeffs)

df = pd.DataFrame(harmonics_data, columns=['Note', 'Harmonic 1', 'Harmonic 2', 'Harmonic 3', 'Harmonic 4', 'Harmonic 5', 'Harmonic 6'])

df.to_excel('harmonic_coefficients.xlsx', index=False)

#######################
##### part 2.4
# Final step: Optimize sound sequence with harmonics and damping


final_optimized_sequence = []
alpha_damp = 6

for note in optional_notes:

    note_name = note.split()[0] + note.split()[1]
    duration = float(note.split()[2])

    fs, audio = wavfile.read(FOLDER_PATH+note_name+'.wav')

    # If the audio is stereo, convert it to mono by averaging the channels
    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)

    # Normalizing
    audio = audio / np.max(np.abs(audio))

    # Perform FFT to extract the frequencies and magnitudes
    n = len(audio)
    fft_spectrum = np.fft.fft(audio)
    frequencies = np.fft.fftfreq(n, 1/fs)

    # Only consider the positive half of the frequencies
    positive_frequencies = frequencies[:n//2]
    magnitude = np.abs(fft_spectrum)[:n//2]

    # Find peaks in the FFT spectrum (harmonics)
    peaks, _ = find_peaks(magnitude, height=np.max(magnitude)*0.1, distance=100)
    harmonic_frequencies = positive_frequencies[peaks[:6]]  # First 6 harmonics
    harmonic_amplitudes = magnitude[peaks[:6]]

    t = np.linspace(0, duration, int(fs * duration), endpoint=False)

    synthesized_signal = np.zeros_like(t)
    for i in range(len(harmonic_frequencies)):  # Summing the sine signals
        synthesized_signal += harmonic_amplitudes[i] * np.sin(2 * np.pi * harmonic_frequencies[i] * t)

    # Apply a damping factor to the signal
    damping_factor = np.exp(-alpha_damp * t)
    synthesized_signal *= damping_factor

    # Normalize the signal
    synthesized_signal /= np.max(np.abs(synthesized_signal))

    # Append the synthesized signal to the final sequence
    final_optimized_sequence.extend(synthesized_signal)

    final_optimized_sequence.extend(silence_samples)

final_optimized_sequence = np.array(final_optimized_sequence)

wavfile.write('noteOptimized.wav', fs, (final_optimized_sequence * scale_factor).astype(np.int16))  # Convert to 16-bit PCM

print('Done')



###############################

###############################
# faze 3
# optional/score-based part

def generate_reference_wave(note_name, duration, sample_rate, note_freqs):
    frequency = note_freqs.get(note_name)
    if frequency:
        return generate_sine_wave(frequency, duration, sample_rate)
    return np.zeros(int(sample_rate * duration))



def predict_notes_from_wav(wav_file, note_freqs, sample_rate, silence_duration=0.025):
    rate, audio_data = read(wav_file)

    if len(audio_data.shape) > 1:
        audio_data = audio_data.mean(axis=1)

    predicted_notes = []
    note_duration = silence_duration
    previous_note = None
    accumulated_duration = 0

    for start in range(0, len(audio_data), int(sample_rate * note_duration)):
        end = start + int(sample_rate * note_duration)
        segment = audio_data[start:end]

        correlations = {}
        for note_name, freq in note_freqs.items():
            ref_wave = generate_reference_wave(note_name, note_duration, sample_rate, note_freqs)
            correlation = correlate(segment, ref_wave, mode='same')
            correlations[note_name] = np.max(correlation)

        predicted_note = max(correlations, key=correlations.get)

        if predicted_note == previous_note:
            accumulated_duration += note_duration
        else:
            if previous_note is not None:
                predicted_notes.append((previous_note, accumulated_duration))
            previous_note = predicted_note
            accumulated_duration = note_duration

    if previous_note is not None:
        predicted_notes.append((previous_note, accumulated_duration))

    return predicted_notes


def write_predicted_notes_to_file(predicted_notes, file_name="predictedNotes.txt"):
    with open(file_name, "w") as f:
        for note, duration in predicted_notes:
            f.write(f"{note} {duration}\n")


wav_file = 'noteHarryPoter.wav'
predicted_notes = predict_notes_from_wav(wav_file, note_freqs, sample_rate)

write_predicted_notes_to_file(predicted_notes)

print(f"Predicted notes with durations have been written to 'predictedNotes.txt'.")
