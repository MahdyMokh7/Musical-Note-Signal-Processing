# note_frequencies.py

# Define the notes in octave 0
base_octave_frequencies = {
     "C": 16.352, "B#": 16.352, "C#": 17.324,
     "Db": 17.324, "D": 18.354, "D#": 19.445,
     "Eb": 19.445, "E": 20.602, "Fb": 20.602,
     "F": 21.827, "E#": 21.827, "F#": 23.125,
     "Gb": 23.125, "G": 24.500, "G#": 25.957,
     "Ab": 25.957, "A": 27.500, "A#": 29.135,
     "Bb": 29.135, "B": 30.868, "Cb": 30.868
}

# Calculate the frequencies for other octaves
note_freqs = {}

# Populate the dictionary with frequencies for all octaves (from 0 to 10)
for octave in range(11):
    for note, base_freq in base_octave_frequencies.items():
        note_freqs[f'{note} {octave}'] = base_freq * (2 ** octave)

# Print the note frequencies for verification
# for note, freq in note_freqs.items():
#     print(f'{note}: {freq:.3f}')
