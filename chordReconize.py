import numpy as np
import matplotlib.pyplot as plt


from madmom.audio.chroma import DeepChromaProcessor
from madmom.features.chords import DeepChromaChordRecognitionProcessor

def main():
    # wav, fs = madmom.audio.ffmpeg.load_ffmpeg_file('data/song.mp3')
    dcp = DeepChromaProcessor()
    decode = DeepChromaChordRecognitionProcessor()
    chroma = dcp('data/song.mp3')
    chords = decode(chroma)
    print(chords)

if __name__ == '__main__':
    main()