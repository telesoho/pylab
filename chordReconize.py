import numpy as np

from madmom.audio.ffmpeg import load_ffmpeg_file
from madmom.audio.chroma import DeepChromaProcessor
from madmom.features.chords import DeepChromaChordRecognitionProcessor


def main():
    dcp = DeepChromaProcessor()
    decode = DeepChromaChordRecognitionProcessor()
    chroma = dcp('data/song.mp3')
    chords = decode(chroma)
    print(chords)
   

if __name__ == '__main__':
    main()