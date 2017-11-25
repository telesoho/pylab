import numpy as np

from madmom.audio.ffmpeg import load_ffmpeg_file

from madmom.features.chords import CNNChordFeatureProcessor
from madmom.features.chords import CRFChordRecognitionProcessor

def main():
    proc = CNNChordFeatureProcessor()
    features = proc('data/song.mp3')
    decode = CRFChordRecognitionProcessor()
    chords = decode(features)
    print(chords)

if __name__ == '__main__':
    main()