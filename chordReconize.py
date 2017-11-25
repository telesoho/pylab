import numpy as np
import matplotlib.pyplot as plt

from madmom.audio.ffmpeg import load_ffmpeg_file
from madmom.audio.chroma import DeepChromaProcessor
from madmom.features.chords import DeepChromaChordRecognitionProcessor

def get_center_sample(wav, fs, cuttime):
    """
    wav: 波形数据
    fs: 采样率
    cuttime: 采样长度（单位：s)
    """
    # 音声波形の中心部分を切り出す
    center = len(wav) / 2  # 中心のサンプル番号
    start = int(center - cuttime/2*fs)
    end = int(center + cuttime/2*fs)
    wavedata = wav[ start : end ]
    t = np.arange(0.0, len(wav) / fs, 1/fs) # 生成时间坐标轴，单位（秒）
    time = t[ start : end ]
    return wavedata, time

def main():
    wav, fs = load_ffmpeg_file('data/song.mp3')
    dcp = DeepChromaProcessor()
    decode = DeepChromaChordRecognitionProcessor()
    chroma = dcp('data/song.mp3')
    chords = decode(chroma)
    print(chords)
    x, t = get_center_sample(wav, fs, 0.01)
    
    plt.plot(t * 1000, x)
    plt.xlabel("time [ms]")
    plt.ylabel("amplitude")
    plt.show()

if __name__ == '__main__':
    main()