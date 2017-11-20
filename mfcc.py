#coding=utf-8
"""
MFCC (Mel-Frequency Cepstrum Coefficients)
"""
from pydub import AudioSegment
import numpy as np
import matplotlib.pyplot as plt

def open_mp3(filename):
    song = AudioSegment.from_mp3(filename)
    print(
        '音乐文件名:', filename,
        '样本宽度:', song.sample_width,
        '声道数:', song.channels,
        '采样率:', song.frame_rate,
        '帧宽:', song.frame_width,
        'array type:', song.array_type
    )
    wav = song.get_array_of_samples()   # 获取声音数据数组
    wav = np.frombuffer(wav, dtype=song.array_type) / 32768.0  # 正则化为(-1, 1)
    return wav, song.frame_rate

def main():
    wav, fs = open_mp3('data/song.mp3')
    t = np.arange(0.0, len(wav) / fs, 1/fs) # 生成时间坐标轴，单位（秒）

    # 音声波形の中心部分を切り出す
    center = len(wav) / 2  # 中心のサンプル番号
    cuttime = 0.04         # 切り出す長さ [s]
    start = int(center - cuttime/2*fs)
    end = int(center + cuttime/2*fs)
    wavdata = wav[ start : end ]
    time = t[ start : end ]

    # 波形をプロット
    plt.plot(time * 1000, wavdata)
    plt.xlabel("time [ms]")
    plt.ylabel("amplitude")
    plt.show()

if __name__ == '__main__':
    main()