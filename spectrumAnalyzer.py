#coding=utf-8
"""
Spectrum Analyzer
"""
import sys
from pydub import AudioSegment
import numpy as np
import matplotlib
matplotlib.use("WXAgg")
import matplotlib.pyplot as plt


song = AudioSegment.from_mp3('data/song.mp3')
print(
    '样本宽度:', song.sample_width,
    '声道数:', song.channels,
    '采样率:', song.frame_rate,
    '帧宽:', song.frame_width,
    'array type:', song.array_type
)

wav = song.get_array_of_samples()   # 获取声音数据数组
x = np.frombuffer(wav, dtype=song.array_type) / 32768.0  # 正则化为(-1, 1)

print(len(x))

fig = plt.figure()
sp1 = fig.add_subplot(211)
sp2 = fig.add_subplot(212)
fs = song.frame_rate

start = 12990800    # サンプリングする開始位置
N = 512      # FFTのサンプル数
SHIFT = 128  # 窓関数をずらすサンプル数

hamming_window = np.hamming(N)
freq_list = np.fft.fftfreq(N, d=1.0/fs)  # 周波数軸の値を計算


def update(idleevent):
    global start

    windowed_data = hamming_window * x[start:start+N]  # 切り出した波形データ（窓関数あり）
    X = np.fft.fft(windowed_data)  # FFT

    amplitude_spectrum = [np.sqrt(c.real ** 2 + c.imag ** 2) for c in X]  # 振幅スペクトル

    # 波形を更新
    sp1.cla()  # クリア
    sp1.plot(range(start, start+N), x[start:start+N])
    sp1.axis([start, start+N, -0.9, 0.9])
    sp1.set_xlabel("time [sample]")
    sp1.set_ylabel("amplitude")

    # 振幅スペクトルを描画
    sp2.cla()
    sp2.plot(freq_list, amplitude_spectrum, marker= '.', linestyle='-')
    sp2.axis([0, fs/2, 0, 40])
    sp2.set_xlabel("frequency [Hz]")
    sp2.set_ylabel("amplitude spectrum")

    fig.canvas.draw_idle()
    start += SHIFT  # 窓関数をかける範囲をずらす
    if start + N > len(x):
        sys.exit()

import wx

wx.EVT_IDLE(wx.GetApp(), update)
plt.show()
