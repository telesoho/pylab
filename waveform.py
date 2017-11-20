#coding:utf-8
"""
filename: mp3-to-waveform.py
"""
from pydub import AudioSegment
import numpy as np
import matplotlib.pyplot as plt

def main():
    """
    读取mp3片段，并显示波形
    """
    song = AudioSegment.from_mp3('data/song.mp3')
    print(
        '样本宽度:', song.sample_width,
        '声道数:', song.channels,
        '采样率:', song.frame_rate,
        '帧宽:', song.frame_width,
        'song.array_type', song.array_type
    )

    wav = song.get_array_of_samples()   # 获取声音数据数组
    wav = np.frombuffer(wav, dtype=song.array_type) / 32768.0  # 正则化为(-1, 1)

    print(wav[1:10])

    full_time = np.arange(0.0, len(wav)/song.frame_rate, 1/song.frame_rate) #生成时间坐标

    # 取出中心部分波形
    center = len(wav) / 2   # 中心样本编号
    cuttime = 0.02         # 声音切片长度 [s]
    start = int(center - cuttime/2*song.frame_rate)
    end = int(center + cuttime/2*song.frame_rate)

    wav_fft = np.fft.fft(wav[start: end])     # 高速傅立叶变换
    freq_list = np.fft.fftfreq(end - start, d=1.0/song.frame_rate)  # 计算出频率

    amplitude_spectrum = [np.sqrt(c.real ** 2 + c.imag ** 2) for c in wav_fft] # 计算振幅
    phase_spectrum = [np.arctan2(int(c.imag), int(c.real)) for c in wav_fft] # 计算相位

    # 画出波形
    plt.subplot(311)  # 3行1列
    plt.plot(full_time[start: end]*1000, wav[start : end])
    plt.xlabel("time [ms]")
    plt.ylabel("amplitude")
    plt.plot()

    plt.subplot(312)
    plt.plot(freq_list, amplitude_spectrum, marker= '.', linestyle='-')
    plt.axis([0, song.frame_rate/2, 0, 50])
    plt.xlabel("frequency [Hz]")
    plt.ylabel("amplitude spectrum")

    # 位相描画
    plt.subplot(313)
    plt.plot(freq_list, phase_spectrum, marker='.', linestyle='-')
    plt.axis([0, song.frame_rate/2, -np.pi, np.pi])
    plt.xlabel("frequency [Hz]")
    plt.ylabel("phase spectrum")

    plt.show()

if __name__ == '__main__':
    main()
