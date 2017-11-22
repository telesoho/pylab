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

def hz2mel(f):
    """Hzをmelに変換"""
    return 1127.01048 * np.log(f / 700.0 + 1.0)

def mel2hz(m):
    """melをhzに変換"""
    return 700.0 * (np.exp(m / 1127.01048) - 1.0)

def melFilterBank(fs, nfft, numChannels):
    """メルフィルタバンクを作成"""
    # ナイキスト周波数（Hz）
    fmax = fs / 2
    # ナイキスト周波数（mel）
    melmax = hz2mel(fmax)
    # 周波数インデックスの最大数
    nmax = int(nfft / 2)
    # 周波数解像度（周波数インデックス1あたりのHz幅）
    df = fs / nfft
    # メル尺度における各フィルタの中心周波数を求める
    dmel = melmax / (numChannels + 1)
    melcenters = np.arange(1, numChannels + 1) * dmel
    # 各フィルタの中心周波数をHzに変換
    fcenters = mel2hz(melcenters)
    # 各フィルタの中心周波数を周波数インデックスに変換
    indexcenter = np.round(fcenters / df)
    # 各フィルタの開始位置のインデックス
    indexstart = np.hstack(([0], indexcenter[0:numChannels - 1]))
    # 各フィルタの終了位置のインデックス
    indexstop = np.hstack((indexcenter[1:numChannels], [nmax]))

    filterbank = np.zeros((numChannels, nmax))
    for c in np.arange(0, numChannels):
        # 三角フィルタの左の直線の傾きから点を求める
        increment= 1.0 / (indexcenter[c] - indexstart[c])
        for i in np.arange(indexstart[c], indexcenter[c]):
            filterbank[c: int(i)] = (i - indexstart[c]) * increment
        # 三角フィルタの右の直線の傾きから点を求める
        decrement = 1.0 / (indexstop[c] - indexcenter[c])
        for i in np.arange(indexcenter[c], indexstop[c]):
            filterbank[c, int(i)] = 1.0 - ((i - indexcenter[c]) * decrement)

    return filterbank, fcenters

def main():
    wav, fs = open_mp3('data/song.mp3')

    signal, time = get_center_sample(wav, fs, 0.004)

    # ハミング窓をかける
    hammingWindow = np.hamming(len(signal))
    signal2 = signal * hammingWindow
    
    # 振幅スペクトルを求める
    nfft = 2048  # FFTのサンプル数
    spec = np.abs(np.fft.fft(signal, nfft))[:int(nfft/2)]
    fscale = np.fft.fftfreq(nfft, d = 1.0 / fs)[:int(nfft/2)]

    # メルフィルタバンクを作成
    numChannels = 20  # メルフィルタバンクのチャネル数
    df = fs / nfft   # 周波数解像度（周波数インデックス1あたりのHz幅）
    filterbank, fcenters = melFilterBank(fs, int(nfft), numChannels)


    # 波形をプロット
    plt.figure(1)
    plt.subplot(311)
    plt.plot(time * 1000, signal)
    plt.xlabel("time [ms]")
    plt.ylabel("amplitude")

    plt.subplot(312)
    plt.plot(time * 1000, signal2)
    plt.xlabel("time [ms]")
    plt.ylabel("amplitude")

    plt.subplot(313)
    plt.plot(fscale, spec)
    plt.xlabel("frequency [Hz]")
    plt.ylabel("amplitude spectrum")


    plt.figure(2)
    # メルフィルタバンクのプロット
    for c in np.arange(0, numChannels):
        plt.plot(np.arange(0, nfft / 2) * df, filterbank[c])

    print(filterbank.shape)

    # 振幅スペクトルに対してフィルタバンクの各フィルタをかけ、
    # 振幅の和の対数をとる
    # mspec = []
    # for c in np.arange(0, numChannels):
    #     mspec.append(np.log10(sum(spec * filterbank[c])))
    # mspec = np.array(mspec)

    # 振幅スペクトルにメルフィルタバンクを適用
    mspec = np.log10(np.dot(spec, filterbank.T))

    plt.figure(3)
    # 元の振幅スペクトルとフィルタバンクをかけて圧縮したスペクトルを表示
    plt.subplot(211)
    plt.plot(fscale, np.log10(spec))
    plt.xlabel("frequency")
    plt.xlim(0, 25000)

    plt.subplot(212)
    plt.plot(fcenters, mspec, "o-")
    plt.xlabel("frequency")
    plt.xlim(0, 25000)

    plt.show()
if __name__ == '__main__':
    main()