import os
import wave
import glob
import pathlib
import warnings
import splitfolders
import pandas as pd
import numpy as np
import librosa, librosa.display
from pydub import AudioSegment
import matplotlib.pyplot as plt
from pydub.silence import detect_nonsilent
warnings.filterwarnings('ignore')


def remove_sil(path_in, path_out, format="wav"):
    """去除单条语音的静音段"""
    sound = AudioSegment.from_file(path_in, format=format)
    non_sil_times = detect_nonsilent(sound, min_silence_len=30, silence_thresh=sound.dBFS * 1.5)
    print(len(non_sil_times) > 0)
    if len(non_sil_times) > 0:
        non_sil_times_concat = [non_sil_times[0]]
        if len(non_sil_times) > 1:
            for t in non_sil_times[1:]:
                if t[0] - non_sil_times_concat[-1][-1] < 200:
                    non_sil_times_concat[-1][-1] = t[1]
                else:
                    non_sil_times_concat.append(t)
        non_sil_times = [t for t in non_sil_times_concat]
        print(non_sil_times)
        sound[non_sil_times[0][0]: non_sil_times[-1][1]].export(path_out, format='wav')


def is_0s(filename):
    """判断语音是否是0s"""
    f = wave.open(filename)
    duration = f.getnframes()/float(f.getframerate())
    if duration == 0:
        return True
    return False


def batch_remove_sil(src_path="./wav_data", dest_path="./nonsilent_wav"):
    """去除文件夹中所有语音静音段"""
    cnt_0s = 0
    results = ['positive', 'negative']
    for res in results:
        pathlib.Path(f"{dest_path}/{res}").mkdir(parents=True, exist_ok=True)
        for files in os.listdir(f"{src_path}/{res}"):
            filename = f"{src_path}/{res}/{files}"
            wav_out = f'{dest_path}/{res}/{files}'
            if is_0s(filename):
                continue
            print(filename, wav_out)
            remove_sil(filename, wav_out)
    print(f"Number of audio with 0s: {cnt_0s}")


def read_wave_from_file(audio_file):
    """
    return 一维numpy数组，如（584,） 采样率"""
    wav = wave.open(audio_file, 'rb')
    num_frames = wav.getnframes()
    framerate = wav.getframerate()
    str_data = wav.readframes(num_frames)
    wav.close()
    wave_data = np.frombuffer(str_data, dtype=np.short)
    return wave_data, framerate


def save_wav(file_name, audio_data, channels=1, sample_width=2, rate=16000):
    wf = wave.open(file_name, 'wb')
    wf.setnchannels(channels)
    wf.setsampwidth(sample_width)
    wf.setframerate(rate)
    wf.writeframes(b''.join(audio_data))
    wf.close()


def gaussian_white_noise_numpy(samples, min_db=10, max_db=200):
    """
    高斯白噪声
    噪声音量db
        db = 10, 听不见
        db = 100,可以听见，很小
        db = 500,大
        人声都很清晰
    :param samples:
    :param max_db:
    :param min_db:
    :return:
    """
    samples = samples.copy()  # frombuffer()导致数据不可更改因此使用拷贝
    data_type = samples[0].dtype
    db = np.random.randint(low=min_db, high=max_db)
    noise = db * np.random.normal(0, 1, len(samples))  # 高斯分布
    samples = samples + noise
    samples = samples.astype(data_type)
    return samples


def time_shift_numpy(samples, max_ratio=0.05):
    """
    时间变化是在时间轴的±5％范围内的随机滚动。环绕式转换以保留所有信息。
    Shift a spectrogram along the frequency axis in the spectral-domain at random
    :param max_ratio:
    :param samples: 音频数据，一维(序列长度,) 或 特征数据(序列长度,特征维度)
    :return:
    """
    samples = samples.copy()  # frombuffer()导致数据不可更改因此使用拷贝
    data_type = samples[0].dtype
    frame_num = samples.shape[0]
    max_shifts = frame_num * max_ratio  # around 5% shift
    nb_shifts = np.random.randint(-max_shifts, max_shifts)
    samples = np.roll(samples, nb_shifts, axis=0)
    samples = samples.astype(data_type)
    return samples


def wav_augment(src_path='./wav_augment/positive', dest_path='./wav_augment/positive'):
    for files in os.listdir(src_path):
        file = f'{src_path}/{files}'
        # 1. 高斯白噪声
        audio_data, _ = read_wave_from_file(file)
        audio_data1 = gaussian_white_noise_numpy(audio_data)
        out_file1 = f'{dest_path}/{files[:-4]}_gaussian_white_noise.wav'
        save_wav(out_file1, audio_data1)
        # 2.时间变化
        audio_data2 = time_shift_numpy(audio_data)
        out_file2 = f'{dest_path}/{files[:-4]}_time_shift.wav'
        save_wav(out_file2, audio_data2)


def spectrogram_from_wav(wav_path, spec_path):
    cnt_0s = 0
    results = ['positive', 'negative']
    for res in results:
        pathlib.Path(f"{spec_path}/{res}").mkdir(parents=True, exist_ok=True)
        for files in os.listdir(f"{wav_path}/{res}"):
            filename = f"{wav_path}/{res}/{files}"
            f = wave.open(filename)
            duration = f.getnframes()/float(f.getframerate())
            if duration == 0:
                cnt_0s += 1
                continue
            x, _ = librosa.load(filename, mono=True)
            plt.specgram(x, NFFT=2048, Fs=2, Fc=0, noverlap=128, cmap='inferno', sides='default', mode='default', scale='dB');
            plt.axis('off');
            # plt.savefig(f"./img/spectrograms/{res}/{files[:-4]}.png")
            plt.savefig(f"{spec_path}/{res}/{files[:-4]}.png")
            plt.clf()
    print(f"Number of audio with 0s: {cnt_0s}")


def create_annotations_csv(file_path):
    """创建注释文件"""
    spec_list = []
    labels = ['negative', 'positive']
    for label in labels:
        for spec in glob.glob(f"{file_path}{label}/*"):
            spec_list.append([spec, labels.index(label), label])
    spec_list = pd.DataFrame(spec_list)
    spec_list.to_csv(f'{file_path}annotations.csv', index=None, header=None)


if __name__ == '__main__':

    # 1. 去静音段
    batch_remove_sil(src_path="./data/cleaned_data", dest_path="./data/nonsilent_wav")

    # 2. 数据增强
    wav_augment(src_path='./data/nonsilent_wav/positive', dest_path='./data/nonsilent_wav/positive')

    # 3. 提取语谱图
    spectrogram_from_wav(wav_path='./data/nonsilent_wav', spec_path="./data/spectrograms")

    # 统计正负样本数目
    positive_spec = [spec for spec in os.listdir("./spectrograms/positive/")]
    negative_spec = [spec for spec in os.listdir("./spectrograms/negative/")]
    print(f"Number of Negative spectrograms: {len(positive_spec)}\n"
          f"Number of Positive spectrograms: {len(negative_spec)}")

    # 4.划分训练集和测试集
    RATIO = (0.8, 0.2)
    base_path = r"D:/Courses/AAEA/code/COVID_Cough-master/"
    splitfolders.ratio(base_path + "spectrograms", output=base_path + "./spectrograms_split",
                       seed=1337, ratio=RATIO)
    # 创建注释文件
    create_annotations_csv("./data/spectrograms_split/train/")
    create_annotations_csv("./data/spectrograms_split/val/")