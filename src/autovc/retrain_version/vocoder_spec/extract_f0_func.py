import os
import numpy as np
import soundfile as sf
import pdb
from scipy import signal
from librosa.filters import mel
from numpy.random import RandomState
from pysptk import sptk
from src.autovc.retrain_version.vocoder_spec.utils import butter_highpass
from src.autovc.retrain_version.vocoder_spec.utils import speaker_normalization
from scipy.signal import get_window
import glob


def pySTFT(x, fft_length=1024, hop_length=256):
    x = np.pad(x, int(fft_length // 2), mode='reflect')

    noverlap = fft_length - hop_length
    shape = x.shape[:-1] + ((x.shape[-1] - noverlap) // hop_length, fft_length)
    strides = x.strides[:-1] + (hop_length * x.strides[-1], x.strides[-1])
    result = np.lib.stride_tricks.as_strided(x, shape=shape,
                                             strides=strides)

    fft_window = get_window('hann', fft_length, fftbins=True)
    result = np.fft.rfft(fft_window * result, n=fft_length).T

    return np.abs(result)


def extract_f0_func(gender):
    floor_sp, ceil_sp = -80, 30
    mel_basis = mel(16000, 1024, fmin=90, fmax=7600, n_mels=80).T
    min_level = np.exp(-100 / 20 * np.log(10))
    b, a = butter_highpass(30, 16000, order=5)

    # Set the directory you want to start from
    ROOT = r'E:\Dataset\VCTK\test_audio'
    rootDir = os.path.join(ROOT, 'audio')
    targetDir_f0 = os.path.join(ROOT, 'f0')
    targetDir = os.path.join(ROOT, 'mel-sp')

    pt = glob.glob1(rootDir, '*')

    cep_all = []
    dirName, subdirList, _ = next(os.walk(rootDir))
    print('Found directory: %s' % dirName)
    for subdir in sorted(pt):
        print(subdir)
        if not os.path.exists(os.path.join(targetDir, subdir)):
            os.makedirs(os.path.join(targetDir, subdir))
        if not os.path.exists(os.path.join(targetDir_f0, subdir)):
            os.makedirs(os.path.join(targetDir_f0, subdir))
        _, _, fileList = next(os.walk(os.path.join(dirName, subdir)))
        if gender == 'M':
            lo, hi = 50, 250
        elif gender == 'F':
            lo, hi = 100, 600
        else:
            raise ValueError
        prng = RandomState(0)
        for fileName in sorted(fileList):
            print(subdir, fileName)
            x, fs = sf.read(os.path.join(dirName, subdir, fileName))
            if(len(x.shape) >= 2):
                x = x[:, 0]
            if x.shape[0] % 256 == 0:
                x = np.concatenate((x, np.array([1e-06])), axis=0)
            y = signal.filtfilt(b, a, x)
            wav = y * 0.95 + (prng.rand(y.shape[0]) - 0.5) * 1e-06
            D = pySTFT(wav).T
            D_mel = np.dot(D, mel_basis)
            D_db = 20 * np.log10(np.maximum(min_level, D_mel)) - 16
            S = (D_db + 100) / 100

            f0_rapt = sptk.rapt(wav.astype(np.float32) *
                                32768, fs, 256, min=lo, max=hi, otype=2)
            index_nonzero = (f0_rapt != -1e10)
            tmp = f0_rapt[index_nonzero]
            mean_f0, std_f0 = np.mean(tmp), np.std(tmp)

            f0_norm = speaker_normalization(
                f0_rapt, index_nonzero, mean_f0, std_f0)

            if len(S) != len(f0_norm):
                pdb.set_trace()

            np.save(os.path.join(targetDir, subdir, fileName[:-4]),
                    S.astype(np.float32), allow_pickle=False)

            np.save(os.path.join(targetDir_f0, subdir, fileName[:-4]),
                    f0_norm.astype(np.float32), allow_pickle=False)

            print(S.shape)
            print(f0_norm.shape)
            # exit(0)


def extract_f0_func_audiofile(audio_file, gender='M'):
    floor_sp, ceil_sp = -80, 30
    mel_basis = mel(16000, 1024, fmin=90, fmax=7600, n_mels=80).T
    min_level = np.exp(-100 / 20 * np.log(10))
    b, a = butter_highpass(30, 16000, order=5)

    if gender == 'M':
        lo, hi = 50, 250
    elif gender == 'F':
        lo, hi = 100, 600
    else:
        raise ValueError
    prng = RandomState(0)
    x, fs = sf.read(audio_file)
    if(len(x.shape) >= 2):
        x = x[:, 0]
    if x.shape[0] % 256 == 0:
        x = np.concatenate((x, np.array([1e-06])), axis=0)
    y = signal.filtfilt(b, a, x)
    wav = y * 0.95 + (prng.rand(y.shape[0]) - 0.5) * 1e-06
    D = pySTFT(wav).T
    D_mel = np.dot(D, mel_basis)
    D_db = 20 * np.log10(np.maximum(min_level, D_mel)) - 16
    S = (D_db + 100) / 100

    f0_rapt = sptk.rapt(wav.astype(np.float32) * 32768,
                        fs, 256, min=lo, max=hi, otype=2)
    index_nonzero = (f0_rapt != -1e10)
    tmp = f0_rapt[index_nonzero]
    mean_f0, std_f0 = np.mean(tmp), np.std(tmp)

    f0_norm = speaker_normalization(f0_rapt, index_nonzero, mean_f0, std_f0)

    return S, f0_norm


if __name__ == '__main__':
    extract_f0_func('M')
