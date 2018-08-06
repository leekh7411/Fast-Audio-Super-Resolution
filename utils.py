import librosa
import numpy as np
from scipy import interpolate
from scipy.signal import decimate
from matplotlib import pyplot as plt
from scipy import signal
import h5py
import wave

def wav2plot(signal):
    #Extract Raw Audio from Wav File
    #signal = spf.readframes(-1)
    signal = np.fromstring(signal, 'Int16')
    plt.figure(1)
    plt.title('Signal Wave...')
    plt.plot(signal)
    plt.show()

def float2complex(fdata):
    compx = fdata[...,0] + fdata[...,1] * 1j
    return compx

def upsample(x_lr, r):
    x_lr = x_lr.flatten()
    x_hr_len = len(x_lr) * r
    x_sp = np.zeros(x_hr_len)

    i_lr = np.arange(x_hr_len, step=r)
    i_hr = np.arange(x_hr_len)

    f = interpolate.splrep(i_lr, x_lr)
    x_sp = interpolate.splev(i_hr, f)

    return x_sp

def pad_with(vector, pad_width, iaxis, kwargs):
    pad_value = kwargs.get('padder', 0)
    vector[:pad_width[0]] = pad_value
    return vector

def STFT(X, title,n_fft=2048, show=False):
    S = librosa.stft(X,n_fft=n_fft)
    S = S.T
    S_real = np.expand_dims(S.real, axis=-1) 
    S_imag = np.expand_dims(S.imag, axis=-1)
    S_data = np.concatenate((S_real, S_imag),axis=-1)
    if show:
        showSpectrum(S,title)
        
    return S_data, S

def showSpectrum(S,title,aspect=0.1):
    aS = np.abs(S)
    iS = np.log1p(aS)
    plt.imshow(iS.T, aspect=aspect)
    plt.tight_layout()
    plt.title(title)
    plt.show()

def iSFTF(S):
    X = librosa.istft(S)
    return X