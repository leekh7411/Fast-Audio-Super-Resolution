import os
import numpy as np
import librosa
from utils import upsample
import h5py
import random

my_wav_dir = '/mnt/GB/wav/' # set your local path containing .wav data 
my_list_path = '/mnt/GB/wav_list.txt' # set your wav list file path

def save(dataset_name, X, Y):
    with h5py.File(dataset_name, 'w') as f:
        data_set = f.create_dataset('data', X.shape, np.float32) # lr
        label_set = f.create_dataset('label', Y.shape, np.float32) # hr
        data_set[...] = X
        label_set[...] = Y
    print('save complete -> %s'%(dataset_name))

def preprocess(file_list, start, end, sr=16000, scale=4, in_dim=64, out_dim=8, tag='train'):
    random.shuffle(file_list)
    data_size = end - start + 1
    lr_patches = list()
    hr_patches = list()
    dataset_name = None
    for i, wav_path in enumerate(file_list[start:end+1]):
        if i % 10 == 0 : print("%s - %d/%d"%(wav_path,i+1+start,len(file_list)))

        # Get low sample rate version data for training
        x_hr, fs = librosa.load(wav_path, sr=sr)
        x_len = len(x_hr)
        x_hr = x_hr[ : x_len - (x_len % scale)]
        
        # Down sampling for Low res version
        #x_lr = decimate(x, scale)
        x_lr = np.array(x_hr[0::scale])
        
        # Upscale using cubic spline Interpolation
        x_lr = upsample(x_lr, scale)
        
        x_lr = np.reshape(x_lr,(len(x_lr),1))
        #x_hr = np.reshape(x_hr,(len(x_hr),1))
        
        for i in range(0, x_lr.shape[0]-in_dim , out_dim):
            lr_patch = x_lr[i:i+in_dim]
            mid = in_dim // 2 - out_dim // 2
            hr_patch = x_hr[i+mid:i+mid+out_dim]
            lr_patches.append(lr_patch)
            hr_patches.append(hr_patch)
    
    hr_len = len(hr_patches)
    lr_len = len(lr_patches)
    
    hr_patches = np.array(hr_patches[0:hr_len])
    lr_patches = np.array(lr_patches[0:lr_len])
    
    print('high resolution(Y) dataset shape is ',hr_patches.shape) # (None, 64, 1)
    print('low resolution(X) dataset shape is ',lr_patches.shape) # (None, 8)
                
    dataset_name = 'data/asr-ex%d-start%d-end%d-scale%d-sr%d-in%d-out%d-%s.h5'%(data_size,
                                                                                    start,
                                                                                    end,
                                                                                    scale,
                                                                                    sr,
                                                                                    in_dim,
                                                                                    out_dim,
                                                                                    tag
                                                                                   )

    return lr_patches, hr_patches, dataset_name

def load_wav_list(local_dir, list_path):
    file_list = []    
    # Before using it,
    # make wav list file 
    # ex) ls .../wav/ > wav_list.txt
    with open(list_path) as f:
        for line in f:
            filename = line.strip()
            file_list.append(os.path.join(local_dir, filename))
    
    print('load wav list examples..')
    for i in range(5):
        print(file_list[i])
    
    return file_list


def run():
    convert_limit = 10 # All  dataset size
    dataset_size  = 10 # each dataset size
    sr = 48000 # sampling rate
    scale = 6 # down scaling ratio
    dsr = sr // scale # 48000 // 6 = 8000(hz)
    in_dim = 64 
    out_dim = 8
    
    file_list = load_wav_list(local_dir=my_wav_dir, list_path=my_list_path)
    
    for i in range(0, len(file_list), dataset_size):
        if i == convert_limit: break
        train_X, train_Y, trainset_name = preprocess(file_list = file_list,
                                                     start   =  i, 
                                                     end     =  i + dataset_size - 1,
                                                     sr      =  sr, 
                                                     scale   =  scale,
                                                     in_dim  =  in_dim,
                                                     out_dim =  out_dim,
                                                     tag     =  'train-s2p')
        save(trainset_name, train_X, train_Y)

        
if __name__ == '__main__':
    run()