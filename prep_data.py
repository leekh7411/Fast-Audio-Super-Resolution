import os
import numpy as np
import librosa
from utils import upsample
import h5py
import random

my_wav_dir = '/mnt/dataSet/VCTK-Corpus-Subset/' # set your local path containing .wav data 

def save(dataset_name, X, Y):
    with h5py.File(dataset_name, 'w') as f:
        data_set = f.create_dataset('data', X.shape, np.float32) # lr
        label_set = f.create_dataset('label', Y.shape, np.float32) # hr
        data_set[...] = X
        label_set[...] = Y
    print('save complete -> %s'%(dataset_name))

def preprocess(file_list, start, end, sr=48000, scale=6, dimension=64, stride=8, tag='train'):
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
        x_hr = np.reshape(x_hr,(len(x_hr),1))
        
        for i in range(0, x_lr.shape[0]-dimension , stride):
            lr_patch = x_lr[i:i+dimension]
            
            #mid = dimension // 2 - stride // 2
            #hr_patch = x_hr[i+mid:i+mid+stride]
            
            hr_patch = x_hr[i:i+dimension]
            
            lr_patches.append(lr_patch)
            hr_patches.append(hr_patch)
    
    hr_len = len(hr_patches)
    lr_len = len(lr_patches)
    
    hr_patches = np.array(hr_patches[0:hr_len])
    lr_patches = np.array(lr_patches[0:lr_len])
    
    print('high resolution(Y) dataset shape is ',hr_patches.shape)
    print('low resolution(X) dataset shape is ',lr_patches.shape)
                
    dataset_name = 'data/asr-ex%d-start%d-end%d-scale%d-sr%d-dim%d-strd%d-%s.h5'%(data_size,
                                                                                    start,
                                                                                    end,
                                                                                    scale,
                                                                                    sr,
                                                                                    dimension,
                                                                                    stride,
                                                                                    tag
                                                                                   )

    return lr_patches, hr_patches, dataset_name

def load_wav_list(dirname='data/'):
    file_list = []    
    filenames = os.listdir(dirname)
    file_extensions = set(['.wav'])
    for filename in filenames:
        ext = os.path.splitext(filename)[-1]
        if ext in file_extensions:
            full_filename = os.path.join(dirname, filename)
            file_list.append(full_filename)
    
    print('load wav list examples..')
    
    for i, file in enumerate(file_list):
        print(file)
        if i > 5: break

    return file_list


def run():
    convert_limit = 10  # All  dataset size
    dataset_size  = 10  # each dataset size
    sr = 48000          # sampling rate
    scale = 8           # down scaling ratio
    dsr = sr // scale   # 48000 // 8 = 6000(hz)
    dimension = 256     # Input & Output size 
    stride = 64         # stride size
    
    file_list = load_wav_list(dirname=my_wav_dir)
    
    for i in range(0, len(file_list), dataset_size):
        if i == convert_limit: break
        train_X, train_Y, trainset_name = preprocess(file_list = file_list,
                                                     start   =  i, 
                                                     end     =  i + dataset_size - 1,
                                                     sr      =  sr, 
                                                     scale   =  scale,
                                                     dimension  =  dimension,
                                                     stride  =  stride,
                                                     tag     =  'train')
        save(trainset_name, train_X, train_Y)

        
if __name__ == '__main__':
    run()