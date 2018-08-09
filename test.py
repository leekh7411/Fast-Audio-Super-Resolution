import os
from keras.callbacks import ModelCheckpoint, EarlyStopping
from model import base_model
import h5py
from utils import load_model, compile_model, STFT
from prep_data import preprocess, load_wav_list

def test_samples():
    # Hyper params
    BATCH_SIZE = 4000
    LOAD_WEIGHTS = True
    WEIGHTS_PATH = 'weights/'
    WEIGHTS_FILE = 'asr-weights.hdf5'
    
    model = base_model()
    model = load_model(model, os.path.join(WEIGHTS_PATH,WEIGHTS_FILE) , load_weights=LOAD_WEIGHTS)
    model = compile_model(model)
    
    
    # load test wav samples
    test_samples = load_wav_list('test-samples/')
    
    # patch sample data
    X, Y, _ = preprocess(test_samples, start=0, end=len(test_samples)-1, sr=48000, scale=6, in_dim=64, out_dim=8, tag='test')
    
    print(X.shape)
    print(Y.shape)
    
    # predict
    pred = model.predict(X)
    
    # evaluate
    scores = model.evaluate(X,Y)
    print('Evaluate scores')
    for score in scores:
        print('- %10f'%(score))
    
    STFT(Y.flatten(), title='Original',n_fft=2048, show=True)
    STFT(X.flatten(), title='Downsampled',n_fft=2048, show=True)
    STFT(pred.flatten(), title='Upsampled',n_fft=2048, show=True)   
    