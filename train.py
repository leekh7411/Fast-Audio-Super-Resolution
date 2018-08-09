import os
from keras.callbacks import ModelCheckpoint, EarlyStopping
from model import base_model
from utils import load_model, SNR, sum_loss, compile_model
import h5py

def load_h5_list(dirname):
    datasets = []
    filenames = os.listdir(dirname)
    file_extensions = set(['.h5'])
    for filename in filenames:
        ext = os.path.splitext(filename)[-1]
        if ext in file_extensions:
            full_filename = os.path.join(dirname, filename)
            datasets.append(full_filename)
    return datasets

def load_h5(dataset_name):
    print('Loading dataset ',dataset_name)
    with h5py.File(dataset_name, 'r') as hf:
        X = (hf['data'][:])
        Y = (hf['label'][:])
    print(X.shape)
    print(Y.shape)
    return X, Y


def run():
    # Hyper params
    EPOCHS = 10
    BATCH_SIZE = 4000
    LOAD_WEIGHTS = True
    WEIGHTS_PATH = 'weights/'
    WEIGHTS_FILE = 'asr-weights.hdf5'
    VALID_SPLIT = 0.05
    SHUFFLE = True
    MINI_EPOCH = 1 # set each dataset's epochs
    
    model = base_model()
    model = load_model(model, os.path.join(WEIGHTS_PATH, WEIGHTS_FILE), load_weights=LOAD_WEIGHTS)
    model = compile_model(model)
    
    datasets = load_h5_list('data/')
    checkpointer = ModelCheckpoint(filepath=os.path.join(WEIGHTS_PATH, WEIGHTS_FILE), verbose=1, save_best_only=True) 
    earlystopper = EarlyStopping(monitor='val_loss', patience=3, verbose=0)
    
    for i in range(EPOCHS):
        print('#REAL EPOCH(%3d/%3d)'%(i+1,EPOCHS))
        for dataset in datasets:
            X,Y = load_h5(dataset)
            model.fit(X, Y,
                    batch_size=BATCH_SIZE,
                    epochs=MINI_EPOCH, 
                    shuffle=SHUFFLE,
                    callbacks=[checkpointer, earlystopper],
                    validation_split=VALID_SPLIT)
    
    
if __name__ == '__main__':
    run()    
    
    
    
    
    
    
    
    
    
