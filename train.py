import os
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras import backend as K
from model import base_model
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

# load before compile
def load_model(model, weights_file, keep_train=False):
    if keep_train: model.load_weights(weights_file)
    return model


def SNR(y_true,y_pred):
    P = y_pred
    Y = y_true
    sqrt_l2_loss = K.sqrt(K.mean((P-Y)**2 + 1e-6))
    sqrn_l2_norm = K.sqrt(K.mean(Y**2))
    snr = 20 * K.log(sqrn_l2_norm / sqrt_l2_loss + 1e-8) / K.log(10.)
    avg_snr = K.mean(snr)
    return avg_snr

def sum_loss(y_true,y_pred):
    P = y_pred
    Y = y_true
    loss = K.sum((P-Y)**2)
    return loss

def compile_model(model):
    model.compile(loss='mse', optimizer="adam", metrics=[sum_loss, SNR])
    return model


def run():
    # Hyper params
    EPOCHS = 10
    BATCH_SIZE = 4000
    KEEP_TRAIN = False
    WEIGHTS_PATH = 'weights/'
    WEIGHTS_FILE = 'asr-weights.hdf5'
    VALID_SPLIT = 0.05
    SHUFFLE = True
    MINI_EPOCH = 1 # set each dataset's epochs
    
    model = base_model()
    model = load_model(model, WEIGHTS_FILE, keep_train=KEEP_TRAIN)
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
    
    
    
    
    
    
    
    
    
