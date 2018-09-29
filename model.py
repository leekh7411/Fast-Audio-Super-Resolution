import keras
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM, GRU
from keras.layers import Lambda, Conv1D, Lambda
from keras.layers.advanced_activations import LeakyReLU
import tensorflow as tf

def split(x):
    return x[:,28:36] # it is fixed range for input(64) & output(8) dataset


def SubPixel1D(input_shape, r, color=False):
    def _phase_shift(I, r=2):
        X = tf.transpose(I, [2,1,0]) # (r, w, b)
        X = tf.batch_to_space_nd(X, [r], [[0,0]]) # (1, r*w, b)
        X = tf.transpose(X, [2,1,0])
        return X

    def subpixel_shape(input_shape):
        dims = [input_shape[0],
            input_shape[1] * r,
            int(input_shape[2] / (r))]
        output_shape = tuple(dims)
        return output_shape
  
    def subpixel(x):
        # only single channel!
        x_upsampled = _phase_shift(x, r)
        return x_upsampled

    return Lambda(subpixel, output_shape=subpixel_shape)

'''
Update 2018.09.29
- Input & output Shape -> (256,1)
- Fast and Upsample much better
- If you would like to use this model, set dataset size to (64,1) -> (256,1)
'''
def base_model(summary=True):
    print('load base model..')
    x = keras.layers.Input((256,1))
    main_input = x
    
    # 128 256 512 512
    # 65 31 15 15

    
    # Donwsampling layer 1
    x = Conv1D(padding='same', kernel_initializer='Orthogonal', filters=16, kernel_size=16, activation=None, strides=2)(x)
    x = LeakyReLU(0.2)(x)
    x1 = x # 128
    
    # Donwsampling layer 2
    x = Conv1D(padding='same', kernel_initializer='Orthogonal', filters=32, kernel_size=8, activation=None, strides=2)(x)
    x = LeakyReLU(0.2)(x)
    x2 = x # 64
    
    # Donwsampling layer 3
    x = Conv1D(padding='same', kernel_initializer='Orthogonal', filters=32, kernel_size=4, activation=None, strides=2)(x)
    x = LeakyReLU(0.2)(x)
    x3 = x # 32
    
    # Donwsampling layer 4
    x = Conv1D(padding='same', kernel_initializer='Orthogonal', filters=32, kernel_size=4, activation=None, strides=2)(x)
    x = LeakyReLU(0.2)(x)
    x4 = x # 16
    
    # Donwsampling layer 5
    x = Conv1D(padding='same', kernel_initializer='Orthogonal', filters=32, kernel_size=4, activation=None, strides=2)(x)
    x = LeakyReLU(0.2)(x)
    x5 = x # 8
    
    # Donwsampling layer 6
    x = Conv1D(padding='same', kernel_initializer='Orthogonal', filters=32, kernel_size=4, activation=None, strides=2)(x)
    x = LeakyReLU(0.2)(x)
    x6 = x # 4
    
    # Bottleneck layer 
    x = Conv1D(padding='same', kernel_initializer='Orthogonal', filters=32, kernel_size=4, activation=None, strides=2)(x)
    x = LeakyReLU(0.2)(x)
    
    # Upsampling layer 6
    x = Conv1D(padding='same', kernel_initializer='Orthogonal',filters=2*32, kernel_size=4, activation=None)(x)
    x = Activation('relu')(x)
    x = Dropout(rate=0.5)(x)
    x = SubPixel1D(x.shape, r=2, color=False)(x)
    x = keras.layers.concatenate([x, x6])
    
    # Upsampling layer 5
    x = Conv1D(padding='same', kernel_initializer='Orthogonal',filters=2*32, kernel_size=4, activation=None)(x)
    x = Activation('relu')(x)
    x = Dropout(rate=0.5)(x)
    x = SubPixel1D(x.shape, r=2, color=False)(x)
    x = keras.layers.concatenate([x, x5])
     
    # Upsampling layer 4
    x = Conv1D(padding='same', kernel_initializer='Orthogonal',filters=2*32, kernel_size=4, activation=None)(x)
    x = Activation('relu')(x)
    x = Dropout(rate=0.5)(x)
    x = SubPixel1D(x.shape, r=2, color=False)(x)
    x = keras.layers.concatenate([x, x4])
    
    
    # Upsampling layer 3
    x = Conv1D(padding='same', kernel_initializer='Orthogonal',filters=2*32, kernel_size=4, activation=None)(x)
    x = Activation('relu')(x)
    x = Dropout(rate=0.5)(x)
    x = SubPixel1D(x.shape, r=2, color=False)(x)
    x = keras.layers.concatenate([x, x3])
    
    
    # Upsampling layer 2
    x = Conv1D(padding='same', kernel_initializer='Orthogonal',filters=2*32, kernel_size=8, activation=None)(x)
    x = Activation('relu')(x)
    x = Dropout(rate=0.5)(x)
    x = SubPixel1D(x.shape, r=2, color=False)(x)
    x = keras.layers.concatenate([x, x2])
    
    # Upsampling layer 1
    x = Conv1D(padding='same', kernel_initializer='Orthogonal',filters=2*16, kernel_size=16, activation=None)(x)
    x = Activation('relu')(x)
    x = Dropout(rate=0.5)(x)
    x = SubPixel1D(x.shape, r=2, color=False)(x)
    x = keras.layers.concatenate([x, x1])
   
    
    # SubPixel-1D Final
    x = Conv1D(padding='same', kernel_initializer='he_normal',filters=2, kernel_size=8, activation=None)(x)     
    x = SubPixel1D(x.shape, r=2, color=False)(x)
    output = keras.layers.add([x, main_input])
    model  = keras.models.Model(main_input,output)
    
    if summary: 
        model.summary()       
        
    return model

'''
def base_model(summary=True):
    x = keras.layers.Input((64,1))
    main_input = x
    
    # Dim -> (None, 64, 1) --> (None, 32, 32) # Donwsampling layer 1
    x = Conv1D(padding='same', kernel_initializer='Orthogonal', filters=32, kernel_size=8, activation=None, strides=2)(x)
    x = LeakyReLU(0.2)(x)
    #x1 = x
    
    # Dim -> (None, 32, 32) --> (None, 16, 32) # Donwsampling layer 2
    x = Conv1D(padding='same', kernel_initializer='Orthogonal', filters=32, kernel_size=8, activation=None, strides=2)(x)
    x = LeakyReLU(0.2)(x)
    #x2 = x
    
    # Dim -> (None, 16, 32) --> (None, 8, 32) # Donwsampling layer 3
    x = Conv1D(padding='same', kernel_initializer='Orthogonal', filters=64, kernel_size=8, activation=None, strides=2)(x)
    x = LeakyReLU(0.2)(x)
    #x3 = x
    
    # Dim -> (None, 8, 32) --> (None, 4, 32) # Bottleneck layer 
    x = Conv1D(padding='same', kernel_initializer='Orthogonal', filters=96, kernel_size=8, activation=None, strides=2)(x)
    x = LeakyReLU(0.2)(x)
       
    # SubPixel-1D Final
    # Dim -> (None, 4, 32) --> (None, 8, 1)
    x = Conv1D(padding='same', kernel_initializer='he_normal',filters=2, kernel_size=3, activation=None)(x)     
    x = SubPixel1D(x.shape, r=2, color=False)(x)
    split_input = Lambda(split)(main_input)
    split_input = keras.layers.Reshape((8,))(split_input)
    x = keras.layers.Reshape((8,))(x)
    
    output = keras.layers.add([x,split_input])
    
    model  = keras.models.Model(main_input,output)
    
    if summary: 
        model.summary()       
        
    return model
'''