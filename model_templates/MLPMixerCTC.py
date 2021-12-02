import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Input, Dense, Activation, BatchNormalization, LeakyReLU
from tensorflow.keras.layers import Reshape, Lambda, Permute, Flatten
from tensorflow.keras.layers import MultiHeadAttention, LayerNormalization, Add, Dropout, Embedding
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
from .TransformerEncoder import TransformerEncoder

class MLPBlock(tf.keras.layers.Layer):
    def __init__(self, input_dim, mlp_dim):
        super(MLPBlock).__init__()
        self.dense1 = Dense(mlp_dim, activation='gelu')
        self.dense2 = Dense(input_dim)
    
    def __call__(self, x):
        inner = self.dense1(x)
        inner = self.dense2(inner)

        return x

class MixerBlock(tf.keras.layers.Layer):
    def __init__(self, num_patches, patches_mixer_depth, num_channels, channels_mixer_depth):
        super(MixerBlock).__init__()
        self.ln = LayerNormalization()
        self.permute = Permute((2,1))
        self.mixer1 = MLPBlock(num_patches, patches_mixer_depth)
        self.mixer2 = MLPBlock(num_channels, channels_mixer_depth)
        self.ln2 = LayerNormalization()

    def __call__(self, x):
        y = self.ln(x)
        y = self.permute(y)
        y = self.mixer1(y)
        y = self.permute(y)
        x = x+y
        y = self.ln2(x)
        y = self.mixer2(y)
        return x + y

class MLPMixer(tf.keras.layers.Layer):
    def __init__(self, num_blocks, num_patches, num_channels):
        super(MLPMixer).__init__()
        self.mixer_blocks = []
        for _ in range(num_blocks):
            self.mixer_blocks.append(MixerBlock(num_patches, 2048, num_channels, 256))
        
        self.ln = LayerNormalization()
    
    def __call__(self, x):
        for block in self.mixer_blocks:
            x = block(x)
        x = self.ln(x)
        return x


def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    y_pred = y_pred[:, :, :]
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)


def get_MLPMixer_CTC(input_shape, vocabulary_size):

    features = (input_shape[0]) * 512

    input_data = Input(name='the_input', shape=input_shape, dtype='float32')
    
    convolution = Conv2D(512, kernel_size=[input_shape[0], input_shape[1]], padding='same')(input_data)
    
    inner = Permute((2, 1, 3))(convolution)
    ##inner = PositionEncoding2D(conv_filters[-1])(inner)
    inner = Reshape(target_shape=(-1, features), name='reshape')(inner)
    
    mixer = MLPMixer(8, inner[1], features)

    inner = Dense(vocabulary_size+1, name='dense2')(mixer)
    y_pred = Activation('softmax', name='softmax')(inner)

    model_pr = Model(inputs=input_data, outputs=y_pred)
    model_pr.summary()

    labels = Input(name='the_labels',shape=[None], dtype='float32')
    input_length = Input(name='input_length', shape=[1], dtype='int64')
    label_length = Input(name='label_length', shape=[1], dtype='int64')

    loss_out = Lambda(
        ctc_lambda_func, output_shape=(1,),
        name='ctc')([y_pred, labels, input_length, label_length])

    model_tr = Model(inputs=[input_data, labels, input_length, label_length],
                  outputs=loss_out)

    model_tr.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer='adam')

    # Nota: el modelo que se entrena es el que lleva ctc (model_tr) pero para guardarlo usamos el que no (model_pr)
    
    #model_pr.load_weights("CTCPRIMUS1K0agnostic.h5")
    
    return model_tr, model_pr