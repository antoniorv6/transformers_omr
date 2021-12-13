import numpy as np
import cv2
import os
import itertools
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Input, Dense, Activation, BatchNormalization, LeakyReLU
from tensorflow.keras.layers import Reshape, Lambda, Permute
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GRU, LSTM
from tensorflow.keras.layers import TimeDistributed, Bidirectional
from tensorflow.keras.models import load_model


# ===================================================

def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    y_pred = y_pred[:, :, :]
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)


def get_CRNN_CTC_model(input_shape, vocabulary_size):

    conv_filters = [32,64,64,128]

    input_data = Input(name='the_input', shape=input_shape, dtype='float32')

    inner = Conv2D(conv_filters[0], 5, padding='same', name='conv1')(input_data)
    inner = BatchNormalization()(inner)
    inner = LeakyReLU(alpha=0.2)(inner)
    inner = MaxPooling2D(pool_size=(2, 2), name='max1')(inner)

    inner = Conv2D(conv_filters[1], 3, padding='same', name='conv2')(inner)
    inner = BatchNormalization()(inner)
    inner = LeakyReLU(alpha=0.2)(inner)
    inner = MaxPooling2D(pool_size=(2, 1), name='max2')(inner)

    inner = Conv2D(conv_filters[2], 3, padding='same')(inner)
    inner = BatchNormalization()(inner)
    inner = LeakyReLU(alpha=0.2)(inner)
    inner = MaxPooling2D(pool_size=(2, 1))(inner)

    inner = Conv2D(conv_filters[3], 3, padding='same')(inner)
    inner = BatchNormalization()(inner)
    inner = LeakyReLU(alpha=0.2)(inner)
    inner = MaxPooling2D(pool_size=(2, 1))(inner)

    inner = Permute((2, 1, 3))(inner)
    inner = Reshape(target_shape=(-1, (input_shape[0] // (2 ** 4)) * conv_filters[-1]), name='reshape')(inner)

    inner = Bidirectional(LSTM(512, return_sequences = True, dropout=0.25))(inner)

    inner = Dense(vocabulary_size+1, name='dense2')(inner)
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




