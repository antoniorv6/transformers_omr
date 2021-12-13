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

#class PositionEncoding2D(tf.keras.layers.Layer):
#    
#    def positional_encoding(self, d_model, height, width):
#        pe = np.zeros((d_model, height, width))
#        d_model = int(d_model / 2)
#        div_term = np.exp(np.arange(0., d_model, 2) *-(np.log(10000.0) / d_model))
#        pos_w = np.expand_dims(np.arange(0., width), 1)
#        pos_h = np.expand_dims(np.arange(0., height), 1)
#        pe[0:d_model:2, :, :] = np.expand_dims(np.sin(pos_w * div_term).transpose(0, 1),1).repeat(1, axis=0).repeat(height,axis=1).repeat(1,axis=2).reshape(-1, height,width)
#        pe[1:d_model:2, :, :] = np.expand_dims(np.cos(pos_w * div_term).transpose(0, 1), 1).repeat(1, axis=0).repeat(height,axis=1).repeat(1,axis=2).reshape(-1, height,width)
#        pe[d_model::2, :, :] = np.expand_dims(np.sin(pos_h * div_term).transpose(0, 1), 2).repeat(1, axis=0).repeat(1,axis=1).repeat(width,axis=2).reshape(-1, height,width)
#        pe[d_model + 1::2, :, :] = np.expand_dims(np.cos(pos_h * div_term).transpose(0, 1),2).repeat(1, axis=0).repeat(1,axis=1).repeat(width,axis=2).reshape(-1, height,width)
#        return pe.reshape(height, width, -1)#
#

#    def __init__(self, d_model):
#        super(PositionEncoding2D, self).__init__()
#        if d_model % 2 != 0:
#            raise ValueError("Cannot use sin and cos with an odd d_model value")
#        
#        self.d_model = d_model
#        pass
#    
#    def __call__(self, input):
#        return input + self.positional_encoding(self.d_model, tf.shape(input)[2], tf.shape(input)[1])

#class PositionalEncoding(tf.keras.layers.Layer):
#    @staticmethod
#    def get_angles(pos, i, model_depth):
#        angle_rates = 1 / np.power(10000, (2*(i//2)) / np.float32(model_depth))
#        return pos * angle_rates
#    @staticmethod
#    def positional_encoding(position, model_depth):
#        angle_rads = PositionalEncoding.get_angles(np.arange(position)[:,np.newaxis],
#                                np.arange(model_depth)[np.newaxis, :],
#                                model_depth)
#        angle_rads[:, 0::2] = np.sin(angle_rads[: , 0::2])
#        angle_rads[:, 1::2] = np.cos(angle_rads[: , 1::2])
#
#        pos_encoding = angle_rads[np.newaxis, ...]
#
#        return tf.cast(pos_encoding, dtype=tf.float32)
#    
#    def __init__(self, maximum_pos_encoding, model_depth):
#        super(PositionalEncoding, self).__init__()
#        self.position_encoding = PositionalEncoding.positional_encoding(maximum_pos_encoding, model_depth)
#        self.model_depth = model_depth
#    
#    def __call__(self, inputs):
#        seq_len = tf.shape(inputs)[1]
#        #inputs *= tf.math.sqrt(tf.cast(self.model_depth, tf.float32))
#        inputs += self.position_encoding[:, :seq_len, :]
#
#        return inputs

def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    y_pred = y_pred[:, :, :]
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)


#class TransformerEncoderLayer(tf.keras.layers.Layer):
#    def __init__(self, n_heads, d_model, hidden_ff, drop=0.1):
#        super(TransformerEncoderLayer, self).__init__()
#        self.mha = MultiHeadAttention(num_heads=n_heads, key_dim=d_model, dropout=drop)
#        self.add1 = Add()
#        self.ln1 = LayerNormalization(epsilon=1e-6)
#        self.ff1 = Dense(hidden_ff, activation=tf.nn.gelu)
#        self.drop1 = Dropout(drop)
#        self.ff2 = Dense(d_model)
#        self.drop2 = Dropout(drop)
#        self.add2 = Add()
#        self.ln2 = LayerNormalization(epsilon=1e-6)
#
#    def __call__(self, inputs):
#        x = self.mha(inputs, inputs)
#        x2 = self.add1([x, inputs])
#        x2 = self.ln1(x2)
#        x3 = self.ff1(x2)
#        x3 = self.drop1(x3)
#        x3 = self.ff2(x3)
#        x3 = self.drop2(x3)
#        output = self.add2([x3, x2])
#        output = self.ln2(output)
#        return output
        


def get_CNNTransformer_CTC_model(input_shape, vocabulary_size, transf_nheads, transf_depth, transf_ffunits):

    conv_filters = [32,32,64,64]
    features = (input_shape[0] // (2 ** len(conv_filters))) * conv_filters[-1]

    input_data = Input(name='the_input', shape=input_shape, dtype='float32')

    inner = Conv2D(conv_filters[0], 5, padding='same', name='conv1')(input_data)
    inner = BatchNormalization()(inner)
    inner = LeakyReLU(alpha=0.2)(inner)
    inner = MaxPooling2D(pool_size=(2, 2), name='max1')(inner)

    inner = Conv2D(conv_filters[1], 5, padding='same', name='conv2')(inner)
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
    ##inner = PositionEncoding2D(conv_filters[-1])(inner)
    inner = Reshape(target_shape=(-1, features), name='reshape')(inner)
    #inner = Flatten()(inner)

    #inner = Embedding(features, transf_depth)(inner)
    #inner = Dense(transf_depth)(inner)
    #inner = PositionalEncoding(1024, features)(inner)
    #inner = TransformerEncoderLayer(n_heads=transf_nheads, d_model=features, hidden_ff=transf_ffunits)(inner)
    inner = TransformerEncoder(1, features, transf_nheads, transf_ffunits, 1100)(inner, mask=None)

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
