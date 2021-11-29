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

def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    y_pred = y_pred[:, :, :]
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)


class Patches(tf.keras.layers.Layer):
    def __init__(self, patch_size):
        super(Patches, self).__init__()
        self.patch_size = patch_size

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="SAME",
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches

#class PatchEncoder(tf.keras.layers.Layer):
#    def __init__(self, num_patches, projection_dim):
#        super(PatchEncoder, self).__init__()
#        self.num_patches = num_patches
#        self.projection = Dense(units=projection_dim)
#        self.position_embedding = Embedding(
#            input_dim=num_patches, output_dim=projection_dim
#        )
#
#    def call(self, patch):
#        positions = tf.range(start=0, limit=self.num_patches, delta=1)
#        encoded = self.projection(patch) + self.position_embedding(positions)
#        return encoded


def get_vit_model(input_shape, model_depth, vocabulary_size):
    
    input_data = Input(name='the_input', shape=input_shape, dtype='float32')
    #num_patches = (input_shape[0] // patch_size) * (max_image_width // patch_size)
    
    patches = Permute((2, 1, 3))(input_data)
    ##inner = PositionEncoding2D(conv_filters[-1])(inner)
    patches = Reshape(target_shape=(-1, input_shape[0]), name='reshape')(patches)
    encoded_patches = Dense(model_depth)(patches)

    encoder_out = TransformerEncoder(4, model_depth, 8, 2048, 2048)(encoded_patches, mask=None)

    output = Dense(vocabulary_size+1, name='dense2')(encoder_out)
    y_pred = Activation('softmax', name='softmax')(output)

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

