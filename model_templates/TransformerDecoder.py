import tensorflow as tf
from tensorflow.keras.layers import Embedding, Dropout
import numpy as np
from .DecoderLayer import DecoderLayer

class TransformerDecoder(tf.keras.layers.Layer):

    @staticmethod
    def get_angles(pos, i, model_depth):
        angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(model_depth))
        return pos * angle_rates

    @staticmethod
    def positional_encoding(position, model_depth):
        angle_rads = TransformerDecoder.get_angles(np.arange(position)[:, np.newaxis],
                                                   np.arange(model_depth)[np.newaxis, :],
                                                   model_depth)
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

        pos_encoding = angle_rads[np.newaxis, ...]

        return tf.cast(pos_encoding, dtype=tf.float32)

    def __init__(self, num_layers, model_depth, num_heads, feed_forward_depth, target_vocab_size, maximum_position_encoding, dropout_rate=0.1):
        super(TransformerDecoder, self).__init__()
        
        self.model_depth = model_depth
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.feed_forward_depth = feed_forward_depth
        self.target_vocab_size = target_vocab_size
        self.maximum_position_encoding = maximum_position_encoding
        self.dropout_rate = dropout_rate

        self.embedding = Embedding(self.target_vocab_size, self.model_depth)
        self.position_encoding = TransformerDecoder.positional_encoding(self.maximum_position_encoding, self.model_depth)

        self.decoder_layers = [DecoderLayer(self.model_depth, self.num_heads, self.feed_forward_depth, self.dropout_rate)
                               for _ in range(self.num_layers)]

        self.dropout = Dropout(self.dropout_rate)

    def call(self, inputs, encoder_output, look_ahead_mask, padding_mask):

        seq_len = tf.shape(inputs)[1]
        attention_weights = {}

        x = self.embedding(inputs) # (batch_size, target_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.model_depth, tf.float32))
        x += self.position_encoding[:, :seq_len, :]

        x = self.dropout(x)

        for i in range(self.num_layers):
            x, block1, block2 = self.decoder_layers[i](x,
                                                       encoder_output=encoder_output,
                                                       look_ahead_mask=look_ahead_mask,
                                                       padding_mask = padding_mask)
            attention_weights['decoder_layer{}_block1'.format(i+1)] = block1
            attention_weights['decoder_layer{}_block2'.format(i+1)] = block2

        return x, attention_weights

    
    def get_config(self):
            return {
                "num_layers": self.num_layers,
                "model_depth": self.model_depth,
                "num_heads" : self.num_heads,
                "feed_forward_depth": self.feed_forward_depth,
                "maximum_position_encoding": self.maximum_position_encoding,
                "dropout_rate": self.dropout_rate,
                "target_vocab_size": self.target_vocab_size
            }