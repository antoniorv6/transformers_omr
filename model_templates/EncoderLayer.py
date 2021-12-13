from .MHA import MHA
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, LayerNormalization

class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, model_depth, num_heads, feed_forward_depth, dropout_rate=0.1):
        super(EncoderLayer, self).__init__()

        self.model_depth = model_depth
        self.num_heads = num_heads
        self.feed_forward_depth = feed_forward_depth
        self.dropout_rate = dropout_rate


        self.multi_headed_attention = MHA(self.model_depth, self.num_heads)
        self.pw_feedf_net_relu = Dense(self.feed_forward_depth, activation='relu') #First layer must have a ReLu
        self.pw_feedf_net_out = Dense(self.model_depth) #Output of the point wise feed forward net that we are interested of

        self.layerNormalization1 = LayerNormalization(epsilon=1e-6)
        self.layerNormalization2 = LayerNormalization(epsilon=1e-6)

        self.dropout1 = Dropout(self.dropout_rate)
        self.dropout2 = Dropout(self.dropout_rate)

    def call(self, inputs, mask):
        attn_output, _ = self.multi_headed_attention(inputs, k=inputs, q=inputs, mask=mask) # (batch_size, input_seq_len, self.model_depth)
        attn_output = self.dropout1(attn_output)
        out1 = self.layerNormalization1(inputs + attn_output)  # (batch_size, input_seq_len, model_depth)

        ffn_output = self.pw_feedf_net_relu(out1) # (batch_size, input_seq_len, d_model)
        ffn_output = self.pw_feedf_net_out(ffn_output)
        ffn_output = self.dropout2(ffn_output) # (batch_size, input_seq_len, d_model)
        out2 = self.layerNormalization2(out1 + ffn_output)

        return out2
    

    def get_config(self):
        return {
            "model_depth": self.model_depth,
            "num_heads" : self.num_heads,
            "feed_forward_depth": self.feed_forward_depth,
            "dropout_rate": self.dropout_rate
        }