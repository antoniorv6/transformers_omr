import tensorflow as tf
from .MHA import MHA
from tensorflow.keras.layers import Dense, LayerNormalization, Dropout

class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, model_depth, num_heads, feed_forward_depth, dropout_rate=0.1):
        super(DecoderLayer, self).__init__()

        self.model_depth = model_depth
        self.num_heads = num_heads
        self.feed_forward_depth = feed_forward_depth
        self.dropout_rate = dropout_rate

        self.mha1 = MHA(self.model_depth, self.num_heads)
        self.mha2 = MHA(self.model_depth, self.num_heads)

        self.pw_feedf_net_relu = Dense(self.feed_forward_depth, activation='gelu')  # First layer must have a ReLu
        self.pw_feedf_net_out = Dense(self.model_depth)  # Output of the point wise feed forward net that we are interested of

        self.layerNormalization1 = LayerNormalization(epsilon=1e-6)
        self.layerNormalization2 = LayerNormalization(epsilon=1e-6)
        self.layerNormalization3 = LayerNormalization(epsilon=1e-6)

        self.dropout1 = Dropout(self.dropout_rate)
        self.dropout2 = Dropout(self.dropout_rate)
        self.dropout3 = Dropout(self.dropout_rate)

    def call(self, inputs, encoder_output, look_ahead_mask, padding_mask):
        attn1, attn_weights_block1 = self.mha1(inputs, k=inputs, q=inputs, mask=look_ahead_mask) # (batch_size, target_seq_len, d_model)
        attn1 = self.dropout1(attn1)
        out1 = self.layerNormalization1(attn1 + inputs)

        attn2, attn_weights_block2 = self.mha2(encoder_output, k=encoder_output, q=out1, mask=padding_mask) # (batch_size, target_seq_len, d_model)
        attn2 = self.dropout2(attn2)
        out2 = self.layerNormalization2(attn2 + out1) # (batch_size, target_seq_len, d_model)

        ffn_output = self.pw_feedf_net_relu(out2) # (batch_size, target_seq_len, d_model)
        ffn_output = self.pw_feedf_net_out(ffn_output)
        output = self.layerNormalization3(ffn_output + out2) # (batch_size, target_seq_len, d_model)

        return output, attn_weights_block1, attn_weights_block2


    def get_config(self):
        return {
            "model_depth": self.model_depth,
            "num_heads" : self.num_heads,
            "feed_forward_depth": self.feed_forward_depth,
            "dropout_rate": self.dropout_rate
        }