import tensorflow as tf
from tensorflow.keras.layers import Dense, Softmax

class MHA(tf.keras.layers.Layer):
    def __init__(self, model_depth, num_heads):
        super(MHA, self).__init__()
        
        self.num_heads = num_heads
        self.model_depth = model_depth
        
        assert self.model_depth % self.num_heads == 0
        self.depth = self.model_depth // self.num_heads

        self.wQ = Dense(self.model_depth) #Q weights
        self.wK = Dense(self.model_depth) #K weights
        self.wV = Dense(self.model_depth) #V weights

        self.output_linear = Dense(self.model_depth)

    def split_heads(self, x, batch_size):
        #Split the last dimension into (num_heads, depth) where depth is the model_depth // num_heads
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        #We transpose the result that the shape is (batch_size, num_heads, seq_len, depth)
        return tf.transpose(x, perm=[0,2,1,3])

    @staticmethod
    def scaled_dot_product_attention(q, k, v, mask):
        matmul_qk = tf.matmul(q, k, transpose_b=True) #(..., seq_len_q, seq_len_k)
        #Scale the matmul
        k_dim = tf.cast(tf.shape(k)[-1], tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(k_dim)

        #add mask to the scaled tensor (this will happen in the decoder layers)
        if mask is not None:
            scaled_attention_logits += (mask * -1e9)

        attention_weights = Softmax(axis=-1)(scaled_attention_logits) # (..., seq_len_q, seq_len_k)
        output = tf.matmul(attention_weights, v) #Multiply by the V value tu get the attention effect on the real values

        return output, attention_weights


    #kwargs -> (v,k,q,mask)
    def __call__(self, inputs, k, q, mask):
        v = inputs
        batch_size = tf.shape(q)[0]

        Q = self.wQ(q) #(batch_size, seq_len, d_model)
        K = self.wK(k) #(batch_size, seq_len, d_model)
        V = self.wV(v) #(batch_size, seq_len, d_model)

        Q = self.split_heads(Q, batch_size) # (batch_size, num_heads, seq_len_q, depth)
        K = self.split_heads(K, batch_size) # (batch_size, num_heads, seq_len_k, depth)
        V = self.split_heads(V, batch_size) # (batch_size, num_heads, seq_len_v, depth)
        #mask = self.split_heads(mask, batch_size)

        # scaled_attention.shape -> (batch_size, num_heads, seq_len_q, depth)
        # attention_weights.shape -> (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention, attention_weights = MHA.scaled_dot_product_attention(Q, K, V, mask)

        scaled_attention = tf.transpose(scaled_attention, perm=[0,2,1,3]) # (batch_size, seq_len_q, num_heads, depth)
        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.model_depth)) #(batch_size, seq_len_q, model_depth)

        output = self.output_linear(concat_attention)

        return output, attention_weights
    
    def get_config(self):
        return {
            "model_depth" : self.model_depth,
            "num_heads" : self.num_heads
        }