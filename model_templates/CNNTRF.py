
import tensorflow as tf
from tensorflow.keras.optimizers.schedules import LearningRateSchedule
from tensorflow.keras.optimizers import Adam

from .TransformerEncoder import TransformerEncoder
from .TransformerDecoder import TransformerDecoder
from tensorflow.keras.layers import Input, Dense, LeakyReLU, Conv2D, MaxPooling2D, BatchNormalization, Permute, Reshape, Concatenate, Softmax, Flatten
from tensorflow.keras.models import Model

loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False, reduction='none')

class TransformerLearningRate(LearningRateSchedule):
    def __init__(self, model_depth, warmup_steps=4000):
        super(TransformerLearningRate, self).__init__()

        self.model_depth = model_depth
        self.model_depth_tensor = tf.cast(model_depth, tf.float32)
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.model_depth_tensor) * tf.math.minimum(arg1, arg2)
    
    def get_config(self):
        return {
            "model_depth" : self.model_depth,
            "warmup_steps" : self.warmup_steps
        }


def Get_Custom_Adam_Optimizer(model_depth):
    scheduler = TransformerLearningRate(model_depth)
    t_optimizer = Adam(scheduler, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
    return t_optimizer

def Transformer_Loss_AIAYN(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real,0))
    loss_ = loss_object(real, pred)
    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_sum(loss_)/tf.reduce_sum(mask)

def get_cnn_transformer(conv_filters, pool_layers, image_input_shape, MAX_SEQ_LEN, VOCAB_SIZE,
                      transformer_encoder_layers, transformer_decoder_layers, model_depth, ff_depth, num_heads, POS_ENCODING_INPUT, POS_ENCODING_TARGET):

    #Define convolutional block, remember that we have to scale the mask too
    image_input   =      Input(name='image_input', shape = image_input_shape, dtype=tf.float32)
    decoder_input =      Input(name='decoder_input', shape= (None, ))
    look_ahead_mask    = Input(name='look_ah_mask', shape=(None, MAX_SEQ_LEN+1, MAX_SEQ_LEN+1))

    #=== CONVOLUTIONAL BLOCK ===#
    convNet  = None
    for i in range(len(conv_filters)):
        if i == 0:
            convNet = Conv2D(conv_filters[i], 5, padding='same', name=f'conv{i+1}')(image_input)
        else:
            convNet = Conv2D(conv_filters[i], 3, padding='same', name=f'conv{i+1}')(convNet)

        convNet = BatchNormalization()(convNet)
        convNet = LeakyReLU(alpha=0.2)(convNet)
        convNet = MaxPooling2D(pool_size=(pool_layers[i]), name=f'pool{i+1}')(convNet)

    #=== CONVOLUTIONAL TO TRANSFORMER CONVERSION ===#
    conversion = Permute((2,1,3))(convNet)
    ##(-1, (fixed_height / pool_heights ** pool_layers)*last_conv_filters)

    reshape = (image_input_shape[0] // 2**len(pool_layers)) * conv_filters[-1] #column-level analysis
    #model_depth = conv_filters[-1] #pixel-level analysis
    encoder_input = Reshape(target_shape=(-1, reshape), name='reshape_layer')(conversion) #(batch_size, seq_len, d_model)
    ### NEED TO CONCATENATE (LAST_CONV_SIZE) DUPLIACTES IN THE MASK TO ACHIEVE A CORRECT RESHAPE OF THE MASK
    if(reshape!=model_depth):
        encoder_input = Dense(model_depth)(encoder_input) 

    #encoder_input = Dense(reshape_val, activation="softmax")(encoder_input)

    encoder = TransformerEncoder(transformer_encoder_layers, model_depth, num_heads, ff_depth, POS_ENCODING_INPUT)(encoder_input, mask=None)
    decoder, _ = TransformerDecoder(transformer_decoder_layers, model_depth, num_heads, ff_depth, VOCAB_SIZE, POS_ENCODING_TARGET)(decoder_input,
                                                                                                                                encoder_output = encoder,
                                                                                                                                look_ahead_mask = look_ahead_mask,
                                                                                                                                padding_mask = None)
    #=== TRANSFORMER MODEL IMPLEMENTATION
    out = Dense(VOCAB_SIZE, activation='softmax')(decoder)
    model = Model(inputs=[image_input, decoder_input, look_ahead_mask], outputs=out)
    model.compile(optimizer=Get_Custom_Adam_Optimizer(model_depth), loss=Transformer_Loss_AIAYN)
    model.summary()

    return model
