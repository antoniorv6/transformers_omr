import tensorflow as tf
from tensorflow.keras.layers import BatchNormalization, Conv2D, Activation, Add, MaxPool2D, AveragePooling2D

class ResNetIdentityBlock(tf.keras.layers.Layer):
    def __init__(self, filters):
        super(ResNetIdentityBlock, self).__init__()

        self.conv2D1 = Conv2D(filters, (3,3), padding='same')
        self.bn1 = BatchNormalization(axis=3)
        self.relu1 = Activation('relu')

        self.conv2D2 = Conv2D(filters, (3,3), padding='same')
        self.bn2 = BatchNormalization(axis=3)

        self.add = Add()
        self.relu2 = Activation('relu')

    def __call__(self, x):
        x_skip = x
        
        # First layer
        x = self.conv2D1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        
        # Second layer
        x = self.conv2D2(x)
        x = self.bn2(x)

        # Skip connection
        x = self.add([x, x_skip])
        x = self.relu2(x)
        return x

class ResNetConvBlock(tf.keras.layers.Layer):
    def __init__(self, filters):
        super(ResNetConvBlock, self).__init__()

        self.conv2D1 = Conv2D(filters, (3,3), padding='same', strides=(2,2))
        self.bn1 = BatchNormalization(axis=3)
        self.relu1 = Activation('relu')

        self.conv2D2 = Conv2D(filters, (3,3), padding='same')
        self.bn2 = BatchNormalization(axis=3)

        self.conv2D_skip = Conv2D(filters, (1,1), padding='same', strides=(2,2))

        self.add = Add()
        self.relu2 = Activation('relu')

    def __call__(self, x):
        x_skip = x
        
        # First layer
        x = self.conv2D1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        
        # Second layer
        x = self.conv2D2(x)
        x = self.bn2(x)

        x_skip = self.conv2D_skip(x_skip)

        # Skip connection
        x = self.add([x, x_skip])
        x = self.relu2(x)
        return x


def ResNet34(input):
    x = Conv2D(64, kernel_size=7, strides=2, padding='same')(input)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPool2D(pool_size=3, strides=2, padding='same')(x)

    block_layers = [3,4,6,3]
    filter_size = 64

    for i in range(4):
        if i == 0:
            for j in range(block_layers[i]):
                x = ResNetIdentityBlock(filter_size)(x)
        else:
            filter_size = filter_size*2
            x = ResNetConvBlock(filter_size)(x)
            for _ in range(block_layers[i]-1):
                x = ResNetIdentityBlock(filter_size)(x)
    
    return x
