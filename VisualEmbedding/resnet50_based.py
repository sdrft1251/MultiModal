import tensorflow as tf


class BottleNeck(tf.keras.layers.Layer):
    def __init__(self, filter_num, stride=1, **kwargs):
        super(BottleNeck, self).__init__(**kwargs)
        self.filter_num = filter_num
        self.stride = stride

        self.conv1 = tf.keras.layers.Conv1D(filters=filter_num, kernel_size=1, strides=1, padding='same')
        self.conv2 = tf.keras.layers.Conv1D(filters=filter_num, kernel_size=3, strides=stride, padding='same')
        self.conv3 = tf.keras.layers.Conv1D(filters=filter_num * 4, kernel_size=1, strides=1, padding='same')

        self.down_conv1 = tf.keras.layers.Conv1D(filters=filter_num*4, kernel_size=1, strides=stride)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'filter_num': self.filter_num,
            'stride': self.stride
        })
        return config

    def call(self, inputs, training=None, **kwargs):
        residual = self.down_conv1(inputs)

        x = self.conv1(inputs)
        x = tf.nn.relu(x)
        x = self.conv2(x)
        x = tf.nn.relu(x)
        x = self.conv3(x)

        output = tf.nn.relu(tf.keras.layers.add([residual, x]))
        return output


def make_bottleneck_layer(filter_num, blocks, name_end, stride=1):
    res_block = tf.keras.Sequential(name=f"Resblock_{name_end}")
    res_block.add(BottleNeck(filter_num, stride=stride))
    for _ in range(1, blocks):
        res_block.add(BottleNeck(filter_num, stride=1))
    return res_block


class VisualEncodingLayer(tf.keras.layers.Layer):
    def __init__(self, num_layers, first_dims=64, filters=[64,128,256,512], blocks=[3,4,6,3], strides=[1,2,2,2], name="Visual_Encoding_Layer", **kwargs):
        super(VisualEncodingLayer, self).__init__(name=name, **kwargs)
        self.num_layers = num_layers
        self.first_dims = first_dims
        self.filters = filters
        self.blocks = blocks
        self.strides = strides

        self.interp_layer = tf.keras.Sequential(
            [
                tf.keras.layers.Conv1D(first_dims, kernel_size=7, strides=2, padding='same'),
                tf.keras.layers.ReLU(),
                tf.keras.layers.MaxPool1D(pool_size=3, strides=2, padding='same')
            ]
        )

        self.res_blocks = tf.keras.Sequential(
            [make_bottleneck_layer(filter_num=filters[i], name_end=f"{i}", blocks=blocks[i], stride=strides[i]) for i in range(num_layers)]
        )

    def call(self, inputs):
        outputs = self.interp_layer(inputs)
        outputs = self.res_blocks(outputs)
        return outputs

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'num_layers': self.num_layers,
            'first_dims': self.first_dims,
            'filters': self.filters,
            'blocks': self.blocks,
            'strides': self.strides,
        })
        return config


def get_model(time_len, dims, first_dims=64, filters=[64,128,256,512], blocks=[3,4,6,3], strides=[1,2,2,2], name="VisEncoderResnet50Based"):
    input_tens = tf.keras.Input(shape=(time_len, dims))

    x = tf.keras.layers.Conv1D(first_dims, kernel_size=7, strides=2, padding='same')(input_tens)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.MaxPool1D(pool_size=3, strides=2, padding='same')(x)

    for i in range(len(filters)):
        x = make_bottleneck_layer(filter_num=filters[i], name_end=f"{i}", blocks=blocks[i], stride=strides[i])(x)
        
    model = tf.keras.Model(inputs=input_tens, outputs=x, name=name)
    return model





