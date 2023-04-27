import tensorflow as tf
import numpy as np


class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, embedding_dim, num_heads=8, name="Multi_Head_Attention", **kwargs):
        super(MultiHeadAttention, self).__init__(name=name, **kwargs)
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads

        assert embedding_dim % self.num_heads == 0

        self.projection_dim = embedding_dim // num_heads
        self.query_dense = tf.keras.layers.Dense(embedding_dim)
        self.key_dense = tf.keras.layers.Dense(embedding_dim)
        self.value_dense = tf.keras.layers.Dense(embedding_dim)
        self.dense = tf.keras.layers.Dense(embedding_dim)

    def scaled_dot_product_attention(self, query, key, value, mask):
        matmul_qk = tf.matmul(query, key, transpose_b=True)
        depth = tf.cast(tf.shape(key)[-1], tf.float32)
        logits = matmul_qk / tf.math.sqrt(depth)
        if mask is not None:
            logits += (mask * -1e9)
            # print(mask.shape, logits.shape)
        attention_weights = tf.nn.softmax(logits, axis=-1)
        output = tf.matmul(attention_weights, value)
        return output, attention_weights

    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.projection_dim))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, inputs):
        # x.shape = [batch_size, seq_len, embedding_dim]
        features, mask = inputs['inputs'], inputs['mask']
        batch_size = tf.shape(features)[0]
        seq_len = tf.shape(features)[1]

        # (batch_size, seq_len, embedding_dim)
        query = self.query_dense(features)
        key = self.key_dense(features)
        value = self.value_dense(features)

        # (batch_size, num_heads, seq_len, projection_dim)
        query = self.split_heads(query, batch_size)  
        key = self.split_heads(key, batch_size)
        value = self.split_heads(value, batch_size)

        scaled_attention, scaled_attention_weights = self.scaled_dot_product_attention(query, key, value, mask)

        # (batch_size, seq_len, num_heads, projection_dim)
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
        
        # (batch_size, seq_len, embedding_dim)
        concat_attention = tf.reshape(scaled_attention, (batch_size, seq_len, self.embedding_dim))

        outputs = self.dense(concat_attention)
        return outputs, scaled_attention_weights

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'embedding_dim': self.embedding_dim,
            'num_heads': self.num_heads,
            "name": self.name
        })
        return config


class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, embedding_dim, num_heads, dff, rate=0.1, name="Transformer_Block", **kwargs):
        super(TransformerBlock, self).__init__(name=name, **kwargs)
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.dff = dff
        self.rate = rate

        self.att = MultiHeadAttention(embedding_dim, num_heads, f"{name}_MultiHeadPart")
        self.ffn = tf.keras.Sequential(
            [tf.keras.layers.Dense(dff, activation="relu"),
             tf.keras.layers.Dense(embedding_dim),]
        )
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, inputs, training):
        attn_output, attn_prob = self.att(inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs["inputs"] + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output), attn_prob

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'embedding_dim': self.embedding_dim,
            'num_heads': self.num_heads,
            'dff': self.dff,
            'rate': self.rate,
            "name": self.name
        })
        return config


class JointEmbeddingLayer(tf.keras.layers.Layer):
    def __init__(self, dims, dff, num_heads, num_layers, lang_len, name="Joint_Embedding_Layer", **kwargs):
        super(JointEmbeddingLayer, self).__init__(name=name, **kwargs)
        self.dims = dims
        self.dff = dff
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.lang_len = lang_len
        self.transformer_blocks = [
            TransformerBlock(dims, num_heads, dff, name=f"Transformer_Block_{i}") for i in range(num_layers)
        ]

    def call(self, inputs):
        attn_probs = []
        outputs, attn_prob = self.transformer_blocks[0](
            {"inputs":inputs["inputs"], "mask":inputs["padding_mask"]}
        )
        attn_probs.append(tf.expand_dims(attn_prob, axis=1))

        if self.num_layers >1:
            for layer_ in self.transformer_blocks[1:]:
                outputs, attn_prob = layer_(
                    {"inputs":outputs, "mask":inputs["padding_mask"]}
                )
                attn_probs.append(tf.expand_dims(attn_prob, axis=1))

        outputs_attn_prob = tf.concat(attn_probs, axis=1)
        return outputs[:,-self.lang_len:], outputs[:,0], outputs_attn_prob

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'dims': self.dims,
            'dff': self.dff,
            'num_heads': self.num_heads,
            'num_layers': self.num_layers,
            'lang_len': self.lang_len,
            "name": self.name
        })
        return config


def get_model(max_len, dims, num_heads, dff, num_layers, name="Joint_Embedding"):
    inputs = tf.keras.layers.Input(shape=(max_len, dims), name=f"{name}_inputs")
    padding_mask = tf.keras.Input(shape=(1, max_len, max_len), name='padding_mask')

    attn_probs = []
    outputs, attn_prob = TransformerBlock(dims, num_heads, dff, name="Transformer_Block_0")({"inputs":inputs, "mask":padding_mask})
    attn_probs.append(tf.expand_dims(attn_prob, axis=1))

    if num_layers>1:
        for i in range(num_layers-1):
            outputs, attn_prob = TransformerBlock(dims, num_heads, dff, name=f"Transformer_Block_{i+1}")({"inputs":outputs, "mask":padding_mask})
            attn_probs.append(tf.expand_dims(attn_prob, axis=1))

    outputs_attn_prob = tf.keras.layers.Concatenate(axis=1)(attn_probs)
    outputs_cls = tf.keras.layers.Dense(dims, activation="tanh")(outputs[:,0])
    return tf.keras.Model(inputs=[inputs, padding_mask], outputs=[outputs, outputs_cls, outputs_attn_prob], name=name)







