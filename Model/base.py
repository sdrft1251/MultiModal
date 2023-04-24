import tensorflow as tf
import numpy as np

from JointEmbedding import bert_based
from VisualEmbedding import resnet50_based

# TokenAndPositionEmbedding
class LanguageEmbedding(tf.keras.layers.Layer):
    def __init__(self, max_len, vocab_size, embedding_dim, name="Language_Embedding"):
        super(LanguageEmbedding, self).__init__(name=name)
        self.token_emb = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.pos_emb = tf.keras.layers.Embedding(max_len, embedding_dim)
        self.seg_emb = tf.keras.layers.Embedding(2, embedding_dim)

    def call(self, x):
        max_len = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=max_len, delta=1)
        seg = tf.ones(max_len)

        lang_enc_emb = self.token_emb(x)
        lang_pos_emb = self.pos_emb(positions)
        lang_seg_emb = self.seg_emb(seg)
        return lang_enc_emb + lang_pos_emb + lang_seg_emb


# TokenAndPositionEmbedding
class VisualEmbedding(tf.keras.layers.Layer):
    def __init__(self, sig_len, hid_dims, embedding_dim, name="Visual_Embedding"):
        super(VisualEmbedding, self).__init__(name=name)
        self.pos_emb = tf.keras.layers.Embedding(sig_len+2, hid_dims)
        self.seg_emb = tf.keras.layers.Embedding(2, hid_dims)

        self.projection_emb = tf.keras.layers.Dense(embedding_dim)

    def call(self, x):
        batch_size = tf.shape(x)[0]
        sig_len = tf.shape(x)[-2]
        hid_dims = tf.shape(x)[-1]

        CLS_TOKEN = 1
        SEP_TOKEN = 2
        cls_token = tf.ones((batch_size, 1, hid_dims))*CLS_TOKEN
        sep_token = tf.ones((batch_size, 1, hid_dims))*SEP_TOKEN

        positions = tf.range(start=0, limit=sig_len+2, delta=1)
        seg = tf.zeros(sig_len+2)
        vis_pos_emb = self.pos_emb(positions)
        vis_seg_emb = self.seg_emb(seg)

        vis_emb = tf.concat([cls_token, x, sep_token], -2)
        vis_emb = vis_emb + vis_pos_emb + vis_seg_emb
        vis_emb = self.projection_emb(vis_emb)
        return vis_emb


def create_padding_mask(x):
    mask = tf.cast(tf.math.equal(x, 0), tf.float32)
    return mask[:, tf.newaxis, tf.newaxis, :]

    
def create_seq2seq_mask(x):
    vis_emb, lang_inputs = x["vis_emb"], x["lang_inputs"]

    vis_len = tf.shape(vis_emb)[-2]
    lang_len = tf.shape(lang_inputs)[-1]
    batch_size = tf.shape(lang_inputs)[0]
    
    # Make Vis Mask
    padd_for_vis_vis = tf.zeros((vis_len, vis_len))
    padd_for_vis_lang = tf.ones((vis_len, lang_len))
    padd_for_vis = tf.concat([padd_for_vis_vis, padd_for_vis_lang], -1)
    # Make Lang Mask
    padd_for_lang_lang = 1 - tf.linalg.band_part(tf.ones((lang_len, lang_len)), -1, 0)
    padd_for_lang_vis = tf.zeros((lang_len, vis_len))
    padd_for_lang = tf.concat([padd_for_lang_vis, padd_for_lang_lang], -1)
    # Make All Mask
    padd_for_all = tf.concat([padd_for_vis, padd_for_lang], -2)
    # Make Padd Mask
    padd_ = tf.ones((batch_size, vis_len))
    new_lang_input = tf.concat([padd_, lang_inputs], axis=-1)
    
    padding_mask = create_padding_mask(new_lang_input)
    return tf.maximum(padd_for_all, padding_mask)


def get_model(sig_size, lang_len, vocab_size, embedding_dim, dff, num_heads, bert_layer_num,
vis_first_dims=64, vis_filters=[64,128,256,512], vis_blocks=[3,4,6,3], vis_strides=[1,2,2,2], name="BaseModel"):
    
    vis_inputs = tf.keras.Input(shape=sig_size, name="Visual_Input")
    lang_inputs = tf.keras.Input(shape=(lang_len), name="Language_Input")

    ### Language Embedding
    lang_emb = LanguageEmbedding(lang_len, vocab_size, embedding_dim)(lang_inputs)
    ### Signal Embedding
    vis_emb = resnet50_based.VisualEncodingLayer(
        num_layers=len(vis_filters), first_dims=vis_first_dims, filters=vis_filters, blocks=vis_blocks, strides=vis_strides
    )(vis_inputs)
    vis_emb = VisualEmbedding(vis_emb.shape[-2], vis_emb.shape[-1], embedding_dim)(vis_emb)

    ### Get All Embedded vector
    outputs = tf.keras.layers.Concatenate(axis=-2)([vis_emb, lang_emb])

    ### Create Padding Mask
    padding_mask = tf.keras.layers.Lambda(
        create_seq2seq_mask, output_shape=(1, outputs.shape[-2], outputs.shape[-2]),
        name='Seq2Seq_Mask')({"vis_emb":vis_emb, "lang_inputs":lang_inputs})

    ### Joint Embedding
    outputs, outputs_cls, outputs_attn_prob = bert_based.JointEmbeddingLayer(dims=embedding_dim, dff=dff, num_heads=num_heads, num_layers=bert_layer_num, lang_len=lang_len)({
        "inputs": outputs, "padding_mask": padding_mask
    })

    ### Get Results
    for_mlm = tf.keras.layers.Dense(units=vocab_size, name="Output_for_MLM")(outputs)
    for_irm = tf.keras.layers.Dense(units=embedding_dim, name="for_IRM")(outputs_cls)
    for_irm = tf.keras.layers.Dense(units=2, name="Output_for_IRM")(for_irm)

    return tf.keras.Model(inputs=[vis_inputs, lang_inputs], outputs=[for_mlm, for_irm, outputs_attn_prob], name=name)



    


