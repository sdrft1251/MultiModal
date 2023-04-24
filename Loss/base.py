import tensorfow as tf


def cal_loss(model, inputs, targets):
    for_mlm, for_irm, outputs_attn_prob = model(inputs)

    mlm_loss = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction='none')(targets["ans_enc"], for_mlm)
    mlm_mask = tf.cast(tf.not_equal(targets["mask_idx"], 0), tf.float32)
    mlm_loss = tf.reduce_mean(tf.multiply(mlm_loss, mlm_mask))

    irm_loss = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction='none')(targets["cls_ans"], for_irm)
    irm_loss = tf.reduce_mean(irm_loss)

    tot_loss = mlm_loss+irm_loss
    return tot_loss, mlm_loss, irm_loss

