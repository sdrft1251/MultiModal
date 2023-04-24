import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

from Model import base as model_ob
from Loss import base as loss_ob
from DataLoader import base dataloader_ob


sig_len = 250*10
sig_dims = 12
text_len = 100
vocab_size = 9000
dims = 256
ffd_dims = 128
num_head = 8
transformer_layer_nums = 5

epochs = 500
batch_size = 128

def grad(model, inputs, targets):
    with tf.GradientTape() as tape:
        tot_loss, mlm_loss, irm_loss = loss_ob.cal_loss(model, inputs, targets)
    return tot_loss, mlm_loss, irm_loss, tape.gradient(tot_loss, model.trainable_variables)


if __name__ == "__main__":
    base_model = model_ob.get_model((sig_len, sig_dims), text_len, vocab_size, dims, ffd_dims, num_head, transformer_layer_nums)
    print(base_model.summary())

    training_generator = dataloader_ob.Dataloader(id_list=None, batch_size=batch_size, vocab_size=vocab_size, sig_len=sig_len, sig_dims=sig_dims, lang_len=text_len, shuffle=True)
    total_len = len(training_generator)
    for ep_ in range(epochs):
        for idx_ in range(total_len):
            vis_inputs, lang_inputs, lang_outputs, mask_idxs  = training_generator.__getitem__(idx_)
            inputs_ = [vis_inputs, lang_inputs]
            targets_ = {
                "mask_idx": mask_idxs,
                "ans_enc": lang_outputs,
                "cls_ans": None
            }
            tot_loss, mlm_loss, irm_loss, grads = grad(model, inputs_, targets_)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))



