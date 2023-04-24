from tensorflow.keras.utils import Sequence
import numpy as np

import re


class Dataloader(Sequence):
    def __init__(self, id_list, batch_size, vocab_size, tokenizer, sig_len=int(250*10), sig_dims=12, lang_len=100, shuffle=True):
        self.id_list = id_list
        self.batch_size = batch_size
        self.vocab_size = vocab_size
        self.tokenizer = tokenizer
        self.sig_len = sig_len
        self.sig_dims = sig_dims
        self.lang_len = lang_len
        self.shuffle = shuffle

    def __len__(self):
        return int(np.floor(len(self.id_list) / self.batch_size))

    def on_epoch_end(self):
        self.indices = np.arange(len(self.id_list))
        if self.shuffle == True:
            np.random.shuffle(self.indices)

    def __getitem__(self, idx):
        inds_ = self.indices[idx*self.batch_size:(idx+1)*self.batch_size]
        target_id_list = [self.id_list[i] for i in inds_]
        vis_inputs, lang_inputs, lang_outputs, mask_idxs = self.__data_generation(target_id_list)
        return vis_inputs, lang_inputs, lang_outputs, mask_idxs
        
    def __data_generation(self, target_id_list):
        vis_inputs = np.zeros((self.batch_size, self.sig_len, self.sig_dims), dtype=np.float32)
        lang_inputs = np.zeros((self.batch_size, self.lang_len), dtype=np.float32)
        lang_outputs = np.zeros((self.batch_size, self.lang_len), dtype=np.float32)
        mask_idxs = np.zeros((self.batch_size, self.lang_len), dtype=np.float32)
        for idx, target_id in enumerate(target_id_list):
            sig = None
            text = None

            lang_tokens = tokenizer.encode(self.preprocess_sentence(text))
            lang_tokens_origin = lang_tokens.copy()
            lang_tokens_origin = np.array(list(lang_tokens_origin) + [SEP_TOKEN])

            lang_tokens, mask_idx = text_input_gen(lang_tokens)
            lang_tokens = np.array(list(lang_tokens) + [SEP_TOKEN])

            vis_inputs[idx,:,:] = sig
            lang_inputs[idx,:len(lang_tokens)] = lang_tokens
            lang_outputs[idx,:len(lang_tokens_origin)] = lang_tokens_origin
            mask_idxs[idx,:] = mask_idx
        return vis_inputs, lang_inputs, lang_outputs, mask_idxs

    def preprocess_sentence(self, sentence):
        sentence = re.sub(r"([?.!,])", r" \1 ", sentence)
        sentence = sentence.strip()
        return sentence

    def text_input_gen(self, lang_tokens):
        tokens_num = len(lang_tokens)
        change_num = int(tokens_num*0.15)

        tokens_idx = np.arange(tokens_num)
        np.random.shuffle(tokens_idx)

        change_idxs = tokens_idx[:change_num]
        for idx_ in change_idxs:
            rand_val = np.random.randint(100)
            if rand_val<80:     # Mask Token
                lang_tokens[idx_] = MASK_TOKEN
            elif rand_val<90:   # Random Token
                lang_tokens[idx_] = np.random.randint(vocab_size)
            # else ->           # Original Token

        mask_idx = np.zeros(self.lang_len)
        np.put(mask_idx, change_idxs, 1)
        return lang_tokens, mask_idx

