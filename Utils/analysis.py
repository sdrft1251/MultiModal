import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import seaborn as sns


def head_fusion(attention_scores, method="mean"):
    results = np.identity(attention_scores.shape[-1])
    for layer_idx in range(attention_scores.shape[1]):
        layer_attention_scores = attention_scores[:,layer_idx,:,:,:]   # (1, 1, num_heads, seq_len, seq_len)
        if method == "mean":
            layer_attention_scores_fusion = layer_attention_scores.mean(axis=-3)
        elif method == "min":
            layer_attention_scores_fusion = layer_attention_scores.min(axis=-3)
        elif method == "max":
            layer_attention_scores_fusion = layer_attention_scores.max(axis=-3)
            
        layer_attention_scores_fusion_med = np.median(layer_attention_scores_fusion, axis=-1).reshape(1,-1,1)
        layer_attention_scores_fusion[layer_attention_scores_fusion<layer_attention_scores_fusion_med] = 0
        
        I_vector = np.identity(layer_attention_scores_fusion.shape[-1])
        
        A_vector = (layer_attention_scores_fusion+1.0*I_vector)/2
        A_vector = A_vector / A_vector.sum(axis=-1)
        results = np.matmul(A_vector, results)
    mask = results[0,0,1:-1]
    mask = mask/np.max(mask)
    return mask


def make_new_input(sig_part, lang_pre, model, lang_len):
    sig_part_reshape  = sig_part.reshape(-1,1280,12)
    lang_pre_reshape = lang_pre.reshape(-1,lang_len)    
    for_mlm, outputs_attn_prob = model([sig_part_reshape,lang_pre_reshape])
    return for_mlm, outputs_attn_prob


def resampling_sig(sig, time_size):
    input_beat_length = len(sig)
    x = np.linspace(0, input_beat_length, num=input_beat_length, endpoint=True)
    f = interp1d(x, sig, axis=0)
    interpolated_beat_length = int(time_size)
    interpolated_beat = np.linspace(0, input_beat_length, num=interpolated_beat_length, endpoint=True)
    return f(interpolated_beat)


def inerence_process(sig_part, model_obj, tokenizer_obj, lang_len):
    lang_for_infer = np.zeros(lang_len, dtype=np.float32)
    lang_for_infer[0] = 2.0
    
    mask_dumps = []
    for i in range(1, lang_len):
        lang_for_infer[i] = 4.0
        for_mlm, outputs_attn_prob = make_new_input(sig_part, lang_for_infer, model_obj, tokenizer_obj)
        predicted_token = np.argmax(for_mlm[0][i])
        if predicted_token == 3.:
            break
        else:
            lang_for_infer[i] = predicted_token
    
        mask = head_fusion(outputs_attn_prob.numpy()[:,:,:,:-lang_len,:-lang_len])
        mask_resampled = resampling_sig(mask, 1280)
        mask_dumps.append(mask_resampled.reshape(1, 1280))
    return lang_for_infer.astype(np.int32), np.concatenate(mask_dumps, axis=0)



def ploting_12leads(sigs, mask, ans_text, pred_text):
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Answer text ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print(ans_text)
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Output text ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print(pred_text)
    grid_h_size = [3]
    for _ in range(12):
        grid_h_size.append(1)
    lead_name = ["I","II","III","aVR","aVL","aVF","V1","V2","V3","V4","V5","V6"]
    fig, axes_all = plt.subplots(13,1, sharex=True, constrained_layout=True, figsize=(10,12), gridspec_kw={'height_ratios': grid_h_size})
    sns.heatmap(mask, cmap="bwr", ax=axes_all[0])
    for idx, axes in enumerate(axes_all[1:]):
        axes.plot(sigs[:,idx])
        axes.set_title(lead_name[idx])
    plt.show()