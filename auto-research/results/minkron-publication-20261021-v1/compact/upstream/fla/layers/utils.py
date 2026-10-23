"""Minimal explicit cache API for this fixed-length CPU experiment."""
def get_layer_cache(layer,cache):
    return None if cache is None else cache.get(layer.layer_idx)
def update_layer_cache(layer,cache,recurrent_state,conv_state,offset):
    if cache is not None:cache[layer.layer_idx]={'recurrent_state':recurrent_state,'conv_state':conv_state}
def unpad_hidden_states(hidden,cu_seqlens,attention_mask,q_len):
    if cu_seqlens is not None or attention_mask is not None:raise NotImplementedError('Use unpadded equal-length batches')
    return hidden,None,None
def repad_hidden_states(hidden,indices,batch_size,q_len):
    assert indices is None
    return hidden
