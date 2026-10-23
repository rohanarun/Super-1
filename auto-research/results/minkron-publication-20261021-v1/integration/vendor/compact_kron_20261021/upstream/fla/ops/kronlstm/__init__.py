"""Authors' differentiable PyTorch reference operations; no Triton execution."""
from .naive import naive_chunk_kronlstm,naive_recurrent_kronlstm

def chunk_kronlstm(**kwargs):
    assert kwargs.pop('cu_seqlens',None) is None
    return naive_chunk_kronlstm(**kwargs,chunk_size=32)

def fused_recurrent_kronlstm(**kwargs):
    assert kwargs.pop('cu_seqlens',None) is None
    return naive_recurrent_kronlstm(**kwargs)
