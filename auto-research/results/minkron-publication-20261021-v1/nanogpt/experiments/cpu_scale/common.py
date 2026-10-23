"""Explicit ~124M CPU exploration, separate from the equal-state pilot."""
from dataclasses import asdict
import os
import random
import time
import numpy as np
import torch
from experiments.nanogpt_memory.model import Config, build

LAYOUTS = {
    'h12': dict(kron_heads=12, mlp_hidden=4352),
    'h24': dict(kron_heads=24, mlp_hidden=4096),
    'h48': dict(kron_heads=48, mlp_hidden=3584),
    'h72': dict(kron_heads=72, mlp_hidden=3072),
}


def configuration(layout, chunk=32, core='stable'):
    return Config(layers=12, width=768, transformer_heads=12, key_dim=8, value_dim=8,
                  enforce_equal_state=False, chunk_size=chunk, kron_core=core, **LAYOUTS[layout])


def make_optimizer(model, lr, mode='foreach'):
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        excluded = p.ndim < 2 or name == 'embedding.weight' or name.endswith(('A_log', 'dt_bias'))
        (no_decay if excluded else decay).append(p)
    return torch.optim.AdamW([dict(params=decay, weight_decay=.1), dict(params=no_decay, weight_decay=0.)],
                            lr=lr, betas=(.9, .95), eps=1e-8, foreach=mode=='foreach', fused=mode=='fused')


def configure_cpu(threads):
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    torch.set_flush_denormal(True)


def autocast(precision):
    return torch.autocast('cpu', dtype=torch.bfloat16, enabled=precision=='bf16')


def rng_state():
    return dict(torch=torch.get_rng_state(), numpy=np.random.get_state(), python=random.getstate())


def restore_rng(state):
    torch.set_rng_state(state['torch'])
    np.random.set_state(state['numpy'])
    random.setstate(state['python'])


def state_bytes(cfg, batch=1):
    return cfg.layers*cfg.kron_heads*cfg.key_dim*cfg.value_dim*4*batch
