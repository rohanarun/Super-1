import hashlib
import json
from dataclasses import asdict
from pathlib import Path
import torch
from model import Config, build

ROOT=Path(__file__).resolve().parent
ARMS=['k8', 'k16', 'k32', 'k16_lean', 'k16_lean_conv', 'transformer_lean']
SEEDS=[25101, 25102, 25103]

def config(arm):
    key=32 if arm=='k32' else 8 if arm=='k8' else 16
    return Config(vocab_size=4096, layers=3, width=192, mlp_hidden=384 if 'lean' in arm else 768,
        transformer_heads=6, kron_heads=96//key, key_dim=key, value_dim=key,
        chunk_size=32, arm='transformer' if arm.startswith('transformer') else 'minkron_8x8',
        enforce_equal_state=False, kron_core='factored', short_conv=4 if 'conv' in arm else 0)

def make_model(arm, seed):
    model=build(config(arm), seed)
    # Name-keyed seeds keep the embedding and shared tensors paired even when
    # MLP dimensions differ. No pretrained weights are used.
    with torch.no_grad():
        for name,p in model.named_parameters():
            if '.mixer.' in name: continue
            if p.ndim==1: p.fill_(1.); continue
            keyed=int.from_bytes(hashlib.sha256(f'{seed}:{name}'.encode()).digest()[:8], 'little')%(2**63)
            gen=torch.Generator().manual_seed(keyed)
            std=.02/(6**.5 if '.fc2.' in name else 1.)
            p.copy_(torch.randn(p.shape, generator=gen)*std)
    return model

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
