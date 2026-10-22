"""Common decoder contract, with minKron's October 21 recurrence.

Native PyTorch correctness backend. No fused CUDA kernel is claimed.
Kron's pairwise tensors are bounded by chunk_size, never the context length.
All recurrent state and recurrence arithmetic use FP32 (FP64 for checks).
"""
import math
from dataclasses import dataclass
import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class Config:
    vocab_size: int = 50257
    layers: int = 4
    width: int = 256
    mlp_multiplier: int = 4
    transformer_heads: int = 4
    kron_heads: int = 4
    key_dim: int = 8
    value_dim: int = 8
    chunk_size: int = 32
    arm: str = 'minkron_8x8'


def acc_dtype(x):
    return torch.float64 if x.dtype == torch.float64 else torch.float32


class RMSNorm(nn.Module):
    def __init__(self, width, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.eps = eps

    def forward(self, x):
        a = x.to(acc_dtype(x))
        return (a * torch.rsqrt(a.square().mean(-1, keepdim=True) + self.eps)
                * self.weight).to(x.dtype)


def kron_recurrent(q, k, v, g, beta, write, state=None, reset=None):
    """Independent sequential recurrence with explicit before-token resets."""
    dtype = q.dtype
    q, k, v, g, beta, write = [t.to(acc_dtype(q)) for t in (q, k, v, g, beta, write)]
    B, T, H, K = q.shape
    s = q.new_zeros(B, H, K, v.shape[-1]) if state is None else state.to(q.dtype)
    outputs = []
    for t in range(T):
        if reset is not None:
            s = s * (~reset[:, t])[:, None, None, None]
        s = s * g[:, t].exp().unsqueeze(-1)
        vt, kt = v[:, t], k[:, t]
        norm = vt.square().sum(-1)
        inv = norm.clamp_min(1e-18).reciprocal()
        score = ((kt.unsqueeze(-1) * s) * vt.unsqueeze(-2)).sum((-2, -1))
        u = write[:, t] * norm * inv - beta[:, t] * inv * score
        s = s + u[..., None, None] * kt.unsqueeze(-1) * vt.unsqueeze(-2)
        outputs.append((q[:, t].unsqueeze(-1) * s).sum(-2) * K**-.5)
    return torch.stack(outputs, 1).to(dtype), s


def kron_chunk(q, k, v, g, beta, write, state=None, reset=None):
    """Direct triangular solve, extended with per-example document masks.

    A reset zeros the incoming state before that token. Pairwise interactions
    only connect positions in the same document; the carried state contributes
    only before the first reset. No division by cumulative decay is used.
    """
    orig = q.dtype
    q, k, v, g, beta, write = [t.transpose(1, 2).to(acc_dtype(q))
                              for t in (q, k, v, g, beta, write)]
    B, H, T, K = q.shape
    s = q.new_zeros(B, H, K, v.shape[-1]) if state is None else state.to(q.dtype)
    if T == 1:
        if reset is not None:
            s = s * (~reset[:, 0])[:, None, None, None]
        s = s * g[:, :, 0].exp().unsqueeze(-1)
        vt, kt = v[:, :, 0], k[:, :, 0]
        norm = vt.square().sum(-1)
        inv = norm.clamp_min(1e-18).reciprocal()
        score = ((kt.unsqueeze(-1) * s) * vt.unsqueeze(-2)).sum((-2, -1))
        u = write[:, :, 0] * norm * inv - beta[:, :, 0] * inv * score
        s = s + u[..., None, None] * kt.unsqueeze(-1) * vt.unsqueeze(-2)
        out = (q[:, :, 0].unsqueeze(-1) * s).sum(-2) * K**-.5
        return out[:, None].to(orig), s
    docs = torch.zeros(B, T, dtype=torch.long, device=q.device) if reset is None else reset.long().cumsum(1)
    causal = torch.ones(T, T, device=q.device, dtype=torch.bool).tril()
    valid = (docs[:, :, None] == docs[:, None, :])[:, None] & causal
    gc = g.cumsum(-2)
    # Mask BEFORE exp: upper-triangle positive differences can otherwise overflow.
    relative = (gc.unsqueeze(-2) - gc.unsqueeze(-3)).masked_fill(~valid[..., None], 0).exp()
    relative = relative * valid[..., None]
    kk = (k.unsqueeze(-2) * k.unsqueeze(-3) * relative).sum(-1)
    qk = (q.unsqueeze(-2) * k.unsqueeze(-3) * relative).sum(-1) * K**-.5
    norm = v.square().sum(-1)
    inv = norm.clamp_min(1e-18).reciprocal()
    bs = beta * inv
    matrix = (bs.unsqueeze(-1) * kk * (v @ v.transpose(-1, -2))).tril(-1)
    matrix = matrix + torch.eye(T, device=q.device, dtype=q.dtype)
    origin_decay = gc.exp() * (docs == 0)[:, None, :, None]
    score = (((k * origin_decay) @ s) * v).sum(-1)
    rhs = write * norm * inv - bs * score
    u = torch.linalg.solve_triangular(matrix, rhs[..., None], upper=False, unitriangular=True)
    uv = u * v
    out = K**-.5 * ((q * origin_decay) @ s) + qk @ uv
    tail = k * (gc[..., -1:, :] - gc).exp() * (docs == docs[:, -1:])[:, None, :, None]
    final = origin_decay[..., -1, :, None] * s + tail.transpose(-1, -2) @ uv
    return out.transpose(1, 2).contiguous().to(orig), final


class MinKron(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        D, H, K, V = cfg.width, cfg.kron_heads, cfg.key_dim, cfg.value_dim
        assert H * K * V == D, 'Equal-state contract requires H*K*V == width'
        self.H, self.K, self.V, self.chunk_size = H, K, V, cfg.chunk_size
        self.backend = 'chunk'
        # Construct in source order to preserve the October 21 initialization.
        qkv = nn.Linear(D, H*(2*K+V), bias=False)
        erase = nn.Linear(D, H, bias=False)
        write = nn.Linear(D, H, bias=False)
        forget0 = nn.Linear(D, V, bias=False)
        self.forget1 = nn.Linear(V, H*K, bias=False)
        self.A_log = nn.Parameter(torch.empty(H))
        self.dt_bias = nn.Parameter(torch.empty(H*K))
        with torch.no_grad():
            self.A_log.uniform_(0, 16).log_()
            self.dt_bias.uniform_()
            dt = torch.exp(self.dt_bias*(math.log(.1)-math.log(.001))+math.log(.001)).clamp_min(1e-4)
            self.dt_bias.copy_(dt + torch.log(-torch.expm1(-dt)))
        gate = nn.Linear(D, H*V, bias=True)
        self.out_norm = RMSNorm(V)
        self.out_proj = nn.Linear(H*V, D, bias=False)
        nn.init.zeros_(gate.weight)
        nn.init.constant_(gate.bias, 3.)
        # Permanent packing: same per-element optimizer semantics, no per-call cat.
        self.packed_weight = nn.Parameter(torch.cat([qkv.weight, erase.weight, write.weight,
                                                   forget0.weight, gate.weight]).detach())
        self.gate_bias = nn.Parameter(gate.bias.detach().clone())

    def forward(self, x, state=None, reset=None):
        B, T, _ = x.shape
        H, K, V = self.H, self.K, self.V
        projection = F.linear(x, self.packed_weight)
        qkv, erase, write, f, gate = projection.split([H*(2*K+V), H, H, V, H*V], -1)
        f = self.forget1(f)
        # AMP applies to projections only. Normalization, decay, solve and state
        # remain FP32. This mixed precision policy still requires GPU qualification.
        with torch.autocast(device_type=x.device.type, enabled=False):
            dtype = acc_dtype(x)
            q, k, v = F.silu(qkv.to(dtype)).split([H*K, H*K, H*V], -1)
            q, k, v = [F.normalize(t.reshape(B, T, H, d), dim=-1)
                       for t, d in ((q, K), (k, K), (v, V))]
            g = -self.A_log.to(dtype).exp()[None, None, :, None] * F.softplus(
                f.to(dtype) + self.dt_bias.to(dtype)).reshape(B, T, H, K)
            beta, write = erase.to(dtype).sigmoid(), write.to(dtype).sigmoid()
            outputs = []
            core = kron_recurrent if self.backend == 'recurrent' else kron_chunk
            for start in range(0, T, self.chunk_size):
                stop = start + self.chunk_size
                args = [a[:, start:stop] for a in (q, k, v, g, beta, write)]
                out, state = core(*args, state, None if reset is None else reset[:, start:stop])
                outputs.append(out)
            out = torch.cat(outputs, 1)
            out = self.out_norm(out) * (gate.to(dtype) + self.gate_bias).reshape(B, T, H, V).sigmoid()
        return self.out_proj(out.reshape(B, T, H*V).to(projection.dtype)), state


def affine_scan(a, b, state):
    distance = 1
    while distance < a.shape[1]:
        b = torch.cat((b[:, :distance], b[:, distance:] + a[:, distance:]*b[:, :-distance]), 1)
        a = torch.cat((a[:, :distance], a[:, distance:]*a[:, :-distance]), 1)
        distance *= 2
    return b + a*state[:, None]


class MinGRU(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.projection = nn.Linear(cfg.width, 2*cfg.width)
        nn.init.zeros_(self.projection.bias)
        self.chunk_size = cfg.chunk_size
        self.backend = 'scan'

    def forward(self, x, state=None, reset=None):
        p = self.projection(x)
        with torch.autocast(device_type=x.device.type, enabled=False):
            candidate, gate = p.to(acc_dtype(x)).chunk(2, -1)
            z = gate.sigmoid()
            a, b = 1-z, z*candidate
            if reset is not None:
                a = a * (~reset)[..., None]
            state = a.new_zeros(a.shape[0], a.shape[-1]) if state is None else state.to(a.dtype)
            outputs = []
            for start in range(0, x.shape[1], self.chunk_size):
                aa, bb = a[:, start:start+self.chunk_size], b[:, start:start+self.chunk_size]
                if self.backend == 'recurrent':
                    out = []
                    for at, bt in zip(aa.unbind(1), bb.unbind(1)):
                        state = at*state + bt
                        out.append(state)
                    out = torch.stack(out, 1)
                else:
                    out = affine_scan(aa, bb, state)
                state = out[:, -1].clone()  # own only the last state
                outputs.append(out)
        return torch.cat(outputs, 1).to(p.dtype), state


def rope(x, positions):
    half = x.shape[-1] // 2
    freq = torch.exp(-math.log(10000) * torch.arange(half, device=x.device).float() / half)
    angle = positions[:, None, :, None].float() * freq
    cosine, sine = angle.cos().to(x.dtype), angle.sin().to(x.dtype)
    a, b = x.chunk(2, -1)
    return torch.cat((a*cosine-b*sine, a*sine+b*cosine), -1)


class Attention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.heads = cfg.transformer_heads
        assert cfg.width % (2*self.heads) == 0
        self.qkv = nn.Linear(cfg.width, 3*cfg.width, bias=False)
        self.out_proj = nn.Linear(cfg.width, cfg.width, bias=False)

    def forward(self, x, state=None, reset=None):
        B, T, D = x.shape
        q, k, v = self.qkv(x).reshape(B, T, 3, self.heads, D//self.heads).unbind(2)
        q, k, v = [a.transpose(1, 2) for a in (q, k, v)]
        zero = torch.zeros(B, dtype=torch.long, device=x.device)
        start = zero if state is None else state['last_position']+1
        prior_doc = zero if state is None else state['last_document']
        idx = start[:, None]+torch.arange(T, device=x.device)
        resets = torch.zeros(B, T, device=x.device, dtype=torch.bool) if reset is None else reset
        reset_idx = torch.where(resets, idx, -1).cummax(1).values
        positions = torch.where(reset_idx >= 0, idx-reset_idx, idx)
        docs = resets.long().cumsum(1)+prior_doc[:, None]
        q, k = rope(q, positions), rope(k, positions)
        old_len = 0 if state is None else state['key'].shape[2]
        all_docs = docs if state is None else torch.cat((state['documents'], docs), 1)
        if state is not None:
            k, v = torch.cat((state['key'], k), 2), torch.cat((state['value'], v), 2)
        if state is None and reset is None:
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            causal = torch.arange(k.shape[2], device=x.device)[None, :] <= (old_len+torch.arange(T, device=x.device))[:, None]
            mask = (docs[:, :, None] == all_docs[:, None, :]) & causal
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask[:, None])
        new_state = dict(key=k.clone(), value=v.clone(), documents=all_docs.clone(),
                         last_position=positions[:, -1].clone(), last_document=docs[:, -1].clone())
        return self.out_proj(out.transpose(1, 2).reshape(B, T, D)), new_state


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.norm1 = RMSNorm(cfg.width)
        self.mixer = {'minkron_8x8': MinKron, 'mingru_linear': MinGRU, 'transformer': Attention}[cfg.arm](cfg)
        self.norm2 = RMSNorm(cfg.width)
        self.fc1 = nn.Linear(cfg.width, cfg.width*cfg.mlp_multiplier, bias=False)
        self.fc2 = nn.Linear(cfg.width*cfg.mlp_multiplier, cfg.width, bias=False)

    def forward(self, x, state, reset):
        y, state = self.mixer(self.norm1(x), state, reset)
        x = x+y
        return x+self.fc2(F.relu(self.fc1(self.norm2(x))).square()), state


class MemoryLM(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.embedding = nn.Embedding(cfg.vocab_size, cfg.width)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.layers)])
        self.final_norm = RMSNorm(cfg.width)

    def forward(self, tokens, state=None, reset=None):
        if tokens.shape[1] == 0:
            raise ValueError('At least one token is required')
        x = self.embedding(tokens)
        new_state = []
        for i, block in enumerate(self.blocks):
            x, s = block(x, None if state is None else state[i], reset)
            new_state.append(s)
        return F.linear(self.final_norm(x), self.embedding.weight), new_state


def build(cfg, seed):
    # Separate, deterministic mixer RNG and shared RNG. Copy identical shared
    # tensors generated in a common order, independent of mixer parameter shapes.
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        model = MemoryLM(cfg)
        torch.manual_seed(seed+1000000)
        common = {}
        for name, p in model.named_parameters():
            if '.mixer.' in name:
                continue
            value = torch.empty_like(p)
            if p.ndim == 1:
                value.fill_(1.)
            else:
                nn.init.normal_(value, std=.02/(math.sqrt(2*cfg.layers) if '.fc2.' in name else 1.))
            common[name] = value
        with torch.no_grad():
            for name, p in model.named_parameters():
                if name in common:
                    p.copy_(common[name])
    return model


def optimizer(model, lr, weight_decay=.1):
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        excluded = p.ndim < 2 or name == 'embedding.weight' or name.endswith(('A_log', 'dt_bias'))
        (no_decay if excluded else decay).append(p)
    return torch.optim.AdamW([{'params': decay, 'weight_decay': weight_decay},
                             {'params': no_decay, 'weight_decay': 0.}], lr=lr, betas=(.9, .95))


def tensor_bytes(state):
    if isinstance(state, torch.Tensor):
        return state.numel()*state.element_size(), state.untyped_storage().nbytes()
    items = state.values() if isinstance(state, dict) else state
    sizes = [tensor_bytes(x) for x in items]
    return tuple(sum(s[i] for s in sizes) for i in (0, 1))
