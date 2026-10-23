"""Unqualified same-math core variants for the proposed synthetic canary only.

Adapted from the preserved minKron direct core, credited to the parent PR #15
and its NX-AI/KronLSTM provenance. No architecture, gate or norm-floor change.
"""


def chunk(q, k, v, g, beta, write, state=None, reset=None, *, interval=False, scaled=False):
    import torch

    original_dtype = q.dtype
    dtype = torch.float64 if q.dtype == torch.float64 else torch.float32
    q, k, v, g = [item.to(dtype).transpose(1, 2) for item in (q, k, v, g)]
    beta, write = [item.to(dtype).transpose(1, 2) for item in (beta, write)]
    batch, heads, length, keys = q.shape
    current = q.new_zeros(batch, heads, keys, v.shape[-1]) if state is None else state.to(dtype)
    docs = torch.zeros(batch, length, dtype=torch.long, device=q.device) if reset is None else reset.long().cumsum(1)
    causal = torch.ones(length, length, dtype=torch.bool, device=q.device).tril()
    valid = (docs[:, :, None] == docs[:, None, :])[:, None] & causal
    if interval:
        decay = g.exp()
        positions = torch.arange(length, device=q.device)
        after_source = positions[None, :] > positions[:, None]
        factors = torch.where(after_source[..., None], decay.unsqueeze(-3), torch.ones_like(decay).unsqueeze(-3))
        relative = factors.cumprod(-2).transpose(-3, -2).masked_fill(~valid[..., None], 0)
        origin = decay.cumprod(-2) * (docs == 0)[:, None, :, None]
        tail = k * relative[..., -1, :, :]
    else:
        cumulative = g.cumsum(-2)
        relative = (cumulative.unsqueeze(-2) - cumulative.unsqueeze(-3)).masked_fill(~valid[..., None], 0).exp()
        relative = relative * valid[..., None]
        origin = cumulative.exp() * (docs == 0)[:, None, :, None]
        tail = k * (cumulative[..., -1:, :] - cumulative).exp() * (docs == docs[:, -1:])[:, None, :, None]
    kk = (k.unsqueeze(-2) * k.unsqueeze(-3) * relative).sum(-1)
    qk = (q.unsqueeze(-2) * k.unsqueeze(-3) * relative).sum(-1) * keys**-.5
    norm = v.square().sum(-1)
    denominator = norm.clamp_min(1e-18)
    if scaled:
        inverse_root = denominator.rsqrt()
        vectors = v * inverse_root[..., None]
        coefficient = beta
        drive = write * norm * inverse_root
    else:
        inverse = denominator.reciprocal()
        vectors = v
        coefficient = beta * inverse
        drive = write * norm * inverse
    matrix = (coefficient.unsqueeze(-1) * kk * (vectors @ vectors.transpose(-1, -2))).tril(-1)
    matrix = matrix + torch.eye(length, dtype=q.dtype, device=q.device)
    score = (((k * origin) @ current) * vectors).sum(-1)
    rhs = drive - coefficient * score
    updates = torch.linalg.solve_triangular(matrix, rhs[..., None], upper=False, unitriangular=True)
    weighted = updates * vectors
    outputs = keys**-.5 * ((q * origin) @ current) + qk @ weighted
    final = origin[..., -1, :, None] * current + tail.transpose(-1, -2) @ weighted
    return outputs.transpose(1, 2).contiguous().to(original_dtype), final
