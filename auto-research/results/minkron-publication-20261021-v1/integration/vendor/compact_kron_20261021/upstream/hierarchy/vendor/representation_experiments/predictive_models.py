"""Equal-storage students for predictive-state compilation."""
from __future__ import annotations

import math
from dataclasses import dataclass
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F

from representation_experiments.common import parameter_count


class StudentBase(nn.Module):
    def __init__(self, vocab_size: int, width: int, max_context: int):
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.width = int(width)
        self.max_context = int(max_context)
        self.embedding = nn.Embedding(vocab_size, width)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=width ** -0.5)
        self.final_norm = nn.LayerNorm(width)

    def logits(self, hidden: torch.Tensor) -> torch.Tensor:
        return F.linear(self.final_norm(hidden), self.embedding.weight)

    def stored_state_elements(self, batch_size: int = 1, context: int = 512) -> int:
        raise NotImplementedError

    def output_head_macs_per_token(self) -> int:
        return self.width * self.vocab_size

    def estimated_macs_per_token(self, context: int = 512) -> int:
        raise NotImplementedError


class TransformerStudent(StudentBase):
    kind = "transformer"

    def __init__(self, vocab_size: int, width: int, max_context: int,
                 layers: int, heads: int, expansion: int = 4,
                 position_encoding: str = "learned"):
        super().__init__(vocab_size, width, max_context)
        if width % heads:
            raise ValueError("transformer width must divide evenly into heads")
        if position_encoding not in ("learned", "sinusoidal"):
            raise ValueError("position_encoding must be learned or sinusoidal")
        self.layers = int(layers)
        self.heads = int(heads)
        self.expansion = int(expansion)
        self.position_encoding = position_encoding
        self.position = (nn.Embedding(max_context, width)
                         if position_encoding == "learned" else None)
        block = nn.TransformerEncoderLayer(
            d_model=width, nhead=heads, dim_feedforward=expansion * width,
            activation="gelu", batch_first=True, norm_first=True)
        self.blocks = nn.TransformerEncoder(block, num_layers=layers)

    def forward(self, input_ids: torch.Tensor):
        length = input_ids.shape[1]
        if length > self.max_context:
            input_ids = input_ids[:, -self.max_context:]
            length = self.max_context
        positions = torch.arange(length, device=input_ids.device)
        if self.position is not None:
            position_values = self.position(positions)
        else:
            half = self.width // 2
            frequencies = torch.exp(
                -math.log(10000.0) * torch.arange(
                    half, device=input_ids.device, dtype=torch.float32)
                / max(half - 1, 1))
            angles = positions.float().unsqueeze(1) * frequencies.unsqueeze(0)
            position_values = torch.cat((angles.sin(), angles.cos()), dim=-1)
            if position_values.shape[-1] < self.width:
                position_values = F.pad(position_values, (0, 1))
            position_values = position_values.to(self.embedding.weight.dtype)
        hidden = self.embedding(input_ids) + position_values.unsqueeze(0)
        mask = torch.ones(length, length, device=input_ids.device, dtype=torch.bool).triu(1)
        hidden = self.blocks(hidden, mask=mask, is_causal=True)
        return SimpleNamespace(logits=self.logits(hidden), hidden_states=hidden)

    def stored_state_elements(self, batch_size: int = 1, context: int = 512) -> int:
        # Standard autoregressive KV cache, two tensors per layer.
        return 2 * batch_size * min(context, self.max_context) * self.width * self.layers

    def estimated_macs_per_token(self, context: int = 512) -> int:
        feedforward = 2 * self.expansion * self.width * self.width
        projections = 4 * self.width * self.width
        attention = 2 * min(context, self.max_context) * self.width
        return (self.output_head_macs_per_token()
                + self.layers * (feedforward + projections + attention))


class RecurrentLogitStudent(StudentBase):
    kind = "recurrent_logits"

    def __init__(self, vocab_size: int, width: int, max_context: int, layers: int):
        super().__init__(vocab_size, width, max_context)
        self.layers = int(layers)
        self.cells = nn.ModuleList([
            nn.GRUCell(width, width) for _ in range(layers)
        ])

    def initial_state(self, batch_size: int, device: torch.device,
                      dtype: torch.dtype | None = None):
        dtype = dtype or self.embedding.weight.dtype
        return [torch.zeros(batch_size, self.width, device=device, dtype=dtype)
                for _ in range(self.layers)]

    def step(self, token_ids: torch.Tensor, state=None):
        if state is None:
            state = self.initial_state(token_ids.shape[0], token_ids.device)
        hidden = self.embedding(token_ids)
        new_state = []
        for cell, previous in zip(self.cells, state):
            hidden = cell(hidden, previous)
            new_state.append(hidden)
        return self.logits(hidden), new_state, hidden

    def forward(self, input_ids: torch.Tensor):
        state = None
        hidden_values = []
        logits = []
        for index in range(input_ids.shape[1]):
            token_logits, state, hidden = self.step(input_ids[:, index], state)
            logits.append(token_logits)
            hidden_values.append(hidden)
        return SimpleNamespace(
            logits=torch.stack(logits, dim=1),
            hidden_states=torch.stack(hidden_values, dim=1),
            final_state=state)

    def last_logits(self, input_ids: torch.Tensor):
        state = None
        token_logits = None
        for index in range(input_ids.shape[1]):
            token_logits, state, _ = self.step(input_ids[:, index], state)
        if token_logits is None:
            raise ValueError("last_logits requires at least one input token")
        return token_logits.unsqueeze(1)

    def stored_state_elements(self, batch_size: int = 1, context: int = 512) -> int:
        del context
        return batch_size * self.width * self.layers

    def estimated_macs_per_token(self, context: int = 512) -> int:
        del context
        gru = 6 * self.width * self.width
        return self.output_head_macs_per_token() + self.layers * gru


class HierarchicalPredictiveStudent(StudentBase):
    kind = "hierarchical_predictive"

    def __init__(self, vocab_size: int, width: int, max_context: int,
                 atoms: int, segment_length: int, document_segments: int,
                 future_horizons: list[int]):
        super().__init__(vocab_size, width, max_context)
        self.atoms = int(atoms)
        self.segment_length = int(segment_length)
        self.document_segments = int(document_segments)
        self.future_horizons = tuple(int(value) for value in future_horizons)
        self.fast_cell = nn.GRUCell(3 * width, width)
        self.segment_cell = nn.GRUCell(width, width)
        self.document_cell = nn.GRUCell(width, width)
        self.combine = nn.Sequential(
            nn.Linear(3 * width, 2 * width), nn.SiLU(), nn.Linear(2 * width, width))
        self.atom_head = nn.Linear(width, atoms)
        self.future_heads = nn.ModuleList([
            nn.Linear(width, atoms) for _ in self.future_horizons
        ])

    def initial_state(self, batch_size: int, device: torch.device,
                      dtype: torch.dtype | None = None):
        dtype = dtype or self.embedding.weight.dtype
        zero = lambda: torch.zeros(batch_size, self.width, device=device, dtype=dtype)
        return {
            "fast": zero(), "segment": zero(), "document": zero(),
            "segment_sum": zero(), "segment_tokens": 0, "segments": 0,
        }

    def step(self, token_ids: torch.Tensor, state=None):
        if state is None:
            state = self.initial_state(token_ids.shape[0], token_ids.device)
        token = self.embedding(token_ids)
        fast_input = torch.cat((token, state["segment"], state["document"]), dim=-1)
        fast = self.fast_cell(fast_input, state["fast"])
        segment_sum = state["segment_sum"] + fast
        segment_tokens = int(state["segment_tokens"]) + 1
        segment = state["segment"]
        document = state["document"]
        segments = int(state["segments"])
        if segment_tokens == self.segment_length:
            segment = self.segment_cell(segment_sum / segment_tokens, segment)
            segment_sum = torch.zeros_like(segment_sum)
            segment_tokens = 0
            segments += 1
            if segments == self.document_segments:
                document = self.document_cell(segment, document)
                segments = 0
        combined = self.combine(torch.cat((fast, segment, document), dim=-1))
        new_state = {
            "fast": fast, "segment": segment, "document": document,
            "segment_sum": segment_sum, "segment_tokens": segment_tokens,
            "segments": segments,
        }
        return {
            "logits": self.logits(combined),
            "atom_logits": self.atom_head(combined),
            "future_atom_logits": [head(combined) for head in self.future_heads],
            "hidden": combined,
        }, new_state

    def forward(self, input_ids: torch.Tensor):
        state = None
        records = []
        for index in range(input_ids.shape[1]):
            record, state = self.step(input_ids[:, index], state)
            records.append(record)
        return SimpleNamespace(
            logits=torch.stack([item["logits"] for item in records], dim=1),
            atom_logits=torch.stack([item["atom_logits"] for item in records], dim=1),
            future_atom_logits=[
                torch.stack([item["future_atom_logits"][head] for item in records], dim=1)
                for head in range(len(self.future_horizons))
            ],
            hidden_states=torch.stack([item["hidden"] for item in records], dim=1),
            final_state=state)

    def stored_state_elements(self, batch_size: int = 1, context: int = 512) -> int:
        del context
        # fast, segment, document, and the active segment accumulator.
        return 4 * batch_size * self.width

    def estimated_macs_per_token(self, context: int = 512) -> int:
        del context
        fast = 12 * self.width * self.width
        slower = (6 * self.width * self.width / self.segment_length
                   + 6 * self.width * self.width
                   / (self.segment_length * self.document_segments))
        combine = 8 * self.width * self.width
        probes = self.width * self.atoms * (1 + len(self.future_horizons))
        return int(self.output_head_macs_per_token() + fast + slower + combine + probes)


class HierarchicalDistributionStudent(StudentBase):
    """Three-timescale recurrent state without the failed discrete atom heads."""

    kind = "hierarchical_distribution"

    def __init__(self, vocab_size: int, width: int, max_context: int,
                 segment_length: int, document_segments: int):
        super().__init__(vocab_size, width, max_context)
        self.segment_length = int(segment_length)
        self.document_segments = int(document_segments)
        self.fast_cell = nn.GRUCell(3 * width, width)
        self.segment_cell = nn.GRUCell(width, width)
        self.document_cell = nn.GRUCell(width, width)
        self.combine = nn.Sequential(
            nn.Linear(3 * width, 2 * width), nn.SiLU(), nn.Linear(2 * width, width))

    def initial_state(self, batch_size: int, device: torch.device,
                      dtype: torch.dtype | None = None):
        dtype = dtype or self.embedding.weight.dtype
        zero = lambda: torch.zeros(batch_size, self.width, device=device, dtype=dtype)
        return {
            "fast": zero(), "segment": zero(), "document": zero(),
            "segment_sum": zero(), "segment_tokens": 0, "segments": 0,
        }

    def transition(self, token_ids: torch.Tensor, state=None):
        if state is None:
            state = self.initial_state(token_ids.shape[0], token_ids.device)
        token = self.embedding(token_ids)
        fast_input = torch.cat((token, state["segment"], state["document"]), dim=-1)
        fast = self.fast_cell(fast_input, state["fast"])
        segment_sum = state["segment_sum"] + fast
        segment_tokens = int(state["segment_tokens"]) + 1
        segment = state["segment"]
        document = state["document"]
        segments = int(state["segments"])
        if segment_tokens == self.segment_length:
            segment = self.segment_cell(segment_sum / segment_tokens, segment)
            segment_sum = torch.zeros_like(segment_sum)
            segment_tokens = 0
            segments += 1
            if segments == self.document_segments:
                document = self.document_cell(segment, document)
                segments = 0
        combined = self.combine(torch.cat((fast, segment, document), dim=-1))
        new_state = {
            "fast": fast, "segment": segment, "document": document,
            "segment_sum": segment_sum, "segment_tokens": segment_tokens,
            "segments": segments,
        }
        return combined, new_state

    def step(self, token_ids: torch.Tensor, state=None):
        hidden, new_state = self.transition(token_ids, state)
        return self.logits(hidden), new_state, hidden

    def forward(self, input_ids: torch.Tensor):
        state = None
        hidden_values = []
        logits = []
        for index in range(input_ids.shape[1]):
            token_logits, state, hidden = self.step(input_ids[:, index], state)
            logits.append(token_logits)
            hidden_values.append(hidden)
        if not logits:
            raise ValueError("forward requires at least one input token")
        return SimpleNamespace(
            logits=torch.stack(logits, dim=1),
            hidden_states=torch.stack(hidden_values, dim=1),
            final_state=state)

    def last_logits(self, input_ids: torch.Tensor):
        state = None
        hidden = None
        for index in range(input_ids.shape[1]):
            hidden, state = self.transition(input_ids[:, index], state)
        if hidden is None:
            raise ValueError("last_logits requires at least one input token")
        return self.logits(hidden).unsqueeze(1)

    def stored_state_elements(self, batch_size: int = 1, context: int = 512) -> int:
        del context
        return 4 * batch_size * self.width

    def estimated_macs_per_token(self, context: int = 512) -> int:
        del context
        fast = 12 * self.width * self.width
        slower = (6 * self.width * self.width / self.segment_length
                   + 6 * self.width * self.width
                   / (self.segment_length * self.document_segments))
        combine = 8 * self.width * self.width
        return int(self.output_head_macs_per_token() + fast + slower + combine)


class CompressedLocalAttention(nn.Module):
    """Exact causal softmax attention over a fixed compressed local cache."""

    def __init__(self, width: int, local_width: int, heads: int, window: int,
                 gate_bias: float):
        super().__init__()
        if min(width, local_width, heads, window) <= 0:
            raise ValueError("local attention dimensions must be positive")
        if local_width % heads:
            raise ValueError("local attention width must divide evenly into heads")
        self.local_width = int(local_width)
        self.heads = int(heads)
        self.head_width = self.local_width // self.heads
        self.window = int(window)
        self.input_norm = nn.LayerNorm(width)
        self.query = nn.Linear(width, self.local_width, bias=False)
        self.key = nn.Linear(width, self.local_width, bias=False)
        self.value = nn.Linear(width, self.local_width, bias=False)
        self.output = nn.Linear(self.local_width, width, bias=False)
        self.output_norm = nn.LayerNorm(width)
        self.gate = nn.Parameter(torch.tensor(float(gate_bias)))

    def forward(self, hidden: torch.Tensor, keys: torch.Tensor | None,
                values: torch.Tensor | None):
        batch = hidden.shape[0]
        normalized = self.input_norm(hidden)
        query = self.query(normalized).view(
            batch, self.heads, self.head_width)
        key = self.key(normalized).view(
            batch, self.heads, 1, self.head_width)
        value = self.value(normalized).view(
            batch, self.heads, 1, self.head_width)
        keys = key if keys is None else torch.cat((keys, key), dim=2)
        values = value if values is None else torch.cat((values, value), dim=2)
        keys = keys[:, :, -self.window:]
        values = values[:, :, -self.window:]
        scores = torch.einsum("bhd,bhtd->bht", query, keys)
        scores = scores / math.sqrt(self.head_width)
        probabilities = torch.softmax(scores.float(), dim=-1).to(values.dtype)
        context = torch.einsum("bht,bhtd->bhd", probabilities, values)
        context = context.reshape(batch, self.local_width)
        result = self.output_norm(
            hidden + torch.sigmoid(self.gate) * self.output(context))
        return result, keys, values

    def stored_state_elements(self, batch_size: int = 1) -> int:
        return 2 * batch_size * self.window * self.local_width

    def estimated_macs_per_token(self, model_width: int) -> int:
        projections = 4 * model_width * self.local_width
        attention = 2 * self.window * self.local_width
        return projections + attention


class HierarchicalLocalDistributionStudent(HierarchicalDistributionStudent):
    """Hierarchical recurrence plus an exact, fixed-size local attention cache."""

    kind = "hierarchical_local_distribution"

    def __init__(self, vocab_size: int, width: int, max_context: int,
                 segment_length: int, document_segments: int,
                 local_attention: dict):
        super().__init__(
            vocab_size, width, max_context, segment_length, document_segments)
        self.local_attention = CompressedLocalAttention(
            width=width,
            local_width=int(local_attention["width"]),
            heads=int(local_attention["heads"]),
            window=int(local_attention["window"]),
            gate_bias=float(local_attention["gate_bias"]),
        )

    def transition(self, token_ids: torch.Tensor, state=None):
        hidden, new_state = super().transition(token_ids, state)
        keys = None if state is None else state.get("local_keys")
        values = None if state is None else state.get("local_values")
        hidden, keys, values = self.local_attention(hidden, keys, values)
        new_state["local_keys"] = keys
        new_state["local_values"] = values
        return hidden, new_state

    def stored_state_elements(self, batch_size: int = 1, context: int = 512) -> int:
        del context
        return (4 * batch_size * self.width
                + self.local_attention.stored_state_elements(batch_size))

    def estimated_macs_per_token(self, context: int = 512) -> int:
        return (super().estimated_macs_per_token(context)
                + self.local_attention.estimated_macs_per_token(self.width))


def clamped_cubic_design(points: int, controls: int,
                         dtype=torch.float32) -> torch.Tensor:
    """Open-uniform cubic B-spline controls sampled at virtual depths."""
    degree = 3
    if points < 1:
        raise ValueError("virtual depth must be positive")
    if controls <= degree:
        raise ValueError("a cubic spline needs at least four controls")
    interior_count = controls - degree - 1
    if interior_count:
        interior = torch.arange(
            1, interior_count + 1, dtype=dtype, device="cpu") / (
                interior_count + 1)
    else:
        interior = torch.empty(0, dtype=dtype, device="cpu")
    knots = torch.cat((
        torch.zeros(degree + 1, dtype=dtype, device="cpu"), interior,
        torch.ones(degree + 1, dtype=dtype, device="cpu")))
    depths = (torch.linspace(0, 1, points, dtype=dtype, device="cpu")
              if points > 1 else torch.zeros(1, dtype=dtype, device="cpu"))
    rows = []
    for index, depth in enumerate(depths):
        basis = ((depth >= knots[:-1]) & (depth < knots[1:])).to(dtype)
        if index + 1 == points:
            basis.zero_()
            basis[controls - 1] = 1
        for order in range(1, degree + 1):
            count = basis.numel() - 1
            left_den = knots[order:order + count] - knots[:count]
            right_den = (
                knots[order + 1:order + 1 + count] - knots[1:1 + count])
            left = torch.where(
                left_den > 0,
                (depth - knots[:count]) / left_den.clamp_min(1e-30),
                torch.zeros_like(left_den))
            right = torch.where(
                right_den > 0,
                (knots[order + 1:order + 1 + count] - depth)
                / right_den.clamp_min(1e-30),
                torch.zeros_like(right_den))
            basis = left * basis[:count] + right * basis[1:1 + count]
        rows.append(basis)
    design = torch.stack(rows)
    if design.shape != (points, controls):
        raise AssertionError("cubic design shape mismatch")
    if not torch.allclose(
            design.sum(1), torch.ones(
                points, dtype=dtype, device="cpu"), atol=1e-6):
        raise AssertionError("cubic design partition of unity failed")
    return design


class SmoothFunctionalAffine(nn.Module):
    """Dense private map plus a spline-generated low-rank residual family."""

    def __init__(self, input_size: int, output_size: int, rank: int,
                 controls: int, virtual_depth: int):
        super().__init__()
        if min(input_size, output_size, rank) <= 0:
            raise ValueError("functional affine dimensions must be positive")
        self.input_size = int(input_size)
        self.output_size = int(output_size)
        self.controls = int(controls)
        self.virtual_depth = int(virtual_depth)
        self.base = nn.Linear(input_size, output_size)
        self.control_left = nn.Parameter(
            torch.empty(controls, input_size, rank))
        self.control_right = nn.Parameter(
            torch.empty(controls, rank, output_size))
        self.register_buffer(
            "design", clamped_cubic_design(virtual_depth, controls),
            persistent=True)
        nn.init.normal_(self.control_left, std=input_size ** -0.5)
        nn.init.normal_(self.control_right, std=1e-3)

    def forward(self, values: torch.Tensor, phase: int) -> torch.Tensor:
        coefficients = self.design[int(phase) % self.virtual_depth].to(
            device=values.device, dtype=values.dtype)
        coordinates = torch.einsum(
            "bi,kir->bkr", values, self.control_left)
        control_outputs = torch.einsum(
            "bkr,kro->bko", coordinates, self.control_right)
        residual = torch.einsum("k,bko->bo", coefficients, control_outputs)
        return self.base(values) + residual

    def estimated_macs(self) -> int:
        rank = self.control_left.shape[-1]
        return int(
            self.input_size * self.output_size
            + self.controls * rank * (self.input_size + self.output_size)
            + self.controls * self.output_size)


class SmoothFunctionGRUCell(nn.Module):
    """GRU whose phase-specific maps come from smooth functional bases."""

    def __init__(self, input_size: int, hidden_size: int, rank: int,
                 controls: int, virtual_depth: int):
        super().__init__()
        self.input_function = SmoothFunctionalAffine(
            input_size, 3 * hidden_size, rank, controls, virtual_depth)
        self.hidden_function = SmoothFunctionalAffine(
            hidden_size, 3 * hidden_size, rank, controls, virtual_depth)

    def forward(self, values: torch.Tensor, previous: torch.Tensor,
                phase: int) -> torch.Tensor:
        input_gates = self.input_function(values, phase)
        hidden_gates = self.hidden_function(previous, phase)
        input_reset, input_update, input_new = input_gates.chunk(3, dim=-1)
        hidden_reset, hidden_update, hidden_new = hidden_gates.chunk(3, dim=-1)
        reset = torch.sigmoid(input_reset + hidden_reset)
        update = torch.sigmoid(input_update + hidden_update)
        candidate = torch.tanh(input_new + reset * hidden_new)
        return candidate + update * (previous - candidate)

    def estimated_macs(self) -> int:
        return (self.input_function.estimated_macs()
                + self.hidden_function.estimated_macs())


class HierarchicalLocalFunctionWeightsStudent(StudentBase):
    """Local hierarchical memory with smooth function-generated transitions."""

    kind = "hierarchical_local_function_weights"

    def __init__(self, vocab_size: int, width: int, max_context: int,
                 segment_length: int, document_segments: int,
                 local_attention: dict, function_weights: dict):
        super().__init__(vocab_size, width, max_context)
        self.segment_length = int(segment_length)
        self.document_segments = int(document_segments)
        self.virtual_depth = int(function_weights["virtual_depth"])
        arguments = {
            "hidden_size": width,
            "rank": int(function_weights["rank"]),
            "controls": int(function_weights["controls"]),
            "virtual_depth": self.virtual_depth,
        }
        self.fast_cell = SmoothFunctionGRUCell(
            input_size=3 * width, **arguments)
        self.segment_cell = SmoothFunctionGRUCell(
            input_size=width, **arguments)
        self.document_cell = SmoothFunctionGRUCell(
            input_size=width, **arguments)
        self.cross_scale = SmoothFunctionalAffine(
            3 * width, 2 * width, int(function_weights["rank"]),
            int(function_weights["controls"]), self.virtual_depth)
        self.readout = SmoothFunctionalAffine(
            2 * width, width, int(function_weights["rank"]),
            int(function_weights["controls"]), self.virtual_depth)
        self.local_attention = CompressedLocalAttention(
            width=width,
            local_width=int(local_attention["width"]),
            heads=int(local_attention["heads"]),
            window=int(local_attention["window"]),
            gate_bias=float(local_attention["gate_bias"]),
        )

    def initial_state(self, batch_size: int, device: torch.device,
                      dtype: torch.dtype | None = None):
        dtype = dtype or self.embedding.weight.dtype
        zero = lambda: torch.zeros(batch_size, self.width, device=device, dtype=dtype)
        return {
            "fast": zero(), "segment": zero(), "document": zero(),
            "segment_sum": zero(), "segment_tokens": 0, "segments": 0,
            "position": 0, "segment_updates": 0, "document_updates": 0,
            "local_keys": None, "local_values": None,
        }

    def transition(self, token_ids: torch.Tensor, state=None):
        if state is None:
            state = self.initial_state(token_ids.shape[0], token_ids.device)
        position = int(state["position"])
        token = self.embedding(token_ids)
        fast_input = torch.cat(
            (token, state["segment"], state["document"]), dim=-1)
        fast = self.fast_cell(fast_input, state["fast"], position)
        segment_sum = state["segment_sum"] + fast
        segment_tokens = int(state["segment_tokens"]) + 1
        segment = state["segment"]
        document = state["document"]
        segments = int(state["segments"])
        segment_updates = int(state["segment_updates"])
        document_updates = int(state["document_updates"])
        if segment_tokens == self.segment_length:
            segment = self.segment_cell(
                segment_sum / segment_tokens, segment, segment_updates)
            segment_sum = torch.zeros_like(segment_sum)
            segment_tokens = 0
            segment_updates += 1
            segments += 1
            if segments == self.document_segments:
                document = self.document_cell(
                    segment, document, document_updates)
                document_updates += 1
                segments = 0
        combined = torch.cat((fast, segment, document), dim=-1)
        hidden = self.readout(
            F.silu(self.cross_scale(combined, position)), position)
        hidden, local_keys, local_values = self.local_attention(
            hidden, state["local_keys"], state["local_values"])
        new_state = {
            "fast": fast, "segment": segment, "document": document,
            "segment_sum": segment_sum, "segment_tokens": segment_tokens,
            "segments": segments, "position": position + 1,
            "segment_updates": segment_updates,
            "document_updates": document_updates,
            "local_keys": local_keys, "local_values": local_values,
        }
        return hidden, new_state

    def step(self, token_ids: torch.Tensor, state=None):
        hidden, new_state = self.transition(token_ids, state)
        return self.logits(hidden), new_state, hidden

    def forward(self, input_ids: torch.Tensor):
        state = None
        hidden_values = []
        logits = []
        for index in range(input_ids.shape[1]):
            token_logits, state, hidden = self.step(input_ids[:, index], state)
            logits.append(token_logits)
            hidden_values.append(hidden)
        if not logits:
            raise ValueError("forward requires at least one input token")
        return SimpleNamespace(
            logits=torch.stack(logits, dim=1),
            hidden_states=torch.stack(hidden_values, dim=1),
            final_state=state)

    def last_logits(self, input_ids: torch.Tensor):
        state = None
        hidden = None
        for index in range(input_ids.shape[1]):
            hidden, state = self.transition(input_ids[:, index], state)
        if hidden is None:
            raise ValueError("last_logits requires at least one input token")
        return self.logits(hidden).unsqueeze(1)

    def stored_state_elements(self, batch_size: int = 1, context: int = 512) -> int:
        del context
        return (4 * batch_size * self.width
                + self.local_attention.stored_state_elements(batch_size))

    def estimated_macs_per_token(self, context: int = 512) -> int:
        del context
        transitions = (
            self.fast_cell.estimated_macs()
            + self.segment_cell.estimated_macs() / self.segment_length
            + self.document_cell.estimated_macs()
            / (self.segment_length * self.document_segments))
        return int(
            self.output_head_macs_per_token()
            + transitions
            + self.cross_scale.estimated_macs()
            + self.readout.estimated_macs()
            + self.local_attention.estimated_macs_per_token(self.width))


def build_student(kind: str, vocab_size: int, width: int, max_context: int,
                  config: dict) -> StudentBase:
    if kind == "transformer":
        return TransformerStudent(
            vocab_size, width, max_context, int(config["layers"]),
            int(config["heads"]), int(config.get("expansion", 4)),
            str(config.get("position_encoding", "learned")))
    if kind == "recurrent_logits":
        return RecurrentLogitStudent(
            vocab_size, width, max_context, int(config["layers"]))
    if kind in ("hierarchical_predictive", "hierarchical_logit_control"):
        return HierarchicalPredictiveStudent(
            vocab_size, width, max_context, int(config["atoms"]),
            int(config["segment_length"]), int(config["document_segments"]),
            list(config["future_horizons"]))
    if kind == "hierarchical_distribution":
        return HierarchicalDistributionStudent(
            vocab_size, width, max_context, int(config["segment_length"]),
            int(config["document_segments"]))
    if kind == "hierarchical_local_distribution":
        return HierarchicalLocalDistributionStudent(
            vocab_size, width, max_context, int(config["segment_length"]),
            int(config["document_segments"]), dict(config["local_attention"]))
    if kind == "hierarchical_local_function_weights":
        return HierarchicalLocalFunctionWeightsStudent(
            vocab_size, width, max_context, int(config["segment_length"]),
            int(config["document_segments"]), dict(config["local_attention"]),
            dict(config["function_weights"]))
    raise ValueError(f"unknown student kind {kind}")


@dataclass
class BudgetedStudent:
    model: StudentBase
    width: int
    parameters: int
    target_parameters: int


def build_to_budget(kind: str, vocab_size: int, max_context: int, config: dict,
                    target_parameters: int) -> BudgetedStudent:
    """Choose the widest multiple-of-32 model that does not exceed the budget."""
    widths = list(range(int(config["minimum_width"]),
                        int(config["maximum_width"]) + 1,
                        int(config.get("width_step", 32))))
    if kind == "transformer":
        heads = int(config["heads"])
        widths = [width for width in widths if width % heads == 0]
    chosen = None
    for width in widths:
        with torch.device("meta"):
            candidate = build_student(kind, vocab_size, width, max_context, config)
        count = parameter_count(candidate)
        if count <= target_parameters:
            chosen = (width, count)
        else:
            break
    if chosen is None:
        raise ValueError(f"{kind} cannot fit minimum width under parameter budget")
    width, count = chosen
    model = build_student(kind, vocab_size, width, max_context, config)
    actual = parameter_count(model)
    if actual != count:
        raise RuntimeError("meta-device parameter count disagrees with real model")
    return BudgetedStudent(model, width, actual, int(target_parameters))
