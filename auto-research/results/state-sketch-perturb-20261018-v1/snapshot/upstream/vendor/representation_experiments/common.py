"""Shared, dependency-light utilities for representation experiments."""
from __future__ import annotations

import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Mapping

import numpy as np
import torch


def atomic_json(payload: Mapping, path: str | Path) -> None:
    output = Path(path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, output)


def load_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def unique_parameters(module: torch.nn.Module) -> list[torch.nn.Parameter]:
    seen: set[int] = set()
    result = []
    for parameter in module.parameters():
        if id(parameter) not in seen:
            seen.add(id(parameter))
            result.append(parameter)
    return result


def parameter_count(module: torch.nn.Module, trainable_only: bool = False) -> int:
    return sum(
        parameter.numel()
        for parameter in unique_parameters(module)
        if not trainable_only or parameter.requires_grad
    )


def normalized_mse(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    numerator = (prediction.float() - target.float()).square().mean()
    denominator = target.float().square().mean().clamp_min(1e-8)
    return numerator / denominator


def symmetric_geometry_loss(student: torch.Tensor, teacher: torch.Tensor,
                            samples: int = 64) -> torch.Tensor:
    """Match representation geometry without assuming matching coordinates."""
    student = student.reshape(-1, student.shape[-1]).float()
    teacher = teacher.reshape(-1, teacher.shape[-1]).float()
    count = min(samples, student.shape[0])
    if count < student.shape[0]:
        indices = torch.linspace(
            0, student.shape[0] - 1, count, device=student.device).long()
        student = student.index_select(0, indices)
        teacher = teacher.index_select(0, indices)
    student = torch.nn.functional.normalize(student, dim=-1)
    teacher = torch.nn.functional.normalize(teacher, dim=-1)
    return (student @ student.t() - teacher @ teacher.t()).square().mean()


def teacher_kl(student_logits: torch.Tensor, teacher_logits: torch.Tensor,
               temperature: float = 1.0) -> torch.Tensor:
    scale = float(temperature)
    teacher_probability = torch.softmax(teacher_logits.float() / scale, dim=-1)
    student_log_probability = torch.log_softmax(student_logits.float() / scale, dim=-1)
    teacher_log_probability = torch.log_softmax(teacher_logits.float() / scale, dim=-1)
    return (
        teacher_probability * (teacher_log_probability - student_log_probability)
    ).sum(dim=-1).mean() * (scale * scale)


@dataclass
class TokenMemmap:
    path: Path
    dtype: np.dtype

    @classmethod
    def open(cls, path: str | Path, dtype: str) -> "TokenMemmap":
        resolved = Path(path).expanduser().resolve()
        return cls(resolved, np.dtype(dtype))

    @property
    def tokens(self) -> np.memmap:
        return np.memmap(self.path, dtype=self.dtype, mode="r")

    def batches(self, context: int, batch_size: int, steps: int, seed: int,
                sequential: bool = False) -> Iterator[torch.Tensor]:
        tokens = self.tokens
        if len(tokens) <= context + 1:
            raise ValueError(f"{self.path} is shorter than one context")
        generator = np.random.default_rng(seed)
        cursor = 0
        for _ in range(steps):
            if sequential:
                starts = (cursor + np.arange(batch_size) * (context + 1))
                starts %= len(tokens) - context - 1
                cursor = int(starts[-1] + context + 1)
            else:
                starts = generator.integers(
                    0, len(tokens) - context - 1, size=batch_size)
            batch = np.stack([
                tokens[int(start):int(start) + context + 1]
                for start in starts
            ]).astype(np.int64, copy=False)
            yield torch.from_numpy(batch)

    def batch_for_step(self, context: int, batch_size: int, step: int,
                       seed: int) -> torch.Tensor:
        """Return a restart-stable random batch determined only by seed and step."""
        tokens = self.tokens
        if len(tokens) <= context + 1:
            raise ValueError(f"{self.path} is shorter than one context")
        generator = np.random.default_rng(int(seed) + 1_000_003 * int(step))
        starts = generator.integers(
            0, len(tokens) - context - 1, size=batch_size)
        batch = np.stack([
            tokens[int(start):int(start) + context + 1]
            for start in starts
        ]).astype(np.int64, copy=False)
        return torch.from_numpy(batch)


def cosine_with_warmup(step: int, total_steps: int, warmup: int,
                       minimum_ratio: float = 0.1) -> float:
    if warmup > 0 and step < warmup:
        return max(1e-8, (step + 1) / warmup)
    progress = (step - warmup) / max(1, total_steps - warmup)
    cosine = 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))
    return minimum_ratio + (1.0 - minimum_ratio) * cosine


def chunks(values: Iterable, size: int) -> Iterator[list]:
    bucket = []
    for value in values:
        bucket.append(value)
        if len(bucket) == size:
            yield bucket
            bucket = []
    if bucket:
        yield bucket
