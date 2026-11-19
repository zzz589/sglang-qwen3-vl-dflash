"""M-RoPE position construction shared by DSpark runtime paths."""
from __future__ import annotations
from typing import TYPE_CHECKING, Sequence
import torch
if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import ScheduleBatch

def build_dspark_rope_positions(*, batch: "ScheduleBatch", logical_positions: torch.Tensor, token_counts: Sequence[int], use_mrope: bool) -> torch.Tensor:
    """Build rotary positions without changing logical cache positions."""
    if not use_mrope:
        return logical_positions
    if logical_positions.ndim != 1:
        raise ValueError("logical_positions must be one-dimensional")
    counts = tuple(int(count) for count in token_counts)
    if sum(counts) != logical_positions.numel():
        raise ValueError("token_counts must cover logical_positions")
    logical_positions = logical_positions.to(dtype=torch.int64)
    mm_inputs = batch.multimodal_inputs
    if mm_inputs is None:
        mm_inputs = [None] * len(counts)
    if len(mm_inputs) != len(counts):
        raise ValueError("multimodal_inputs and token_counts have different lengths")
    chunks = []
    offset = 0
    for count, mm_input in zip(counts, mm_inputs):
        request_positions = logical_positions[offset : offset + count]
        offset += count
        request_rope = request_positions.unsqueeze(0).expand(3, -1).clone()
        stored = None if mm_input is None else getattr(mm_input, "mrope_positions", None)
        prompt_mask = torch.zeros(count, dtype=torch.bool, device=logical_positions.device)
        if stored is not None:
            stored = torch.as_tensor(stored, device=logical_positions.device, dtype=torch.int64)
            if stored.ndim == 3 and stored.shape[0] == 1:
                stored = stored.squeeze(0)
            if stored.ndim != 2 or stored.shape[0] != 3:
                raise ValueError("stored mrope_positions must have shape [3, seq]")
            prompt_mask = (request_positions >= 0) & (request_positions < stored.shape[1])
            prompt_indices = torch.arange(count, device=logical_positions.device)[prompt_mask]
            request_rope[:, prompt_indices] = stored[:, request_positions[prompt_mask]]
        delta = 0 if mm_input is None else getattr(mm_input, "mrope_position_delta", 0)
        delta = torch.as_tensor(delta, device=logical_positions.device, dtype=torch.int64).reshape(-1)[0]
        continuation_mask = ~prompt_mask
        continuation_indices = torch.arange(count, device=logical_positions.device)[continuation_mask]
        continuation = (request_positions[continuation_mask] + delta).unsqueeze(0).expand(3, -1)
        request_rope[:, continuation_indices] = continuation
        chunks.append(request_rope)
    if not chunks:
        return logical_positions.new_empty((3, 0))
    return torch.cat(chunks, dim=1)
