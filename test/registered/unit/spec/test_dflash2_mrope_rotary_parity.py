"""Compare SpecForge draft mRoPE with SGLang draft RoPE numerically."""

import torch
from types import SimpleNamespace
from transformers import Qwen3Config

from specforge.modeling.draft.dflash import (
    Qwen3InterleavedMultiRotaryEmbedding,
    apply_rotary_pos_emb,
)
from sglang.srt.layers.rotary_embedding import get_rope
from sglang.srt.layers.rotary_embedding import base as rope_base
from sglang.srt.layers.rotary_embedding import mrope as rope_mrope


rope_base.get_exec = lambda: SimpleNamespace(
    deterministic=SimpleNamespace(rl_on_policy_target=None)
)
rope_mrope.get_exec = rope_base.get_exec


torch.manual_seed(47)
device = "cuda"
positions = torch.tensor([[0, 1, 2, 3], [0, 10, 11, 12], [0, 20, 21, 22]], device=device)
config = Qwen3Config(
    hidden_size=256,
    num_attention_heads=2,
    num_key_value_heads=2,
    head_dim=128,
    rope_theta=5000000,
    rope_scaling={"rope_type": "default", "mrope_section": [24, 20, 20], "mrope_interleaved": True},
)
train_rope = Qwen3InterleavedMultiRotaryEmbedding(config).to(device)
q = torch.randn(1, 2, 4, 128, device=device)
k = torch.randn(1, 2, 4, 128, device=device)
cos, sin = train_rope(torch.zeros(1, 4, 256, device=device), positions[:, None, :])
train_q, train_k = apply_rotary_pos_emb(q, k, cos, sin)

infer_rope = get_rope(
    128,
    rotary_dim=128,
    max_position=32768,
    base=5000000,
    rope_scaling={"rope_type": "default", "mrope_section": [24, 20, 20], "mrope_interleaved": True},
    is_neox_style=True,
).to(device)
infer_q, infer_k = infer_rope(
    positions,
    q.permute(0, 2, 1, 3).reshape(4, 256).contiguous(),
    k.permute(0, 2, 1, 3).reshape(4, 256).contiguous(),
)
infer_q = infer_q.reshape(1, 4, 2, 128).permute(0, 2, 1, 3)
infer_k = infer_k.reshape(1, 4, 2, 128).permute(0, 2, 1, 3)
torch.testing.assert_close(train_q, infer_q, atol=1e-5, rtol=1e-5)
torch.testing.assert_close(train_k, infer_k, atol=1e-5, rtol=1e-5)
print("DFlash2 mRoPE Q/K parity: PASS")
