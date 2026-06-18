from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


def apply_rope(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    q = (q * cos) + (_rotate_half(q) * sin)
    k = (k * cos) + (_rotate_half(k) * sin)
    return q, k


class VanillaRoPE(nn.Module):
    def __init__(self, hidden_size: int, base: float = 1000.0, device=None):
        super().__init__()
        if hidden_size % 2 != 0:
            raise ValueError(f"hidden_size must be even for RoPE, got {hidden_size}")
        dim = hidden_size
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64).float().to(device) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, x: torch.Tensor, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded @ position_ids_expanded).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class R3LatentThoughtAttention(nn.Module):
    """R3-style latent thought generator.

    Given final-layer hidden states H (B, L, H), and a special <|Thought|> position t per sample,
    produce a continuous latent vector r by attending over the prefix before t.

    - Query uses position (t - 1)
    - Keys/values come from all positions, but the mask restricts to < t (optionally last end_k tokens)
    """

    def __init__(self, hidden_size: int, end_k: int = -1, rope_base: float = 1000.0):
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.end_k = int(end_k)

        self.query = nn.Linear(self.hidden_size, self.hidden_size)
        self.key = nn.Linear(self.hidden_size, self.hidden_size)
        self.value = nn.Linear(self.hidden_size, self.hidden_size)
        self.scale = 1.0 / (self.hidden_size ** 0.5)
        self.rope = VanillaRoPE(self.hidden_size, base=rope_base)

    @staticmethod
    def _mask_to_bias(attention_mask: torch.Tensor, thought_pos: torch.Tensor, end_k: int) -> torch.Tensor:
        if attention_mask.dim() != 2:
            raise ValueError(f"attention_mask must be (B, L), got {attention_mask.shape}")

        bsz, seq_len = attention_mask.shape
        if thought_pos.shape[0] != bsz:
            raise ValueError("thought_pos must have shape (B,)")

        mask = attention_mask.clone().bool()

        # Restrict to prefix before <|Thought|> position.
        for i in range(bsz):
            idx = int(thought_pos[i].item())
            if idx < 0:
                idx = seq_len
            idx = max(0, min(idx, seq_len))

            # Only attend to [0, idx) (exclude idx and after)
            if idx < seq_len:
                mask[i, idx:] = False

            if end_k != -1:
                start = max(0, idx - end_k)
                keep = torch.zeros(seq_len, device=mask.device, dtype=torch.bool)
                keep[start:idx] = True
                mask[i] = mask[i] & keep

        bias = torch.zeros_like(attention_mask, dtype=torch.float32)
        bias = bias.masked_fill(~mask, float("-inf"))
        return bias

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor, thought_pos: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, _ = hidden_states.shape

        attn_bias = self._mask_to_bias(attention_mask, thought_pos, self.end_k).to(hidden_states.device)

        position_ids = torch.arange(0, seq_len, device=hidden_states.device).unsqueeze(0)
        cos, sin = self.rope(hidden_states, position_ids=position_ids)

        q = self.query(hidden_states)
        k = self.key(hidden_states)
        v = self.value(hidden_states)
        q, k = apply_rope(q, k, cos, sin)

        # Query index = thought_pos - 1 (clamped)
        q_idx = thought_pos.to(torch.long) - 1
        q_idx = torch.clamp(q_idx, 0, seq_len - 1)

        q_sel = q[torch.arange(bsz, device=hidden_states.device), q_idx].unsqueeze(1)  # (B, 1, H)
        scores = torch.matmul(q_sel, k.transpose(-2, -1)) * self.scale  # (B, 1, L)
        scores = scores + attn_bias.unsqueeze(1)

        # attn_bias is float32, so scores/softmax will typically be float32 under bf16 training.
        # Cast weights back to v.dtype before matmul to avoid dtype mismatch.
        weights = F.softmax(scores, dim=-1).to(dtype=v.dtype)
        out = torch.matmul(weights, v).squeeze(1)  # (B, H)
        return out
