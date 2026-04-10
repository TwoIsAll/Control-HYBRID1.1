from __future__ import annotations
import io
import json
import math
import os
import struct
import time
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import DEFAULT_HYBRID_CONFIG
MODEL_MAGIC = b"HWCF"
MODEL_VERSION = 1
DTYPE_TO_CODE = {
    torch.float16: 1,
    torch.float32: 2,
    torch.bfloat16: 3,
}
CODE_TO_DTYPE = {v: k for k, v in DTYPE_TO_CODE.items()}

def _stable_hash_text(text: str) -> int:
    h = 1469598103934665603
    for ch in text.encode("utf-8", errors="ignore"):
        h ^= ch
        h *= 1099511628211
        h &= 0xFFFFFFFFFFFFFFFF
    return h

def _align16(x: int) -> int:
    rem = x % 16
    if rem == 0:
        return x
    return x + (16 - rem)
@dataclass

class HybridConfig:
    vocab_size: int = DEFAULT_HYBRID_CONFIG["vocab_size"]
    context_len: int = DEFAULT_HYBRID_CONFIG["context_len"]
    d_model: int = DEFAULT_HYBRID_CONFIG["d_model"]
    n_layers: int = DEFAULT_HYBRID_CONFIG["n_layers"]
    n_heads: int = DEFAULT_HYBRID_CONFIG["n_heads"]
    d_ff: int = DEFAULT_HYBRID_CONFIG["d_ff"]
    dropout: float = DEFAULT_HYBRID_CONFIG["dropout"]
    rope_base: float = DEFAULT_HYBRID_CONFIG["rope_base"]
    rms_eps: float = DEFAULT_HYBRID_CONFIG["rms_eps"]
    max_patch_len: int = DEFAULT_HYBRID_CONFIG["max_patch_len"]
    min_patch_len: int = DEFAULT_HYBRID_CONFIG["min_patch_len"]
    patch_entropy_threshold: float = DEFAULT_HYBRID_CONFIG["patch_entropy_threshold"]
    local_encoder_layers: int = DEFAULT_HYBRID_CONFIG["local_encoder_layers"]
    local_decoder_layers: int = DEFAULT_HYBRID_CONFIG["local_decoder_layers"]
    latent_dim: int = DEFAULT_HYBRID_CONFIG["latent_dim"]
    mamba_state_dim: int = DEFAULT_HYBRID_CONFIG["mamba_state_dim"]
    mamba_conv_kernel: int = DEFAULT_HYBRID_CONFIG["mamba_conv_kernel"]
    mamba_expand: int = DEFAULT_HYBRID_CONFIG["mamba_expand"]
    hybrid_pattern: str = DEFAULT_HYBRID_CONFIG["hybrid_pattern"]
    tie_embeddings: bool = DEFAULT_HYBRID_CONFIG["tie_embeddings"]
    init_std: float = DEFAULT_HYBRID_CONFIG["init_std"]
    layer_scale_init: float = DEFAULT_HYBRID_CONFIG["layer_scale_init"]
    use_bias: bool = DEFAULT_HYBRID_CONFIG["use_bias"]
    gradient_checkpointing: bool = DEFAULT_HYBRID_CONFIG["gradient_checkpointing"]
    use_fp32_residual: bool = DEFAULT_HYBRID_CONFIG["use_fp32_residual"]
    ssm_scan_mode: str = DEFAULT_HYBRID_CONFIG["ssm_scan_mode"]
    ssm_chunk_size: int = DEFAULT_HYBRID_CONFIG["ssm_chunk_size"]
    ssm_dt_min: float = DEFAULT_HYBRID_CONFIG["ssm_dt_min"]
    ssm_dt_max: float = DEFAULT_HYBRID_CONFIG["ssm_dt_max"]
    ssm_a_init_min: float = DEFAULT_HYBRID_CONFIG["ssm_a_init_min"]
    ssm_a_init_max: float = DEFAULT_HYBRID_CONFIG["ssm_a_init_max"]
    ssd_rank: int = DEFAULT_HYBRID_CONFIG["ssd_rank"]
    ssd_mix_init: float = DEFAULT_HYBRID_CONFIG["ssd_mix_init"]
    entropy_predictor_hidden: int = DEFAULT_HYBRID_CONFIG["entropy_predictor_hidden"]
    patch_embed_dim: int = DEFAULT_HYBRID_CONFIG["patch_embed_dim"]
    patch_agg: str = DEFAULT_HYBRID_CONFIG["patch_agg"]
    byte_dropout: float = DEFAULT_HYBRID_CONFIG["byte_dropout"]
    patch_dropout: float = DEFAULT_HYBRID_CONFIG["patch_dropout"]
    final_norm: bool = DEFAULT_HYBRID_CONFIG["final_norm"]

    def to_dict(self) -> Dict[str, object]:
        return {
            k: getattr(self, k)
            for k in self.__dataclass_fields__.keys()
        }
    @staticmethod

    def from_dict(data: Dict[str, object]) -> "HybridConfig":
        configuration = HybridConfig()
        for k in configuration.__dataclass_fields__.keys():
            if k in data:
                setattr(configuration, k, data[k])
        return configuration
    @staticmethod

    def tiny_2m_context2k() -> "HybridConfig":
        return HybridConfig(
            vocab_size=256,
            context_len=2048,
            d_model=64,
            latent_dim=64,
            n_layers=6,
            n_heads=4,
            d_ff=192,
            local_encoder_layers=1,
            local_decoder_layers=1,
            mamba_state_dim=16,
            mamba_expand=2,
            hybrid_pattern="tmtmtm",
            max_patch_len=12,
        )

class RMSNorm(nn.Module):

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x * norm * self.weight

def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    output_tensor = torch.stack((-x2, x1), dim=-1)
    return output_tensor.flatten(-2)

class RotaryEmbedding(nn.Module):

    def __init__(self, dim: int, base: float = 10000.0, maximum_sequence_length: int = 8192) -> None:
        super().__init__()
        self.dim = dim
        self.base = base
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._set_cache(maximum_sequence_length)

    def _set_cache(self, sequence_length: int) -> None:
        t = torch.arange(sequence_length, dtype=torch.float32, device=self.inv_freq.device)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)

    def _maybe_refresh_cache(self, sequence_length: int) -> None:
        if sequence_length > self.cos_cached.size(2):
            self._set_cache(int(2 ** math.ceil(math.log2(sequence_length))))

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        offset: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        t = q.size(-2)
        self._maybe_refresh_cache(t + offset)
        cos = self.cos_cached[:, :, offset: offset + t, :].to(dtype=q.dtype, device=q.device)
        sin = self.sin_cached[:, :, offset: offset + t, :].to(dtype=q.dtype, device=q.device)
        q = (q * cos) + (rotate_half(q) * sin)
        k = (k * cos) + (rotate_half(k) * sin)
        return q, k

class SwiGLU(nn.Module):

    def __init__(self, d_model: int, d_ff: int, bias: bool = False) -> None:
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=bias)
        self.w2 = nn.Linear(d_model, d_ff, bias=bias)
        self.w3 = nn.Linear(d_ff, d_model, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w3(F.silu(self.w1(x)) * self.w2(x))

class LayerScale(nn.Module):

    def __init__(self, dim: int, init: float = 1.0) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.full((dim,), init))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.scale

class CausalSelfAttention(nn.Module):

    def __init__(self, configuration: HybridConfig) -> None:
        super().__init__()
        self.d_model = configuration.d_model
        self.n_heads = configuration.n_heads
        self.head_dim = configuration.d_model // configuration.n_heads
        assert configuration.d_model % configuration.n_heads == 0, "d_model must divide n_heads"
        self.query_projection = nn.Linear(configuration.d_model, configuration.d_model, bias=configuration.use_bias)
        self.key_projection = nn.Linear(configuration.d_model, configuration.d_model, bias=configuration.use_bias)
        self.value_projection = nn.Linear(configuration.d_model, configuration.d_model, bias=configuration.use_bias)
        self.output_projection = nn.Linear(configuration.d_model, configuration.d_model, bias=configuration.use_bias)
        self.attn_drop = nn.Dropout(configuration.dropout)
        self.res_drop = nn.Dropout(configuration.dropout)
        self.rope = RotaryEmbedding(self.head_dim, base=configuration.rope_base, maximum_sequence_length=max(8192, configuration.context_len))

    def _split(self, x: torch.Tensor) -> torch.Tensor:
        b, t, c = x.shape
        x = x.view(b, t, self.n_heads, self.head_dim)
        return x.transpose(1, 2)

    def _merge(self, x: torch.Tensor) -> torch.Tensor:
        b, h, t, d = x.shape
        return x.transpose(1, 2).contiguous().view(b, t, h * d)

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        q = self._split(self.query_projection(x))
        k = self._split(self.key_projection(x))
        v = self._split(self.value_projection(x))
        q, k = self.rope(q, k)
        scale = 1.0 / math.sqrt(self.head_dim)
        att = torch.matmul(q, k.transpose(-2, -1)) * scale
        t = x.size(1)
        causal = torch.triu(torch.ones(t, t, device=x.device, dtype=torch.bool), diagonal=1)
        att = att.masked_fill(causal[None, None, :, :], float("-inf"))
        if attn_mask is not None:
            att = att + attn_mask
        p = F.softmax(att, dim=-1)
        p = self.attn_drop(p)
        y = torch.matmul(p, v)
        y = self._merge(y)
        y = self.output_projection(y)
        return self.res_drop(y)

class TransformerBlock(nn.Module):

    def __init__(self, configuration: HybridConfig) -> None:
        super().__init__()
        self.n1 = RMSNorm(configuration.d_model, configuration.rms_eps)
        self.attn = CausalSelfAttention(configuration)
        self.ls1 = LayerScale(configuration.d_model, configuration.layer_scale_init)
        self.n2 = RMSNorm(configuration.d_model, configuration.rms_eps)
        self.ff = SwiGLU(configuration.d_model, configuration.d_ff, bias=configuration.use_bias)
        self.ls2 = LayerScale(configuration.d_model, configuration.layer_scale_init)
        self.drop = nn.Dropout(configuration.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.attn(self.n1(x))
        x = x + self.drop(self.ls1(h))
        h = self.ff(self.n2(x))
        x = x + self.drop(self.ls2(h))
        return x

class DepthwiseConv1d(nn.Module):

    def __init__(self, channels: int, kernel_size: int) -> None:
        super().__init__()
        pad = kernel_size - 1
        self.conv = nn.Conv1d(
            channels,
            channels,
            kernel_size,
            groups=channels,
            padding=pad,
            bias=True,
        )
        self.kernel_size = kernel_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x.transpose(1, 2)
        y = self.conv(y)
        y = y[:, :, : x.size(1)]
        return y.transpose(1, 2)

class SelectiveSSMCore(nn.Module):

    def __init__(
        self,
        d_inner: int,
        state_dim: int,
        scan_mode: str = "auto",
        chunk_size: int = 256,
        dt_min: float = 1e-4,
        dt_max: float = 1.0,
        a_init_min: float = 0.1,
        a_init_max: float = 16.0,
    ) -> None:
        super().__init__()
        self.d_inner = d_inner
        self.state_dim = state_dim
        self.scan_mode = scan_mode
        self.chunk_size = chunk_size
        self.dt_min = dt_min
        self.dt_max = dt_max
        u = torch.rand(d_inner, state_dim)
        a_init = torch.exp(torch.log(torch.tensor(a_init_min)) + u * (math.log(a_init_max) - math.log(a_init_min)))
        self.state_decay_log = nn.Parameter(torch.log(a_init))
        self.D = nn.Parameter(torch.ones(d_inner))
        self.input_to_state_projection = nn.Linear(d_inner, d_inner * state_dim, bias=False)
        self.state_to_output_projection = nn.Linear(d_inner, d_inner * state_dim, bias=False)
        self.time_step_projection = nn.Linear(d_inner, d_inner, bias=True)
        self.out_proj = nn.Linear(d_inner, d_inner, bias=False)
        self._reset_parameters()

    def _inv_softplus(self, y: float) -> float:
        y = max(y, 1e-8)
        return math.log(math.exp(y) - 1.0)

    def _reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.time_step_projection.weight)
        target_dt = (self.dt_min + self.dt_max) * 0.5
        nn.init.constant_(self.time_step_projection.bias, self._inv_softplus(target_dt))
        nn.init.xavier_uniform_(self.input_to_state_projection.weight)
        nn.init.xavier_uniform_(self.state_to_output_projection.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)

    def _step(
        self,
        x_t: torch.Tensor,
        h_prev: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        b, c = x_t.shape
        n = self.state_dim
        dt = F.softplus(self.time_step_projection(x_t)) + 1e-4
        a = -torch.exp(self.state_decay_log).unsqueeze(0)
        a_bar = torch.exp(dt.unsqueeze(-1) * a)
        B = self.input_to_state_projection(x_t).view(b, c, n)
        C = self.state_to_output_projection(x_t).view(b, c, n)
        h = a_bar * h_prev + B
        y = (C * h).sum(dim=-1) + self.D * x_t
        y = self.out_proj(y)
        return y, h

    def _discretize(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        b, t, c = x.shape
        n = self.state_dim
        dt = F.softplus(self.time_step_projection(x)) + self.dt_min
        dt = dt.clamp(max=self.dt_max)
        a = -torch.exp(self.state_decay_log).unsqueeze(0).unsqueeze(0)
        a_bar = torch.exp(dt.unsqueeze(-1) * a)
        B = self.input_to_state_projection(x).view(b, t, c, n)
        C = self.state_to_output_projection(x).view(b, t, c, n)
        return a_bar, B, C

    def _forward_parallel_scan(self, x: torch.Tensor) -> torch.Tensor:
        a_bar, B, C = self._discretize(x)
        eps = 1e-12
        P = torch.cumprod(a_bar.clamp_min(eps), dim=1)
        invP = torch.reciprocal(P.clamp_min(eps))
        H = P * torch.cumsum(B * invP, dim=1)
        y = (C * H).sum(dim=-1) + self.D[None, None, :] * x
        return self.out_proj(y)

    def _forward_chunk_scan(self, x: torch.Tensor) -> torch.Tensor:
        b, t, c = x.shape
        n = self.state_dim
        h_prev = x.new_zeros((b, c, n))
        ys = []
        for s in range(0, t, self.chunk_size):
            e = min(s + self.chunk_size, t)
            xc = x[:, s:e, :]
            a_bar, B, C = self._discretize(xc)
            eps = 1e-12
            P = torch.cumprod(a_bar.clamp_min(eps), dim=1)
            invP = torch.reciprocal(P.clamp_min(eps))
            H = P * (h_prev[:, None, :, :] + torch.cumsum(B * invP, dim=1))
            h_prev = H[:, -1, :, :]
            y = (C * H).sum(dim=-1) + self.D[None, None, :] * xc
            ys.append(self.out_proj(y))
        return torch.cat(ys, dim=1)

    def _forward_recurrent(self, x: torch.Tensor) -> torch.Tensor:
        b, t, c = x.shape
        h = x.new_zeros((b, c, self.state_dim))
        ys = []
        for i in range(t):
            y, h = self._step(x[:, i, :], h)
            ys.append(y)
        return torch.stack(ys, dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mode = self.scan_mode
        if mode == "auto":
            mode = "parallel" if x.size(1) <= 2048 else "chunk"
        if mode == "parallel":
            return self._forward_parallel_scan(x)
        if mode == "chunk":
            return self._forward_chunk_scan(x)
        if mode == "recurrent":
            return self._forward_recurrent(x)
        raise ValueError(f"Unknown SSM scan mode: {mode}")

class SSDDualMixer(nn.Module):

    def __init__(self, d_inner: int, rank: int) -> None:
        super().__init__()
        self.rank = rank
        self.query_projection = nn.Linear(d_inner, rank, bias=False)
        self.key_projection = nn.Linear(d_inner, rank, bias=False)
        self.value_projection = nn.Linear(d_inner, d_inner, bias=False)
        self.output_projection = nn.Linear(d_inner, d_inner, bias=False)

    def _phi(self, x: torch.Tensor) -> torch.Tensor:
        return F.elu(x) + 1.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = self._phi(self.query_projection(x))
        k = self._phi(self.key_projection(x))
        v = self.value_projection(x)
        key_value = torch.einsum("btr,btc->btrc", k, v)
        key_value_prefix = torch.cumsum(key_value, dim=1)
        key_prefix = torch.cumsum(k, dim=1)
        numerator = torch.einsum("btr,btrc->btc", q, key_value_prefix)
        denominator = torch.einsum("btr,btr->bt", q, key_prefix).clamp_min(1e-6).unsqueeze(-1)
        y = numerator / denominator
        return self.output_projection(y)

class Mamba2Block(nn.Module):

    def __init__(self, configuration: HybridConfig) -> None:
        super().__init__()
        d_inner = configuration.d_model * configuration.mamba_expand
        self.norm = RMSNorm(configuration.d_model, configuration.rms_eps)
        self.in_proj = nn.Linear(configuration.d_model, d_inner * 2, bias=configuration.use_bias)
        self.conv = DepthwiseConv1d(d_inner, configuration.mamba_conv_kernel)
        self.ssm = SelectiveSSMCore(
            d_inner,
            configuration.mamba_state_dim,
            scan_mode=configuration.ssm_scan_mode,
            chunk_size=configuration.ssm_chunk_size,
            dt_min=configuration.ssm_dt_min,
            dt_max=configuration.ssm_dt_max,
            a_init_min=configuration.ssm_a_init_min,
            a_init_max=configuration.ssm_a_init_max,
        )
        self.ssd = SSDDualMixer(d_inner, rank=min(configuration.ssd_rank, d_inner))
        self.ssd_mix = nn.Parameter(torch.tensor(float(configuration.ssd_mix_init)))
        self.gate_proj = nn.Linear(d_inner, d_inner, bias=configuration.use_bias)
        self.out_proj = nn.Linear(d_inner, configuration.d_model, bias=configuration.use_bias)
        self.drop = nn.Dropout(configuration.dropout)
        self.ls = LayerScale(configuration.d_model, configuration.layer_scale_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        v, z = self.in_proj(h).chunk(2, dim=-1)
        v = self.conv(v)
        v = F.silu(v)
        v_ssm = self.ssm(v)
        v_ssd = self.ssd(v)
        mix = torch.sigmoid(self.ssd_mix)
        v = (1.0 - mix) * v_ssm + mix * v_ssd
        gate = torch.sigmoid(self.gate_proj(z))
        y = self.out_proj(v * gate)
        x = x + self.drop(self.ls(y))
        return x

class ByteEmbedding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.emb = nn.Embedding(256, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x_bytes: torch.Tensor) -> torch.Tensor:
        return self.drop(self.emb(x_bytes))

class MultiHeadAttention(nn.Module):

    def __init__(self, d_model: int, n_heads: int) -> None:
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.query_projection = nn.Linear(d_model, d_model)
        self.key_projection = nn.Linear(d_model, d_model)
        self.value_projection = nn.Linear(d_model, d_model)
        self.output_projection = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, sequence_length, _ = x.shape
        query = self.query_projection(x).view(batch_size, sequence_length, self.n_heads, self.d_k).transpose(1, 2)
        key = self.key_projection(x).view(batch_size, sequence_length, self.n_heads, self.d_k).transpose(1, 2)
        value = self.value_projection(x).view(batch_size, sequence_length, self.n_heads, self.d_k).transpose(1, 2)
        attention_scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, float("-inf"))
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_output = torch.matmul(attention_weights, value)
        attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, sequence_length, self.d_model)
        return self.output_projection(attention_output)

class FeedForward(nn.Module):

    def __init__(self, d_model: int, d_ff: int) -> None:
        super().__init__()
        self.first_linear = nn.Linear(d_model, d_ff)
        self.second_linear = nn.Linear(d_ff, d_model)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.second_linear(self.activation(self.first_linear(x)))

class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model, n_heads)
        self.feed_forward_network = FeedForward(d_model, d_ff)
        self.attention_layer_norm = nn.LayerNorm(d_model)
        self.feed_forward_layer_norm = nn.LayerNorm(d_model)
        self.attention_dropout = nn.Dropout(dropout)
        self.feed_forward_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attention_output = self.self_attention(x)
        x = x + self.attention_dropout(attention_output)
        x = self.attention_layer_norm(x)
        feed_forward_output = self.feed_forward_network(x)
        x = x + self.feed_forward_dropout(feed_forward_output)
        x = self.feed_forward_layer_norm(x)
        return x

class EntropyPatchPredictor(nn.Module):

    def __init__(self, d_model: int, hidden: int = 128, n_heads: int = 4, n_layers: int = 2) -> None:
        super().__init__()
        self.input_projection = nn.Linear(d_model, hidden)
        self.transformer_layers = nn.ModuleList([
            TransformerEncoderLayer(hidden, n_heads, hidden * 4, dropout=0.1)
            for _ in range(n_layers)
        ])
        self.output_projection = nn.Linear(hidden, 1)
        self.layer_normalization = nn.LayerNorm(hidden)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        x = self.input_projection(h)
        x = self.layer_normalization(x)
        for transformer_layer in self.transformer_layers:
            x = transformer_layer(x)
        entropy = self.output_projection(x).squeeze(-1)
        return F.softplus(entropy)

def _segment_patches_from_entropy(
    entropy: torch.Tensor,
    min_len: int,
    max_len: int,
    threshold: float,
) -> List[List[Tuple[int, int]]]:
    b, t = entropy.shape
    output_tensor: List[List[Tuple[int, int]]] = []
    for bi in range(b):
        cur: List[Tuple[int, int]] = []
        s = 0
        while s < t:
            e = min(s + min_len, t)
            while e < min(s + max_len, t):
                val = float(entropy[bi, e - 1].item())
                if val >= threshold:
                    break
                e += 1
            cur.append((s, e))
            s = e
        output_tensor.append(cur)
    return output_tensor

class PatchAggregator(nn.Module):

    def __init__(self, d_in: int, d_out: int, mode: str = "mean") -> None:
        super().__init__()
        self.mode = mode
        self.proj = nn.Linear(d_in, d_out)
        self.gate = nn.Linear(d_in, d_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == "mean":
            p = x.mean(dim=0)
        elif self.mode == "max":
            p = x.max(dim=0).values
        elif self.mode == "sum":
            p = x.sum(dim=0)
        else:
            w = torch.softmax((x * x).mean(dim=-1), dim=0)
            p = (w[:, None] * x).sum(dim=0)
        return self.proj(p) * torch.sigmoid(self.gate(p))

class LocalEncoderBlock(nn.Module):

    def __init__(self, d_model: int, d_ff: int, dropout: float, eps: float) -> None:
        super().__init__()
        local_cfg = HybridConfig(d_model=d_model, n_heads=max(1, d_model // 32), d_ff=d_ff, dropout=dropout, rms_eps=eps)
        self.block = TransformerBlock(local_cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)

class LocalDecoderBlock(nn.Module):

    def __init__(self, d_model: int, d_ff: int, dropout: float, eps: float) -> None:
        super().__init__()
        local_cfg = HybridConfig(d_model=d_model, n_heads=max(1, d_model // 32), d_ff=d_ff, dropout=dropout, rms_eps=eps)
        self.block = TransformerBlock(local_cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)

class BLTFrontEnd(nn.Module):

    def __init__(self, configuration: HybridConfig) -> None:
        super().__init__()
        self.configuration = configuration
        self.byte_emb = ByteEmbedding(configuration.d_model, dropout=configuration.byte_dropout)
        self.local_enc = nn.ModuleList(
            [
                LocalEncoderBlock(configuration.d_model, configuration.d_ff, configuration.dropout, configuration.rms_eps)
                for _ in range(configuration.local_encoder_layers)
            ]
        )
        self.entropy_pred = EntropyPatchPredictor(configuration.d_model, configuration.entropy_predictor_hidden)
        self.patch_agg = PatchAggregator(configuration.d_model, configuration.latent_dim, configuration.patch_agg)
        self.patch_drop = nn.Dropout(configuration.patch_dropout)
        self.post_norm = RMSNorm(configuration.latent_dim, configuration.rms_eps)

    def forward(
        self,
        x_bytes: torch.Tensor,
        force_uniform_patch: bool = False,
    ) -> Tuple[torch.Tensor, List[List[Tuple[int, int]]], torch.Tensor]:
        b, t = x_bytes.shape
        h = self.byte_emb(x_bytes)
        for blk in self.local_enc:
            h = blk(h)
        entropy = self.entropy_pred(h).detach()
        if force_uniform_patch:
            segments: List[List[Tuple[int, int]]] = []
            step = self.configuration.max_patch_len
            for bi in range(b):
                cur = []
                s = 0
                while s < t:
                    e = min(s + step, t)
                    cur.append((s, e))
                    s = e
                segments.append(cur)
        else:
            segments = _segment_patches_from_entropy(
                entropy=entropy,
                min_len=self.configuration.min_patch_len,
                max_len=self.configuration.max_patch_len,
                threshold=self.configuration.patch_entropy_threshold,
            )
        max_patches = max(len(sg) for sg in segments)
        latent = h.new_zeros((b, max_patches, self.configuration.latent_dim))
        patch_mask = torch.zeros((b, max_patches), device=h.device, dtype=torch.bool)
        for bi in range(b):
            for pi, (s, e) in enumerate(segments[bi]):
                patch_h = h[bi, s:e, :]
                latent[bi, pi, :] = self.patch_agg(patch_h)
                patch_mask[bi, pi] = True
        latent = self.patch_drop(self.post_norm(latent))
        return latent, segments, patch_mask

class BLTBackEnd(nn.Module):

    def __init__(self, configuration: HybridConfig) -> None:
        super().__init__()
        self.configuration = configuration
        self.latent_to_local = nn.Linear(configuration.latent_dim, configuration.d_model)
        self.local_dec = nn.ModuleList(
            [
                LocalDecoderBlock(configuration.d_model, configuration.d_ff, configuration.dropout, configuration.rms_eps)
                for _ in range(configuration.local_decoder_layers)
            ]
        )
        self.final_norm = RMSNorm(configuration.d_model, configuration.rms_eps)
        self.head = nn.Linear(configuration.d_model, configuration.vocab_size, bias=False)

    def _scatter_latent(
        self,
        latent: torch.Tensor,
        segments: List[List[Tuple[int, int]]],
        t_bytes: int,
    ) -> torch.Tensor:
        b, _, _ = latent.shape
        output_tensor = latent.new_zeros((b, t_bytes, self.configuration.d_model))
        local_latent = self.latent_to_local(latent)
        for bi in range(b):
            for pi, (s, e) in enumerate(segments[bi]):
                output_tensor[bi, s:e, :] = local_latent[bi, pi, :]
        return output_tensor

    def forward(
        self,
        latent: torch.Tensor,
        segments: List[List[Tuple[int, int]]],
        t_bytes: int,
        residual_local: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        h = self._scatter_latent(latent, segments, t_bytes)
        if residual_local is not None:
            h = h + residual_local
        for blk in self.local_dec:
            h = blk(h)
        h = self.final_norm(h)
        logits = self.head(h)
        return logits

class LatentAdapter(nn.Module):

    def __init__(self, configuration: HybridConfig) -> None:
        super().__init__()
        if configuration.latent_dim == configuration.d_model:
            self.up = nn.Identity()
            self.down = nn.Identity()
        else:
            self.up = nn.Linear(configuration.latent_dim, configuration.d_model, bias=False)
            self.down = nn.Linear(configuration.d_model, configuration.latent_dim, bias=False)

    def to_model(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(x)

    def to_latent(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(x)

class HybridMiddleStack(nn.Module):

    def __init__(self, configuration: HybridConfig) -> None:
        super().__init__()
        self.configuration = configuration
        self.adapter = LatentAdapter(configuration)
        pattern = configuration.hybrid_pattern.lower()
        if len(pattern) < configuration.n_layers:
            reps = (configuration.n_layers + len(pattern) - 1) // len(pattern)
            pattern = (pattern * reps)[: configuration.n_layers]
        else:
            pattern = pattern[: configuration.n_layers]
        self.pattern = pattern
        layers: List[nn.Module] = []
        for ch in pattern:
            if ch == "t":
                layers.append(TransformerBlock(configuration))
            elif ch == "m":
                layers.append(Mamba2Block(configuration))
            else:
                raise ValueError(f"Unknown layer code {ch} in hybrid_pattern")
        self.layers = nn.ModuleList(layers)
        self.norm = RMSNorm(configuration.d_model, configuration.rms_eps) if configuration.final_norm else nn.Identity()

    def forward(self, latent: torch.Tensor, patch_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.adapter.to_model(latent)
        if patch_mask is not None:
            m = patch_mask[:, :, None].to(dtype=x.dtype)
            x = x * m
        for layer in self.layers:
            x = layer(x)
            if patch_mask is not None:
                x = x * patch_mask[:, :, None].to(dtype=x.dtype)
        x = self.norm(x)
        x = self.adapter.to_latent(x)
        return x

class ControlH1Model(nn.Module):

    def __init__(self, configuration: HybridConfig) -> None:
        super().__init__()
        self.configuration = configuration
        self.front = BLTFrontEnd(configuration)
        self.middle = HybridMiddleStack(configuration)
        self.back = BLTBackEnd(configuration)
        self._init_weights(configuration.init_std)
        if configuration.tie_embeddings and configuration.d_model == configuration.latent_dim:
            self.back.head.weight = self.front.byte_emb.emb.weight

    def _init_weights(self, standard_deviation: float) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=standard_deviation)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=standard_deviation)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        input_bytes: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        force_uniform_patch: bool = False,
        return_aux: bool = False,
    ) -> Dict[str, torch.Tensor]:
        if input_bytes.dtype != torch.long:
            x = input_bytes.long()
        else:
            x = input_bytes
        assert x.dim() == 2, "Expected [B, T]"
        assert x.size(1) <= self.configuration.context_len, "Sequence exceeds context_len"
        latent, segments, patch_mask = self.front(x, force_uniform_patch=force_uniform_patch)
        latent = self.middle(latent, patch_mask=patch_mask)
        logits = self.back(latent, segments, t_bytes=x.size(1), residual_local=None)
        output_tensor: Dict[str, torch.Tensor] = {"logits": logits}
        if labels is not None:
            loss = F.cross_entropy(
                logits[:, :-1, :].reshape(-1, logits.size(-1)),
                labels[:, 1:].reshape(-1),
                reduction="mean",
            )
            output_tensor["loss"] = loss
        if return_aux:
            patch_counts = torch.tensor([len(s) for s in segments], device=x.device, dtype=torch.float32)
            output_tensor["avg_patch_count"] = patch_counts.mean()
            output_tensor["patch_mask"] = patch_mask
        return output_tensor
    @torch.no_grad()

    def generate(
        self,
        prompt_bytes: torch.Tensor,
        max_new_tokens: int = 128,
        temperature: float = 1.0,
        top_k: int = 0,
        force_uniform_patch: bool = False,
    ) -> torch.Tensor:
        self.eval()
        x = prompt_bytes.clone()
        if x.dim() == 1:
            x = x.unsqueeze(0)
        for _ in range(max_new_tokens):
            context_window = x[:, -self.configuration.context_len:]
            output_tensor = self.forward(context_window, force_uniform_patch=force_uniform_patch)
            logits = output_tensor["logits"][:, -1, :] / max(temperature, 1e-5)
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits = logits.masked_fill(logits < v[:, [-1]], float("-inf"))
            p = F.softmax(logits, dim=-1)
            nxt = torch.multinomial(p, num_samples=1)
            x = torch.cat([x, nxt], dim=1)
        return x

    def estimate_num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def save_hwcf(
        self,
        path: str,
        optimizer_state: Optional[Dict[str, object]] = None,
        extra_meta: Optional[Dict[str, str]] = None,
    ) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        cfg_json = json.dumps(self.configuration.to_dict()).encode("utf-8")
        state = self.state_dict()
        keys = sorted(state.keys())
        entries = []
        offset = 0
        blob = io.BytesIO()
        for k in keys:
            t = state[k].detach().cpu().contiguous()
            if t.dtype not in DTYPE_TO_CODE:
                t = t.float()
            raw = t.numpy().tobytes(order="C")
            offset = _align16(offset)
            entries.append((k, DTYPE_TO_CODE[t.dtype], list(t.shape), offset, len(raw)))
            pad = offset - blob.tell()
            if pad > 0:
                blob.write(b"\x00" * pad)
            blob.write(raw)
            offset = blob.tell()
        opt_blob = b""
        if optimizer_state is not None:
            buf = io.BytesIO()
            torch.save(optimizer_state, buf)
            opt_blob = buf.getvalue()
        metadata = {
            "created_unix": str(int(time.time())),
            "model_class": self.__class__.__name__,
            "config_hash": str(_stable_hash_text(str(self.configuration.to_dict()))),
            "param_count": str(self.estimate_num_params()),
        }
        if extra_meta:
            metadata.update(extra_meta)
        meta_blob = json.dumps(metadata).encode("utf-8")
        with open(path, "wb") as f:
            f.write(MODEL_MAGIC)
            f.write(struct.pack("<I", MODEL_VERSION))
            f.write(struct.pack("<I", len(cfg_json)))
            f.write(struct.pack("<I", len(entries)))
            f.write(struct.pack("<I", len(meta_blob)))
            f.write(struct.pack("<I", len(opt_blob)))
            f.write(cfg_json)
            f.write(meta_blob)
            for name, dtype_code, shape, off, nbytes in entries:
                name_b = name.encode("utf-8")
                f.write(struct.pack("<H", len(name_b)))
                f.write(name_b)
                f.write(struct.pack("<B", dtype_code))
                f.write(struct.pack("<B", len(shape)))
                for s in shape:
                    f.write(struct.pack("<I", int(s)))
                f.write(struct.pack("<Q", int(off)))
                f.write(struct.pack("<Q", int(nbytes)))
            f.write(blob.getvalue())
            if opt_blob:
                f.write(opt_blob)
    @staticmethod

    def load_hwcf(
        path: str,
        map_location: str | torch.device = "cpu",
        load_optimizer: bool = False,
    ) -> Tuple["ControlH1Model", Optional[Dict[str, object]], Dict[str, str]]:
        with open(path, "rb") as f:
            magic = f.read(4)
            if magic != MODEL_MAGIC:
                raise ValueError("Invalid HWCF magic")
            version = struct.unpack("<I", f.read(4))[0]
            if version != MODEL_VERSION:
                raise ValueError(f"Unsupported HWCF version: {version}")
            cfg_len = struct.unpack("<I", f.read(4))[0]
            n_tensors = struct.unpack("<I", f.read(4))[0]
            meta_len = struct.unpack("<I", f.read(4))[0]
            opt_len = struct.unpack("<I", f.read(4))[0]
            cfg_blob = f.read(cfg_len)
            meta_blob = f.read(meta_len)
            cfg_dict = json.loads(cfg_blob.decode("utf-8"))
            meta_dict = json.loads(meta_blob.decode("utf-8"))
            configuration = HybridConfig.from_dict(cfg_dict)
            model = ControlH1Model(configuration)
            table = []
            for _ in range(n_tensors):
                name_len = struct.unpack("<H", f.read(2))[0]
                name = f.read(name_len).decode("utf-8")
                dtype_code = struct.unpack("<B", f.read(1))[0]
                ndim = struct.unpack("<B", f.read(1))[0]
                shape = []
                for _j in range(ndim):
                    shape.append(struct.unpack("<I", f.read(4))[0])
                off = struct.unpack("<Q", f.read(8))[0]
                nbytes = struct.unpack("<Q", f.read(8))[0]
                table.append((name, dtype_code, shape, off, nbytes))
            payload_start = f.tell()
            state = {}
            for name, dtype_code, shape, off, nbytes in table:
                f.seek(payload_start + off)
                raw = f.read(nbytes)
                dtype = CODE_TO_DTYPE[dtype_code]
                tensor = torch.frombuffer(bytearray(raw), dtype=dtype).clone().view(*shape)
                state[name] = tensor
            model.load_state_dict(state, strict=True)
            model.to(map_location)
            opt_state = None
            if load_optimizer and opt_len > 0:
                end_payload = payload_start + max((off + nbytes for _, _, _, off, nbytes in table), default=0)
                f.seek(end_payload)
                opt_blob = f.read(opt_len)
                if opt_blob:
                    opt_state = torch.load(io.BytesIO(opt_blob), map_location=map_location)
            return model, opt_state, meta_dict
@dataclass

class ParamGroupStat:
    name: str
    params: int
    trainable: int

def collect_param_stats(model: nn.Module) -> List[ParamGroupStat]:
    groups: Dict[str, ParamGroupStat] = {}
    for name, p in model.named_parameters():
        root = name.split(".")[0]
        if root not in groups:
            groups[root] = ParamGroupStat(root, 0, 0)
        g = groups[root]
        n = p.numel()
        g.params += n
        if p.requires_grad:
            g.trainable += n
    return sorted(groups.values(), key=lambda x: x.params, reverse=True)

def format_param_report(model: nn.Module) -> str:
    statistics = collect_param_stats(model)
    total = sum(s.params for s in statistics)
    trainable = sum(s.trainable for s in statistics)
    lines = []
    lines.append(f"Total params: {total:,}")
    lines.append(f"Trainable params: {trainable:,}")
    lines.append("-" * 64)
    for s in statistics:
        lines.append(f"{s.name:24s} {s.params:12,d} trainable={s.trainable:12,d}")
    return "\n".join(lines)

def create_model(config: Optional[HybridConfig] = None) -> ControlH1Model:
    configuration = config or HybridConfig.tiny_2m_context2k()
    return ControlH1Model(configuration)

def dry_run_shapes(
    model: ControlH1Model,
    batch_size: int = 2,
    seq: int = 128,
    device: str = "cpu",
) -> Dict[str, object]:
    model = model.to(device)
    x = torch.randint(0, 256, (batch_size, seq), device=device, dtype=torch.long)
    output_tensor = model(x, labels=x, return_aux=True)
    return {
        "logits": tuple(output_tensor["logits"].shape),
        "loss": float(output_tensor["loss"].item()),
        "avg_patch_count": float(output_tensor["avg_patch_count"].item()),
        "params": model.estimate_num_params(),
    }

def hybrid_pattern_from_ratio(n_layers: int, mamba_ratio: float) -> str:
    mamba_ratio = max(0.0, min(1.0, mamba_ratio))
    n_m = int(round(n_layers * mamba_ratio))
    n_t = n_layers - n_m
    if n_layers == 0:
        return ""
    pattern = []
    rem_m = n_m
    rem_t = n_t
    for i in range(n_layers):
        m_score = rem_m / max(1, (n_layers - i))
        t_score = rem_t / max(1, (n_layers - i))
        if m_score >= t_score and rem_m > 0:
            pattern.append("m")
            rem_m -= 1
        else:
            pattern.append("t")
            rem_t -= 1
    return "".join(pattern)

def configure_2m_model(
    context_len: int = 2048,
    mamba_ratio: float = 0.4,
) -> HybridConfig:
    configuration = HybridConfig.tiny_2m_context2k()
    configuration.context_len = context_len
    configuration.hybrid_pattern = hybrid_pattern_from_ratio(configuration.n_layers, mamba_ratio)
    return configuration

def estimate_flops_per_token(configuration: HybridConfig) -> float:
    d = configuration.d_model
    h = configuration.n_heads
    t = configuration.context_len
    ff = configuration.d_ff
    attn = 4 * d * d + 2 * t * d
    ffn = 3 * d * ff
    mamba = (configuration.mamba_expand * d) * (configuration.mamba_state_dim + configuration.mamba_conv_kernel + d)
    n_t = configuration.hybrid_pattern.lower().count("t")
    n_m = configuration.hybrid_pattern.lower().count("m")
    return float(n_t * (attn + ffn) + n_m * mamba)

def build_optimizer_param_groups(
    model: nn.Module,
    weight_decay: float = 0.1,
) -> List[Dict[str, object]]:
    decay_params = []
    no_decay_params = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim < 2 or name.endswith(".bias") or "norm" in name.lower():
            no_decay_params.append(p)
        else:
            decay_params.append(p)
    return [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

def bytes_from_text(text: str) -> List[int]:
    return list(text.encode("utf-8", errors="replace"))

def text_from_bytes(byte_ids: Sequence[int]) -> str:
    arr = bytes(int(x) % 256 for x in byte_ids)
    return arr.decode("utf-8", errors="replace")

def special_tag_bytes() -> Dict[str, List[int]]:
    tags = {
        "<|assistant|>": bytes_from_text("<|assistant|>"),
        "<|user|>": bytes_from_text("<|user|>"),
        "<|system|>": bytes_from_text("<|system|>"),
        "<|end|>": bytes_from_text("<|end|>"),
    }
    return tags

def append_tagged_turn(buffer: List[int], role: str, text: str, add_end: bool = True) -> None:
    tags = special_tag_bytes()
    role_tag = f"<|{role}|>"
    if role_tag not in tags:
        raise ValueError(f"Unsupported role {role}")
    buffer.extend(tags[role_tag])
    buffer.extend(bytes_from_text("\n"))
    buffer.extend(bytes_from_text(text))
    if add_end:
        buffer.extend(bytes_from_text("\n"))
        buffer.extend(tags["<|end|>"])
        buffer.extend(bytes_from_text("\n"))

def build_chat_sample(messages: List[Dict[str, str]]) -> List[int]:
    output_tensor: List[int] = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        append_tagged_turn(output_tensor, role=role, text=content, add_end=True)
    return output_tensor

def causal_lm_targets(x: torch.Tensor) -> torch.Tensor:
    return x.clone()

class RunningStats:

    def __init__(self) -> None:
        self.n = 0
        self.mean = 0.0
        self.m2 = 0.0

    def update(self, x: float) -> None:
        self.n += 1
        d = x - self.mean
        self.mean += d / self.n
        d2 = x - self.mean
        self.m2 += d * d2
    @property

    def variance(self) -> float:
        if self.n < 2:
            return 0.0
        return self.m2 / (self.n - 1)
    @property

    def standard_deviation(self) -> float:
        return math.sqrt(self.variance)

def model_diagnostics(model: ControlH1Model) -> Dict[str, float]:
    d = {}
    with torch.no_grad():
        statistics = RunningStats()
        for _n, p in model.named_parameters():
            if p.numel() == 0:
                continue
            statistics.update(float(p.float().abs().mean().item()))
        d["param_abs_mean_avg"] = statistics.mean
        d["param_abs_mean_std"] = statistics.standard_deviation
        d["param_count"] = float(model.estimate_num_params())
    return d

class IdentityBlock(nn.Module):

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x

class ResidualGate(nn.Module):

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.g = nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        gate = torch.sigmoid(self.g)[None, None, :]
        return x + gate * y

class MultiQueryAttention(nn.Module):

    def __init__(self, configuration: HybridConfig, n_kv_heads: int = 1) -> None:
        super().__init__()
        self.d_model = configuration.d_model
        self.n_heads = configuration.n_heads
        self.n_kv = n_kv_heads
        self.head_dim = configuration.d_model // configuration.n_heads
        self.q = nn.Linear(configuration.d_model, configuration.d_model, bias=configuration.use_bias)
        self.k = nn.Linear(configuration.d_model, self.n_kv * self.head_dim, bias=configuration.use_bias)
        self.v = nn.Linear(configuration.d_model, self.n_kv * self.head_dim, bias=configuration.use_bias)
        self.o = nn.Linear(configuration.d_model, configuration.d_model, bias=configuration.use_bias)
        self.rope = RotaryEmbedding(self.head_dim, configuration.rope_base, max(8192, configuration.context_len))

    def _repeat_kv(self, x: torch.Tensor) -> torch.Tensor:
        if x.size(1) == self.n_heads:
            return x
        rep = self.n_heads // x.size(1)
        return x.repeat_interleave(rep, dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        q = self.q(x).view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k(x).view(b, t, self.n_kv, self.head_dim).transpose(1, 2)
        v = self.v(x).view(b, t, self.n_kv, self.head_dim).transpose(1, 2)
        q, k = self.rope(q, k)
        k = self._repeat_kv(k)
        v = self._repeat_kv(v)
        att = torch.matmul(q, k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        causal = torch.triu(torch.ones(t, t, device=x.device, dtype=torch.bool), diagonal=1)
        att = att.masked_fill(causal[None, None, :, :], float("-inf"))
        att = torch.softmax(att, dim=-1)
        y = torch.matmul(att, v).transpose(1, 2).contiguous().view(b, t, self.d_model)
        return self.o(y)

class TransformerBlockMQA(nn.Module):

    def __init__(self, configuration: HybridConfig) -> None:
        super().__init__()
        self.n1 = RMSNorm(configuration.d_model, configuration.rms_eps)
        self.attn = MultiQueryAttention(configuration, n_kv_heads=1)
        self.n2 = RMSNorm(configuration.d_model, configuration.rms_eps)
        self.ff = SwiGLU(configuration.d_model, configuration.d_ff, bias=configuration.use_bias)
        self.g1 = ResidualGate(configuration.d_model)
        self.g2 = ResidualGate(configuration.d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.g1(x, self.attn(self.n1(x)))
        x = self.g2(x, self.ff(self.n2(x)))
        return x

class GLAProjection(nn.Module):

    def __init__(self, dim: int, expand: int = 2) -> None:
        super().__init__()
        d = dim * expand
        self.w = nn.Linear(dim, d, bias=False)
        self.g = nn.Linear(dim, d, bias=False)
        self.o = nn.Linear(d, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.silu(self.w(x)) * torch.sigmoid(self.g(x))
        return self.o(y)

class EntropyAuxHead(nn.Module):

    def __init__(self, dim: int, hidden: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.softplus(self.net(x).squeeze(-1))

def entropy_regularization_loss(
    predicted_entropy: torch.Tensor,
    target_entropy: torch.Tensor,
    weight: float = 0.01,
) -> torch.Tensor:
    return weight * F.mse_loss(predicted_entropy, target_entropy)

def estimate_byte_entropy_targets(x_bytes: torch.Tensor, window: int = 16) -> torch.Tensor:
    b, t = x_bytes.shape
    output_tensor = x_bytes.new_zeros((b, t), dtype=torch.float32)
    for i in range(t):
        s = max(0, i - window + 1)
        chunk = x_bytes[:, s: i + 1]
        uniq = []
        for bi in range(b):
            u = chunk[bi].unique().numel()
            uniq.append(float(u))
        output_tensor[:, i] = torch.tensor(uniq, device=x_bytes.device) / math.log2(256.0)
    return output_tensor

def save_state_dict_pt(model: nn.Module, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(model.state_dict(), path)

def load_state_dict_pt(model: nn.Module, path: str, map_location: Union[str, torch.device] = "cpu") -> None:
    sd = torch.load(path, map_location=map_location)
    model.load_state_dict(sd, strict=True)

def convert_pt_to_hwcf(
    pt_path: str,
    nvcf_path: str,
    config: Optional[HybridConfig] = None,
) -> None:
    configuration = config or HybridConfig.tiny_2m_context2k()
    model = ControlH1Model(configuration)
    load_state_dict_pt(model, pt_path)
    model.save_hwcf(nvcf_path)

def convert_hwcf_to_pt(
    nvcf_path: str,
    pt_path: str,
) -> None:
    model, _opt, _meta = ControlH1Model.load_hwcf(nvcf_path)
    save_state_dict_pt(model, pt_path)

class InferenceSession:

    def __init__(self, model: ControlH1Model, device: str = "cpu") -> None:
        self.model = model.to(device)
        self.device = device
        self.buffer: List[int] = []

    def reset(self) -> None:
        self.buffer = []

    def feed_text(self, text: str) -> None:
        self.buffer.extend(bytes_from_text(text))

    def generate_text(
        self,
        max_new_tokens: int = 128,
        temperature: float = 0.9,
        top_k: int = 40,
    ) -> str:
        if len(self.buffer) == 0:
            self.buffer = bytes_from_text("")
        x = torch.tensor([self.buffer], dtype=torch.long, device=self.device)
        y = self.model.generate(x, max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k)
        output_tensor = y[0].tolist()
        self.buffer = output_tensor
        return text_from_bytes(output_tensor)

def _config_grid() -> List[HybridConfig]:
    output_tensor = []
    for d in [128, 160, 192]:
        for layers in [8, 10, 12]:
            configuration = HybridConfig.tiny_2m_context2k()
            configuration.d_model = d
            configuration.latent_dim = d
            configuration.n_layers = layers
            configuration.n_heads = max(1, d // 32)
            configuration.d_ff = int(d * 2.6)
            configuration.hybrid_pattern = hybrid_pattern_from_ratio(layers, mamba_ratio=0.4)
            output_tensor.append(configuration)
    return output_tensor

def sweep_param_counts() -> List[Tuple[HybridConfig, int]]:
    results = []
    for configuration in _config_grid():
        m = ControlH1Model(configuration)
        results.append((configuration, m.estimate_num_params()))
    return results

def pick_closest_to_target(target_params: int = 2_000_000) -> HybridConfig:
    best_cfg = None
    best_dist = 10**18
    for configuration, n in sweep_param_counts():
        dist = abs(n - target_params)
        if dist < best_dist:
            best_dist = dist
            best_cfg = configuration
    assert best_cfg is not None
    return best_cfg

def build_default_2m_model() -> ControlH1Model:
    configuration = HybridConfig.from_dict(DEFAULT_HYBRID_CONFIG)
    return ControlH1Model(configuration)
@dataclass

class SegmentCacheEntry:
    segments: List[Tuple[int, int]]
    patch_count: int

class RollingByteBuffer:

    def __init__(self, max_len: int) -> None:
        self.max_len = max_len
        self.data: List[int] = []

    def extend(self, xs: Sequence[int]) -> None:
        self.data.extend(int(x) & 0xFF for x in xs)
        if len(self.data) > self.max_len:
            self.data = self.data[-self.max_len:]

    def clear(self) -> None:
        self.data = []

    def as_tensor(self, device: torch.device) -> torch.Tensor:
        if len(self.data) == 0:
            return torch.zeros((1, 1), dtype=torch.long, device=device)
        return torch.tensor([self.data], dtype=torch.long, device=device)

class StreamStateCache:

    def __init__(self, context_len: int) -> None:
        self.context_len = context_len
        self.buffer = RollingByteBuffer(context_len)
        self.last_segments: Optional[SegmentCacheEntry] = None
        self.total_tokens_seen = 0

    def reset(self) -> None:
        self.buffer.clear()
        self.last_segments = None
        self.total_tokens_seen = 0

    def append(self, byte_ids: Sequence[int]) -> None:
        self.buffer.extend(byte_ids)
        self.total_tokens_seen += len(byte_ids)

    def snapshot(self) -> Dict[str, object]:
        output_tensor: Dict[str, object] = {
            "total_tokens_seen": self.total_tokens_seen,
            "buffer_len": len(self.buffer.data),
        }
        if self.last_segments is not None:
            output_tensor["patch_count"] = self.last_segments.patch_count
        return output_tensor

class StreamingGenerator:

    def __init__(
        self,
        model: ControlH1Model,
        device: str = "cpu",
        temperature: float = 0.9,
        top_k: int = 40,
    ) -> None:
        self.model = model.to(device)
        self.device = torch.device(device)
        self.temperature = temperature
        self.top_k = top_k
        self.cache = StreamStateCache(model.configuration.context_len)
    @torch.no_grad()

    def prime(self, text: str) -> None:
        self.cache.append(bytes_from_text(text))
    @torch.no_grad()

    def step(self) -> int:
        x = self.cache.buffer.as_tensor(self.device)
        output_tensor = self.model(x, return_aux=True)
        logits = output_tensor["logits"][:, -1, :] / max(1e-5, self.temperature)
        if self.top_k > 0:
            v, _ = torch.topk(logits, min(self.top_k, logits.size(-1)))
            logits = logits.masked_fill(logits < v[:, [-1]], float("-inf"))
        p = torch.softmax(logits, dim=-1)
        nxt = int(torch.multinomial(p, num_samples=1).item())
        self.cache.append([nxt])
        if "patch_mask" in output_tensor:
            pm = output_tensor["patch_mask"]
            patch_count = int(pm.sum(dim=1).float().mean().item())
            self.cache.last_segments = SegmentCacheEntry([], patch_count)
        return nxt
    @torch.no_grad()

    def generate_bytes(self, n_tokens: int) -> List[int]:
        output_tensor = []
        for _ in range(n_tokens):
            output_tensor.append(self.step())
        return output_tensor
    @torch.no_grad()

    def generate_text(self, n_tokens: int) -> str:
        new_ids = self.generate_bytes(n_tokens)
        return text_from_bytes(new_ids)

def _infer_rank_world() -> Tuple[int, int]:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank(), torch.distributed.get_world_size()
    return 0, 1

class ShardInfo:

    def __init__(self, rank: int, world_size: int) -> None:
        self.rank = rank
        self.world_size = world_size
    @staticmethod

    def from_dist() -> "ShardInfo":
        r, w = _infer_rank_world()
        return ShardInfo(r, w)

    def is_sharded(self) -> bool:
        return self.world_size > 1

    def split_range(self, n: int) -> Tuple[int, int]:
        base = n // self.world_size
        rem = n % self.world_size
        start = self.rank * base + min(self.rank, rem)
        end = start + base + (1 if self.rank < rem else 0)
        return start, end

class TensorParallelLinear(nn.Module):

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        mode: str = "column",
        gather_output: bool = True,
        shard: Optional[ShardInfo] = None,
    ) -> None:
        super().__init__()
        self.mode = mode
        self.gather_output = gather_output
        self.shard = shard or ShardInfo.from_dist()
        if mode == "column":
            s, e = self.shard.split_range(out_features)
            self.local_out = e - s
            self.linear = nn.Linear(in_features, self.local_out, bias=bias)
            self.global_out = out_features
        elif mode == "row":
            s, e = self.shard.split_range(in_features)
            self.local_in = e - s
            self.linear = nn.Linear(self.local_in, out_features, bias=bias)
            self.global_in = in_features
        else:
            raise ValueError("mode must be column or row")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == "column":
            y_local = self.linear(x)
            if self.shard.world_size == 1 or not self.gather_output:
                return y_local
            parts = [torch.empty_like(y_local) for _ in range(self.shard.world_size)]
            torch.distributed.all_gather(parts, y_local)
            return torch.cat(parts, dim=-1)
        s, e = self.shard.split_range(x.size(-1))
        x_local = x[..., s:e]
        y = self.linear(x_local)
        if self.shard.world_size > 1:
            torch.distributed.all_reduce(y)
        return y

class TPAttentionAdapter(nn.Module):

    def __init__(self, configuration: HybridConfig, shard: Optional[ShardInfo] = None) -> None:
        super().__init__()
        self.shard = shard or ShardInfo.from_dist()
        self.head_dim = configuration.d_model // configuration.n_heads
        self.n_heads = configuration.n_heads
        self.query_projection = TensorParallelLinear(configuration.d_model, configuration.d_model, bias=configuration.use_bias, mode="column", shard=self.shard)
        self.key_projection = TensorParallelLinear(configuration.d_model, configuration.d_model, bias=configuration.use_bias, mode="column", shard=self.shard)
        self.value_projection = TensorParallelLinear(configuration.d_model, configuration.d_model, bias=configuration.use_bias, mode="column", shard=self.shard)
        self.output_projection = TensorParallelLinear(configuration.d_model, configuration.d_model, bias=configuration.use_bias, mode="row", shard=self.shard)
        self.rope = RotaryEmbedding(self.head_dim, base=configuration.rope_base, maximum_sequence_length=max(8192, configuration.context_len))

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        b, t, c = x.shape
        h = self.n_heads
        d = c // h
        return x.view(b, t, h, d).transpose(1, 2)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        b, h, t, d = x.shape
        return x.transpose(1, 2).reshape(b, t, h * d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = self._split_heads(self.query_projection(x))
        k = self._split_heads(self.key_projection(x))
        v = self._split_heads(self.value_projection(x))
        q, k = self.rope(q, k)
        t = x.size(1)
        att = torch.matmul(q, k.transpose(-2, -1)) * (1.0 / math.sqrt(q.size(-1)))
        causal = torch.triu(torch.ones(t, t, device=x.device, dtype=torch.bool), diagonal=1)
        att = att.masked_fill(causal[None, None, :, :], float("-inf"))
        p = torch.softmax(att, dim=-1)
        y = torch.matmul(p, v)
        y = self._merge_heads(y)
        y = self.output_projection(y)
        return y

class TPFeedForwardAdapter(nn.Module):

    def __init__(self, configuration: HybridConfig, shard: Optional[ShardInfo] = None) -> None:
        super().__init__()
        self.shard = shard or ShardInfo.from_dist()
        self.w1 = TensorParallelLinear(configuration.d_model, configuration.d_ff, bias=configuration.use_bias, mode="column", shard=self.shard)
        self.w2 = TensorParallelLinear(configuration.d_model, configuration.d_ff, bias=configuration.use_bias, mode="column", shard=self.shard)
        self.w3 = TensorParallelLinear(configuration.d_ff, configuration.d_model, bias=configuration.use_bias, mode="row", shard=self.shard)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w3(F.silu(self.w1(x)) * self.w2(x))

class TPTransformerBlock(nn.Module):

    def __init__(self, configuration: HybridConfig, shard: Optional[ShardInfo] = None) -> None:
        super().__init__()
        self.n1 = RMSNorm(configuration.d_model, configuration.rms_eps)
        self.attn = TPAttentionAdapter(configuration, shard=shard)
        self.n2 = RMSNorm(configuration.d_model, configuration.rms_eps)
        self.ff = TPFeedForwardAdapter(configuration, shard=shard)
        self.drop = nn.Dropout(configuration.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop(self.attn(self.n1(x)))
        x = x + self.drop(self.ff(self.n2(x)))
        return x

class DistributedHybridStack(nn.Module):

    def __init__(self, configuration: HybridConfig) -> None:
        super().__init__()
        self.configuration = configuration
        self.shard = ShardInfo.from_dist()
        pattern = configuration.hybrid_pattern.lower()
        if len(pattern) < configuration.n_layers:
            pattern = (pattern * ((configuration.n_layers + len(pattern) - 1) // len(pattern)))[: configuration.n_layers]
        else:
            pattern = pattern[: configuration.n_layers]
        self.layers = nn.ModuleList()
        for ch in pattern:
            if ch == "t":
                self.layers.append(TPTransformerBlock(configuration, shard=self.shard))
            elif ch == "m":
                self.layers.append(Mamba2Block(configuration))
            else:
                raise ValueError(f"Unknown pattern char: {ch}")
        self.norm = RMSNorm(configuration.d_model, configuration.rms_eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
            if self.shard.world_size > 1 and isinstance(layer, Mamba2Block):
                torch.distributed.all_reduce(x)
                x = x / float(self.shard.world_size)
        return self.norm(x)
@dataclass

class QuantTensor:
    q: torch.Tensor
    scale: torch.Tensor
    zero: torch.Tensor
    bits: int
    axis: int

def quantize_per_channel(x: torch.Tensor, bits: int = 8, axis: int = 0) -> QuantTensor:
    assert bits in (4, 8)
    qmin = 0
    qmax = (1 << bits) - 1
    x_perm = x.transpose(0, axis).contiguous()
    flat = x_perm.view(x_perm.size(0), -1)
    x_min = flat.min(dim=1).values
    x_max = flat.max(dim=1).values
    scale = (x_max - x_min).clamp_min(1e-8) / float(qmax - qmin)
    zero = torch.round(qmin - x_min / scale).clamp(qmin, qmax)
    q = torch.round(flat / scale[:, None] + zero[:, None]).clamp(qmin, qmax)
    q = q.to(torch.uint8).view_as(x_perm).transpose(0, axis).contiguous()
    return QuantTensor(q=q, scale=scale, zero=zero, bits=bits, axis=axis)

def dequantize_per_channel(qt: QuantTensor) -> torch.Tensor:
    q = qt.q.transpose(0, qt.axis).contiguous()
    flat = q.view(q.size(0), -1).float()
    x = (flat - qt.zero[:, None]) * qt.scale[:, None]
    x = x.view_as(q).transpose(0, qt.axis).contiguous()
    return x

class QuantizedStateDict:

    def __init__(self) -> None:
        self.tensors: Dict[str, QuantTensor] = {}

    def add(self, name: str, t: torch.Tensor, bits: int = 8, axis: int = 0) -> None:
        self.tensors[name] = quantize_per_channel(t, bits=bits, axis=axis)

    def reconstruct(self) -> Dict[str, torch.Tensor]:
        return {k: dequantize_per_channel(v) for k, v in self.tensors.items()}

def quantize_model_state_dict(
    model: nn.Module,
    bits: int = 8,
    skip_small: int = 128,
) -> QuantizedStateDict:
    qsd = QuantizedStateDict()
    for name, t in model.state_dict().items():
        if t.numel() < skip_small:
            continue
        if not t.is_floating_point():
            continue
        axis = 0 if t.ndim >= 2 else 0
        qsd.add(name, t.float().cpu(), bits=bits, axis=axis)
    return qsd

def apply_quantized_state_dict(model: nn.Module, qsd: QuantizedStateDict) -> None:
    sd = model.state_dict()
    rec = qsd.reconstruct()
    for k, v in rec.items():
        if k in sd and sd[k].shape == v.shape:
            sd[k] = v.to(sd[k].dtype)
    model.load_state_dict(sd, strict=False)
@dataclass

class TensorRecord:
    name: str
    dtype_code: int
    shape: List[int]
    offset: int
    nbytes: int
@dataclass

class HWCFHeader:
    version: int
    config_text: str
    metadata_text: str
    tensor_records: List[TensorRecord]
    payload_start: int
    optimizer_len: int

def parse_hwcf_header(path: str) -> HWCFHeader:
    with open(path, "rb") as f:
        magic = f.read(4)
        if magic != MODEL_MAGIC:
            raise ValueError("Invalid HWCF file")
        version = struct.unpack("<I", f.read(4))[0]
        cfg_len = struct.unpack("<I", f.read(4))[0]
        n_tensors = struct.unpack("<I", f.read(4))[0]
        meta_len = struct.unpack("<I", f.read(4))[0]
        opt_len = struct.unpack("<I", f.read(4))[0]
        cfg_text = f.read(cfg_len).decode("utf-8", errors="replace")
        meta_text = f.read(meta_len).decode("utf-8", errors="replace")
        records: List[TensorRecord] = []
        for _ in range(n_tensors):
            name_len = struct.unpack("<H", f.read(2))[0]
            name = f.read(name_len).decode("utf-8", errors="replace")
            dtype_code = struct.unpack("<B", f.read(1))[0]
            ndim = struct.unpack("<B", f.read(1))[0]
            shape = [struct.unpack("<I", f.read(4))[0] for _j in range(ndim)]
            off = struct.unpack("<Q", f.read(8))[0]
            nbytes = struct.unpack("<Q", f.read(8))[0]
            records.append(TensorRecord(name, dtype_code, shape, off, nbytes))
        payload_start = f.tell()
    return HWCFHeader(
        version=version,
        config_text=cfg_text,
        metadata_text=meta_text,
        tensor_records=records,
        payload_start=payload_start,
        optimizer_len=opt_len,
    )

def hwcf_tensor_index(path: str) -> Dict[str, TensorRecord]:
    h = parse_hwcf_header(path)
    return {r.name: r for r in h.tensor_records}

def validate_hwcf_integrity(path: str, strict_alignment: bool = True) -> Dict[str, object]:
    h = parse_hwcf_header(path)
    errors: List[str] = []
    warnings: List[str] = []
    offsets = []
    for r in h.tensor_records:
        if r.dtype_code not in CODE_TO_DTYPE:
            errors.append(f"Unknown dtype_code for {r.name}: {r.dtype_code}")
        if len(r.shape) == 0:
            warnings.append(f"Scalar tensor {r.name} has empty shape list")
        if strict_alignment and (r.offset % 16 != 0):
            errors.append(f"Unaligned tensor offset {r.offset} for {r.name}")
        offsets.append((r.offset, r.offset + r.nbytes, r.name))
    offsets = sorted(offsets)
    for i in range(1, len(offsets)):
        if offsets[i][0] < offsets[i - 1][1]:
            errors.append(f"Overlapping tensor regions: {offsets[i-1][2]} and {offsets[i][2]}")
    return {
        "ok": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "tensor_count": len(h.tensor_records),
        "version": h.version,
    }

def compare_hwcf_models(path_a: str, path_b: str) -> Dict[str, object]:
    idx_a = hwcf_tensor_index(path_a)
    idx_b = hwcf_tensor_index(path_b)
    names_a = set(idx_a.keys())
    names_b = set(idx_b.keys())
    only_a = sorted(list(names_a - names_b))
    only_b = sorted(list(names_b - names_a))
    common = sorted(list(names_a & names_b))
    shape_mismatch = []
    dtype_mismatch = []
    for n in common:
        ra, rb = idx_a[n], idx_b[n]
        if ra.shape != rb.shape:
            shape_mismatch.append(n)
        if ra.dtype_code != rb.dtype_code:
            dtype_mismatch.append(n)
    return {
        "only_in_a": only_a,
        "only_in_b": only_b,
        "common": len(common),
        "shape_mismatch": shape_mismatch,
        "dtype_mismatch": dtype_mismatch,
    }

def extract_hwcf_tensor(path: str, tensor_name: str) -> torch.Tensor:
    h = parse_hwcf_header(path)
    record = None
    for r in h.tensor_records:
        if r.name == tensor_name:
            record = r
            break
    if record is None:
        raise KeyError(f"Tensor not found in HWCF: {tensor_name}")
    with open(path, "rb") as f:
        f.seek(h.payload_start + record.offset)
        raw = f.read(record.nbytes)
    dtype = CODE_TO_DTYPE[record.dtype_code]
    t = torch.frombuffer(bytearray(raw), dtype=dtype).clone().view(*record.shape)
    return t

def rewrite_hwcf_metadata(path: str, new_meta: Dict[str, str], out_path: str) -> None:
    model, optimizer_state, metadata = ControlH1Model.load_hwcf(path, load_optimizer=True)
    merged = dict(metadata)
    merged.update(new_meta)
    model.save_hwcf(out_path, optimizer_state=optimizer_state, extra_meta=merged)
@dataclass

class PerfSample:
    batch_size: int
    sequence_length: int
    step_ms: float
    toks_per_sec: float
    mem_alloc_mb: float
    mem_peak_mb: float

def _cuda_mem_stats(device: torch.device) -> Tuple[float, float]:
    if device.type != "cuda":
        return 0.0, 0.0
    alloc = torch.cuda.memory_allocated(device) / (1024.0 * 1024.0)
    peak = torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0)
    return float(alloc), float(peak)

def benchmark_train_step(
    model: ControlH1Model,
    device: str = "cpu",
    batch_size: int = 4,
    sequence_length: int = 512,
    n_warmup: int = 5,
    n_steps: int = 20,
    lr: float = 1e-3,
) -> PerfSample:
    dev = torch.device(device)
    model = model.to(dev).train()
    optimizer_state = torch.optim.AdamW(model.parameters(), lr=lr)
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats(dev)
    x = torch.randint(0, 256, (batch_size, sequence_length), device=dev, dtype=torch.long)
    y = x.clone()
    for _ in range(n_warmup):
        optimizer_state.zero_grad(set_to_none=True)
        output_tensor = model(x, labels=y)
        output_tensor["loss"].backward()
        optimizer_state.step()
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)
    t0 = time.perf_counter()
    for _ in range(n_steps):
        optimizer_state.zero_grad(set_to_none=True)
        output_tensor = model(x, labels=y)
        output_tensor["loss"].backward()
        optimizer_state.step()
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)
    t1 = time.perf_counter()
    ms = (t1 - t0) * 1000.0 / max(1, n_steps)
    tps = (batch_size * sequence_length) / max(1e-9, ms / 1000.0)
    alloc, peak = _cuda_mem_stats(dev)
    return PerfSample(
        batch_size=batch_size,
        sequence_length=sequence_length,
        step_ms=float(ms),
        toks_per_sec=float(tps),
        mem_alloc_mb=alloc,
        mem_peak_mb=peak,
    )
@dataclass

class AblationResult:
    label: str
    params: int
    loss: float
    avg_patch_count: float

def quick_ablation_step(
    configuration: HybridConfig,
    batch_size: int = 2,
    sequence_length: int = 128,
    device: str = "cpu",
) -> AblationResult:
    model = ControlH1Model(configuration).to(device)
    x = torch.randint(0, 256, (batch_size, sequence_length), device=device, dtype=torch.long)
    output_tensor = model(x, labels=x, return_aux=True)
    return AblationResult(
        label=f"pattern={configuration.hybrid_pattern},scan={configuration.ssm_scan_mode}",
        params=model.estimate_num_params(),
        loss=float(output_tensor["loss"].item()),
        avg_patch_count=float(output_tensor["avg_patch_count"].item()),
    )

def run_ablation_suite(device: str = "cpu") -> List[AblationResult]:
    base = HybridConfig.tiny_2m_context2k()
    variants = []
    c1 = HybridConfig.from_dict(base.to_dict())
    c1.ssm_scan_mode = "parallel"
    c1.hybrid_pattern = hybrid_pattern_from_ratio(c1.n_layers, 0.4)
    variants.append(c1)
    c2 = HybridConfig.from_dict(base.to_dict())
    c2.ssm_scan_mode = "chunk"
    c2.hybrid_pattern = hybrid_pattern_from_ratio(c2.n_layers, 0.6)
    variants.append(c2)
    c3 = HybridConfig.from_dict(base.to_dict())
    c3.ssm_scan_mode = "recurrent"
    c3.hybrid_pattern = hybrid_pattern_from_ratio(c3.n_layers, 0.2)
    variants.append(c3)
    return [quick_ablation_step(configuration, device=device) for configuration in variants]

def format_ablation_table(results: Sequence[AblationResult]) -> str:
    lines = []
    lines.append("label | params | loss | avg_patch_count")
    lines.append("-" * 72)
    for r in results:
        lines.append(f"{r.label} | {r.params:,} | {r.loss:.4f} | {r.avg_patch_count:.2f}")
    return "\n".join(lines)

def benchmark_scan_modes(
    configuration: Optional[HybridConfig] = None,
    seq_lens: Sequence[int] = (256, 512, 1024, 2048),
    device: str = "cpu",
) -> Dict[str, List[PerfSample]]:
    base = configuration or HybridConfig.tiny_2m_context2k()
    output_tensor: Dict[str, List[PerfSample]] = {}
    for mode in ("parallel", "chunk", "recurrent"):
        c = HybridConfig.from_dict(base.to_dict())
        c.ssm_scan_mode = mode
        model = ControlH1Model(c)
        perf = []
        for seq in seq_lens:
            perf.append(
                benchmark_train_step(
                    model=model,
                    device=device,
                    batch_size=2,
                    sequence_length=min(seq, c.context_len),
                    n_warmup=2,
                    n_steps=5,
                    lr=1e-3,
                )
            )
        output_tensor[mode] = perf
    return output_tensor

def format_scan_benchmarks(data: Dict[str, List[PerfSample]]) -> str:
    lines = []
    for mode, rows in data.items():
        lines.append(f"[mode={mode}]")
        lines.append("seq | step_ms | toks_per_sec | mem_alloc_mb | mem_peak_mb")
        for r in rows:
            lines.append(
                f"{r.sequence_length:4d} | {r.step_ms:8.2f} | {r.toks_per_sec:12.1f} | "
                f"{r.mem_alloc_mb:11.1f} | {r.mem_peak_mb:10.1f}"
            )
        lines.append("")
    return "\n".join(lines)

class GradMonitor:

    def __init__(self, model: nn.Module) -> None:
        self.model = model

    def global_grad_norm(self) -> float:
        total = 0.0
        for p in self.model.parameters():
            if p.grad is None:
                continue
            g = p.grad.detach().float()
            total += float((g * g).sum().item())
        return math.sqrt(total)

    def per_module_norms(self) -> Dict[str, float]:
        output_tensor: Dict[str, float] = {}
        for name, p in self.model.named_parameters():
            if p.grad is None:
                continue
            root = name.split(".")[0]
            g = p.grad.detach().float()
            val = float((g * g).sum().sqrt().item())
            output_tensor[root] = output_tensor.get(root, 0.0) + val
        return output_tensor

class ActivationMonitor:

    def __init__(self, model: nn.Module) -> None:
        self.model = model
        self.handles = []
        self.statistics: Dict[str, Tuple[float, float]] = {}

    def _hook(self, name: str):

        def fn(_m, _inp, output_tensor):
            if not torch.is_tensor(output_tensor):
                return
            x = output_tensor.detach().float()
            self.statistics[name] = (float(x.mean().item()), float(x.std().item()))
        return fn

    def attach(self, module_filter: Optional[callable] = None) -> None:
        for name, mod in self.model.named_modules():
            if module_filter is not None and not module_filter(name, mod):
                continue
            if isinstance(mod, (nn.Linear, RMSNorm, nn.Conv1d)):
                self.handles.append(mod.register_forward_hook(self._hook(name)))

    def clear(self) -> None:
        self.statistics.clear()

    def remove(self) -> None:
        for h in self.handles:
            h.remove()
        self.handles = []

def assert_model_contract(model: ControlH1Model) -> None:
    configuration = model.configuration
    if configuration.context_len != 2048:
        raise AssertionError("Model context_len is not 2048 as requested target default.")
    if configuration.vocab_size != 256:
        raise AssertionError("Model vocab_size must be 256 for raw byte modeling.")
    if "t" not in configuration.hybrid_pattern.lower():
        raise AssertionError("hybrid_pattern must include Transformer blocks.")
    if "m" not in configuration.hybrid_pattern.lower():
        raise AssertionError("hybrid_pattern must include Mamba blocks.")
    if configuration.max_patch_len < configuration.min_patch_len:
        raise AssertionError("Invalid patch length bounds.")

def assert_forward_backward(model: ControlH1Model, device: str = "cpu") -> None:
    model = model.to(device).train()
    x = torch.randint(0, 256, (2, 64), device=device, dtype=torch.long)
    output_tensor = model(x, labels=x)
    loss = output_tensor["loss"]
    loss.backward()
    gn = 0.0
    for p in model.parameters():
        if p.grad is not None:
            gn += float(p.grad.detach().float().pow(2).sum().item())
    if not math.isfinite(gn):
        raise AssertionError("Non-finite gradient norm.")

class ByteTransform:

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()

class RandomByteDrop(ByteTransform):

    def __init__(self, p: float = 0.01, fill: int = 32) -> None:
        self.p = p
        self.fill = fill

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if self.p <= 0:
            return x
        m = torch.rand_like(x.float()) < self.p
        y = x.clone()
        y[m] = self.fill
        return y

class RandomByteSpanMask(ByteTransform):

    def __init__(self, p: float = 0.05, span: int = 3, fill: int = 32) -> None:
        self.p = p
        self.span = span
        self.fill = fill

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if self.p <= 0:
            return x
        y = x.clone()
        b, t = y.shape
        for bi in range(b):
            i = 0
            while i < t:
                if random.random() < self.p:
                    e = min(t, i + self.span)
                    y[bi, i:e] = self.fill
                    i = e
                else:
                    i += 1
        return y

class ByteTransformChain(ByteTransform):

    def __init__(self, transforms: Sequence[ByteTransform]) -> None:
        self.transforms = list(transforms)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        y = x
        for tr in self.transforms:
            y = tr(y)
        return y
@dataclass

class CurriculumStage:
    max_seq_len: int
    epochs: int
    lr_scale: float = 1.0
    patch_entropy_threshold: Optional[float] = None

class CurriculumPlan:

    def __init__(self, stages: Sequence[CurriculumStage]) -> None:
        self.stages = list(stages)
        if len(self.stages) == 0:
            raise ValueError("Curriculum plan requires at least one stage.")

    def stage_for_epoch(self, epoch: int) -> CurriculumStage:
        e = epoch
        for st in self.stages:
            if e < st.epochs:
                return st
            e -= st.epochs
        return self.stages[-1]

def default_curriculum(context_len: int = 2048) -> CurriculumPlan:
    stages = [
        CurriculumStage(max_seq_len=min(256, context_len), epochs=1, lr_scale=1.0, patch_entropy_threshold=2.4),
        CurriculumStage(max_seq_len=min(512, context_len), epochs=1, lr_scale=0.9, patch_entropy_threshold=2.3),
        CurriculumStage(max_seq_len=min(1024, context_len), epochs=1, lr_scale=0.8, patch_entropy_threshold=2.2),
        CurriculumStage(max_seq_len=context_len, epochs=1000, lr_scale=0.7, patch_entropy_threshold=2.2),
    ]
    return CurriculumPlan(stages)

class SequencePacker:

    def __init__(self, window: int, stride: Optional[int] = None) -> None:
        self.window = window
        self.stride = stride or window

    def pack(self, sequences: Sequence[Sequence[int]]) -> List[List[int]]:
        stream: List[int] = []
        for seq in sequences:
            stream.extend(int(x) & 0xFF for x in seq)
            stream.append(ord("\n"))
        output_tensor = []
        for s in range(0, max(1, len(stream) - self.window), self.stride):
            e = s + self.window
            if e > len(stream):
                break
            output_tensor.append(stream[s:e])
        return output_tensor

def make_next_token_batch(
    windows: Sequence[Sequence[int]],
    device: str = "cpu",
) -> Tuple[torch.Tensor, torch.Tensor]:
    x = torch.tensor(windows, dtype=torch.long, device=device)
    y = x.clone()
    return x, y

class ScanKernelInterface:

    def forward(
        self,
        a_bar: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        D: torch.Tensor,
        x: torch.Tensor,
        out_proj: nn.Linear,
    ) -> torch.Tensor:
        raise NotImplementedError()

class TorchScanKernel(ScanKernelInterface):

    def __init__(self, mode: str = "parallel", chunk_size: int = 256) -> None:
        self.mode = mode
        self.chunk_size = chunk_size

    def _parallel(self, a_bar: torch.Tensor, B: torch.Tensor, C: torch.Tensor, D: torch.Tensor, x: torch.Tensor, out_proj: nn.Linear) -> torch.Tensor:
        eps = 1e-12
        P = torch.cumprod(a_bar.clamp_min(eps), dim=1)
        invP = torch.reciprocal(P.clamp_min(eps))
        H = P * torch.cumsum(B * invP, dim=1)
        y = (C * H).sum(dim=-1) + D[None, None, :] * x
        return out_proj(y)

    def _chunk(self, a_bar: torch.Tensor, B: torch.Tensor, C: torch.Tensor, D: torch.Tensor, x: torch.Tensor, out_proj: nn.Linear) -> torch.Tensor:
        b, t, c, n = B.shape
        h_prev = x.new_zeros((b, c, n))
        ys = []
        for s in range(0, t, self.chunk_size):
            e = min(s + self.chunk_size, t)
            ab = a_bar[:, s:e, :, :]
            bb = B[:, s:e, :, :]
            cc = C[:, s:e, :, :]
            xx = x[:, s:e, :]
            eps = 1e-12
            P = torch.cumprod(ab.clamp_min(eps), dim=1)
            invP = torch.reciprocal(P.clamp_min(eps))
            H = P * (h_prev[:, None, :, :] + torch.cumsum(bb * invP, dim=1))
            h_prev = H[:, -1, :, :]
            y = (cc * H).sum(dim=-1) + D[None, None, :] * xx
            ys.append(out_proj(y))
        return torch.cat(ys, dim=1)

    def forward(
        self,
        a_bar: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        D: torch.Tensor,
        x: torch.Tensor,
        out_proj: nn.Linear,
    ) -> torch.Tensor:
        if self.mode == "parallel":
            return self._parallel(a_bar, B, C, D, x, out_proj)
        if self.mode == "chunk":
            return self._chunk(a_bar, B, C, D, x, out_proj)
        raise ValueError(f"Unknown torch kernel mode: {self.mode}")

class TritonScanKernel(ScanKernelInterface):

    def __init__(self, mode: str = "parallel", chunk_size: int = 256) -> None:
        self.mode = mode
        self.chunk_size = chunk_size
        self.available = False
        try:
            import triton
            self.available = True
        except Exception:
            self.available = False
        self.fallback = TorchScanKernel(mode=mode, chunk_size=chunk_size)

    def forward(
        self,
        a_bar: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        D: torch.Tensor,
        x: torch.Tensor,
        out_proj: nn.Linear,
    ) -> torch.Tensor:
        return self.fallback.forward(a_bar, B, C, D, x, out_proj)

class KernelRegistry:

    def __init__(self) -> None:
        self.kernels: Dict[str, ScanKernelInterface] = {}

    def register(self, name: str, kernel: ScanKernelInterface) -> None:
        self.kernels[name] = kernel

    def get(self, name: str) -> ScanKernelInterface:
        if name not in self.kernels:
            raise KeyError(f"Kernel not registered: {name}")
        return self.kernels[name]

def default_kernel_registry(configuration: HybridConfig) -> KernelRegistry:
    reg = KernelRegistry()
    reg.register("torch_parallel", TorchScanKernel(mode="parallel", chunk_size=configuration.ssm_chunk_size))
    reg.register("torch_chunk", TorchScanKernel(mode="chunk", chunk_size=configuration.ssm_chunk_size))
    reg.register("triton_parallel", TritonScanKernel(mode="parallel", chunk_size=configuration.ssm_chunk_size))
    reg.register("triton_chunk", TritonScanKernel(mode="chunk", chunk_size=configuration.ssm_chunk_size))
    return reg

class HybridMiddleStackWithKernels(nn.Module):

    def __init__(self, configuration: HybridConfig, kernel_name: str = "torch_parallel") -> None:
        super().__init__()
        self.configuration = configuration
        self.adapter = LatentAdapter(configuration)
        self.registry = default_kernel_registry(configuration)
        self.kernel_name = kernel_name
        pattern = configuration.hybrid_pattern.lower()
        if len(pattern) < configuration.n_layers:
            pattern = (pattern * ((configuration.n_layers + len(pattern) - 1) // len(pattern)))[: configuration.n_layers]
        else:
            pattern = pattern[: configuration.n_layers]
        self.pattern = pattern
        self.layers = nn.ModuleList()
        for ch in pattern:
            if ch == "t":
                self.layers.append(TransformerBlock(configuration))
            elif ch == "m":
                m = Mamba2Block(configuration)
                self.layers.append(m)
            else:
                raise ValueError(f"Unknown hybrid code: {ch}")
        self.norm = RMSNorm(configuration.d_model, configuration.rms_eps) if configuration.final_norm else nn.Identity()

    def _inject_kernel(self, layer: nn.Module) -> None:
        if not isinstance(layer, Mamba2Block):
            return
        ker = self.registry.get(self.kernel_name)
        ssm = layer.ssm
        if not hasattr(ssm, "_kernel_override"):
            setattr(ssm, "_kernel_override", ker)

    def forward(self, latent: torch.Tensor, patch_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.adapter.to_model(latent)
        for layer in self.layers:
            self._inject_kernel(layer)
            x = layer(x)
            if patch_mask is not None:
                x = x * patch_mask[:, :, None].to(dtype=x.dtype)
        x = self.norm(x)
        return self.adapter.to_latent(x)

class DistillAdapter(nn.Module):

    def __init__(self, temperature: float = 2.0, alpha: float = 0.5) -> None:
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha

    def kl_logits(self, student_logits: torch.Tensor, teacher_logits: torch.Tensor) -> torch.Tensor:
        t = self.temperature
        s = F.log_softmax(student_logits / t, dim=-1)
        q = F.softmax(teacher_logits / t, dim=-1)
        return F.kl_div(s, q, reduction="batchmean") * (t * t)

    def latent_mse(self, student_latent: torch.Tensor, teacher_latent: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(student_latent, teacher_latent)

    def total_loss(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        student_latent: torch.Tensor,
        teacher_latent: torch.Tensor,
    ) -> torch.Tensor:
        return self.alpha * self.kl_logits(student_logits, teacher_logits) + (1.0 - self.alpha) * self.latent_mse(student_latent, teacher_latent)

class PatchSchedule:

    def __init__(self, start: float = 2.4, end: float = 2.0, total_steps: int = 10000) -> None:
        self.start = start
        self.end = end
        self.total_steps = max(1, total_steps)

    def value(self, step: int) -> float:
        p = min(max(step / float(self.total_steps), 0.0), 1.0)
        return self.start + (self.end - self.start) * p

def apply_patch_schedule(configuration: HybridConfig, schedule: PatchSchedule, step: int) -> HybridConfig:
    c = HybridConfig.from_dict(configuration.to_dict())
    c.patch_entropy_threshold = schedule.value(step)
    return c

def model_clone(model: ControlH1Model, device: str = "cpu") -> ControlH1Model:
    clone = ControlH1Model(HybridConfig.from_dict(model.configuration.to_dict()))
    clone.load_state_dict(model.state_dict())
    return clone.to(device)

def ema_update(target: nn.Module, source: nn.Module, decay: float = 0.999) -> None:
    with torch.no_grad():
        for p_t, p_s in zip(target.parameters(), source.parameters()):
            p_t.data.mul_(decay).add_(p_s.data, alpha=(1.0 - decay))

def save_text(path: str, text: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)

def load_text(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return f.read()

def dump_model_summary(path: str, model: ControlH1Model) -> None:
    report = []
    report.append(format_param_report(model))
    report.append("")
    report.append("Diagnostics:")
    diagnostics = model_diagnostics(model)
    for k, v in diagnostics.items():
        report.append(f"- {k}: {v}")
    save_text(path, "\n".join(report))

def save_ablation_report(path: str, results: Sequence[AblationResult]) -> None:
    save_text(path, format_ablation_table(results))

def save_scan_benchmark_report(path: str, table: Dict[str, List[PerfSample]]) -> None:
    save_text(path, format_scan_benchmarks(table))

def ensure_reproducibility(seed: int = 1337) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def smoke_test_forward() -> None:
    configuration = HybridConfig.tiny_2m_context2k()
    model = ControlH1Model(configuration)
    x = torch.randint(0, 256, (2, 64), dtype=torch.long)
    output_tensor = model(x, labels=x, return_aux=True)
    assert output_tensor["logits"].shape == (2, 64, 256)
    assert "loss" in output_tensor
    assert "avg_patch_count" in output_tensor

def smoke_test_generate() -> None:
    configuration = HybridConfig.tiny_2m_context2k()
    model = ControlH1Model(configuration)
    x = torch.tensor([bytes_from_text("hello")], dtype=torch.long)
    y = model.generate(x, max_new_tokens=8)
    assert y.shape[1] == x.shape[1] + 8

def smoke_test_save_load_hwcf(tmp_path: str = "tmp_test_model.hwcf") -> None:
    configuration = HybridConfig.tiny_2m_context2k()
    model = ControlH1Model(configuration)
    model.save_hwcf(tmp_path, extra_meta={"case": "smoke"})
    m2, _opt, metadata = ControlH1Model.load_hwcf(tmp_path, load_optimizer=False)
    assert "case" in metadata
    assert m2.estimate_num_params() == model.estimate_num_params()
    try:
        os.remove(tmp_path)
    except Exception:
        pass

def smoke_test_quantization() -> None:
    configuration = HybridConfig.tiny_2m_context2k()
    model = ControlH1Model(configuration)
    qsd = quantize_model_state_dict(model, bits=8, skip_small=2048)
    assert len(qsd.tensors) > 0
    apply_quantized_state_dict(model, qsd)

def smoke_test_streaming() -> None:
    configuration = HybridConfig.tiny_2m_context2k()
    model = ControlH1Model(configuration)
    generated_text = StreamingGenerator(model)
    generated_text.prime("<|user|>\nhello\n<|end|>\n<|assistant|>\n")
    b = generated_text.step()
    assert 0 <= b <= 255

def smoke_test_ablation_suite() -> None:
    results = run_ablation_suite(device="cpu")
    assert len(results) >= 3
    txt = format_ablation_table(results)
    assert "label" in txt

def run_all_smoke_tests() -> None:
    smoke_test_forward()
    smoke_test_generate()
    smoke_test_save_load_hwcf()
    smoke_test_quantization()
    smoke_test_streaming()
    smoke_test_ablation_suite()

def _parse_cli_args(argv: Optional[Sequence[str]] = None) -> Dict[str, object]:
    import argparse
    p = argparse.ArgumentParser(
        prog="model.py",
        description="Utility CLI for the Novel Byte-Latent Hybrid model.",
    )
    sub = p.add_subparsers(dest="cmd", required=False)
    s_info = sub.add_parser("info", help="Print model param report and diagnostics.")
    s_info.add_argument("--target-params", type=int, default=2_000_000)
    s_info.add_argument("--device", type=str, default="cpu")
    s_dry = sub.add_parser("dryrun", help="Run a dry forward pass.")
    s_dry.add_argument("--bsz", type=int, default=2)
    s_dry.add_argument("--seq", type=int, default=128)
    s_dry.add_argument("--device", type=str, default="cpu")
    s_gen = sub.add_parser("generate", help="Generate text bytes from prompt.")
    s_gen.add_argument("--prompt", type=str, default="<|user|>\nhello\n<|end|>\n<|assistant|>\n")
    s_gen.add_argument("--max-new", type=int, default=64)
    s_gen.add_argument("--temp", type=float, default=0.9)
    s_gen.add_argument("--top-k", type=int, default=40)
    s_gen.add_argument("--device", type=str, default="cpu")
    s_bench = sub.add_parser("bench", help="Benchmark scan modes quickly.")
    s_bench.add_argument("--device", type=str, default="cpu")
    s_abl = sub.add_parser("ablate", help="Run tiny ablation suite.")
    s_abl.add_argument("--device", type=str, default="cpu")
    s_smoke = sub.add_parser("smoke", help="Run all smoke tests.")
    s_export = sub.add_parser("export", help="Export model to HWCF.")
    s_export.add_argument("--out", type=str, default="export/model.hwcf")
    s_validate = sub.add_parser("validate", help="Validate HWCF integrity.")
    s_validate.add_argument("--path", type=str, required=True)
    args = p.parse_args(list(argv) if argv is not None else None)
    return vars(args)

def _cli_info(target_params: int, device: str) -> int:
    configuration = HybridConfig.from_dict(DEFAULT_HYBRID_CONFIG)
    model = ControlH1Model(configuration).to(device)
    print(format_param_report(model))
    print("Diagnostics:", model_diagnostics(model))
    print("Estimated FLOPs/token (rough):", estimate_flops_per_token(configuration))
    return 0

def _cli_dryrun(batch_size: int, seq: int, device: str) -> int:
    configuration = HybridConfig.from_dict(DEFAULT_HYBRID_CONFIG)
    model = ControlH1Model(configuration).to(device)
    print(dry_run_shapes(model, batch_size=batch_size, seq=seq, device=device))
    return 0

def _cli_generate(prompt: str, max_new: int, temp: float, top_k: int, device: str) -> int:
    configuration = HybridConfig.from_dict(DEFAULT_HYBRID_CONFIG)
    model = ControlH1Model(configuration).to(device)
    input_tensor = torch.tensor([bytes_from_text(prompt)], dtype=torch.long, device=device)
    output_tensor = model.generate(input_tensor, max_new_tokens=max_new, temperature=temp, top_k=top_k)
    print(text_from_bytes(output_tensor[0].tolist()))
    return 0

def _cli_bench(device: str) -> int:
    configuration = HybridConfig.tiny_2m_context2k()
    rows = benchmark_scan_modes(configuration=configuration, device=device)
    print(format_scan_benchmarks(rows))
    return 0

def _cli_ablate(device: str) -> int:
    rows = run_ablation_suite(device=device)
    print(format_ablation_table(rows))
    return 0

def _cli_smoke() -> int:
    run_all_smoke_tests()
    print("All smoke tests passed.")
    return 0

def _cli_export(output_tensor: str) -> int:
    model = build_default_2m_model()
    model.save_hwcf(output_tensor, extra_meta={"exported_by": "model.py cli"})
    print(f"Saved: {output_tensor}")
    return 0

def _cli_validate(path: str) -> int:
    info = validate_hwcf_integrity(path)
    print(info)
    return 0 if bool(info.get("ok")) else 2

def model_cli_main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_cli_args(argv)
    cmd = args.get("cmd")
    if cmd is None:
        configuration = HybridConfig.from_dict(DEFAULT_HYBRID_CONFIG)
        model = ControlH1Model(configuration)
        print(format_param_report(model))
        print("Tip: run `python model.py info|dryrun|generate|bench|ablate|smoke|export|validate`")
        return 0
    if cmd == "info":
        return _cli_info(target_params=int(args["target_params"]), device=str(args["device"]))
    if cmd == "dryrun":
        return _cli_dryrun(batch_size=int(args["bsz"]), seq=int(args["seq"]), device=str(args["device"]))
    if cmd == "generate":
        return _cli_generate(
            prompt=str(args["prompt"]),
            max_new=int(args["max_new"]),
            temp=float(args["temp"]),
            top_k=int(args["top_k"]),
            device=str(args["device"]),
        )
    if cmd == "bench":
        return _cli_bench(device=str(args["device"]))
    if cmd == "ablate":
        return _cli_ablate(device=str(args["device"]))
    if cmd == "smoke":
        return _cli_smoke()
    if cmd == "export":
        return _cli_export(output_tensor=str(args["out"]))
    if cmd == "validate":
        return _cli_validate(path=str(args["path"]))
    raise ValueError(f"Unknown command: {cmd}")

def notebook_cell_intro() -> str:
    return (
        "# Novel Byte-Latent Hybrid Model\n"
        "This helper text is generated from model.py for quick copy into notebooks.\n"
    )

def notebook_cell_build_model() -> str:
    return (
        "from model import HybridConfig, ControlH1Model\n"
        "cfg = HybridConfig.tiny_2m_context2k()\n"
        "model = ControlH1Model(cfg)\n"
        "print(model.estimate_num_params())\n"
    )

def notebook_cell_profile() -> str:
    return (
        "from model import benchmark_train_step, HybridConfig, ControlH1Model\n"
        "cfg = HybridConfig.tiny_2m_context2k()\n"
        "model = ControlH1Model(cfg)\n"
        "print(benchmark_train_step(model, device='cpu'))\n"
    )

def notebook_cell_generation() -> str:
    return (
        "from model import HybridConfig, ControlH1Model, bytes_from_text, text_from_bytes\n"
        "import torch\n"
        "cfg = HybridConfig.tiny_2m_context2k()\n"
        "m = ControlH1Model(cfg)\n"
        "x = torch.tensor([bytes_from_text('<|user|>\\nhello\\n<|end|>\\n<|assistant|>\\n')], dtype=torch.long)\n"
        "y = m.generate(x, max_new_tokens=64)\n"
        "print(text_from_bytes(y[0].tolist()))\n"
    )

def notebook_full_template() -> str:
    parts = [
        notebook_cell_intro(),
        notebook_cell_build_model(),
        notebook_cell_profile(),
        notebook_cell_generation(),
    ]
    return "\n\n".join(parts)

def enforce_config_guards(configuration: HybridConfig) -> None:
    if configuration.context_len > 8192:
        raise ValueError("context_len too large for default memory budget in this implementation.")
    if configuration.vocab_size != 256:
        raise ValueError("Byte modeling requires vocab_size=256.")
    if configuration.n_layers < 2:
        raise ValueError("n_layers must be >= 2.")
    if configuration.n_heads < 1 or (configuration.d_model % configuration.n_heads != 0):
        raise ValueError("n_heads must divide d_model.")
    if configuration.min_patch_len < 1:
        raise ValueError("min_patch_len must be >= 1.")
    if configuration.max_patch_len < configuration.min_patch_len:
        raise ValueError("max_patch_len must be >= min_patch_len.")
    if not (0.0 <= configuration.dropout <= 1.0):
        raise ValueError("dropout out of range.")
    if not (0.0 < configuration.ssm_dt_min <= configuration.ssm_dt_max):
        raise ValueError("Invalid SSM dt bounds.")

def enforce_runtime_guards(input_bytes: torch.Tensor, configuration: HybridConfig) -> None:
    if input_bytes.dtype not in (torch.long, torch.int64, torch.uint8, torch.int32):
        raise TypeError("input_bytes must be integer type.")
    if input_bytes.dim() != 2:
        raise ValueError("input_bytes must be [B, T].")
    if input_bytes.size(1) > configuration.context_len:
        raise ValueError("input sequence exceeds configured context length.")
    if input_bytes.min().item() < 0 or input_bytes.max().item() > 255:
        raise ValueError("Byte ids must be in [0,255].")
@dataclass

class LayerTypeStats:
    transformer_layers: int
    mamba_layers: int
    other_layers: int

def count_layer_types(model: ControlH1Model) -> LayerTypeStats:
    t = 0
    m = 0
    o = 0
    for mod in model.middle.layers:
        if isinstance(mod, TransformerBlock):
            t += 1
        elif isinstance(mod, Mamba2Block):
            m += 1
        else:
            o += 1
    return LayerTypeStats(transformer_layers=t, mamba_layers=m, other_layers=o)

def summarize_model_contract(model: ControlH1Model) -> Dict[str, object]:
    statistics = count_layer_types(model)
    configuration = model.configuration
    return {
        "context_len": configuration.context_len,
        "vocab_size": configuration.vocab_size,
        "hybrid_pattern": configuration.hybrid_pattern,
        "transformer_layers": statistics.transformer_layers,
        "mamba_layers": statistics.mamba_layers,
        "other_layers": statistics.other_layers,
        "params": model.estimate_num_params(),
        "scan_mode": configuration.ssm_scan_mode,
    }

def print_model_contract(model: ControlH1Model) -> None:
    info = summarize_model_contract(model)
    print("Model Contract")
    print("-" * 40)
    for k, v in info.items():
        print(f"{k:20s}: {v}")

def compare_forward_modes(
    configuration: Optional[HybridConfig] = None,
    sequence_length: int = 128,
    device: str = "cpu",
) -> Dict[str, float]:
    base = configuration or HybridConfig.tiny_2m_context2k()
    x = torch.randint(0, 256, (2, sequence_length), dtype=torch.long, device=device)
    losses: Dict[str, float] = {}
    for mode in ("parallel", "chunk", "recurrent"):
        c = HybridConfig.from_dict(base.to_dict())
        c.ssm_scan_mode = mode
        m = ControlH1Model(c).to(device)
        output_tensor = m(x, labels=x)
        losses[mode] = float(output_tensor["loss"].item())
    return losses
if __name__ == "__main__":
    try:
        raise SystemExit(model_cli_main())
    except Exception as e:
        print("Fatal error:", str(e))
        raise
