from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn.functional as F
from diffusers import Flux2KleinPipeline
from diffusers.models.embeddings import apply_rotary_emb
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.transformers.transformer_flux2 import (
    Flux2Attention,
    Flux2FeedForward,
    Flux2Modulation,
    Flux2ParallelSelfAttention,
    Flux2SingleTransformerBlock,
    Flux2Transformer2DModel,
    Flux2TransformerBlock,
)
from safetensors import safe_open
from safetensors.torch import save_file
from torch import nn


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

def parse_torch_dtype(value: str) -> torch.dtype:
    dtypes = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    try:
        return dtypes[value]
    except KeyError as exc:
        raise argparse.ArgumentTypeError(f"Unsupported dtype: {value}") from exc


def parse_args() -> argparse.Namespace:
    checkpoint_path = Path("/home/hanlab/nunchaku-flux.2-klein-4b-int4/nunchaku-int4_r128-flux.2-klein-4b.safetensors")
    # This cache stores the converted tensors used by the pure PyTorch reference path.
    dense_cache_path = checkpoint_path.with_name("nunchaku-flux.2-klein-4b-w4a4-emulated.safetensors")
    # If this local snapshot exists, the script avoids a fresh Hugging Face download.
    local_model_path = Path(
        "/root/.cache/huggingface/hub/models--black-forest-labs--FLUX.2-klein-4B/snapshots/e7b7dc27f91deacad38e78976d1f2b499d76a294"
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="black-forest-labs/FLUX.2-klein-4B")
    parser.add_argument("--checkpoint-path", type=Path, default=checkpoint_path)
    parser.add_argument("--dense-cache-path", type=Path, default=dense_cache_path)
    parser.add_argument("--local-model-path", type=Path, default=local_model_path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", type=parse_torch_dtype, default=torch.bfloat16)
    parser.add_argument("--prompt", default="A cat holding a sign that says hello world")
    parser.add_argument("--use-w4a8-main-matmul", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--w4a8-main-matmul-out-chunk", type=int, default=256)
    parser.add_argument("--use-dummy-prompt-embeds", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--dummy-prompt-length", type=int, default=512)
    parser.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--guidance-scale", type=float, default=1.0)
    parser.add_argument("--num-inference-steps", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-path", type=Path, default=Path("flux2-klein-4b-w4a8-ref.png"))
    return parser.parse_args()


def format_gib(num_bytes: int) -> str:
    return f"{num_bytes / 1024**3:.2f} GiB"


def load_state_dict_in_safetensors(
    path: str | Path, device: str | torch.device = "cpu", return_metadata: bool = False
) -> dict[str, torch.Tensor] | tuple[dict[str, torch.Tensor], dict[str, str]]:
    state_dict = {}
    with safe_open(str(path), framework="pt", device=str(device)) as handle:
        metadata = handle.metadata()
        for key in handle.keys():
            state_dict[key] = handle.get_tensor(key)
    if return_metadata:
        return state_dict, metadata
    return state_dict


def convert_flux_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    new_state_dict: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        if "single_transformer_blocks." in key:
            if ".qkv_proj." in key:
                new_key = key.replace(".qkv_proj.", ".attn.to_qkv.")
            elif ".out_proj." in key:
                new_key = key.replace(".out_proj.", ".attn.to_out.")
            elif ".norm_q." in key or ".norm_k." in key:
                new_key = key.replace(".norm_k.", ".attn.norm_k.")
                new_key = new_key.replace(".norm_q.", ".attn.norm_q.")
            else:
                new_key = key
            new_key = new_key.replace(".lora_down", ".proj_down")
            new_key = new_key.replace(".lora_up", ".proj_up")
            if ".smooth_orig" in key:
                new_key = new_key.replace(".smooth_orig", ".smooth_factor_orig")
            elif ".smooth" in key and ".smooth_factor" not in key:
                new_key = new_key.replace(".smooth", ".smooth_factor")
            new_state_dict[new_key] = value
        elif "transformer_blocks." in key:
            if ".mlp_context_fc1" in key:
                new_key = key.replace(".mlp_context_fc1.", ".ff_context.linear_in.")
            elif ".mlp_context_fc2" in key:
                new_key = key.replace(".mlp_context_fc2.", ".ff_context.linear_out.")
            elif ".mlp_fc1" in key:
                new_key = key.replace(".mlp_fc1.", ".ff.linear_in.")
            elif ".mlp_fc2" in key:
                new_key = key.replace(".mlp_fc2.", ".ff.linear_out.")
            elif ".qkv_proj_context." in key:
                new_key = key.replace(".qkv_proj_context.", ".attn.add_qkv_proj.")
            elif ".qkv_proj." in key:
                new_key = key.replace(".qkv_proj.", ".attn.to_qkv.")
            elif ".norm_q." in key or ".norm_k." in key:
                new_key = key.replace(".norm_k.", ".attn.norm_k.")
                new_key = new_key.replace(".norm_q.", ".attn.norm_q.")
            elif ".norm_added_q." in key or ".norm_added_k." in key:
                new_key = key.replace(".norm_added_k.", ".attn.norm_added_k.")
                new_key = new_key.replace(".norm_added_q.", ".attn.norm_added_q.")
            elif ".out_proj." in key:
                new_key = key.replace(".out_proj.", ".attn.to_out.0.")
            elif ".out_proj_context." in key:
                new_key = key.replace(".out_proj_context.", ".attn.to_add_out.")
            else:
                new_key = key
            new_key = new_key.replace(".lora_down", ".proj_down")
            new_key = new_key.replace(".lora_up", ".proj_up")
            if ".smooth_orig" in key:
                new_key = new_key.replace(".smooth_orig", ".smooth_factor_orig")
            elif ".smooth" in key and ".smooth_factor" not in key:
                new_key = new_key.replace(".smooth", ".smooth_factor")
            new_state_dict[new_key] = value
        else:
            new_state_dict[key] = value
    return new_state_dict


class PureTorchSVDQInt4LinearBF16(nn.Module):
    group_size = 64
    use_w4a8_main_matmul = True
    w4a8_main_matmul_out_chunk = 256

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int,
        bias: bool = True,
        torch_dtype: torch.dtype = torch.bfloat16,
        device: str | torch.device | None = None,
    ) -> None:
        super().__init__()
        if device is None:
            device = torch.device("cpu")
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.qweight = nn.Parameter(
            torch.empty(out_features, in_features // 2, dtype=torch.int8, device=device), requires_grad=False
        )
        self.bias = (
            nn.Parameter(torch.empty(out_features, dtype=torch_dtype, device=device), requires_grad=False)
            if bias
            else None
        )
        self.wscales = nn.Parameter(
            torch.empty(in_features // self.group_size, out_features, dtype=torch_dtype, device=device),
            requires_grad=False,
        )
        self.smooth_factor = nn.Parameter(
            torch.ones(in_features, dtype=torch_dtype, device=device), requires_grad=False
        )
        self.smooth_factor_orig = nn.Parameter(
            torch.ones(in_features, dtype=torch_dtype, device=device), requires_grad=False
        )
        self.proj_down = nn.Parameter(torch.empty(in_features, rank, dtype=torch_dtype, device=device), requires_grad=False)
        self.proj_up = nn.Parameter(torch.empty(out_features, rank, dtype=torch_dtype, device=device), requires_grad=False)
        self.register_buffer("main_qweight", torch.empty(0, dtype=torch.int8, device=device), persistent=False)
        self.register_buffer("main_wscale", torch.empty(0, dtype=torch.float32, device=device), persistent=False)
        self.register_buffer("proj_down_dense", torch.empty(0, dtype=torch_dtype, device=device), persistent=False)
        self.register_buffer("proj_up_dense", torch.empty(0, dtype=torch_dtype, device=device), persistent=False)

    @classmethod
    def from_linear(cls, linear: nn.Linear, rank: int, **kwargs) -> "PureTorchSVDQInt4LinearBF16":
        return cls(
            in_features=kwargs.pop("in_features", linear.in_features),
            out_features=kwargs.pop("out_features", linear.out_features),
            bias=kwargs.pop("bias", linear.bias is not None),
            torch_dtype=kwargs.pop("torch_dtype", linear.weight.dtype),
            device=linear.weight.device,
            rank=rank,
            **kwargs,
        )

    def _dequantize_weight(self, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        packed = self.qweight.to(device=device).view(torch.uint8)
        low = (packed & 0x0F).to(torch.int16)
        high = ((packed >> 4) & 0x0F).to(torch.int16)
        low = torch.where(low >= 8, low - 16, low)
        high = torch.where(high >= 8, high - 16, high)
        weight = torch.stack((low, high), dim=-1).reshape(self.out_features, self.in_features).to(dtype)
        scales = self.wscales.to(device=device, dtype=dtype).transpose(0, 1).reshape(
            self.out_features, self.in_features // self.group_size, 1
        )
        weight = weight.reshape(self.out_features, self.in_features // self.group_size, self.group_size)
        weight = (weight * scales).reshape(self.out_features, self.in_features)
        return weight

    def set_reference_weights(
        self,
        *,
        main_qweight: torch.Tensor,
        main_wscale: torch.Tensor,
        proj_down_dense: torch.Tensor,
        proj_up_dense: torch.Tensor,
    ) -> None:
        self.main_qweight = main_qweight.contiguous()
        self.main_wscale = main_wscale.contiguous()
        self.proj_down_dense = proj_down_dense.contiguous()
        self.proj_up_dense = proj_up_dense.contiguous()

    def drop_packed_storage(self) -> None:
        if self.main_qweight.numel():
            device = self.main_qweight.device
        elif self.bias is not None:
            device = self.bias.device
        else:
            device = self.proj_down_dense.device
        if self.bias is not None:
            dtype = self.bias.dtype
        elif self.proj_down_dense.numel():
            dtype = self.proj_down_dense.dtype
        else:
            dtype = torch.bfloat16
        self.qweight = nn.Parameter(torch.empty(0, dtype=torch.int8, device=device), requires_grad=False)
        self.wscales = nn.Parameter(torch.empty(0, dtype=dtype, device=device), requires_grad=False)
        self.smooth_factor = nn.Parameter(torch.empty(0, dtype=dtype, device=device), requires_grad=False)
        self.smooth_factor_orig = nn.Parameter(torch.empty(0, dtype=dtype, device=device), requires_grad=False)
        self.proj_down = nn.Parameter(torch.empty(0, dtype=dtype, device=device), requires_grad=False)
        self.proj_up = nn.Parameter(torch.empty(0, dtype=dtype, device=device), requires_grad=False)

    def _quantize_weight_to_int4(self, weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        weight_grouped = weight.reshape(self.out_features, self.in_features // self.group_size, self.group_size)
        max_abs = weight_grouped.abs().amax(dim=-1, keepdim=True).to(torch.float32)
        scale = torch.where(max_abs > 0, max_abs * (1.0 / 7.0), torch.ones_like(max_abs))
        qweight = torch.round(weight_grouped.to(torch.float32) / scale).clamp_(-8, 7).to(torch.int8)
        return qweight.contiguous(), scale.squeeze(-1).contiguous()

    def _materialize_main_weight(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        qweight = self.main_qweight.to(device=device, dtype=torch.float32)
        scale = self.main_wscale.to(device=device, dtype=torch.float32)
        weight = (qweight * scale.unsqueeze(-1)).reshape(self.out_features, self.in_features)
        return weight.to(dtype)

    def _pack_int4_activation(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.main_qweight.numel() == 0:
            raise RuntimeError("Reference weights are not initialized")
        x_grouped = x.reshape(-1, self.in_features // self.group_size, self.group_size)
        max_abs = x_grouped.abs().amax(dim=-1, keepdim=True).to(torch.float32)
        scale = torch.where(max_abs > 0, max_abs * (1.0 / 7.0), torch.ones_like(max_abs))
        q = torch.round(x_grouped.to(torch.float32) / scale).clamp_(-8, 7).to(torch.int8)
        q_unsigned = (q & 0x0F).to(torch.uint8)
        packed = q_unsigned[..., 0::2] | (q_unsigned[..., 1::2] << 4)
        return packed.reshape(*x.shape[:-1], self.in_features // 2), scale.reshape(
            *x.shape[:-1], self.in_features // self.group_size
        )

    def _dequantize_packed_activation(
        self, packed_x: torch.Tensor, scale: torch.Tensor, output_dtype: torch.dtype
    ) -> torch.Tensor:
        packed = packed_x.reshape(-1, self.in_features // self.group_size, self.group_size // 2).to(torch.uint8)
        low = (packed & 0x0F).to(torch.int16)
        high = ((packed >> 4) & 0x0F).to(torch.int16)
        low = torch.where(low >= 8, low - 16, low)
        high = torch.where(high >= 8, high - 16, high)
        q = torch.stack((low, high), dim=-1).reshape(-1, self.in_features // self.group_size, self.group_size)
        x_dequant = (q.to(torch.float32) * scale.reshape(-1, self.in_features // self.group_size, 1)).reshape(
            *packed_x.shape[:-1], self.in_features
        )
        return x_dequant.to(output_dtype)

    def _quantize_dequantize_activation(self, x: torch.Tensor) -> torch.Tensor:
        packed_x, scale = self._pack_int4_activation(x)
        return self._dequantize_packed_activation(packed_x, scale, x.dtype)

    def _quantize_activation_to_int8(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x_grouped = x.reshape(-1, self.in_features // self.group_size, self.group_size)
        max_abs = x_grouped.abs().amax(dim=-1, keepdim=True).to(torch.float32)
        scale = torch.where(max_abs > 0, max_abs * (1.0 / 127.0), torch.ones_like(max_abs))
        q = torch.round(x_grouped.to(torch.float32) / scale).clamp_(-127, 127).to(torch.int8)
        return q.contiguous(), scale.squeeze(-1).contiguous()

    def _w4a8_main_linear(self, x: torch.Tensor, bias: torch.Tensor | None) -> torch.Tensor:
        qx, ascale = self._quantize_activation_to_int8(x)
        qx = qx.to(torch.float32)
        ascale = ascale.to(torch.float32)
        qw = self.main_qweight.to(device=x.device, dtype=torch.float32)
        wscale = self.main_wscale.to(device=x.device, dtype=torch.float32)
        num_rows = qx.shape[0]
        num_groups = qx.shape[1]
        output = torch.empty(num_rows, self.out_features, device=x.device, dtype=torch.float32)
        for start in range(0, self.out_features, self.w4a8_main_matmul_out_chunk):
            stop = min(start + self.w4a8_main_matmul_out_chunk, self.out_features)
            chunk = torch.zeros(num_rows, stop - start, device=x.device, dtype=torch.float32)
            qweight_chunk = qw[start:stop]
            wscale_chunk = wscale[start:stop]
            for group_idx in range(num_groups):
                prod = qx[:, group_idx, :] @ qweight_chunk[:, group_idx, :].transpose(0, 1)
                chunk += prod * (ascale[:, group_idx : group_idx + 1] * wscale_chunk[:, group_idx].unsqueeze(0))
            output[:, start:stop] = chunk
        if bias is not None:
            output = output + bias.to(torch.float32)
        return output.reshape(*x.shape[:-1], self.out_features).to(x.dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.main_qweight.numel():
            bias = self.bias.to(device=x.device, dtype=x.dtype) if self.bias is not None else None
            if self.use_w4a8_main_matmul:
                output = self._w4a8_main_linear(x, bias)
            else:
                x_main = self._quantize_dequantize_activation(x)
                output = F.linear(x_main, self._materialize_main_weight(x.device, x.dtype), bias)
            if self.proj_down_dense.numel() and self.proj_up_dense.numel():
                low_rank = torch.matmul(x, self.proj_down_dense.to(device=x.device, dtype=x.dtype))
                output = output + torch.matmul(
                    low_rank.to(torch.float32),
                    self.proj_up_dense.to(device=x.device, dtype=torch.float32).transpose(0, 1),
                ).to(x.dtype)
            return output
        dtype = x.dtype
        device = x.device
        weight = self._dequantize_weight(dtype=dtype, device=device)
        x_main = x
        if not torch.all(self.smooth_factor == 1):
            x_main = x_main * self.smooth_factor.to(device=device, dtype=dtype)
        output = F.linear(x_main, weight, self.bias.to(device=device, dtype=dtype) if self.bias is not None else None)
        low_rank = torch.matmul(x_main, self.proj_down.to(device=device, dtype=dtype))
        output = output + torch.matmul(low_rank, self.proj_up.to(device=device, dtype=dtype).transpose(0, 1))
        return output


class PureTorchFlux2Attention(nn.Module):
    def __init__(self, other: Flux2Attention | Flux2ParallelSelfAttention, rank: int, torch_dtype: torch.dtype) -> None:
        super().__init__()
        self.head_dim = other.head_dim
        self.inner_dim = other.inner_dim
        self.query_dim = other.query_dim
        self.out_dim = other.out_dim
        self.heads = other.heads
        self.dropout = other.dropout
        self.norm_q = other.norm_q
        self.norm_k = other.norm_k
        self.added_kv_proj_dim = getattr(other, "added_kv_proj_dim", None)

        with torch.device("meta"):
            to_qkv = nn.Linear(other.query_dim, self.inner_dim * 3, bias=other.use_bias)
        self.to_qkv = PureTorchSVDQInt4LinearBF16.from_linear(to_qkv, rank=rank, torch_dtype=torch_dtype, bias=True)

        if isinstance(other, Flux2Attention):
            self.to_out = nn.ModuleList(
                [
                    PureTorchSVDQInt4LinearBF16.from_linear(
                        other.to_out[0], in_features=self.inner_dim, rank=rank, torch_dtype=torch_dtype, bias=True
                    ),
                    other.to_out[1],
                ]
            )
        else:
            self.to_out = PureTorchSVDQInt4LinearBF16.from_linear(
                nn.Linear(self.inner_dim, self.out_dim, bias=True),
                rank=rank,
                torch_dtype=torch_dtype,
                bias=True,
            )

        if isinstance(other, Flux2Attention) and self.added_kv_proj_dim is not None:
            self.norm_added_q = other.norm_added_q
            self.norm_added_k = other.norm_added_k
            with torch.device("meta"):
                add_qkv = nn.Linear(other.added_kv_proj_dim, self.inner_dim * 3, bias=other.added_proj_bias)
            self.add_qkv_proj = PureTorchSVDQInt4LinearBF16.from_linear(
                add_qkv, rank=rank, torch_dtype=torch_dtype, bias=True
            )
            self.to_add_out = PureTorchSVDQInt4LinearBF16.from_linear(
                other.to_add_out, rank=rank, torch_dtype=torch_dtype, bias=True
            )

    def _run_attention(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        query = query.permute(0, 2, 1, 3)
        key = key.permute(0, 2, 1, 3)
        value = value.permute(0, 2, 1, 3)
        hidden_states = F.scaled_dot_product_attention(query, key, value, dropout_p=0.0, is_causal=False)
        hidden_states = hidden_states.permute(0, 2, 1, 3).flatten(2, 3)
        return hidden_states

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if attention_mask is not None:
            raise NotImplementedError("attention_mask is not supported in the reference path")
        query, key, value = self.to_qkv(hidden_states).chunk(3, dim=-1)
        query = self.norm_q(query.unflatten(-1, (self.heads, -1)))
        key = self.norm_k(key.unflatten(-1, (self.heads, -1)))
        value = value.unflatten(-1, (self.heads, -1))

        text_seq_len = 0
        if encoder_hidden_states is not None and self.added_kv_proj_dim is not None:
            encoder_query, encoder_key, encoder_value = self.add_qkv_proj(encoder_hidden_states).chunk(3, dim=-1)
            encoder_query = self.norm_added_q(encoder_query.unflatten(-1, (self.heads, -1)))
            encoder_key = self.norm_added_k(encoder_key.unflatten(-1, (self.heads, -1)))
            encoder_value = encoder_value.unflatten(-1, (self.heads, -1))
            text_seq_len = encoder_hidden_states.shape[1]
            query = torch.cat([encoder_query, query], dim=1)
            key = torch.cat([encoder_key, key], dim=1)
            value = torch.cat([encoder_value, value], dim=1)

        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb, sequence_dim=1)
            key = apply_rotary_emb(key, image_rotary_emb, sequence_dim=1)

        hidden_states = self._run_attention(query, key, value).to(query.dtype)

        if encoder_hidden_states is not None and self.added_kv_proj_dim is not None:
            encoder_hidden_states = hidden_states[:, :text_seq_len]
            hidden_states = hidden_states[:, text_seq_len:]
            hidden_states = self.to_out[1](self.to_out[0](hidden_states))
            encoder_hidden_states = self.to_add_out(encoder_hidden_states)
            return hidden_states, encoder_hidden_states

        if isinstance(self.to_out, nn.ModuleList):
            hidden_states = self.to_out[1](self.to_out[0](hidden_states))
        else:
            hidden_states = self.to_out(hidden_states)
        return hidden_states


class NunchakuInterleavedSwiGLU(nn.Module):
    interleave_block = 16

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        prefix_shape = x.shape[:-1]
        if x.shape[-1] % (self.interleave_block * 2) != 0:
            raise ValueError(
                f"Expected hidden dim divisible by {self.interleave_block * 2}, got {x.shape[-1]}"
            )
        x = x.view(*x.shape[:-1], x.shape[-1] // (self.interleave_block * 2), 2, self.interleave_block)
        x = F.silu(x[..., 0, :]) * x[..., 1, :]
        return x.reshape(*prefix_shape, -1)


class PureTorchFlux2FeedForward(nn.Module):
    def __init__(self, other: Flux2FeedForward, rank: int, torch_dtype: torch.dtype) -> None:
        super().__init__()
        self.linear_in = PureTorchSVDQInt4LinearBF16.from_linear(
            other.linear_in, rank=rank, torch_dtype=torch_dtype, bias=True
        )
        self.linear_out = PureTorchSVDQInt4LinearBF16.from_linear(
            other.linear_out, rank=rank, torch_dtype=torch_dtype, bias=True
        )
        self.act_fn = NunchakuInterleavedSwiGLU()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.linear_in(hidden_states)
        hidden_states = self.act_fn(hidden_states)
        hidden_states = self.linear_out(hidden_states)
        return hidden_states


class PureTorchFlux2TransformerBlock(nn.Module):
    def __init__(self, block: Flux2TransformerBlock, rank: int, torch_dtype: torch.dtype) -> None:
        super().__init__()
        self.norm1 = block.norm1
        self.norm1_context = block.norm1_context
        self.attn = PureTorchFlux2Attention(block.attn, rank=rank, torch_dtype=torch_dtype)
        self.norm2 = block.norm2
        self.ff = PureTorchFlux2FeedForward(block.ff, rank=rank, torch_dtype=torch_dtype)
        self.norm2_context = block.norm2_context
        self.ff_context = PureTorchFlux2FeedForward(block.ff_context, rank=rank, torch_dtype=torch_dtype)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb_mod_img: torch.Tensor,
        temb_mod_txt: torch.Tensor,
        image_rotary_emb: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        joint_attention_kwargs: Optional[dict[str, Any]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        joint_attention_kwargs = joint_attention_kwargs or {}
        (shift_msa, scale_msa, gate_msa), (shift_mlp, scale_mlp, gate_mlp) = Flux2Modulation.split(temb_mod_img, 2)
        (c_shift_msa, c_scale_msa, c_gate_msa), (c_shift_mlp, c_scale_mlp, c_gate_mlp) = Flux2Modulation.split(
            temb_mod_txt, 2
        )
        norm_hidden_states = (1 + scale_msa) * self.norm1(hidden_states) + shift_msa
        norm_encoder_hidden_states = (1 + c_scale_msa) * self.norm1_context(encoder_hidden_states) + c_shift_msa
        attn_output, context_attn_output = self.attn(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=norm_encoder_hidden_states,
            image_rotary_emb=image_rotary_emb,
            **joint_attention_kwargs,
        )
        hidden_states = hidden_states + gate_msa * attn_output
        norm_hidden_states = (1 + scale_mlp) * self.norm2(hidden_states) + shift_mlp
        hidden_states = hidden_states + gate_mlp * self.ff(norm_hidden_states)
        encoder_hidden_states = encoder_hidden_states + c_gate_msa * context_attn_output
        norm_encoder_hidden_states = (1 + c_scale_mlp) * self.norm2_context(encoder_hidden_states) + c_shift_mlp
        encoder_hidden_states = encoder_hidden_states + c_gate_mlp * self.ff_context(norm_encoder_hidden_states)
        if encoder_hidden_states.dtype == torch.float16:
            encoder_hidden_states = encoder_hidden_states.clip(-65504, 65504)
        return encoder_hidden_states, hidden_states


class PureTorchFlux2SingleTransformerBlock(nn.Module):
    def __init__(self, block: Flux2SingleTransformerBlock, rank: int, torch_dtype: torch.dtype) -> None:
        super().__init__()
        self.norm = block.norm
        self.attn = PureTorchFlux2Attention(block.attn, rank=rank, torch_dtype=torch_dtype)
        self.mlp_hidden_dim = block.attn.mlp_hidden_dim
        self.mlp_fc1 = PureTorchSVDQInt4LinearBF16.from_linear(
            nn.Linear(block.attn.query_dim, self.mlp_hidden_dim * 2, bias=True),
            rank=rank,
            torch_dtype=torch_dtype,
            bias=True,
        )
        self.mlp_fc2 = PureTorchSVDQInt4LinearBF16.from_linear(
            nn.Linear(self.mlp_hidden_dim, block.attn.out_dim, bias=True),
            rank=rank,
            torch_dtype=torch_dtype,
            bias=True,
        )
        self.act_fn = NunchakuInterleavedSwiGLU()

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor],
        temb_mod: torch.Tensor,
        image_rotary_emb: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        joint_attention_kwargs: Optional[dict[str, Any]] = None,
        split_hidden_states: bool = False,
        text_seq_len: Optional[int] = None,
    ) -> tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
        if encoder_hidden_states is not None:
            text_seq_len = encoder_hidden_states.shape[1]
            hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)
        mod_shift, mod_scale, mod_gate = Flux2Modulation.split(temb_mod, 1)[0]
        norm_hidden_states = (1 + mod_scale) * self.norm(hidden_states) + mod_shift
        joint_attention_kwargs = joint_attention_kwargs or {}
        attn_output = self.attn(hidden_states=norm_hidden_states, image_rotary_emb=image_rotary_emb, **joint_attention_kwargs)
        mlp_hidden_states = self.mlp_fc2(self.act_fn(self.mlp_fc1(norm_hidden_states)))
        hidden_states = hidden_states + mod_gate * (attn_output + mlp_hidden_states)
        if hidden_states.dtype == torch.float16:
            hidden_states = hidden_states.clip(-65504, 65504)
        if split_hidden_states:
            assert text_seq_len is not None
            return hidden_states[:, :text_seq_len], hidden_states[:, text_seq_len:]
        return hidden_states


class PureTorchFlux2Transformer2DModel(Flux2Transformer2DModel):
    _is_initialized = False

    def _patch_model(self, rank: int, torch_dtype: torch.dtype) -> None:
        for index, block in enumerate(self.transformer_blocks):
            self.transformer_blocks[index] = PureTorchFlux2TransformerBlock(block, rank=rank, torch_dtype=torch_dtype)
        for index, block in enumerate(self.single_transformer_blocks):
            self.single_transformer_blocks[index] = PureTorchFlux2SingleTransformerBlock(
                block, rank=rank, torch_dtype=torch_dtype
            )
        self._is_initialized = True

    @classmethod
    def from_reference_checkpoint(
        cls,
        checkpoint_path: str | Path,
        torch_dtype: torch.dtype = torch.bfloat16,
    ) -> "PureTorchFlux2Transformer2DModel":
        checkpoint_path = Path(checkpoint_path)
        state_dict, metadata = load_state_dict_in_safetensors(checkpoint_path, return_metadata=True)
        quantization_config = json.loads(metadata.get("quantization_config", "{}"))
        rank = quantization_config.get("rank", 32)
        config = json.loads(metadata["config"])
        transformer = cls.from_config(config)
        transformer = transformer.to(torch_dtype)
        transformer._patch_model(rank=rank, torch_dtype=torch_dtype)
        converted_state_dict = convert_flux_state_dict(state_dict)
        missing, unexpected = transformer.load_state_dict(converted_state_dict, strict=False, assign=True)
        unexpected = [key for key in unexpected if not key.endswith(".smooth_factor_orig")]
        if unexpected:
            raise ValueError(f"Unexpected keys in reference checkpoint: {unexpected}")
        required_missing = [key for key in missing if not key.endswith(".smooth_factor_orig")]
        if required_missing:
            raise ValueError(f"Missing keys in reference checkpoint: {required_missing}")
        return transformer

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        timestep: torch.LongTensor = None,
        img_ids: torch.Tensor = None,
        txt_ids: torch.Tensor = None,
        guidance: torch.Tensor = None,
        joint_attention_kwargs: Optional[dict[str, Any]] = None,
        return_dict: bool = True,
    ) -> torch.Tensor | Transformer2DModelOutput:
        return super().forward(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            timestep=timestep,
            img_ids=img_ids,
            txt_ids=txt_ids,
            guidance=guidance,
            joint_attention_kwargs=joint_attention_kwargs,
            return_dict=return_dict,
        )


module_stats: dict[str, dict[str, int]] = {}
module_state: dict[str, dict[str, int]] = {}


def install_module_profiler(name: str, module: torch.nn.Module):
    hook = module._hf_hook
    orig_pre_forward = hook.pre_forward
    orig_post_forward = hook.post_forward
    module_stats[name] = {
        "calls": 0,
        "max_load_allocated": 0,
        "max_load_reserved": 0,
        "max_activation_allocated": 0,
        "max_activation_reserved": 0,
        "max_peak_allocated": 0,
        "max_peak_reserved": 0,
    }

    def record_stats() -> None:
        state = module_state[name]
        if state.get("finished", False):
            return
        peak_allocated = torch.cuda.max_memory_allocated()
        peak_reserved = torch.cuda.max_memory_reserved()
        stats = module_stats[name]
        stats["calls"] += 1
        stats["max_load_allocated"] = max(stats["max_load_allocated"], state["loaded_allocated"] - state["pre_allocated"])
        stats["max_load_reserved"] = max(stats["max_load_reserved"], state["loaded_reserved"] - state["pre_reserved"])
        stats["max_activation_allocated"] = max(
            stats["max_activation_allocated"], peak_allocated - state["loaded_allocated"]
        )
        stats["max_activation_reserved"] = max(
            stats["max_activation_reserved"], peak_reserved - state["loaded_reserved"]
        )
        stats["max_peak_allocated"] = max(stats["max_peak_allocated"], peak_allocated)
        stats["max_peak_reserved"] = max(stats["max_peak_reserved"], peak_reserved)
        state["finished"] = True

    def profiled_pre_forward(module_, *args, **kwargs):
        torch.cuda.synchronize()
        module_state[name] = {
            "pre_allocated": torch.cuda.memory_allocated(),
            "pre_reserved": torch.cuda.memory_reserved(),
            "finished": False,
        }
        torch.cuda.reset_peak_memory_stats()
        result = orig_pre_forward(module_, *args, **kwargs)
        torch.cuda.synchronize()
        module_state[name]["loaded_allocated"] = torch.cuda.memory_allocated()
        module_state[name]["loaded_reserved"] = torch.cuda.memory_reserved()
        return result

    def profiled_post_forward(module_, output):
        torch.cuda.synchronize()
        record_stats()
        return orig_post_forward(module_, output)

    hook.pre_forward = profiled_pre_forward
    hook.post_forward = profiled_post_forward
    return record_stats


def print_module_stats(name: str) -> None:
    stats = module_stats[name]
    print(
        f"{name} CUDA memory:",
        f"calls={stats['calls']}",
        f"load_allocated={format_gib(stats['max_load_allocated'])}",
        f"activation_allocated={format_gib(stats['max_activation_allocated'])}",
        f"peak_allocated={format_gib(stats['max_peak_allocated'])}",
        f"load_reserved={format_gib(stats['max_load_reserved'])}",
        f"activation_reserved={format_gib(stats['max_activation_reserved'])}",
        f"peak_reserved={format_gib(stats['max_peak_reserved'])}",
    )


def iter_quantized_linears(module: nn.Module) -> dict[str, PureTorchSVDQInt4LinearBF16]:
    return {
        name: submodule for name, submodule in module.named_modules() if isinstance(submodule, PureTorchSVDQInt4LinearBF16)
    }


def load_dense_cache(path: Path) -> dict[str, dict[str, torch.Tensor]]:
    dense_state = load_state_dict_in_safetensors(path, device="cpu")
    cache: dict[str, dict[str, torch.Tensor]] = {}
    for key, value in dense_state.items():
        module_name, tensor_name = key.rsplit(".", 1)
        cache.setdefault(module_name, {})[tensor_name] = value
    return cache


def unpack_lowrank_weight(weight: torch.Tensor, down: bool) -> torch.Tensor:
    c, r = weight.shape
    lane_n, lane_k = 1, 2
    n_pack_size, k_pack_size = 2, 2
    num_n_lanes, num_k_lanes = 8, 4
    frag_n = n_pack_size * num_n_lanes * lane_n
    frag_k = k_pack_size * num_k_lanes * lane_k
    if down:
        r_frags, c_frags = r // frag_n, c // frag_k
    else:
        c_frags, r_frags = c // frag_n, r // frag_k
    weight = weight.view(c_frags, r_frags, num_n_lanes, num_k_lanes, n_pack_size, k_pack_size, lane_k)
    weight = weight.permute(0, 1, 4, 2, 5, 3, 6).contiguous()
    weight = weight.view(c_frags, r_frags, frag_n, frag_k)
    if down:
        weight = weight.permute(1, 2, 0, 3).contiguous().view(r, c)
    else:
        weight = weight.permute(0, 2, 1, 3).contiguous().view(c, r)
    return weight


def build_main_weight_from_quantized_layer(
    source_layer: nn.Module, device: str, dtype: torch.dtype, batch_size: int = 64
) -> torch.Tensor:
    in_features = source_layer.in_features
    out_features = source_layer.out_features
    main_weight = torch.empty(out_features, in_features, dtype=dtype, device="cpu")
    saved_bias = source_layer.bias.detach().clone() if source_layer.bias is not None else None
    saved_proj_down = source_layer.proj_down.detach().clone()
    saved_proj_up = source_layer.proj_up.detach().clone()

    with torch.no_grad():
        if source_layer.bias is not None:
            source_layer.bias.zero_()
        source_layer.proj_down.zero_()
        source_layer.proj_up.zero_()
        for start in range(0, in_features, batch_size):
            stop = min(start + batch_size, in_features)
            chunk = stop - start
            basis = torch.zeros(chunk, in_features, device=device, dtype=dtype)
            basis[torch.arange(chunk, device=device), torch.arange(start, stop, device=device)] = 1
            outputs = source_layer(basis.view(1, chunk, in_features)).view(chunk, out_features).transpose(0, 1)
            main_weight[:, start:stop] = outputs.to(device="cpu", dtype=dtype)
        if source_layer.bias is not None:
            source_layer.bias.copy_(saved_bias)
        source_layer.proj_down.copy_(saved_proj_down)
        source_layer.proj_up.copy_(saved_proj_up)
    return main_weight.contiguous()


def build_dense_cache(
    checkpoint_path: Path,
    cache_path: Path,
    device: str,
    torch_dtype: torch.dtype = torch.bfloat16,
) -> dict[str, dict[str, torch.Tensor]]:
    from nunchaku import NunchakuFlux2Transformer2DModel
    from nunchaku.models.linear import SVDQW4A4Linear

    print(f"Building W4A4 emulation cache at {cache_path} from {checkpoint_path}")
    source_transformer = NunchakuFlux2Transformer2DModel.from_pretrained(
        str(checkpoint_path), device=device, torch_dtype=torch_dtype
    )
    source_transformer.eval()

    dense_state: dict[str, torch.Tensor] = {}
    output_cache: dict[str, dict[str, torch.Tensor]] = {}
    quantized_modules = {
        name: module for name, module in source_transformer.named_modules() if isinstance(module, SVDQW4A4Linear)
    }
    for index, (name, module) in enumerate(quantized_modules.items(), start=1):
        print(f"[{index}/{len(quantized_modules)}] exporting {name}")
        if not torch.all(module.smooth_factor == 1):
            raise ValueError(f"Non-trivial smooth_factor is not supported in the reference path: {name}")
        main_weight = build_main_weight_from_quantized_layer(module, device=device, dtype=torch_dtype)
        main_qweight, main_wscale = PureTorchSVDQInt4LinearBF16._quantize_weight_to_int4(module, main_weight)
        proj_down_dense = unpack_lowrank_weight(module.proj_down.detach().cpu(), down=True).transpose(0, 1).contiguous()
        proj_up_dense = unpack_lowrank_weight(module.proj_up.detach().cpu(), down=False).contiguous()
        dense_state[f"{name}.main_qweight"] = main_qweight
        dense_state[f"{name}.main_wscale"] = main_wscale
        dense_state[f"{name}.proj_down_dense"] = proj_down_dense
        dense_state[f"{name}.proj_up_dense"] = proj_up_dense
        output_cache[name] = {
            "main_qweight": main_qweight,
            "main_wscale": main_wscale,
            "proj_down_dense": proj_down_dense,
            "proj_up_dense": proj_up_dense,
        }
        torch.cuda.empty_cache()

    save_file(dense_state, str(cache_path), metadata={"format": "nunchaku-w4a4-emulation"})
    return output_cache


def attach_dense_weights(
    transformer: nn.Module, dense_cache: dict[str, dict[str, torch.Tensor]], torch_dtype: torch.dtype
) -> None:
    for name, module in iter_quantized_linears(transformer).items():
        cached = dense_cache.get(name)
        if cached is None:
            raise ValueError(f"Missing dense weight for {name}")
        if "main_qweight" not in cached or "main_wscale" not in cached:
            if "main_weight" not in cached:
                raise ValueError(f"Missing quantized or dense main weight for {name}")
            main_qweight, main_wscale = module._quantize_weight_to_int4(cached["main_weight"].to(dtype=torch_dtype))
        else:
            main_qweight = cached["main_qweight"]
            main_wscale = cached["main_wscale"]
        module.set_reference_weights(
            main_qweight=main_qweight.to(dtype=torch.int8),
            main_wscale=main_wscale.to(dtype=torch.float32),
            proj_down_dense=cached["proj_down_dense"].to(dtype=torch_dtype),
            proj_up_dense=cached["proj_up_dense"].to(dtype=torch_dtype),
        )
        module.drop_packed_storage()


def main(args: argparse.Namespace) -> None:
    PureTorchSVDQInt4LinearBF16.use_w4a8_main_matmul = args.use_w4a8_main_matmul
    PureTorchSVDQInt4LinearBF16.w4a8_main_matmul_out_chunk = args.w4a8_main_matmul_out_chunk

    model_source = args.local_model_path if args.local_model_path.exists() else args.model_name
    dense_cache = load_dense_cache(args.dense_cache_path) if args.dense_cache_path.exists() else build_dense_cache(
        checkpoint_path=args.checkpoint_path,
        cache_path=args.dense_cache_path,
        device=args.device,
        torch_dtype=args.dtype,
    )
    transformer = PureTorchFlux2Transformer2DModel.from_reference_checkpoint(
        checkpoint_path=args.checkpoint_path,
        torch_dtype=args.dtype,
    )
    attach_dense_weights(transformer, dense_cache, torch_dtype=args.dtype)
    pipeline = Flux2KleinPipeline.from_pretrained(
        model_source,
        transformer=transformer,
        torch_dtype=args.dtype,
        local_files_only=args.local_files_only,
    )
    pipeline.enable_model_cpu_offload()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    print(
        "Offloaded pipeline idle CUDA memory:",
        f"allocated={format_gib(torch.cuda.memory_allocated())}",
        f"reserved={format_gib(torch.cuda.memory_reserved())}",
    )

    install_module_profiler("text_encoder", pipeline.text_encoder)
    install_module_profiler("transformer", pipeline.transformer)
    finish_vae_stats = install_module_profiler("vae", pipeline.vae)

    orig_vae_decode = pipeline.vae.decode

    def profiled_vae_decode(*args, **kwargs):
        result = orig_vae_decode(*args, **kwargs)
        torch.cuda.synchronize()
        finish_vae_stats()
        return result

    pipeline.vae.decode = profiled_vae_decode

    pipeline_kwargs = {
        "height": args.height,
        "width": args.width,
        "guidance_scale": args.guidance_scale,
        "num_inference_steps": args.num_inference_steps,
        "generator": torch.Generator(device=args.device).manual_seed(args.seed),
    }
    if args.use_dummy_prompt_embeds:
        pipeline_kwargs["prompt"] = None
        pipeline_kwargs["prompt_embeds"] = torch.zeros(
            1,
            args.dummy_prompt_length,
            pipeline.transformer.config.joint_attention_dim,
            dtype=args.dtype,
            device=args.device,
        )
    else:
        pipeline_kwargs["prompt"] = args.prompt

    image = pipeline(**pipeline_kwargs).images[0]
    torch.cuda.synchronize()
    print_module_stats("text_encoder")
    print_module_stats("transformer")
    print_module_stats("vae")
    generation_peak_allocated = max(stats["max_peak_allocated"] for stats in module_stats.values())
    generation_peak_reserved = max(stats["max_peak_reserved"] for stats in module_stats.values())
    print(
        "Generation peak CUDA memory:",
        f"peak={format_gib(generation_peak_allocated)}",
        f"reserved={format_gib(generation_peak_reserved)}",
        f"final_allocated={format_gib(torch.cuda.memory_allocated())}",
    )
    image.save(args.output_path)


if __name__ == "__main__":
    main(parse_args())
