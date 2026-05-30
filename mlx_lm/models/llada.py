# Copyright © 2023-2024 Apple Inc.

from dataclasses import dataclass
from typing import Any, NamedTuple, Optional

import mlx.core as mx
import mlx.nn as nn

from .base import BaseModelArgs, scaled_dot_product_attention


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    d_model: int
    n_heads: int
    n_layers: int
    mlp_hidden_size: int
    vocab_size: int
    max_sequence_length: int
    rms_norm_eps: float
    rope: bool = True
    rope_theta: float = 10000.0
    n_kv_heads: Optional[int] = None
    embedding_size: Optional[int] = None
    include_bias: bool = False
    include_qkv_bias: bool = False
    weight_tying: bool = False
    eos_token_id: Optional[int] = None
    pad_token_id: Optional[int] = None
    mask_token_id: Optional[int] = None

    def __post_init__(self):
        if self.n_kv_heads is None:
            self.n_kv_heads = self.n_heads
        if self.embedding_size is None:
            self.embedding_size = self.vocab_size


def _make_padding_mask(
    attention_mask: Optional[mx.array], dtype: mx.Dtype
) -> Optional[mx.array]:
    if attention_mask is None:
        return None
    attention_mask = attention_mask.astype(dtype)
    return ((1.0 - attention_mask) * mx.finfo(dtype).min)[:, None, None, :]


def _make_block_padding_mask(
    attention_mask: Optional[mx.array], block_length: int, dtype: mx.Dtype
) -> Optional[mx.array]:
    if attention_mask is None:
        return None
    attention_mask = attention_mask.astype(dtype)
    return mx.broadcast_to(
        ((1.0 - attention_mask) * mx.finfo(dtype).min)[:, None, None, :],
        (attention_mask.shape[0], 1, block_length, attention_mask.shape[1]),
    )


class LLaDAOutput(NamedTuple):
    logits: mx.array
    attn_key_values: Optional[list[tuple[mx.array, mx.array]]]
    block_attn_key_values: Optional[list[tuple[mx.array, mx.array]]] = None
    hidden_states: Optional[tuple[mx.array, ...]] = None


def _replace_cache_slice(
    cached: mx.array,
    current: mx.array,
    replace_position: mx.array,
) -> mx.array:
    if replace_position.ndim != 2:
        raise ValueError(
            f"replace_position must have shape (batch, length), got {replace_position.shape}."
        )
    counts = mx.sum(replace_position.astype(mx.int32), axis=1)
    if bool(mx.all(counts == counts[0]).item()):
        replace_count = int(counts[0].item())
        if replace_count == 0:
            return cached
        if replace_count != current.shape[2]:
            raise ValueError(
                "replace_position selected length must match the provided block length: "
                f"{replace_count} != {current.shape[2]}."
            )
        starts = mx.argmax(replace_position.astype(mx.int32), axis=1)
        if bool(mx.all(starts == starts[0]).item()):
            start = int(starts[0].item())
            end = start + replace_count
            return mx.concatenate(
                [
                    cached[:, :, :start, :],
                    current,
                    cached[:, :, end:, :],
                ],
                axis=2,
            )

    updated = []
    for batch_idx in range(cached.shape[0]):
        replace_mask = replace_position[batch_idx].astype(mx.int32)
        replace_count = int(mx.sum(replace_mask).item())
        if replace_count == 0:
            updated.append(cached[batch_idx : batch_idx + 1])
            continue
        if replace_count != current.shape[2]:
            raise ValueError(
                "replace_position selected length must match the provided block length: "
                f"{replace_count} != {current.shape[2]}."
            )
        start = int(mx.argmax(replace_mask).item())
        end = start + replace_count
        updated.append(
            mx.concatenate(
                [
                    cached[batch_idx : batch_idx + 1, :, :start, :],
                    current[batch_idx : batch_idx + 1],
                    cached[batch_idx : batch_idx + 1, :, end:, :],
                ],
                axis=2,
            )
        )
    return mx.concatenate(updated, axis=0)


class LLaDABlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.n_kv_heads = args.n_kv_heads
        self.head_dim = args.d_model // args.n_heads
        self.scale = self.head_dim**-0.5

        qkv_bias = args.include_bias or args.include_qkv_bias
        self.attn_norm = nn.RMSNorm(args.d_model, eps=args.rms_norm_eps)
        self.ff_norm = nn.RMSNorm(args.d_model, eps=args.rms_norm_eps)

        self.q_proj = nn.Linear(args.d_model, args.n_heads * self.head_dim, bias=qkv_bias)
        self.k_proj = nn.Linear(
            args.d_model, args.n_kv_heads * self.head_dim, bias=qkv_bias
        )
        self.v_proj = nn.Linear(
            args.d_model, args.n_kv_heads * self.head_dim, bias=qkv_bias
        )
        self.attn_out = nn.Linear(args.d_model, args.d_model, bias=args.include_bias)

        self.ff_proj = nn.Linear(
            args.d_model, args.mlp_hidden_size, bias=args.include_bias
        )
        self.up_proj = nn.Linear(
            args.d_model, args.mlp_hidden_size, bias=args.include_bias
        )
        self.ff_out = nn.Linear(
            args.mlp_hidden_size, args.d_model, bias=args.include_bias
        )

        self.rope = nn.RoPE(self.head_dim, traditional=False, base=args.rope_theta)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
        *,
        use_cache: bool = False,
        use_block_cache: bool = False,
        replace_position: Optional[mx.array] = None,
        cache_offset: int = 0,
    ) -> mx.array | tuple[mx.array, tuple[mx.array, mx.array], tuple[mx.array, mx.array]]:
        layer_past = cache

        h = self.attn_norm(x)
        B, L, _ = h.shape
        q = self.q_proj(h).reshape(B, L, self.n_heads, -1).transpose(0, 2, 1, 3)
        k = self.k_proj(h).reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        v = self.v_proj(h).reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)

        q = self.rope(q, offset=cache_offset)
        k = self.rope(k, offset=cache_offset)
        block_cache = (k, v)

        if layer_past is not None:
            past_k, past_v = layer_past
            if replace_position is None:
                k = mx.concatenate([past_k, k], axis=2)
                v = mx.concatenate([past_v, v], axis=2)
            else:
                k = _replace_cache_slice(past_k, k, replace_position)
                v = _replace_cache_slice(past_v, v, replace_position)

        attn = scaled_dot_product_attention(
            q,
            k,
            v,
            cache=None,
            scale=self.scale,
            mask=mask,
        )
        attn = attn.transpose(0, 2, 1, 3).reshape(B, L, -1)
        x = x + self.attn_out(attn)

        h = self.ff_norm(x)
        mlp = nn.silu(self.ff_proj(h)) * self.up_proj(h)
        out = x + self.ff_out(mlp)
        if use_cache:
            if use_block_cache:
                return out, (k, v), block_cache
            return out, (k, v), block_cache
        return out

    def forward_block(
        self,
        x: mx.array,
        *,
        prefix: Optional[mx.array] = None,
        mask: Optional[mx.array] = None,
        prefix_length: int = 0,
    ) -> mx.array:
        h = self.attn_norm(x)
        B, L, _ = h.shape
        q = self.q_proj(h).reshape(B, L, self.n_heads, -1).transpose(0, 2, 1, 3)
        k = self.k_proj(h).reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        v = self.v_proj(h).reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)

        q = self.rope(q, offset=prefix_length)
        k = self.rope(k, offset=prefix_length)

        if prefix is not None and prefix.shape[1] > 0:
            prefix_h = self.attn_norm(prefix)
            _, prefix_len, _ = prefix_h.shape
            prefix_k = (
                self.k_proj(prefix_h)
                .reshape(B, prefix_len, self.n_kv_heads, -1)
                .transpose(0, 2, 1, 3)
            )
            prefix_v = (
                self.v_proj(prefix_h)
                .reshape(B, prefix_len, self.n_kv_heads, -1)
                .transpose(0, 2, 1, 3)
            )
            prefix_k = self.rope(prefix_k, offset=0)
            k = mx.concatenate([prefix_k, k], axis=2)
            v = mx.concatenate([prefix_v, v], axis=2)

        attn = scaled_dot_product_attention(
            q,
            k,
            v,
            cache=None,
            scale=self.scale,
            mask=mask,
        )
        attn = attn.transpose(0, 2, 1, 3).reshape(B, L, -1)
        x = x + self.attn_out(attn)

        h = self.ff_norm(x)
        mlp = nn.silu(self.ff_proj(h)) * self.up_proj(h)
        return x + self.ff_out(mlp)


class Transformer(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.wte = nn.Embedding(args.embedding_size, args.d_model)
        self.blocks = [LLaDABlock(args) for _ in range(args.n_layers)]
        self.ln_f = nn.RMSNorm(args.d_model, eps=args.rms_norm_eps)
        if not args.weight_tying:
            self.ff_out = nn.Linear(
                args.d_model,
                args.embedding_size,
                bias=args.include_bias,
            )

    def __call__(
        self,
        inputs: mx.array,
        *,
        attention_mask: Optional[mx.array] = None,
        input_embeddings: Optional[mx.array] = None,
        cache: Optional[Any] = None,
        use_cache: bool = False,
        use_block_cache: bool = False,
        block_past_key_values: Optional[Any] = None,
        block_cache_range: Optional[tuple[int, int]] = None,
        update_past_key_values: bool = True,
        replace_position: Optional[mx.array] = None,
        output_hidden_states: bool = False,
        last_logits_only: bool = False,
        weight_tying: bool = False,
        logits_range: Optional[tuple[int, int]] = None,
    ) -> mx.array | LLaDAOutput:
        past_key_values = cache
        if past_key_values is not None and len(past_key_values) != len(self.blocks):
            raise ValueError(
                "past_key_values must contain one cache entry per transformer block."
            )
        if block_past_key_values is not None and len(block_past_key_values) != len(self.blocks):
            raise ValueError(
                "block_past_key_values must contain one cache entry per transformer block."
            )

        h = self.wte(inputs) if input_embeddings is None else input_embeddings
        if past_key_values is not None or block_past_key_values is not None:
            mask = _make_block_padding_mask(attention_mask, h.shape[1], h.dtype)
        else:
            mask = _make_padding_mask(attention_mask, h.dtype)

        hidden_states = [h] if output_hidden_states else None
        attn_key_values = [] if use_cache else None
        block_attn_key_values = [] if use_cache and use_block_cache else None

        if replace_position is not None:
            non_empty_starts = []
            for batch_idx in range(replace_position.shape[0]):
                replace_mask = replace_position[batch_idx].astype(mx.int32)
                if int(mx.sum(replace_mask).item()) == 0:
                    continue
                non_empty_starts.append(int(mx.argmax(replace_mask).item()))
            cache_offset = min(non_empty_starts) if non_empty_starts else 0
        elif past_key_values is not None:
            cache_offset = int(past_key_values[0][0].shape[2])
        else:
            cache_offset = 0

        for block_idx, block in enumerate(self.blocks):
            layer_past = None if past_key_values is None else past_key_values[block_idx]
            layer_block_past = (
                None if block_past_key_values is None else block_past_key_values[block_idx]
            )
            if (
                use_block_cache
                and layer_block_past is not None
                and layer_past is not None
                and block_cache_range is not None
            ):
                block_start, block_end = block_cache_range
                replace_block = mx.zeros(
                    (h.shape[0], layer_past[0].shape[2]),
                    dtype=mx.bool_,
                )
                replace_block[:, block_start:block_end] = True
                layer_past = (
                    _replace_cache_slice(layer_past[0], layer_block_past[0], replace_block),
                    _replace_cache_slice(layer_past[1], layer_block_past[1], replace_block),
                )
            elif use_block_cache and layer_block_past is not None and layer_past is None:
                layer_past = layer_block_past
            block_out = block(
                h,
                mask=mask,
                cache=layer_past,
                use_cache=use_cache,
                use_block_cache=use_block_cache,
                replace_position=replace_position,
                cache_offset=cache_offset,
            )
            if use_cache:
                h, layer_cache, layer_block_cache = block_out
                attn_key_values.append(
                    layer_cache if update_past_key_values or past_key_values is None else past_key_values[block_idx]
                )
                if block_attn_key_values is not None:
                    if layer_block_past is not None and block_cache_range is not None:
                        block_start, block_end = block_cache_range
                        block_replace = mx.zeros(
                            (h.shape[0], block_end - block_start),
                            dtype=mx.bool_,
                        )
                        if replace_position is None:
                            block_replace[:] = True
                        else:
                            block_replace = replace_position[:, block_start:block_end]
                        block_attn_key_values.append(
                            (
                                _replace_cache_slice(
                                    layer_block_past[0], layer_block_cache[0], block_replace
                                ),
                                _replace_cache_slice(
                                    layer_block_past[1], layer_block_cache[1], block_replace
                                ),
                            )
                        )
                    else:
                        block_attn_key_values.append(layer_block_cache)
            else:
                h = block_out
            if hidden_states is not None:
                hidden_states.append(h)

        h = self.ln_f(h)
        if last_logits_only:
            h = h[:, -1:, :]
        elif logits_range is not None:
            start, end = logits_range
            if not (0 <= start <= end <= h.shape[1]):
                raise ValueError(
                    f"Invalid logits_range {logits_range} for sequence length {h.shape[1]}."
                )
            # Materialize the slice before the LM head so block-local projection
            # does not pay for a strided view in the large vocab matmul.
            h = mx.contiguous(h[:, start:end, :])

        if weight_tying:
            logits = self.wte.as_linear(h)
        else:
            logits = self.ff_out(h)
        if use_cache or output_hidden_states:
            return LLaDAOutput(
                logits=logits,
                attn_key_values=attn_key_values,
                block_attn_key_values=block_attn_key_values,
                hidden_states=tuple(hidden_states) if hidden_states is not None else None,
            )
        return logits

    def prefill_prefix_states(
        self,
        inputs: mx.array,
        *,
        prefix_length: int,
        attention_mask: Optional[mx.array] = None,
        input_embeddings: Optional[mx.array] = None,
    ) -> list[mx.array]:
        if prefix_length <= 0:
            batch_size = inputs.shape[0]
            dtype = (
                input_embeddings.dtype
                if input_embeddings is not None
                else self.wte.weight.dtype
            )
            empty = mx.zeros((batch_size, 0, self.wte.weight.shape[1]), dtype=dtype)
            return [empty] * (len(self.blocks) + 1)

        h = self.wte(inputs[:, :prefix_length]) if input_embeddings is None else input_embeddings[:, :prefix_length, :]
        prefix_mask = (
            _make_padding_mask(attention_mask[:, :prefix_length], h.dtype)
            if attention_mask is not None
            else None
        )
        states = [h]
        for block in self.blocks:
            h = block(h, mask=prefix_mask)
            states.append(h)
        return states

    def forward_block(
        self,
        inputs: mx.array,
        *,
        block_range: tuple[int, int],
        attention_mask: Optional[mx.array] = None,
        input_embeddings: Optional[mx.array] = None,
        prefix_states: Optional[list[mx.array]] = None,
        weight_tying: bool = False,
    ) -> mx.array:
        start, end = block_range
        if not (0 <= start <= end <= inputs.shape[1]):
            raise ValueError(
                f"Invalid block_range {block_range} for sequence length {inputs.shape[1]}."
            )
        if start == 0:
            return self(
                inputs,
                attention_mask=attention_mask,
                input_embeddings=input_embeddings,
                weight_tying=weight_tying,
                logits_range=block_range,
            )

        h = (
            self.wte(inputs[:, start:end])
            if input_embeddings is None
            else input_embeddings[:, start:end, :]
        )
        if prefix_states is None:
            prefix_states = self.prefill_prefix_states(
                inputs,
                prefix_length=start,
                attention_mask=attention_mask,
                input_embeddings=input_embeddings,
            )
        if len(prefix_states) != len(self.blocks) + 1:
            raise ValueError(
                "prefix_states must contain one tensor for the embedding input and one per layer."
            )

        kv_attention_mask = attention_mask[:, :end] if attention_mask is not None else None
        block_mask = _make_block_padding_mask(kv_attention_mask, h.shape[1], h.dtype)

        for layer_idx, block in enumerate(self.blocks):
            h = block.forward_block(
                h,
                prefix=prefix_states[layer_idx],
                mask=block_mask,
                prefix_length=start,
            )

        h = self.ln_f(h)
        if weight_tying:
            return self.wte.as_linear(h)
        return self.ff_out(h)


class Model(nn.Module):
    supports_diffusion = True
    supports_logits_range = True
    supports_block_local = True
    supports_context_window_cache = True
    supports_dual_cache = True
    supports_block_cache = True

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.model_type = args.model_type
        self.args = args
        self.transformer = Transformer(args)

    def __call__(
        self,
        input_ids: mx.array,
        cache: Optional[Any] = None,
        past_key_values: Optional[Any] = None,
        use_cache: bool = False,
        use_block_cache: bool = False,
        block_past_key_values: Optional[Any] = None,
        block_cache_range: Optional[tuple[int, int]] = None,
        update_past_key_values: bool = True,
        replace_position: Optional[mx.array] = None,
        attention_mask: Optional[mx.array] = None,
        input_embeddings: Optional[mx.array] = None,
        last_logits_only: bool = False,
        logits_range: Optional[tuple[int, int]] = None,
        block_range: Optional[tuple[int, int]] = None,
        prefix_states: Optional[list[mx.array]] = None,
        output_hidden_states: bool = False,
    ) -> mx.array | LLaDAOutput:
        if past_key_values is None and cache is not None:
            past_key_values = cache
        if block_range is not None:
            return self.transformer.forward_block(
                input_ids,
                block_range=block_range,
                attention_mask=attention_mask,
                input_embeddings=input_embeddings,
                prefix_states=prefix_states,
                weight_tying=self.args.weight_tying,
            )
        return self.transformer(
            input_ids,
            attention_mask=attention_mask,
            input_embeddings=input_embeddings,
            cache=past_key_values,
            use_cache=use_cache,
            use_block_cache=use_block_cache,
            block_past_key_values=block_past_key_values,
            block_cache_range=block_cache_range,
            update_past_key_values=update_past_key_values,
            replace_position=replace_position,
            output_hidden_states=output_hidden_states,
            last_logits_only=last_logits_only,
            logits_range=logits_range,
            weight_tying=self.args.weight_tying,
        )

    def sanitize(self, weights):
        sanitized = {}
        for key, value in weights.items():
            if key.startswith("model."):
                key = key[len("model.") :]
            sanitized[key] = value
        if self.args.weight_tying:
            sanitized.pop("transformer.ff_out.weight", None)
        return sanitized

    @property
    def layers(self):
        return self.transformer.blocks
