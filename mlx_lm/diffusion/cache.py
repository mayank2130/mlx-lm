from dataclasses import dataclass
from typing import Any, Optional

import mlx.core as mx


@dataclass(frozen=True)
class ExperimentalCacheConfig:
    strategy: str = "none"
    future_blocks: int = 0
    past_blocks: int = 0


@dataclass
class CachedBlockContext:
    context_start: int
    context_end: int
    prefix_states: Optional[list[mx.array]]
    combined_prefix_states: Optional[list[mx.array]]


@dataclass
class DualCacheState:
    full_past_key_values: Optional[list[tuple[mx.array, mx.array]]] = None
    block_past_key_values: Optional[list[tuple[mx.array, mx.array]]] = None


def _slice_cache_rows(
    cache_entries: Optional[list[tuple[mx.array, mx.array]]],
    active_indices: Optional[list[int]],
) -> Optional[list[tuple[mx.array, mx.array]]]:
    if cache_entries is None or active_indices is None:
        return cache_entries
    return [
        (keys[active_indices], values[active_indices])
        for keys, values in cache_entries
    ]


def _merge_cache_rows(
    base_cache: Optional[list[tuple[mx.array, mx.array]]],
    partial_cache: Optional[list[tuple[mx.array, mx.array]]],
    active_indices: Optional[list[int]],
) -> Optional[list[tuple[mx.array, mx.array]]]:
    if partial_cache is None:
        return base_cache
    if active_indices is None or base_cache is None:
        return partial_cache
    merged: list[tuple[mx.array, mx.array]] = []
    for (base_k, base_v), (partial_k, partial_v) in zip(base_cache, partial_cache):
        base_k[active_indices] = partial_k
        base_v[active_indices] = partial_v
        merged.append((base_k, base_v))
    return merged


class DualCacheManager:
    """
    Small Fast-dLLM cache orchestrator.

    It owns:
    - full-sequence cache warmup
    - active-block cache warmup
    - per-batch cache row slicing/merging for dynamic batching
    - replace masks for full-sequence and block-local cache updates
    """

    def __init__(self, runtime):
        self.runtime = runtime

    def replace_mask(
        self,
        *,
        batch_size: int,
        total_length: int,
        start: int,
        end: int,
    ) -> mx.array:
        mask = mx.zeros((batch_size, total_length), dtype=mx.bool_)
        mask[:, start:end] = True
        return mask

    def warm_full_cache(
        self,
        *,
        tokens: mx.array,
        attention_mask: Optional[mx.array],
        active_indices: Optional[list[int]] = None,
    ) -> tuple[mx.array, Optional[list[tuple[mx.array, mx.array]]]]:
        active_tokens = tokens if active_indices is None else tokens[active_indices]
        active_attention = (
            attention_mask
            if attention_mask is None or active_indices is None
            else attention_mask[active_indices]
        )
        output = self.runtime.model(
            active_tokens,
            attention_mask=active_attention,
            use_cache=True,
        )
        return self.runtime._extract_logits(output), self.runtime._extract_cache(output)

    def warm_block_cache(
        self,
        *,
        tokens: mx.array,
        attention_mask: Optional[mx.array],
        full_cache: Optional[list[tuple[mx.array, mx.array]]],
        block_start: int,
        block_end: int,
        active_indices: Optional[list[int]] = None,
    ) -> tuple[mx.array, Optional[list[tuple[mx.array, mx.array]]]]:
        active_tokens = (
            tokens[:, block_start:block_end]
            if active_indices is None
            else tokens[active_indices, block_start:block_end]
        )
        active_attention = (
            attention_mask
            if attention_mask is None or active_indices is None
            else attention_mask[active_indices]
        )
        active_full_cache = _slice_cache_rows(full_cache, active_indices)
        replace_position = self.replace_mask(
            batch_size=active_tokens.shape[0],
            total_length=tokens.shape[1],
            start=block_start,
            end=block_end,
        )
        output = self.runtime.model(
            active_tokens,
            attention_mask=active_attention,
            past_key_values=active_full_cache,
            use_cache=True,
            use_block_cache=True,
            block_cache_range=(block_start, block_end),
            replace_position=replace_position,
            update_past_key_values=False,
        )
        return self.runtime._extract_logits(output), self.runtime._extract_block_cache(output)

    def refine_sub_block(
        self,
        *,
        tokens: mx.array,
        attention_mask: Optional[mx.array],
        full_cache: Optional[list[tuple[mx.array, mx.array]]],
        block_cache: Optional[list[tuple[mx.array, mx.array]]],
        block_start: int,
        block_end: int,
        sub_start: int,
        sub_end: int,
        active_indices: Optional[list[int]] = None,
    ) -> tuple[mx.array, Optional[list[tuple[mx.array, mx.array]]]]:
        active_tokens = (
            tokens[:, sub_start:sub_end]
            if active_indices is None
            else tokens[active_indices, sub_start:sub_end]
        )
        active_attention = (
            attention_mask
            if attention_mask is None or active_indices is None
            else attention_mask[active_indices]
        )
        active_full_cache = _slice_cache_rows(full_cache, active_indices)
        active_block_cache = _slice_cache_rows(block_cache, active_indices)
        replace_position = self.replace_mask(
            batch_size=active_tokens.shape[0],
            total_length=tokens.shape[1],
            start=sub_start,
            end=sub_end,
        )
        output = self.runtime.model(
            active_tokens,
            attention_mask=active_attention,
            past_key_values=active_full_cache,
            block_past_key_values=active_block_cache,
            use_cache=True,
            use_block_cache=True,
            block_cache_range=(block_start, block_end),
            replace_position=replace_position,
            update_past_key_values=False,
        )
        return self.runtime._extract_logits(output), self.runtime._extract_block_cache(output)


class ExperimentalContextCache:
    """
    Experimental, easy-to-delete context reuse layer for diffusion runtime modes.

    This is intentionally small and approximation-oriented:
    - `none`: no extra cached context behavior
    - `window`: keep a local future/past block window in the model forward while
      still reusing prefix states before the window start
    """

    def __init__(self, runtime, config: ExperimentalCacheConfig):
        self.runtime = runtime
        self.config = config

    def context_range(
        self,
        *,
        prompt_length: int,
        total_length: int,
        block_start: int,
        block_end: int,
        block_length: int,
    ) -> tuple[int, int]:
        if self.config.strategy == "none":
            return block_start, block_end

        if self.config.strategy == "window":
            context_start = max(
                prompt_length,
                block_start - self.config.past_blocks * block_length,
            )
            context_end = min(
                total_length,
                block_end + self.config.future_blocks * block_length,
            )
            return context_start, context_end

        raise ValueError(f"Unknown experimental cache strategy: {self.config.strategy}")

    def prepare(
        self,
        *,
        tokens: mx.array,
        attention_mask: Optional[mx.array],
        prompt_index: mx.array,
        mask_id: int,
        context_start: int,
        context_end: int,
        block_local: bool,
        cfg_scale: float,
    ) -> CachedBlockContext:
        if not block_local or context_start <= 0:
            return CachedBlockContext(
                context_start=context_start,
                context_end=context_end,
                prefix_states=None,
                combined_prefix_states=None,
            )

        prefix_states = self.runtime._prefill_prefix_states(
            tokens,
            attention_mask=attention_mask,
            block_start=context_start,
        )
        combined_prefix_states = None
        if cfg_scale > 0.0 and prefix_states is not None:
            unconditional_x = mx.where(
                prompt_index,
                mx.full(tokens.shape, mask_id, dtype=tokens.dtype),
                tokens,
            )
            unconditional_prefix_states = self.runtime._prefill_prefix_states(
                unconditional_x,
                attention_mask=attention_mask,
                block_start=context_start,
            )
            if unconditional_prefix_states is not None:
                combined_prefix_states = [
                    mx.concatenate([cond, uncond], axis=0)
                    for cond, uncond in zip(prefix_states, unconditional_prefix_states)
                ]

        return CachedBlockContext(
            context_start=context_start,
            context_end=context_end,
            prefix_states=prefix_states,
            combined_prefix_states=combined_prefix_states,
        )
