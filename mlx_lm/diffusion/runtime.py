from dataclasses import dataclass
from functools import partial
import time
from typing import Any, Callable, List, Optional, Sequence, Union

import mlx.core as mx
import mlx.nn as nn
from transformers import PreTrainedTokenizer

from ..tokenizer_utils import TokenizerWrapper
from .cache import (
    DualCacheManager,
    ExperimentalCacheConfig,
    ExperimentalContextCache,
    _merge_cache_rows,
    _slice_cache_rows,
)
from .decoder import (
    BaseDiffusionDecoder,
    DiffusionDecoderConfig,
    ThresholdDecoder,
    TopKConfidenceDecoder,
    add_gumbel_noise,
    gather_selected_probs,
)
from .scheduler import BaseDiffusionScheduler
from .state import DiffusionState
from .step import (
    BlockLocalDiffusionStep,
    DiffusionStepContext,
    FullSequenceDiffusionStep,
)


@dataclass
class DiffusionGenerationResult:
    text: Union[str, List[str]]
    token_ids: mx.array
    prompt_length: int
    stats: Optional[dict[str, Any]] = None


@dataclass(frozen=True)
class DiffusionRuntimeConfig:
    mode: str = "faithful_llada"
    use_logits_range: bool = True
    compile_steps: bool = True
    block_local: bool = False
    dual_cache: bool = False
    dynamic_batching: bool = True
    cfg_scale: float = 0.0
    cache_strategy: str = "none"
    cache_future_blocks: int = 0
    cache_past_blocks: int = 0
    use_block_cache: bool = False
    sub_block_size: Optional[int] = None
    quality_gating: bool = False
    quality_min_step_tps: float = 0.0


def _extract_logits(output):
    return output.logits if hasattr(output, "logits") else output


def _extract_cache(output):
    if hasattr(output, "attn_key_values"):
        return output.attn_key_values
    if hasattr(output, "past_key_values"):
        return output.past_key_values
    return None


def _extract_block_cache(output):
    if hasattr(output, "block_attn_key_values"):
        return output.block_attn_key_values
    return None


def _normalize_prompt(
    tokenizer: TokenizerWrapper,
    prompt: Union[str, Sequence[int], mx.array],
) -> mx.array:
    if isinstance(prompt, mx.array):
        return prompt
    if isinstance(prompt, str):
        add_special_tokens = tokenizer.bos_token is None or not prompt.startswith(
            tokenizer.bos_token
        )
        return mx.array(tokenizer.encode(prompt, add_special_tokens=add_special_tokens))
    return mx.array(prompt)


def _is_batched_prompt(prompt) -> bool:
    if isinstance(prompt, mx.array):
        return prompt.ndim == 2
    if isinstance(prompt, str):
        return False
    if isinstance(prompt, Sequence) and len(prompt) > 0:
        first = prompt[0]
        if isinstance(first, str):
            return True
        if isinstance(first, mx.array):
            return first.ndim > 0
        return isinstance(first, Sequence) and not isinstance(first, (str, bytes, int))
    return False


def pad_prompt_batch(
    tokenizer: TokenizerWrapper,
    prompts: Sequence[Union[str, Sequence[int], mx.array]],
    *,
    attention_mask: Optional[Union[Sequence[Sequence[int]], mx.array]] = None,
) -> tuple[mx.array, mx.array, list[int]]:
    prompt_tokens = [_normalize_prompt(tokenizer, prompt) for prompt in prompts]
    prompt_lengths = [int(tokens.shape[0]) for tokens in prompt_tokens]
    max_prompt_len = max(prompt_lengths)
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0

    batch = mx.full(
        (len(prompt_tokens), max_prompt_len),
        pad_id,
        dtype=prompt_tokens[0].dtype,
    )
    batch_attention = mx.zeros((len(prompt_tokens), max_prompt_len), dtype=mx.int32)
    for batch_idx, tokens in enumerate(prompt_tokens):
        prompt_len = tokens.shape[0]
        batch[batch_idx, :prompt_len] = tokens
        batch_attention[batch_idx, :prompt_len] = 1

    if attention_mask is not None:
        batch_attention = (
            attention_mask
            if isinstance(attention_mask, mx.array)
            else mx.array(attention_mask)
        ).astype(mx.int32)
        if batch_attention.shape != batch.shape:
            raise ValueError(
                "attention_mask must match the padded prompt batch shape "
                f"{batch.shape}, got {batch_attention.shape}."
            )

    return batch, batch_attention, prompt_lengths


def _full_like(x: mx.array, value: int) -> mx.array:
    return mx.full(x.shape, value, dtype=x.dtype)


class DiffusionRuntime:
    def __init__(
        self,
        model: nn.Module,
        *,
        scheduler: BaseDiffusionScheduler,
        decoder: BaseDiffusionDecoder,
        config: Optional[DiffusionRuntimeConfig] = None,
    ):
        self.model = model
        self.scheduler = scheduler
        self.decoder = decoder
        self.config = config or DiffusionRuntimeConfig()
        self._compiled_step_cache = {}
        self._full_sequence_step = FullSequenceDiffusionStep(self)
        self._block_local_step = BlockLocalDiffusionStep(self)
        cache_strategy = self.config.cache_strategy
        if (
            cache_strategy != "none"
            and not getattr(self.model, "supports_context_window_cache", False)
        ):
            cache_strategy = "none"
        self.context_cache = ExperimentalContextCache(
            self,
            ExperimentalCacheConfig(
                strategy=cache_strategy,
                future_blocks=self.config.cache_future_blocks,
                past_blocks=self.config.cache_past_blocks,
            ),
        )
        self.dual_cache_manager = DualCacheManager(self)
        self._extract_logits = _extract_logits
        self._extract_cache = _extract_cache
        self._extract_block_cache = _extract_block_cache
        self._compiled_fast_transfer = None

    def _compiled_fast_threshold_transfer_kernel(self):
        if self._compiled_fast_transfer is not None:
            return self._compiled_fast_transfer
        decoder_cfg = self.decoder.config
        remasking = getattr(self.decoder, "remasking", "low_confidence")

        @partial(mx.compile, inputs=mx.random.state, outputs=mx.random.state)
        def _step(logits: mx.array, tokens: mx.array, mask_index: mx.array):
            local_logits = logits
            if decoder_cfg.logits_eos_inf and decoder_cfg.eos_token_id is not None:
                local_logits[:, :, decoder_cfg.eos_token_id] = -mx.inf
            logits_with_noise = add_gumbel_noise(
                local_logits, temperature=decoder_cfg.temperature
            )
            x0 = mx.argmax(logits_with_noise, axis=-1)
            if remasking == "low_confidence":
                x0_p = gather_selected_probs(local_logits, x0)
            else:
                x0_p = mx.random.uniform(shape=x0.shape, dtype=local_logits.dtype)
            x0 = mx.where(mask_index, x0, tokens)
            confidence = mx.where(mask_index, x0_p, -mx.inf)
            threshold_index = mask_index & (confidence >= decoder_cfg.threshold)
            max_conf_indices = mx.argmax(confidence, axis=1, keepdims=True)
            force_mask = mx.put_along_axis(
                mx.zeros(confidence.shape, dtype=mx.bool_),
                max_conf_indices,
                mx.ones(max_conf_indices.shape, dtype=mx.bool_),
                axis=-1,
            )
            transfer_index = (threshold_index | force_mask) & mask_index
            return x0, transfer_index

        self._compiled_fast_transfer = _step
        return _step

    def _can_compile_full_step(
        self,
        *,
        active_prefix_states: Optional[list[mx.array]],
        active_combined_prefix_states: Optional[list[mx.array]],
    ) -> bool:
        return (
            self.config.compile_steps
            and self.config.cfg_scale == 0.0
            and active_prefix_states is None
            and active_combined_prefix_states is None
            and isinstance(self.decoder, (TopKConfidenceDecoder, ThresholdDecoder))
        )

    def _compiled_step_key(
        self,
        *,
        block_start: int,
        block_end: int,
        max_k: int,
        has_attention_mask: bool,
    ):
        return (
            self.config.mode,
            block_start,
            block_end,
            max_k,
            has_attention_mask,
            self.config.use_logits_range,
            type(self.decoder).__name__,
            getattr(self.decoder, "remasking", None),
            self.decoder.config.threshold,
            self.decoder.config.temperature,
            self.decoder.config.logits_eos_inf,
            self.decoder.config.confidence_eos_eot_inf,
            self.decoder.config.eos_token_id,
            self.decoder.config.eot_token_id,
            self.decoder.config.mask_id,
        )

    def _compiled_full_step(
        self,
        *,
        block_start: int,
        block_end: int,
        max_k: int,
        has_attention_mask: bool,
    ):
        key = self._compiled_step_key(
            block_start=block_start,
            block_end=block_end,
            max_k=max_k,
            has_attention_mask=has_attention_mask,
        )
        if key in self._compiled_step_cache:
            return self._compiled_step_cache[key]

        model = self.model
        decoder = self.decoder
        runtime_cfg = self.config
        decoder_cfg = self.decoder.config
        remasking = getattr(self.decoder, "remasking", None)
        use_logits_range = runtime_cfg.use_logits_range
        is_topk = isinstance(decoder, TopKConfidenceDecoder)

        def _decode_block(block_tokens: mx.array, logits: mx.array, step_targets: mx.array):
            block_mask_index = block_tokens == decoder_cfg.mask_id
            if decoder_cfg.logits_eos_inf and decoder_cfg.eos_token_id is not None:
                logits[:, :, decoder_cfg.eos_token_id] = -mx.inf
            logits_with_noise = add_gumbel_noise(
                logits, temperature=decoder_cfg.temperature
            )
            x0_block = mx.argmax(logits_with_noise, axis=-1)
            if decoder_cfg.confidence_eos_eot_inf:
                if decoder_cfg.eos_token_id is not None:
                    logits_with_noise[:, :, decoder_cfg.eos_token_id] = -mx.inf
                if decoder_cfg.eot_token_id is not None:
                    logits[:, :, decoder_cfg.eot_token_id] = -mx.inf
            x0_p = gather_selected_probs(logits, x0_block)
            x0_block = mx.where(block_mask_index, x0_block, block_tokens)
            confidence = mx.where(block_mask_index, x0_p, -mx.inf)

            if is_topk:
                if remasking == "random":
                    confidence = mx.where(
                        block_mask_index,
                        mx.random.uniform(shape=x0_block.shape, dtype=logits.dtype),
                        -mx.inf,
                    )
                transfer_index = mx.zeros(confidence.shape, dtype=mx.bool_)
                if max_k > 0:
                    candidate_index = mx.argpartition(
                        -confidence, kth=max_k - 1, axis=-1
                    )[:, :max_k]
                    selection_order = mx.arange(max_k, dtype=step_targets.dtype)[None, :]
                    selection_mask = selection_order < step_targets[:, None]
                    transfer_index = mx.put_along_axis(
                        transfer_index,
                        candidate_index,
                        selection_mask,
                        axis=-1,
                    )
            else:
                threshold_index = confidence >= decoder_cfg.threshold
                fallback_targets = mx.maximum(
                    step_targets, mx.array(1, dtype=step_targets.dtype)
                )
                candidate_index = mx.argpartition(
                    -confidence, kth=max_k - 1, axis=-1
                )[:, :max_k]
                selection_order = mx.arange(max_k, dtype=fallback_targets.dtype)[None, :]
                fallback_mask = selection_order < fallback_targets[:, None]
                fallback_index = mx.put_along_axis(
                    mx.zeros(confidence.shape, dtype=mx.bool_),
                    candidate_index,
                    fallback_mask,
                    axis=-1,
                )
                transfer_index = mx.where(
                    mx.any(threshold_index, axis=1, keepdims=True),
                    threshold_index,
                    fallback_index,
                )

            updated_block = mx.where(transfer_index, x0_block, block_tokens)
            return updated_block

        if has_attention_mask:

            @partial(mx.compile, inputs=mx.random.state, outputs=mx.random.state)
            def _step(tokens: mx.array, attention_mask: mx.array, step_targets: mx.array):
                logits = _extract_logits(
                    model(
                        tokens,
                        attention_mask=attention_mask,
                        logits_range=(block_start, block_end) if use_logits_range else None,
                    )
                )
                if logits.shape[1] != block_end - block_start:
                    logits = logits[:, block_start:block_end, :]
                block_tokens = tokens[:, block_start:block_end]
                updated_block = _decode_block(block_tokens, logits, step_targets)
                updated_tokens = mx.concatenate(
                    [
                        tokens[:, :block_start],
                        updated_block,
                        tokens[:, block_end:],
                    ],
                    axis=1,
                )
                masks_after = mx.sum(
                    (updated_block == decoder_cfg.mask_id).astype(mx.int32)
                )
                return updated_tokens, updated_block, masks_after

        else:

            @partial(mx.compile, inputs=mx.random.state, outputs=mx.random.state)
            def _step(tokens: mx.array, step_targets: mx.array):
                logits = _extract_logits(
                    model(
                        tokens,
                        logits_range=(block_start, block_end) if use_logits_range else None,
                    )
                )
                if logits.shape[1] != block_end - block_start:
                    logits = logits[:, block_start:block_end, :]
                block_tokens = tokens[:, block_start:block_end]
                updated_block = _decode_block(block_tokens, logits, step_targets)
                updated_tokens = mx.concatenate(
                    [
                        tokens[:, :block_start],
                        updated_block,
                        tokens[:, block_end:],
                    ],
                    axis=1,
                )
                masks_after = mx.sum(
                    (updated_block == decoder_cfg.mask_id).astype(mx.int32)
                )
                return updated_tokens, updated_block, masks_after

        self._compiled_step_cache[key] = _step
        return _step

    def _model_logits(
        self,
        inputs: mx.array,
        *,
        attention_mask: Optional[mx.array] = None,
        logits_range: Optional[tuple[int, int]] = None,
        block_range: Optional[tuple[int, int]] = None,
        prefix_states: Optional[list[mx.array]] = None,
    ):
        if (
            block_range is not None
            and self.config.block_local
            and getattr(self.model, "supports_block_local", False)
        ):
            return _extract_logits(
                self.model(
                    inputs,
                    attention_mask=attention_mask,
                    block_range=block_range,
                    prefix_states=prefix_states,
                )
            )
        kwargs = {"attention_mask": attention_mask}
        if logits_range is not None and getattr(self.model, "supports_logits_range", False):
            kwargs["logits_range"] = logits_range
        logits = _extract_logits(self.model(inputs, **kwargs))
        if "logits_range" in kwargs or logits_range is None:
            return logits
        start, end = logits_range
        return logits[:, start:end, :]

    def _prefill_prefix_states(
        self,
        x: mx.array,
        *,
        attention_mask: Optional[mx.array],
        block_start: int,
    ) -> Optional[list[mx.array]]:
        if (
            not self.config.block_local
            or block_start <= 0
            or not getattr(self.model, "supports_block_local", False)
            or not hasattr(self.model, "transformer")
            or not hasattr(self.model.transformer, "prefill_prefix_states")
        ):
            return None
        return self.model.transformer.prefill_prefix_states(
            x,
            prefix_length=block_start,
            attention_mask=attention_mask,
        )

    def _slice_prefix_states(
        self,
        prefix_states: Optional[list[mx.array]],
        active_indices: Optional[Sequence[int]],
    ) -> Optional[list[mx.array]]:
        if prefix_states is None or active_indices is None:
            return prefix_states
        return [state[active_indices] for state in prefix_states]

    def _active_indices(self, block_tokens: mx.array) -> Optional[list[int]]:
        if not self.config.dynamic_batching or block_tokens.shape[0] == 1:
            return None
        active_mask = mx.any(block_tokens == self.decoder.config.mask_id, axis=1)
        if not bool(mx.any(active_mask).item()):
            return None
        return [i for i, active in enumerate(active_mask.tolist()) if active]

    def _record_step_stats(
        self,
        state: DiffusionState,
        *,
        block_idx: int,
        block_start: int,
        block_end: int,
        step_time: float,
        total_masks_before: int,
        total_masks_after: int,
        logits_span: int,
        context_span: int,
        active_rows: int,
        model_forwards: int = 1,
        compiled: bool = False,
    ) -> tuple[int, float]:
        tokens_committed = max(total_masks_before - total_masks_after, 0)
        step_tps = float(tokens_committed) / step_time if step_time > 0 else 0.0
        state.stats["steps_completed"] += 1
        state.stats["step_times"].append(step_time)
        state.stats["elapsed_time"] += step_time
        state.stats["total_tokens_committed"] += tokens_committed
        state.stats["masks_remaining"].append(total_masks_after)
        state.stats["tokens_committed"].append(tokens_committed)
        state.stats["step_tps"].append(step_tps)
        state.stats["model_forwards"] += model_forwards
        state.stats["compiled_step_hits"] += int(compiled)
        state.stats["compiled_step_misses"] += int(not compiled)
        state.stats["active_rows_history"].append(active_rows)
        state.stats["logits_span_history"].append(logits_span)
        state.stats["context_span_history"].append(context_span)
        state.stats["block_iterations"][block_idx] = (
            state.stats["block_iterations"].get(block_idx, 0) + 1
        )
        return tokens_committed, step_tps

    def _sub_block_ranges(self, block_start: int, block_end: int) -> list[tuple[int, int]]:
        if not self.config.use_block_cache or not self.config.sub_block_size:
            return [(block_start, block_end)]
        sub_block_size = max(1, min(self.config.sub_block_size, block_end - block_start))
        ranges = []
        for sub_start in range(block_start, block_end, sub_block_size):
            ranges.append((sub_start, min(block_end, sub_start + sub_block_size)))
        return ranges

    def _apply_quality_gates(
        self,
        state: DiffusionState,
        *,
        prompt_len: int,
    ) -> None:
        if not self.config.quality_gating:
            return
        final_masks = state.remaining_masks(prompt_len, state.tokens.shape[1])
        state.stats["quality_gate_masks_resolved"] = final_masks == 0
        state.stats["quality_gate_empty_generation"] = bool(
            mx.all(state.tokens[:, prompt_len:] == self.decoder.config.mask_id).item()
        )
        step_tps = state.stats.get("step_tps", [])
        state.stats["quality_gate_min_step_tps"] = (
            min(step_tps) if step_tps else 0.0
        )
        state.stats["quality_gate_passed"] = bool(
            state.stats["quality_gate_masks_resolved"]
            and not state.stats["quality_gate_empty_generation"]
            and state.stats["quality_gate_min_step_tps"] >= self.config.quality_min_step_tps
        )

    def _fast_dllm_transfer_index(
        self,
        logits: mx.array,
        tokens: mx.array,
        mask_index: mx.array,
        *,
        step_targets: Optional[mx.array] = None,
    ) -> tuple[mx.array, mx.array, bool]:
        decoder_cfg = self.decoder.config
        remasking = getattr(self.decoder, "remasking", "low_confidence")

        if decoder_cfg.logits_eos_inf and decoder_cfg.eos_token_id is not None:
            logits[:, :, decoder_cfg.eos_token_id] = -mx.inf

        if (
            self.config.compile_steps
            and decoder_cfg.factor is None
            and decoder_cfg.threshold is not None
        ):
            x0, transfer_index = self._compiled_fast_threshold_transfer_kernel()(
                logits, tokens, mask_index
            )
            return x0, transfer_index, True

        logits_with_noise = add_gumbel_noise(logits, temperature=decoder_cfg.temperature)
        x0 = mx.argmax(logits_with_noise, axis=-1)

        if remasking == "low_confidence":
            x0_p = gather_selected_probs(logits, x0)
        elif remasking == "random":
            x0_p = mx.random.uniform(shape=x0.shape, dtype=logits.dtype)
        else:
            raise NotImplementedError(remasking)

        x0 = mx.where(mask_index, x0, tokens)
        confidence = mx.where(mask_index, x0_p, -mx.inf)

        if decoder_cfg.factor is not None:
            transfer_index = mx.zeros(confidence.shape, dtype=mx.bool_)
            for row_idx in range(confidence.shape[0]):
                num_tokens = int(mx.sum(mask_index[row_idx].astype(mx.int32)).item())
                if num_tokens == 0:
                    continue
                sorted_confidence = mx.sort(confidence[row_idx])[::-1][:num_tokens]
                thresholds = [
                    -1.0 if i == 0 else 1.0 - (decoder_cfg.factor / float(i + 2))
                    for i in range(num_tokens)
                ]
                top_k = num_tokens
                for idx in range(num_tokens):
                    if float(sorted_confidence[idx].item()) < thresholds[idx]:
                        top_k = idx
                        break
                if top_k == 0 or top_k == num_tokens - 1:
                    top_k += 1
                top_k = min(max(top_k, 1), num_tokens)
                select_index = mx.argpartition(-confidence[row_idx], kth=top_k - 1)[:top_k]
                transfer_index[row_idx, select_index] = True
            transfer_index = transfer_index & mask_index
            return x0, transfer_index, False

        if decoder_cfg.threshold is not None:
            threshold_index = mask_index & (confidence >= decoder_cfg.threshold)
            max_conf_indices = mx.argmax(confidence, axis=1, keepdims=True)
            force_mask = mx.put_along_axis(
                mx.zeros(confidence.shape, dtype=mx.bool_),
                max_conf_indices,
                mx.ones(max_conf_indices.shape, dtype=mx.bool_),
                axis=-1,
            )
            transfer_index = (threshold_index | force_mask) & mask_index
            return x0, transfer_index, False

        if step_targets is None:
            raise ValueError("step_targets are required when neither threshold nor factor is set.")
        max_k = int(mx.max(step_targets).item())
        if max_k <= 0:
            return x0, mx.zeros(confidence.shape, dtype=mx.bool_), False
        candidate_index = mx.argpartition(-confidence, kth=max_k - 1, axis=-1)[:, :max_k]
        selection_order = mx.arange(max_k, dtype=step_targets.dtype)[None, :]
        selection_mask = selection_order < step_targets[:, None]
        transfer_index = mx.put_along_axis(
            mx.zeros(confidence.shape, dtype=mx.bool_),
            candidate_index,
            selection_mask,
            axis=-1,
        )
        return x0, transfer_index & mask_index, False

    def _generate_tokens_fast_dllm_v1(
        self,
        state: DiffusionState,
        *,
        prompt_len: int,
        mask_id: int,
        progress_callback: Optional[Callable[[dict[str, Any]], None]] = None,
        verbose: bool = False,
    ) -> mx.array:
        if not getattr(self.model, "supports_dual_cache", False):
            raise ValueError(
                "fast_dllm_v1 requires a model with supports_dual_cache=True."
            )

        for block_idx in range(self.scheduler.num_blocks):
            block_start, block_end = self.scheduler.block_range(prompt_len, block_idx)
            initial_mask_index = state.block_tokens(block_start, block_end) == mask_id
            if not bool(mx.any(initial_mask_index).item()):
                state.stats["blocks_completed"] += 1
                continue

            active_indices = self._active_indices(state.block_tokens(block_start, block_end))
            active_rows = state.tokens.shape[0] if active_indices is None else len(active_indices)
            replace_position = self.dual_cache_manager.replace_mask(
                batch_size=active_rows,
                total_length=state.tokens.shape[1],
                start=block_start,
                end=block_end,
            )
            planned_block_steps = self.scheduler.block_steps(initial_mask_index)
            transfer_schedule = (
                self.scheduler.transfer_schedule(initial_mask_index)
                if hasattr(self.scheduler, "transfer_schedule")
                else None
            )

            # Step 0: warm the full cache once per block and do the initial global transfer.
            step_start = time.perf_counter()
            active_tokens = state.tokens if active_indices is None else state.tokens[active_indices]
            logits_full, past_key_values = self.dual_cache_manager.warm_full_cache(
                tokens=state.tokens,
                attention_mask=state.attention_mask,
                active_indices=active_indices,
            )
            global_mask_index = active_tokens == mask_id
            global_mask_index[:, block_end:] = False
            quota0 = None
            if self.decoder.config.factor is None and self.decoder.config.threshold is None:
                if transfer_schedule is None:
                    raise ValueError("Fast-dLLM quota mode requires a scheduler transfer schedule.")
                quota0 = (
                    transfer_schedule[:, 0].astype(mx.int32)
                    if active_indices is None
                    else transfer_schedule[active_indices, 0].astype(mx.int32)
                )
            total_masks_before = state.remaining_masks(block_start, block_end)
            x0, transfer_index, compiled_transfer = self._fast_dllm_transfer_index(
                logits_full,
                active_tokens,
                global_mask_index,
                step_targets=quota0,
            )
            updated_tokens = mx.where(transfer_index, x0, active_tokens)
            if active_indices is None:
                state.tokens = updated_tokens
            else:
                state.tokens[active_indices] = updated_tokens
            step_time = time.perf_counter() - step_start
            total_masks_after = state.remaining_masks(block_start, block_end)
            tokens_committed, step_tps = self._record_step_stats(
                state,
                block_idx=block_idx,
                block_start=block_start,
                block_end=block_end,
                step_time=step_time,
                total_masks_before=total_masks_before,
                total_masks_after=total_masks_after,
                logits_span=logits_full.shape[1],
                context_span=state.tokens.shape[1],
                active_rows=active_rows,
                model_forwards=1,
                compiled=compiled_transfer,
            )
            if verbose:
                print(
                    f"[diffusion] block={block_idx + 1}/{self.scheduler.num_blocks} "
                    f"step=1/{planned_block_steps} "
                    f"time={step_time:.2f}s masked={total_masks_after}"
                )
            if progress_callback is not None:
                progress_callback(
                    {
                        "mode": self.config.mode,
                        "block_index": block_idx,
                        "num_blocks": self.scheduler.num_blocks,
                        "block_start": block_start,
                        "block_end": block_end,
                        "step_index": 0,
                        "block_steps": planned_block_steps,
                        "step_limit": self.scheduler.step_limit(initial_mask_index),
                        "step_time": step_time,
                        "masks_remaining": total_masks_after,
                        "tokens_committed": tokens_committed,
                        "step_tps": step_tps,
                        "elapsed_time": state.stats["elapsed_time"],
                        "total_tokens_committed": state.stats["total_tokens_committed"],
                        "prompt_length": prompt_len,
                        "mask_id": mask_id,
                        "token_ids": state.tokens.tolist(),
                    }
                )

            block_past_key_values = None
            if self.config.use_block_cache and getattr(self.model, "supports_block_cache", False):
                _, block_past_key_values = self.dual_cache_manager.warm_block_cache(
                    tokens=state.tokens,
                    attention_mask=state.attention_mask,
                    full_cache=past_key_values,
                    block_start=block_start,
                    block_end=block_end,
                    active_indices=active_indices,
                )

            step_idx = 1
            while True:
                block_tokens = state.block_tokens(block_start, block_end)
                block_mask_index = block_tokens == mask_id
                if self.scheduler.should_stop_block(
                    block_mask_index,
                    step_idx=step_idx,
                    initial_mask_index=initial_mask_index,
                ):
                    state.stats["stopped_early"] = True
                    break

                quota_i = None
                if self.decoder.config.factor is None and self.decoder.config.threshold is None:
                    if transfer_schedule is None:
                        raise ValueError("Fast-dLLM quota mode requires a scheduler transfer schedule.")
                    quota_i = (
                        transfer_schedule[:, step_idx].astype(mx.int32)
                        if active_indices is None
                        else transfer_schedule[active_indices, step_idx].astype(mx.int32)
                    )

                sub_ranges = self._sub_block_ranges(block_start, block_end)
                for sub_start, sub_end in sub_ranges:
                    sub_tokens = state.block_tokens(sub_start, sub_end)
                    if not bool(mx.any(sub_tokens == mask_id).item()):
                        continue
                    step_start = time.perf_counter()
                    total_masks_before = state.remaining_masks(block_start, block_end)
                    if self.config.use_block_cache and block_past_key_values is not None:
                        logits, block_past_key_values = self.dual_cache_manager.refine_sub_block(
                            tokens=state.tokens,
                            attention_mask=state.attention_mask,
                            full_cache=past_key_values,
                            block_cache=block_past_key_values,
                            block_start=block_start,
                            block_end=block_end,
                            sub_start=sub_start,
                            sub_end=sub_end,
                            active_indices=active_indices,
                        )
                    else:
                        active_attention = (
                            state.attention_mask
                            if state.attention_mask is None or active_indices is None
                            else state.attention_mask[active_indices]
                        )
                        active_full_cache = _slice_cache_rows(past_key_values, active_indices)
                        active_replace = self.dual_cache_manager.replace_mask(
                            batch_size=active_rows,
                            total_length=state.tokens.shape[1],
                            start=sub_start,
                            end=sub_end,
                        )
                        output = self.model(
                            state.tokens[:, sub_start:sub_end]
                            if active_indices is None
                            else state.tokens[active_indices, sub_start:sub_end],
                            attention_mask=active_attention,
                            past_key_values=active_full_cache,
                            use_cache=True,
                            replace_position=active_replace,
                        )
                        logits = _extract_logits(output)

                    active_sub_tokens = (
                        sub_tokens
                        if active_indices is None
                        else sub_tokens[active_indices]
                    )
                    active_sub_mask = active_sub_tokens == mask_id
                    x0_blk, transfer_idx_blk, compiled_transfer = self._fast_dllm_transfer_index(
                        logits,
                        active_sub_tokens,
                        active_sub_mask,
                        step_targets=quota_i,
                    )
                    updated_sub = mx.where(transfer_idx_blk, x0_blk, active_sub_tokens)
                    if active_indices is None:
                        state.update_block(sub_start, sub_end, updated_sub)
                    else:
                        state.update_block(sub_start, sub_end, updated_sub, active_indices)

                    step_time = time.perf_counter() - step_start
                    total_masks_after = state.remaining_masks(block_start, block_end)
                    tokens_committed, step_tps = self._record_step_stats(
                        state,
                        block_idx=block_idx,
                        block_start=block_start,
                        block_end=block_end,
                        step_time=step_time,
                        total_masks_before=total_masks_before,
                        total_masks_after=total_masks_after,
                        logits_span=logits.shape[1],
                        context_span=(
                            block_end - block_start
                            if self.config.use_block_cache and block_past_key_values is not None
                            else state.tokens.shape[1]
                        ),
                        active_rows=active_rows,
                        model_forwards=1,
                        compiled=compiled_transfer,
                    )
                    if verbose:
                        print(
                            f"[diffusion] block={block_idx + 1}/{self.scheduler.num_blocks} "
                            f"step={step_idx + 1}/{planned_block_steps} "
                            f"time={step_time:.2f}s masked={total_masks_after}"
                        )
                    if progress_callback is not None:
                        progress_callback(
                            {
                                "mode": self.config.mode,
                                "block_index": block_idx,
                                "num_blocks": self.scheduler.num_blocks,
                                "block_start": block_start,
                                "block_end": block_end,
                                "step_index": step_idx,
                                "block_steps": planned_block_steps,
                                "step_limit": self.scheduler.step_limit(initial_mask_index),
                                "step_time": step_time,
                                "masks_remaining": total_masks_after,
                                "tokens_committed": tokens_committed,
                                "step_tps": step_tps,
                                "elapsed_time": state.stats["elapsed_time"],
                                "total_tokens_committed": state.stats["total_tokens_committed"],
                                "prompt_length": prompt_len,
                                "mask_id": mask_id,
                                "token_ids": state.tokens.tolist(),
                            }
                        )
                    step_idx += 1
                    if not bool(mx.any(state.block_tokens(block_start, block_end) == mask_id).item()):
                        break

            state.stats["blocks_completed"] += 1
            if not state.unresolved_suffix(prompt_len):
                state.stats["stopped_early"] = True
                break

        return state.tokens

    def generate_tokens(
        self,
        prompt: mx.array,
        *,
        attention_mask: Optional[mx.array] = None,
        prompt_lengths: Optional[Sequence[int]] = None,
        progress_callback: Optional[Callable[[dict[str, Any]], None]] = None,
        verbose: bool = False,
        return_stats: bool = False,
    ):
        if prompt.ndim == 1:
            prompt = prompt[None, :]
        if attention_mask is not None and attention_mask.ndim == 1:
            attention_mask = attention_mask[None, :]

        mask_id = self.decoder.config.mask_id
        if mask_id is None:
            mask_id = getattr(getattr(self.model, "args", None), "mask_token_id", None)
        if mask_id is None:
            raise ValueError("mask_id must be provided for diffusion generation.")
        self.decoder.config = DiffusionDecoderConfig(
            **{**self.decoder.config.__dict__, "mask_id": mask_id}
        )
        state = DiffusionState.from_prompt_batch(
            prompt,
            gen_length=self.scheduler.config.gen_length,
            mask_id=mask_id,
            attention_mask=attention_mask,
            prompt_lengths=prompt_lengths,
        )
        prompt_len = state.prompt_length
        state.stats["batch_size"] = state.batch_size
        state.stats["runtime_mode"] = self.config.mode
        state.stats["cache_strategy"] = self.context_cache.config.strategy

        if self.config.dual_cache:
            generated = self._generate_tokens_fast_dllm_v1(
                state,
                prompt_len=prompt_len,
                mask_id=mask_id,
                progress_callback=progress_callback,
                verbose=verbose,
            )
            generated_tokens_per_sample = [
                int(generated.shape[1] - prompt_len)
            ] * int(generated.shape[0])
            generated_tokens = int(sum(generated_tokens_per_sample))
            elapsed_time = float(state.stats.get("elapsed_time", 0.0))
            state.stats["generated_tokens_per_sample"] = generated_tokens_per_sample
            state.stats["generated_tokens"] = generated_tokens
            state.stats["tps"] = (
                float(generated_tokens) / elapsed_time if elapsed_time > 0 else 0.0
            )
            self._apply_quality_gates(state, prompt_len=prompt_len)
            if return_stats:
                return generated, state.stats
            return generated

        for block_idx in range(self.scheduler.num_blocks):
            block_start, block_end = self.scheduler.block_range(prompt_len, block_idx)
            initial_mask_index = state.block_tokens(block_start, block_end) == mask_id
            if not bool(mx.any(initial_mask_index).item()):
                state.stats["blocks_completed"] += 1
                continue

            context_start, context_end = self.context_cache.context_range(
                prompt_length=prompt_len,
                total_length=state.tokens.shape[1],
                block_start=block_start,
                block_end=block_end,
                block_length=self.scheduler.config.block_length,
            )
            cached_context = self.context_cache.prepare(
                tokens=state.tokens,
                attention_mask=state.attention_mask,
                prompt_index=state.prompt_index,
                mask_id=mask_id,
                context_start=context_start,
                context_end=context_end,
                block_local=self.config.block_local,
                cfg_scale=self.config.cfg_scale,
            )
            prefix_states = cached_context.prefix_states
            combined_prefix_states = cached_context.combined_prefix_states
            state.block_prefix_states[block_idx] = cached_context

            step_idx = 0
            planned_block_steps = self.scheduler.block_steps(initial_mask_index)
            while True:
                step_start = time.perf_counter()
                block_tokens = state.block_tokens(block_start, block_end)
                block_mask_index = block_tokens == mask_id
                total_masks_before = state.remaining_masks(block_start, block_end)
                if self.scheduler.should_stop_block(
                    block_mask_index,
                    step_idx=step_idx,
                    initial_mask_index=initial_mask_index,
                ):
                    state.stats["stopped_early"] = True
                    break

                active_indices = self._active_indices(block_tokens)
                active_x = state.tokens if active_indices is None else state.tokens[active_indices]
                active_block_tokens = (
                    block_tokens if active_indices is None else block_tokens[active_indices]
                )
                active_attention = (
                    state.attention_mask
                    if state.attention_mask is None or active_indices is None
                    else state.attention_mask[active_indices]
                )
                active_prompt_index = (
                    state.prompt_index
                    if active_indices is None
                    else state.prompt_index[active_indices]
                )
                step_targets = self.scheduler.step_targets(
                    block_mask_index if active_indices is None else block_mask_index[active_indices],
                    step_idx=step_idx,
                    initial_mask_index=(
                        initial_mask_index
                        if active_indices is None
                        else initial_mask_index[active_indices]
                    ),
                )
                active_prefix_states = self._slice_prefix_states(prefix_states, active_indices)
                active_combined_prefix_states = self._slice_prefix_states(
                    combined_prefix_states, active_indices
                )
                step_ctx = DiffusionStepContext(
                    block_start=block_start,
                    block_end=block_end,
                    context_start=context_start,
                    context_end=context_end,
                    active_x=active_x,
                    active_block_tokens=active_block_tokens,
                    active_attention=active_attention,
                    active_prompt_index=active_prompt_index,
                    step_targets=step_targets,
                    active_prefix_states=active_prefix_states,
                    active_combined_prefix_states=active_combined_prefix_states,
                    active_indices=active_indices,
                    mask_id=mask_id,
                )
                if self.config.block_local or self.config.cfg_scale > 0.0:
                    step_result = self._block_local_step.run(step_ctx)
                else:
                    step_result = self._full_sequence_step.run(step_ctx)
                updated_block = step_result.updated_block
                masks_after = step_result.masks_after

                if active_indices is None:
                    state.update_block(block_start, block_end, updated_block)
                else:
                    state.update_block(block_start, block_end, updated_block, active_indices)

                step_time = time.perf_counter() - step_start
                state.stats["steps_completed"] += 1
                state.stats["step_times"].append(step_time)
                total_masks_after = state.remaining_masks(block_start, block_end)
                tokens_committed = max(total_masks_before - total_masks_after, 0)
                step_tps = (
                    float(tokens_committed) / step_time if step_time > 0 else 0.0
                )
                state.stats["elapsed_time"] += step_time
                state.stats["total_tokens_committed"] += tokens_committed
                state.stats["masks_remaining"].append(total_masks_after)
                state.stats["tokens_committed"].append(tokens_committed)
                state.stats["step_tps"].append(step_tps)
                state.stats["model_forwards"] += step_result.model_forwards
                state.stats["compiled_step_hits"] += int(step_result.compiled)
                state.stats["compiled_step_misses"] += int(not step_result.compiled)
                state.stats["active_rows_history"].append(step_result.active_rows)
                state.stats["logits_span_history"].append(step_result.logits_span)
                state.stats["context_span_history"].append(context_end - context_start)
                state.stats["block_iterations"][block_idx] = (
                    state.stats["block_iterations"].get(block_idx, 0) + 1
                )
                if verbose:
                    block_step_label = (
                        f"{step_idx + 1}/{planned_block_steps}"
                        if self.scheduler.step_limit(initial_mask_index) is not None
                        else f"{step_idx + 1}/dynamic"
                    )
                    print(
                        f"[diffusion] block={block_idx + 1}/{self.scheduler.num_blocks} "
                        f"step={block_step_label} "
                        f"time={step_time:.2f}s masked={total_masks_after}"
                    )
                if progress_callback is not None:
                    progress_callback(
                        {
                            "mode": self.config.mode,
                            "block_index": block_idx,
                            "num_blocks": self.scheduler.num_blocks,
                            "block_start": block_start,
                            "block_end": block_end,
                            "step_index": step_idx,
                            "block_steps": planned_block_steps,
                            "step_limit": self.scheduler.step_limit(initial_mask_index),
                            "step_time": step_time,
                            "masks_remaining": total_masks_after,
                            "tokens_committed": tokens_committed,
                            "step_tps": step_tps,
                            "elapsed_time": state.stats["elapsed_time"],
                            "total_tokens_committed": state.stats["total_tokens_committed"],
                            "prompt_length": prompt_len,
                            "mask_id": mask_id,
                            "token_ids": state.tokens.tolist(),
                        }
                    )
                step_idx += 1

            state.stats["blocks_completed"] += 1
            if not state.unresolved_suffix(prompt_len):
                state.stats["stopped_early"] = True
                break

        generated_tokens_per_sample = [
            int(state.tokens.shape[1] - prompt_len)
        ] * int(state.tokens.shape[0])
        generated_tokens = int(sum(generated_tokens_per_sample))
        elapsed_time = float(state.stats.get("elapsed_time", 0.0))
        state.stats["generated_tokens_per_sample"] = generated_tokens_per_sample
        state.stats["generated_tokens"] = generated_tokens
        state.stats["tps"] = (
            float(generated_tokens) / elapsed_time if elapsed_time > 0 else 0.0
        )
        self._apply_quality_gates(state, prompt_len=prompt_len)

        if return_stats:
            return state.tokens, state.stats
        return state.tokens

    def generate(
        self,
        tokenizer: Union[PreTrainedTokenizer, TokenizerWrapper],
        prompt: Union[str, Sequence[int], Sequence[str], Sequence[Sequence[int]], mx.array],
        *,
        attention_mask: Optional[Union[Sequence[int], Sequence[Sequence[int]], mx.array]] = None,
        return_full_sequence: bool = False,
        progress_callback: Optional[Callable[[dict[str, Any]], None]] = None,
        verbose: bool = False,
    ) -> DiffusionGenerationResult:
        if not isinstance(tokenizer, TokenizerWrapper):
            tokenizer = TokenizerWrapper(tokenizer)

        if _is_batched_prompt(prompt):
            prompt, batch_attention_mask, prompt_lengths = pad_prompt_batch(
                tokenizer,
                prompt,
                attention_mask=attention_mask,
            )
            attention_mask = batch_attention_mask
        else:
            prompt = _normalize_prompt(tokenizer, prompt)
            prompt_lengths = [int(prompt.shape[0])] if prompt.ndim == 1 else [prompt.shape[1]]
            if attention_mask is not None and not isinstance(attention_mask, mx.array):
                attention_mask = mx.array(attention_mask)

        generated, stats = self.generate_tokens(
            prompt,
            attention_mask=attention_mask,
            prompt_lengths=prompt_lengths,
            progress_callback=progress_callback,
            verbose=verbose,
            return_stats=True,
        )
        stats["prompt_lengths"] = prompt_lengths
        prompt_length = prompt.shape[-1] if prompt.ndim == 1 else prompt.shape[1]
        if generated.ndim == 1:
            suffix = generated[prompt_length:]
            text = tokenizer.decode(suffix.tolist(), skip_special_tokens=True)
            generated_tokens_per_sample = [int(suffix.shape[0])]
            raw_text = tokenizer.decode(suffix.tolist(), skip_special_tokens=False)
            raw_texts = [raw_text]
            clean_texts = [text]
        else:
            suffix = generated[:, prompt_length:]
            text = [
                tokenizer.decode(sample.tolist(), skip_special_tokens=True)
                for sample in suffix
            ]
            generated_tokens_per_sample = [int(suffix.shape[1])] * int(suffix.shape[0])
            raw_texts = [
                tokenizer.decode(sample.tolist(), skip_special_tokens=False)
                for sample in suffix
            ]
            clean_texts = text

        generated_tokens = int(sum(generated_tokens_per_sample))
        elapsed_time = float(stats.get("elapsed_time", 0.0))
        stats["generated_tokens_per_sample"] = generated_tokens_per_sample
        stats["generated_tokens"] = generated_tokens
        stats["tps"] = (
            float(generated_tokens) / elapsed_time if elapsed_time > 0 else 0.0
        )
        stats["quality_empty_output"] = all(not sample.strip() for sample in clean_texts)
        stats["quality_special_token_heavy"] = any(
            raw.count("<|") > max(len(clean.strip().split()), 1)
            for raw, clean in zip(raw_texts, clean_texts)
        )
        if self.config.quality_gating:
            stats["quality_gate_passed"] = bool(
                stats.get("quality_gate_passed", True)
                and not stats["quality_empty_output"]
                and not stats["quality_special_token_heavy"]
            )

        token_ids = generated if return_full_sequence else suffix
        return DiffusionGenerationResult(
            text=text,
            token_ids=token_ids,
            prompt_length=prompt_length,
            stats=stats,
        )
