from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional, Sequence

import mlx.core as mx


@dataclass
class DiffusionStepContext:
    block_start: int
    block_end: int
    context_start: int
    context_end: int
    active_x: mx.array
    active_block_tokens: mx.array
    active_attention: Optional[mx.array]
    active_prompt_index: mx.array
    step_targets: mx.array
    active_prefix_states: Optional[list[mx.array]]
    active_combined_prefix_states: Optional[list[mx.array]]
    active_indices: Optional[Sequence[int]]
    mask_id: int


@dataclass
class DiffusionStepResult:
    updated_block: mx.array
    masks_after: int
    model_forwards: int
    compiled: bool
    logits_span: int
    active_rows: int


class BaseDiffusionStep(ABC):
    def __init__(self, runtime: Any):
        self.runtime = runtime

    @abstractmethod
    def run(self, ctx: DiffusionStepContext) -> DiffusionStepResult:
        raise NotImplementedError


class FullSequenceDiffusionStep(BaseDiffusionStep):
    def run(self, ctx: DiffusionStepContext) -> DiffusionStepResult:
        runtime = self.runtime
        step_targets = ctx.step_targets.astype(mx.int32)
        use_compiled = runtime._can_compile_full_step(
            active_prefix_states=ctx.active_prefix_states,
            active_combined_prefix_states=ctx.active_combined_prefix_states,
        )
        if use_compiled:
            max_k = max(int(mx.max(step_targets).item()), 1)
            step_kernel = runtime._compiled_full_step(
                block_start=ctx.block_start,
                block_end=ctx.block_end,
                max_k=max_k,
                has_attention_mask=ctx.active_attention is not None,
            )
            if ctx.active_attention is None:
                _, updated_block, masks_after = step_kernel(ctx.active_x, step_targets)
            else:
                _, updated_block, masks_after = step_kernel(
                    ctx.active_x, ctx.active_attention, step_targets
                )
            return DiffusionStepResult(
                updated_block=updated_block,
                masks_after=int(masks_after.item()),
                model_forwards=1,
                compiled=True,
                logits_span=ctx.block_end - ctx.block_start,
                active_rows=ctx.active_x.shape[0],
            )

        logits = runtime._model_logits(
            ctx.active_x,
            attention_mask=ctx.active_attention,
            logits_range=(
                (ctx.block_start, ctx.block_end)
                if runtime.config.use_logits_range and not runtime.config.block_local
                else None
            ),
            block_range=None,
            prefix_states=ctx.active_prefix_states,
        )
        if logits.shape[1] != ctx.active_block_tokens.shape[1]:
            logits = logits[:, ctx.block_start : ctx.block_end, :]
        updated_block, masks_after = runtime.decoder.decode(
            ctx.active_block_tokens,
            logits,
            step_targets,
            compile_steps=runtime.config.compile_steps,
        )
        return DiffusionStepResult(
            updated_block=updated_block,
            masks_after=masks_after,
            model_forwards=1,
            compiled=False,
            logits_span=logits.shape[1],
            active_rows=ctx.active_x.shape[0],
        )


class BlockLocalDiffusionStep(BaseDiffusionStep):
    def run(self, ctx: DiffusionStepContext) -> DiffusionStepResult:
        runtime = self.runtime
        step_targets = ctx.step_targets.astype(mx.int32)
        block_range = (
            (ctx.context_start, ctx.context_end)
            if runtime.config.block_local and ctx.active_prefix_states is not None
            else None
        )

        if runtime.config.cfg_scale > 0.0:
            un_x = mx.where(
                ctx.active_prompt_index,
                mx.full(ctx.active_x.shape, ctx.mask_id, dtype=ctx.active_x.dtype),
                ctx.active_x,
            )
            x_in = mx.concatenate([ctx.active_x, un_x], axis=0)
            attention_in = (
                mx.concatenate([ctx.active_attention, ctx.active_attention], axis=0)
                if ctx.active_attention is not None
                else None
            )
            logits = runtime._model_logits(
                x_in,
                attention_mask=attention_in,
                logits_range=(
                    (ctx.block_start, ctx.block_end)
                    if runtime.config.use_logits_range and not runtime.config.block_local
                    else None
                ),
                block_range=block_range,
                prefix_states=ctx.active_combined_prefix_states,
            )
            logits, un_logits = mx.split(logits, 2, axis=0)
            logits = un_logits + (runtime.config.cfg_scale + 1.0) * (logits - un_logits)
        else:
            logits = runtime._model_logits(
                ctx.active_x,
                attention_mask=ctx.active_attention,
                logits_range=(
                    (ctx.block_start, ctx.block_end)
                    if runtime.config.use_logits_range and not runtime.config.block_local
                    else None
                ),
                block_range=block_range,
                prefix_states=ctx.active_prefix_states,
            )

        if logits.shape[1] != ctx.active_block_tokens.shape[1]:
            if block_range is None:
                logits = logits[:, ctx.block_start : ctx.block_end, :]
            else:
                block_offset = max(ctx.block_start - ctx.context_start, 0)
                logits = logits[
                    :,
                    block_offset : block_offset + ctx.active_block_tokens.shape[1],
                    :,
                ]

        updated_block, masks_after = runtime.decoder.decode(
            ctx.active_block_tokens,
            logits,
            step_targets,
            compile_steps=runtime.config.compile_steps,
        )
        return DiffusionStepResult(
            updated_block=updated_block,
            masks_after=masks_after,
            model_forwards=2 if runtime.config.cfg_scale > 0.0 else 1,
            compiled=False,
            logits_span=logits.shape[1],
            active_rows=ctx.active_x.shape[0],
        )
