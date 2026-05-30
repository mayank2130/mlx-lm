from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import partial
from typing import Optional

import mlx.core as mx


@partial(mx.compile, shapeless=True)
def _transfer_tokens(mask_index: mx.array, steps: int) -> mx.array:
    mask_num = mx.sum(mask_index.astype(mx.int32), axis=1, keepdims=True)
    base = mask_num // steps
    remainder = mask_num % steps
    step_index = mx.arange(steps, dtype=base.dtype)[None, :]
    extra = (step_index < remainder).astype(base.dtype)
    return mx.broadcast_to(base, (mask_num.shape[0], steps)) + extra


@dataclass(frozen=True)
class DiffusionSchedulerConfig:
    steps: Optional[int] = 128
    gen_length: int = 128
    block_length: int = 128
    max_steps_per_block: Optional[int] = None
    min_tokens_per_step: int = 1


class BaseDiffusionScheduler(ABC):
    def __init__(self, config: DiffusionSchedulerConfig):
        self.config = config
        if self.config.gen_length % self.config.block_length != 0:
            raise ValueError("gen_length must be divisible by block_length.")

    @property
    def num_blocks(self) -> int:
        return self.config.gen_length // self.config.block_length

    def block_range(self, prompt_length: int, block_idx: int) -> tuple[int, int]:
        start = prompt_length + block_idx * self.config.block_length
        end = prompt_length + (block_idx + 1) * self.config.block_length
        return start, end

    @abstractmethod
    def block_steps(self, initial_mask_index: mx.array) -> int:
        raise NotImplementedError

    def step_limit(self, initial_mask_index: mx.array) -> Optional[int]:
        return self.block_steps(initial_mask_index)

    @abstractmethod
    def step_targets(
        self,
        current_mask_index: mx.array,
        *,
        step_idx: int,
        initial_mask_index: mx.array,
    ) -> mx.array:
        raise NotImplementedError

    def should_stop_block(
        self,
        current_mask_index: mx.array,
        *,
        step_idx: int,
        initial_mask_index: mx.array,
    ) -> bool:
        has_masks = bool(mx.any(current_mask_index).item())
        step_limit = self.step_limit(initial_mask_index)
        return not has_masks or (
            step_limit is not None and step_idx >= step_limit
        )


class LLaDAFixedScheduler(BaseDiffusionScheduler):
    def __init__(self, config: DiffusionSchedulerConfig):
        super().__init__(config)
        if self.config.steps is None:
            raise ValueError("steps must be provided for the fixed LLaDA scheduler.")
        if self.config.steps % self.num_blocks != 0:
            raise ValueError("steps must be divisible by the number of blocks.")

    def block_steps(self, initial_mask_index: mx.array) -> int:
        return self.config.steps // self.num_blocks

    def transfer_schedule(self, mask_index: mx.array) -> mx.array:
        return _transfer_tokens(mask_index, self.block_steps(mask_index))

    def step_targets(
        self,
        current_mask_index: mx.array,
        *,
        step_idx: int,
        initial_mask_index: mx.array,
    ) -> mx.array:
        schedule = self.transfer_schedule(initial_mask_index)
        return schedule[:, step_idx].astype(mx.int32)


class DynamicBlockDiffusionScheduler(BaseDiffusionScheduler):
    def block_steps(self, initial_mask_index: mx.array) -> int:
        return self.step_limit(initial_mask_index) or max(
            self.config.block_length, self.config.min_tokens_per_step
        )

    def step_limit(self, initial_mask_index: mx.array) -> Optional[int]:
        return self.config.max_steps_per_block

    def step_targets(
        self,
        current_mask_index: mx.array,
        *,
        step_idx: int,
        initial_mask_index: mx.array,
    ) -> mx.array:
        current_mask_count = mx.sum(current_mask_index.astype(mx.int32), axis=1)
        remaining_budget = max(self.block_steps(initial_mask_index) - step_idx, 1)
        adaptive = mx.maximum(
            current_mask_count // remaining_budget,
            mx.array(self.config.min_tokens_per_step, dtype=mx.int32),
        )
        return mx.minimum(adaptive, current_mask_count).astype(mx.int32)


DiffusionScheduler = LLaDAFixedScheduler
