from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

import mlx.core as mx


@dataclass
class DiffusionState:
    tokens: mx.array
    attention_mask: Optional[mx.array]
    prompt_index: mx.array
    prompt_lengths: list[int]
    prompt_width: int
    mask_id: int
    stats: dict[str, Any] = field(default_factory=dict)
    block_prefix_states: dict[int, Any] = field(default_factory=dict)
    request_ids: Optional[list[str]] = None

    @classmethod
    def from_prompt_batch(
        cls,
        prompt: mx.array,
        *,
        gen_length: int,
        mask_id: int,
        attention_mask: Optional[mx.array] = None,
        prompt_lengths: Optional[Sequence[int]] = None,
    ) -> "DiffusionState":
        if prompt.ndim == 1:
            prompt = prompt[None, :]
        batch_size, prompt_len = prompt.shape
        tokens = mx.full((batch_size, prompt_len + gen_length), mask_id, dtype=prompt.dtype)
        tokens[:, :prompt_len] = prompt
        if attention_mask is not None:
            suffix_attention = mx.ones((batch_size, gen_length), dtype=attention_mask.dtype)
            attention_mask = mx.concatenate([attention_mask, suffix_attention], axis=-1)
        prompt_index = tokens != mask_id
        return cls(
            tokens=tokens,
            attention_mask=attention_mask,
            prompt_index=prompt_index,
            prompt_lengths=list(prompt_lengths or [prompt_len] * batch_size),
            prompt_width=prompt_len,
            mask_id=mask_id,
            stats={
                "blocks_completed": 0,
                "steps_completed": 0,
                "step_times": [],
                "masks_remaining": [],
                "tokens_committed": [],
                "step_tps": [],
                "elapsed_time": 0.0,
                "total_tokens_committed": 0,
                "generated_tokens": 0,
                "generated_tokens_per_sample": [],
                "tps": 0.0,
                "stopped_early": False,
                "model_forwards": 0,
                "compiled_step_hits": 0,
                "compiled_step_misses": 0,
                "active_rows_history": [],
                "logits_span_history": [],
                "context_span_history": [],
                "block_iterations": {},
            },
        )

    @property
    def batch_size(self) -> int:
        return self.tokens.shape[0]

    @property
    def prompt_length(self) -> int:
        return self.prompt_width

    def block_tokens(self, start: int, end: int) -> mx.array:
        return self.tokens[:, start:end]

    def remaining_masks(self, start: int, end: int) -> int:
        return int(
            mx.sum((self.tokens[:, start:end] == self.mask_id).astype(mx.int32)).item()
        )

    def update_block(self, start: int, end: int, updated_block: mx.array, indices=None):
        if indices is None:
            self.tokens[:, start:end] = updated_block
        else:
            self.tokens[indices, start:end] = updated_block

    def unresolved_suffix(self, prompt_len: int) -> bool:
        return bool(mx.any(self.tokens[:, prompt_len:] == self.mask_id).item())
