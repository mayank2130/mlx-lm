from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import partial
from typing import Optional

import mlx.core as mx


def add_gumbel_noise(logits: mx.array, temperature: float) -> mx.array:
    if temperature == 0:
        return logits
    noise = mx.random.uniform(shape=logits.shape, dtype=mx.float32)
    gumbel_noise = (-mx.log(noise)) ** temperature
    return mx.exp(logits.astype(mx.float32)) / gumbel_noise


@partial(mx.compile, shapeless=True)
def gather_selected_probs(logits: mx.array, selected: mx.array) -> mx.array:
    probs = mx.softmax(logits, axis=-1)
    return mx.take_along_axis(probs, selected[..., None], axis=-1).squeeze(-1)


@dataclass(frozen=True)
class DiffusionDecoderConfig:
    temperature: float = 0.0
    logits_eos_inf: bool = False
    confidence_eos_eot_inf: bool = False
    eos_token_id: Optional[int] = None
    eot_token_id: Optional[int] = None
    mask_id: Optional[int] = None
    threshold: float = 0.5
    factor: Optional[float] = None


class BaseDiffusionDecoder(ABC):
    def __init__(self, config: DiffusionDecoderConfig):
        self.config = config

    def _prepare_logits(self, logits: mx.array) -> tuple[mx.array, mx.array]:
        if self.config.logits_eos_inf and self.config.eos_token_id is not None:
            logits[:, :, self.config.eos_token_id] = -mx.inf
        logits_with_noise = add_gumbel_noise(logits, temperature=self.config.temperature)
        if self.config.confidence_eos_eot_inf:
            if self.config.eos_token_id is not None:
                logits_with_noise[:, :, self.config.eos_token_id] = -mx.inf
            if self.config.eot_token_id is not None:
                logits[:, :, self.config.eot_token_id] = -mx.inf
        return logits, logits_with_noise

    @abstractmethod
    def decode(
        self,
        block_tokens: mx.array,
        logits: mx.array,
        step_targets: mx.array,
        *,
        compile_steps: bool = True,
    ) -> tuple[mx.array, int]:
        raise NotImplementedError


class TopKConfidenceDecoder(BaseDiffusionDecoder):
    def __init__(self, config: DiffusionDecoderConfig, *, remasking: str = "low_confidence"):
        super().__init__(config)
        self.remasking = remasking
        self._compiled_kernels = {}

    def _compiled_kernel(self, max_k: int):
        if max_k not in self._compiled_kernels:
            self._compiled_kernels[max_k] = self._make_kernel(max_k)
        return self._compiled_kernels[max_k]

    def _make_kernel(self, max_k: int):
        config = self.config
        remasking = self.remasking

        @partial(mx.compile, inputs=mx.random.state, outputs=mx.random.state)
        def _step(
            block_tokens: mx.array,
            logits: mx.array,
            step_targets: mx.array,
        ) -> tuple[mx.array, mx.array]:
            block_mask_index = block_tokens == config.mask_id
            if config.logits_eos_inf and config.eos_token_id is not None:
                logits[:, :, config.eos_token_id] = -mx.inf
            logits_with_noise = add_gumbel_noise(logits, temperature=config.temperature)
            x0_block = mx.argmax(logits_with_noise, axis=-1)
            if config.confidence_eos_eot_inf:
                if config.eos_token_id is not None:
                    logits_with_noise[:, :, config.eos_token_id] = -mx.inf
                if config.eot_token_id is not None:
                    logits[:, :, config.eot_token_id] = -mx.inf
            if remasking == "low_confidence":
                x0_p = gather_selected_probs(logits, x0_block)
            elif remasking == "random":
                x0_p = mx.random.uniform(shape=x0_block.shape, dtype=logits.dtype)
            else:
                raise NotImplementedError(remasking)
            x0_block = mx.where(block_mask_index, x0_block, block_tokens)
            confidence = mx.where(block_mask_index, x0_p, -mx.inf)
            transfer_index = mx.zeros(confidence.shape, dtype=mx.bool_)
            if max_k > 0:
                candidate_index = mx.argpartition(-confidence, kth=max_k - 1, axis=-1)[
                    :, :max_k
                ]
                selection_order = mx.arange(max_k, dtype=step_targets.dtype)[None, :]
                selection_mask = selection_order < step_targets[:, None]
                transfer_index = mx.put_along_axis(
                    transfer_index,
                    candidate_index,
                    selection_mask,
                    axis=-1,
                )
            updated_block = mx.where(transfer_index, x0_block, block_tokens)
            masks_after = mx.sum((updated_block == config.mask_id).astype(mx.int32))
            return updated_block, masks_after

        return _step

    def decode(
        self,
        block_tokens: mx.array,
        logits: mx.array,
        step_targets: mx.array,
        *,
        compile_steps: bool = True,
    ) -> tuple[mx.array, int]:
        if compile_steps:
            max_k = int(mx.max(step_targets).item())
            updated_block, masks_after = self._compiled_kernel(max_k)(
                block_tokens, logits, step_targets
            )
            return updated_block, int(masks_after.item())
        logits, logits_with_noise = self._prepare_logits(logits)
        block_mask_index = block_tokens == self.config.mask_id
        x0_block = mx.argmax(logits_with_noise, axis=-1)
        if self.remasking == "low_confidence":
            x0_p = gather_selected_probs(logits, x0_block)
        elif self.remasking == "random":
            x0_p = mx.random.uniform(shape=x0_block.shape, dtype=logits.dtype)
        else:
            raise NotImplementedError(self.remasking)
        x0_block = mx.where(block_mask_index, x0_block, block_tokens)
        confidence = mx.where(block_mask_index, x0_p, -mx.inf)
        max_k = int(mx.max(step_targets).item())
        transfer_index = mx.zeros(confidence.shape, dtype=mx.bool_)
        if max_k > 0:
            candidate_index = mx.argpartition(-confidence, kth=max_k - 1, axis=-1)[
                :, :max_k
            ]
            selection_order = mx.arange(max_k, dtype=step_targets.dtype)[None, :]
            selection_mask = selection_order < step_targets[:, None]
            transfer_index = mx.put_along_axis(
                transfer_index,
                candidate_index,
                selection_mask,
                axis=-1,
            )
        updated_block = mx.where(transfer_index, x0_block, block_tokens)
        return updated_block, int(
            mx.sum((updated_block == self.config.mask_id).astype(mx.int32)).item()
        )


class ThresholdDecoder(BaseDiffusionDecoder):
    def __init__(self, config: DiffusionDecoderConfig, *, remasking: str = "low_confidence"):
        super().__init__(config)
        self.remasking = remasking

    def _dynamic_transfer_index(
        self,
        confidence: mx.array,
        mask_index: mx.array,
        *,
        factor: float,
    ) -> mx.array:
        transfer_index = mx.zeros(confidence.shape, dtype=mx.bool_)
        for row_idx in range(confidence.shape[0]):
            row_mask = mask_index[row_idx]
            num_tokens = int(mx.sum(row_mask.astype(mx.int32)).item())
            if num_tokens == 0:
                continue
            sorted_confidence = mx.sort(confidence[row_idx])[::-1][:num_tokens]
            top_k = 1
            for candidate_count in range(1, num_tokens + 1):
                threshold = 1.0 - (factor / float(candidate_count + 1))
                if float(sorted_confidence[candidate_count - 1].item()) < threshold:
                    break
                top_k = candidate_count
            candidate_index = mx.argpartition(-confidence[row_idx], kth=top_k - 1)[
                :top_k
            ]
            transfer_index[row_idx, candidate_index] = True
        return transfer_index

    def decode(
        self,
        block_tokens: mx.array,
        logits: mx.array,
        step_targets: mx.array,
        *,
        compile_steps: bool = True,
    ) -> tuple[mx.array, int]:
        logits, logits_with_noise = self._prepare_logits(logits)
        block_mask_index = block_tokens == self.config.mask_id
        x0_block = mx.argmax(logits_with_noise, axis=-1)
        if self.remasking == "low_confidence":
            x0_p = gather_selected_probs(logits, x0_block)
        elif self.remasking == "random":
            x0_p = mx.random.uniform(shape=x0_block.shape, dtype=logits.dtype)
        else:
            raise NotImplementedError(self.remasking)
        x0_block = mx.where(block_mask_index, x0_block, block_tokens)
        confidence = mx.where(block_mask_index, x0_p, -mx.inf)
        if self.config.factor is not None:
            transfer_index = self._dynamic_transfer_index(
                confidence,
                block_mask_index,
                factor=self.config.factor,
            )
        else:
            threshold_index = block_mask_index & (confidence >= self.config.threshold)
            max_confidence_index = mx.argmax(confidence, axis=1, keepdims=True)
            force_index = mx.put_along_axis(
                mx.zeros(confidence.shape, dtype=mx.bool_),
                max_confidence_index,
                mx.ones(max_confidence_index.shape, dtype=mx.bool_),
                axis=-1,
            )
            transfer_index = (threshold_index | force_index) & block_mask_index
        updated_block = mx.where(transfer_index, x0_block, block_tokens)
        return updated_block, int(
            mx.sum((updated_block == self.config.mask_id).astype(mx.int32)).item()
        )


DiffusionDecoder = TopKConfidenceDecoder
