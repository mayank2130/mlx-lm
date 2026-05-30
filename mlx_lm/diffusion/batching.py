from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence, Union

import mlx.core as mx

from ..tokenizer_utils import TokenizerWrapper
from .runtime import DiffusionRuntime


@dataclass(frozen=True)
class DiffusionBatchRequest:
    prompt: Union[str, Sequence[int], mx.array]
    attention_mask: Optional[Union[Sequence[int], mx.array]] = None
    request_id: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)


class ContinuousDiffusionBatcher:
    """
    Queue-driven diffusion batcher.

    Requests are enqueued and then drained in groups of compatible runtime shapes.
    Within each group, the runtime still performs active-row batching during
    refinement, so finished rows fall out of the hot path.
    """

    def __init__(self, runtime: DiffusionRuntime):
        self.runtime = runtime
        self._pending: deque[DiffusionBatchRequest] = deque()

    def enqueue(self, request: DiffusionBatchRequest):
        self._pending.append(request)

    def _batch_key(self, request: DiffusionBatchRequest):
        decoder = self.runtime.decoder
        scheduler = self.runtime.scheduler
        decoder_config = decoder.config
        return (
            self.runtime.config.mode,
            type(scheduler).__name__,
            scheduler.config.steps,
            scheduler.config.gen_length,
            scheduler.config.block_length,
            self.runtime.config.cfg_scale,
            self.runtime.config.compile_steps,
            self.runtime.config.block_local,
            self.runtime.config.dynamic_batching,
            type(decoder).__name__,
            getattr(decoder, "remasking", None),
            decoder_config.temperature,
            decoder_config.threshold,
            decoder_config.logits_eos_inf,
            decoder_config.confidence_eos_eot_inf,
            bool(request.attention_mask is not None),
        )

    def drain(
        self,
        tokenizer: TokenizerWrapper,
        *,
        max_batch_size: Optional[int] = None,
        return_full_sequence: bool = False,
        progress_callback=None,
        verbose: bool = False,
    ) -> dict[str, Any]:
        grouped = defaultdict(list)
        while self._pending:
            request = self._pending.popleft()
            grouped[self._batch_key(request)].append(request)

        results = {}
        for _, requests in grouped.items():
            for start in range(0, len(requests), max_batch_size or len(requests)):
                chunk = requests[start : start + (max_batch_size or len(requests))]
                result = self.generate(
                    tokenizer,
                    chunk,
                    return_full_sequence=return_full_sequence,
                    progress_callback=progress_callback,
                    verbose=verbose,
                )
                texts = result.text if isinstance(result.text, list) else [result.text]
                for i, request in enumerate(chunk):
                    key = request.request_id or f"request_{len(results)}"
                    results[key] = {
                        "text": texts[i],
                        "stats": result.stats,
                        "metadata": request.metadata,
                    }
        return results

    def generate(
        self,
        tokenizer: TokenizerWrapper,
        requests: Sequence[DiffusionBatchRequest],
        *,
        return_full_sequence: bool = False,
        progress_callback=None,
        verbose: bool = False,
    ):
        prompts = [request.prompt for request in requests]
        attention_masks = [
            request.attention_mask if request.attention_mask is not None else None
            for request in requests
        ]
        merged_attention = None
        if any(mask is not None for mask in attention_masks):
            if not all(mask is not None for mask in attention_masks):
                raise ValueError(
                    "Either provide attention_mask for every request or for none."
                )
            merged_attention = attention_masks
        return self.runtime.generate(
            tokenizer,
            prompts,
            attention_mask=merged_attention,
            return_full_sequence=return_full_sequence,
            progress_callback=progress_callback,
            verbose=verbose,
        )
