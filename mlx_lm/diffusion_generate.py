# Copyright © 2023-2024 Apple Inc.

from typing import Any, Callable, Optional, Sequence, Union

import mlx.core as mx
import mlx.nn as nn
from transformers import PreTrainedTokenizer

from .diffusion import (
    ContinuousDiffusionBatcher,
    DiffusionBatchRequest,
    DiffusionGenerationResult,
    DiffusionRuntime,
    DiffusionRuntimeConfig,
    DiffusionRuntimeMode,
    mode_config,
)
from .diffusion.scheduler import _transfer_tokens as get_num_transfer_tokens
from .tokenizer_utils import TokenizerWrapper


def _build_transfer_index(confidence: mx.array, k_per_batch: mx.array) -> mx.array:
    max_k = int(mx.max(k_per_batch).item())
    if max_k <= 0:
        return mx.zeros(confidence.shape, dtype=mx.bool_)
    candidate_index = mx.argpartition(-confidence, kth=max_k - 1, axis=-1)[:, :max_k]
    selection_order = mx.arange(max_k, dtype=k_per_batch.dtype)[None, :]
    selection_mask = selection_order < k_per_batch[:, None]
    return mx.put_along_axis(
        mx.zeros(confidence.shape, dtype=mx.bool_),
        candidate_index,
        selection_mask,
        axis=-1,
    )


def render_diffusion_chat_prompt(
    tokenizer: Union[PreTrainedTokenizer, TokenizerWrapper],
    messages: Sequence[dict[str, str]],
) -> str:
    if not isinstance(tokenizer, TokenizerWrapper):
        tokenizer = TokenizerWrapper(tokenizer)
    return tokenizer.apply_chat_template(
        list(messages),
        tokenize=False,
        add_generation_prompt=True,
    )


def encode_diffusion_chat_prompt(
    tokenizer: Union[PreTrainedTokenizer, TokenizerWrapper],
    messages: Sequence[dict[str, str]],
) -> mx.array:
    if not isinstance(tokenizer, TokenizerWrapper):
        tokenizer = TokenizerWrapper(tokenizer)
    rendered = render_diffusion_chat_prompt(tokenizer, messages)
    return mx.array(
        tokenizer.encode(rendered, add_special_tokens=False),
        dtype=mx.int32,
    )


def _make_runtime(
    model: nn.Module,
    *,
    mode: DiffusionRuntimeMode,
    steps: Optional[int],
    gen_length: int,
    block_length: int,
    temperature: float,
    cfg_scale: float,
    remasking: str,
    mask_id: Optional[int],
    logits_eos_inf: bool,
    confidence_eos_eot_inf: bool,
    eos_token_id: Optional[int],
    eot_token_id: Optional[int],
    use_logits_range: Optional[bool],
    compile_steps: Optional[bool],
    block_local: Optional[bool],
    dynamic_batching: Optional[bool] = None,
    threshold: Optional[float] = None,
    factor: Optional[float] = None,
) -> DiffusionRuntime:
    if mask_id is None:
        mask_id = getattr(getattr(model, "args", None), "mask_token_id", None)

    if block_local and mode == "faithful_llada":
        mode = "experimental_block_local"

    selected_mode = mode_config(
        mode,
        steps=steps,
        gen_length=gen_length,
        block_length=block_length,
        temperature=temperature,
        mask_id=mask_id,
        eos_token_id=eos_token_id,
        eot_token_id=eot_token_id,
        logits_eos_inf=logits_eos_inf,
        confidence_eos_eot_inf=confidence_eos_eot_inf,
        remasking=remasking,
        cfg_scale=cfg_scale,
        threshold=threshold,
        factor=factor,
    )

    decoder_kwargs = {}
    if selected_mode.decoder_cls.__name__ in {"TopKConfidenceDecoder", "ThresholdDecoder"}:
        decoder_kwargs["remasking"] = remasking
    decoder = selected_mode.decoder_cls(selected_mode.decoder_config, **decoder_kwargs)
    scheduler = selected_mode.scheduler_cls(selected_mode.scheduler_config)

    config = DiffusionRuntimeConfig(
        mode=mode,
        use_logits_range=(
            selected_mode.use_logits_range if use_logits_range is None else use_logits_range
        ),
        compile_steps=(
            selected_mode.compile_steps if compile_steps is None else compile_steps
        ),
        block_local=selected_mode.block_local if block_local is None else block_local,
        dual_cache=selected_mode.dual_cache,
        dynamic_batching=(
            selected_mode.dynamic_batching
            if dynamic_batching is None
            else dynamic_batching
        ),
        cfg_scale=selected_mode.cfg_scale,
        cache_strategy=selected_mode.cache_config.strategy,
        cache_future_blocks=selected_mode.cache_config.future_blocks,
        cache_past_blocks=selected_mode.cache_config.past_blocks,
        use_block_cache=selected_mode.use_block_cache,
        sub_block_size=selected_mode.sub_block_size,
        quality_gating=selected_mode.quality_gating,
    )
    return DiffusionRuntime(model, scheduler=scheduler, decoder=decoder, config=config)


def make_llada_runtime(
    model: nn.Module,
    *,
    mode: DiffusionRuntimeMode = "faithful_llada",
    steps: Optional[int] = 128,
    gen_length: int = 128,
    block_length: int = 128,
    temperature: float = 0.0,
    cfg_scale: float = 0.0,
    remasking: str = "low_confidence",
    threshold: Optional[float] = None,
    factor: Optional[float] = None,
    mask_id: Optional[int] = None,
    logits_eos_inf: bool = False,
    confidence_eos_eot_inf: bool = False,
    eos_token_id: Optional[int] = None,
    eot_token_id: Optional[int] = None,
    use_logits_range: Optional[bool] = None,
    compile_steps: Optional[bool] = None,
    block_local: Optional[bool] = None,
    dynamic_batching: Optional[bool] = None,
) -> DiffusionRuntime:
    return _make_runtime(
        model,
        mode=mode,
        steps=steps,
        gen_length=gen_length,
        block_length=block_length,
        temperature=temperature,
        cfg_scale=cfg_scale,
        remasking=remasking,
        threshold=threshold,
        factor=factor,
        mask_id=mask_id,
        logits_eos_inf=logits_eos_inf,
        confidence_eos_eot_inf=confidence_eos_eot_inf,
        eos_token_id=eos_token_id,
        eot_token_id=eot_token_id,
        use_logits_range=use_logits_range,
        compile_steps=compile_steps,
        block_local=block_local,
        dynamic_batching=dynamic_batching,
    )


def llada_generate_tokens(
    model: nn.Module,
    prompt: mx.array,
    *,
    mode: DiffusionRuntimeMode = "faithful_llada",
    attention_mask: Optional[mx.array] = None,
    steps: Optional[int] = 128,
    gen_length: int = 128,
    block_length: int = 128,
    temperature: float = 0.0,
    cfg_scale: float = 0.0,
    remasking: str = "low_confidence",
    threshold: Optional[float] = None,
    factor: Optional[float] = None,
    mask_id: Optional[int] = None,
    logits_eos_inf: bool = False,
    confidence_eos_eot_inf: bool = False,
    eos_token_id: Optional[int] = None,
    eot_token_id: Optional[int] = None,
    use_logits_range: Optional[bool] = None,
    compile_steps: Optional[bool] = None,
    block_local: Optional[bool] = None,
    dynamic_batching: Optional[bool] = None,
    verbose: bool = False,
    progress_callback: Optional[Callable[[dict[str, Any]], None]] = None,
    return_stats: bool = False,
) -> mx.array:
    runtime = _make_runtime(
        model,
        mode=mode,
        steps=steps,
        gen_length=gen_length,
        block_length=block_length,
        temperature=temperature,
        cfg_scale=cfg_scale,
        remasking=remasking,
        threshold=threshold,
        factor=factor,
        mask_id=mask_id,
        logits_eos_inf=logits_eos_inf,
        confidence_eos_eot_inf=confidence_eos_eot_inf,
        eos_token_id=eos_token_id,
        eot_token_id=eot_token_id,
        use_logits_range=use_logits_range,
        compile_steps=compile_steps,
        block_local=block_local,
        dynamic_batching=dynamic_batching,
    )
    return runtime.generate_tokens(
        prompt,
        attention_mask=attention_mask,
        progress_callback=progress_callback,
        verbose=verbose,
        return_stats=return_stats,
    )


def llada_generate(
    model: nn.Module,
    tokenizer: Union[PreTrainedTokenizer, TokenizerWrapper],
    prompt: Union[str, Sequence[int], Sequence[str], Sequence[Sequence[int]], mx.array],
    *,
    mode: DiffusionRuntimeMode = "faithful_llada",
    attention_mask: Optional[Union[Sequence[int], Sequence[Sequence[int]], mx.array]] = None,
    steps: Optional[int] = 128,
    gen_length: int = 128,
    block_length: int = 128,
    temperature: float = 0.0,
    cfg_scale: float = 0.0,
    remasking: str = "low_confidence",
    threshold: Optional[float] = None,
    factor: Optional[float] = None,
    mask_id: Optional[int] = None,
    logits_eos_inf: bool = False,
    confidence_eos_eot_inf: bool = False,
    eos_token_id: Optional[int] = None,
    eot_token_id: Optional[int] = None,
    return_full_sequence: bool = False,
    use_logits_range: Optional[bool] = None,
    compile_steps: Optional[bool] = None,
    block_local: Optional[bool] = None,
    dynamic_batching: Optional[bool] = None,
    verbose: bool = False,
    progress_callback: Optional[Callable[[dict[str, Any]], None]] = None,
) -> DiffusionGenerationResult:
    runtime = _make_runtime(
        model,
        mode=mode,
        steps=steps,
        gen_length=gen_length,
        block_length=block_length,
        temperature=temperature,
        cfg_scale=cfg_scale,
        remasking=remasking,
        threshold=threshold,
        factor=factor,
        mask_id=mask_id,
        logits_eos_inf=logits_eos_inf,
        confidence_eos_eot_inf=confidence_eos_eot_inf,
        eos_token_id=eos_token_id,
        eot_token_id=eot_token_id,
        use_logits_range=use_logits_range,
        compile_steps=compile_steps,
        block_local=block_local,
        dynamic_batching=dynamic_batching,
    )
    return runtime.generate(
        tokenizer,
        prompt,
        attention_mask=attention_mask,
        return_full_sequence=return_full_sequence,
        progress_callback=progress_callback,
        verbose=verbose,
    )


def llada_batch_generate(
    model: nn.Module,
    tokenizer: Union[PreTrainedTokenizer, TokenizerWrapper],
    prompts: Sequence[Union[str, Sequence[int], mx.array]],
    **kwargs,
) -> DiffusionGenerationResult:
    attention_masks = kwargs.pop("attention_mask", None)
    runtime = _make_runtime(
        model,
        mode=kwargs.pop("mode", "faithful_llada"),
        steps=kwargs.pop("steps", 128),
        gen_length=kwargs.pop("gen_length", 128),
        block_length=kwargs.pop("block_length", 128),
        temperature=kwargs.pop("temperature", 0.0),
        cfg_scale=kwargs.pop("cfg_scale", 0.0),
        remasking=kwargs.pop("remasking", "low_confidence"),
        threshold=kwargs.pop("threshold", None),
        factor=kwargs.pop("factor", None),
        mask_id=kwargs.pop("mask_id", None),
        logits_eos_inf=kwargs.pop("logits_eos_inf", False),
        confidence_eos_eot_inf=kwargs.pop("confidence_eos_eot_inf", False),
        eos_token_id=kwargs.pop("eos_token_id", None),
        eot_token_id=kwargs.pop("eot_token_id", None),
        use_logits_range=kwargs.pop("use_logits_range", None),
        compile_steps=kwargs.pop("compile_steps", None),
        block_local=kwargs.pop("block_local", None),
        dynamic_batching=kwargs.pop("dynamic_batching", None),
    )
    if not isinstance(tokenizer, TokenizerWrapper):
        tokenizer = TokenizerWrapper(tokenizer)
    return_full_sequence = kwargs.pop("return_full_sequence", False)
    progress_callback = kwargs.pop("progress_callback", None)
    verbose = kwargs.pop("verbose", False)

    batcher = ContinuousDiffusionBatcher(runtime)
    requests = []
    for idx, prompt in enumerate(prompts):
        attention_mask = None if attention_masks is None else attention_masks[idx]
        request = DiffusionBatchRequest(
            prompt=prompt,
            attention_mask=attention_mask,
            request_id=f"batch_{idx}",
        )
        batcher.enqueue(request)
        requests.append(request)

    return batcher.generate(
        tokenizer,
        requests,
        return_full_sequence=return_full_sequence,
        progress_callback=progress_callback,
        verbose=verbose,
    )
