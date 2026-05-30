from dataclasses import dataclass
from typing import Literal, Optional

from .decoder import (
    DiffusionDecoderConfig,
    ThresholdDecoder,
    TopKConfidenceDecoder,
)
from .cache import ExperimentalCacheConfig
from .scheduler import (
    DiffusionSchedulerConfig,
    DynamicBlockDiffusionScheduler,
    LLaDAFixedScheduler,
)


DiffusionRuntimeMode = Literal[
    "faithful_llada",
    "experimental_block_local",
    "dynamic_block_diffusion",
    "fast_dllm_v1",
    "fast_dllm_v1_dynamic",
    "fast_dllm_v2",
]


@dataclass(frozen=True)
class DiffusionModeConfig:
    scheduler_config: DiffusionSchedulerConfig
    decoder_config: DiffusionDecoderConfig
    scheduler_cls: type
    decoder_cls: type
    block_local: bool
    cache_config: ExperimentalCacheConfig = ExperimentalCacheConfig()
    use_logits_range: bool = True
    compile_steps: bool = True
    dynamic_batching: bool = True
    cfg_scale: float = 0.0
    dual_cache: bool = False
    use_block_cache: bool = False
    sub_block_size: Optional[int] = None
    quality_gating: bool = False


def mode_config(
    mode: DiffusionRuntimeMode,
    *,
    steps: Optional[int],
    gen_length: int,
    block_length: int,
    temperature: float,
    mask_id: Optional[int],
    eos_token_id: Optional[int],
    eot_token_id: Optional[int],
    logits_eos_inf: bool,
    confidence_eos_eot_inf: bool,
    remasking: str,
    cfg_scale: float,
    threshold: Optional[float] = None,
    factor: Optional[float] = None,
) -> DiffusionModeConfig:
    decoder_config = DiffusionDecoderConfig(
        temperature=temperature,
        logits_eos_inf=logits_eos_inf,
        confidence_eos_eot_inf=confidence_eos_eot_inf,
        eos_token_id=eos_token_id,
        eot_token_id=eot_token_id,
        mask_id=mask_id,
    )
    if mode == "faithful_llada":
        if steps is None:
            raise ValueError("faithful_llada requires an explicit steps value.")
        return DiffusionModeConfig(
            scheduler_config=DiffusionSchedulerConfig(
                steps=steps,
                gen_length=gen_length,
                block_length=block_length,
            ),
            decoder_config=decoder_config,
            scheduler_cls=LLaDAFixedScheduler,
            decoder_cls=TopKConfidenceDecoder,
            block_local=False,
            cache_config=ExperimentalCacheConfig(strategy="none"),
            use_logits_range=True,
            compile_steps=True,
            dynamic_batching=True,
            cfg_scale=cfg_scale,
        )
    if mode == "experimental_block_local":
        if steps is None:
            raise ValueError("experimental_block_local requires an explicit steps value.")
        return DiffusionModeConfig(
            scheduler_config=DiffusionSchedulerConfig(
                steps=steps,
                gen_length=gen_length,
                block_length=block_length,
            ),
            decoder_config=decoder_config,
            scheduler_cls=LLaDAFixedScheduler,
            decoder_cls=TopKConfidenceDecoder,
            block_local=True,
            cache_config=ExperimentalCacheConfig(
                strategy="window",
                future_blocks=1,
            ),
            use_logits_range=True,
            compile_steps=True,
            dynamic_batching=True,
            cfg_scale=cfg_scale,
        )
    if mode == "dynamic_block_diffusion":
        return DiffusionModeConfig(
            scheduler_config=DiffusionSchedulerConfig(
                steps=steps,
                gen_length=gen_length,
                block_length=block_length,
                max_steps_per_block=steps,
                min_tokens_per_step=1,
            ),
            decoder_config=DiffusionDecoderConfig(
                **{
                    **decoder_config.__dict__,
                    "threshold": 0.35 if threshold is None else threshold,
                    "factor": factor,
                }
            ),
            scheduler_cls=DynamicBlockDiffusionScheduler,
            decoder_cls=ThresholdDecoder,
            block_local=True,
            cache_config=ExperimentalCacheConfig(
                strategy="window",
                future_blocks=1,
            ),
            use_logits_range=True,
            compile_steps=False,
            dynamic_batching=True,
            cfg_scale=cfg_scale,
        )
    if mode == "fast_dllm_v1":
        if steps is None:
            raise ValueError("fast_dllm_v1 requires an explicit steps value.")
        return DiffusionModeConfig(
            scheduler_config=DiffusionSchedulerConfig(
                steps=steps,
                gen_length=gen_length,
                block_length=block_length,
            ),
            decoder_config=DiffusionDecoderConfig(
                **{
                    **decoder_config.__dict__,
                    "threshold": 0.9 if threshold is None else threshold,
                    "factor": factor,
                }
            ),
            scheduler_cls=LLaDAFixedScheduler,
            decoder_cls=ThresholdDecoder,
            block_local=False,
            cache_config=ExperimentalCacheConfig(strategy="none"),
            use_logits_range=False,
            compile_steps=True,
            dynamic_batching=False,
            cfg_scale=cfg_scale,
            dual_cache=True,
        )
    if mode == "fast_dllm_v1_dynamic":
        if steps is None:
            raise ValueError("fast_dllm_v1_dynamic requires an explicit steps value.")
        return DiffusionModeConfig(
            scheduler_config=DiffusionSchedulerConfig(
                steps=steps,
                gen_length=gen_length,
                block_length=block_length,
            ),
            decoder_config=DiffusionDecoderConfig(
                **{
                    **decoder_config.__dict__,
                    "threshold": 0.9 if threshold is None else threshold,
                    "factor": 1.0 if factor is None else factor,
                }
            ),
            scheduler_cls=LLaDAFixedScheduler,
            decoder_cls=ThresholdDecoder,
            block_local=False,
            cache_config=ExperimentalCacheConfig(strategy="none"),
            use_logits_range=False,
            compile_steps=False,
            dynamic_batching=False,
            cfg_scale=cfg_scale,
            dual_cache=True,
        )
    if mode == "fast_dllm_v2":
        if steps is None:
            raise ValueError("fast_dllm_v2 requires an explicit steps value.")
        return DiffusionModeConfig(
            scheduler_config=DiffusionSchedulerConfig(
                steps=steps,
                gen_length=gen_length,
                block_length=block_length,
            ),
            decoder_config=DiffusionDecoderConfig(
                **{
                    **decoder_config.__dict__,
                    "threshold": 0.9 if threshold is None else threshold,
                    "factor": factor,
                }
            ),
            scheduler_cls=LLaDAFixedScheduler,
            decoder_cls=ThresholdDecoder,
            block_local=False,
            cache_config=ExperimentalCacheConfig(strategy="none"),
            use_logits_range=False,
            compile_steps=True,
            dynamic_batching=True,
            cfg_scale=cfg_scale,
            dual_cache=True,
            use_block_cache=True,
            sub_block_size=max(1, min(block_length, block_length // 2)),
            quality_gating=True,
        )
    raise ValueError(f"Unknown diffusion runtime mode: {mode}")
