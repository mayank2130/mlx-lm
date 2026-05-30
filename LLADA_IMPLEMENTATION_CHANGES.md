# LLaDA Implementation Changes

This document records the code that was added or changed in the current `mlx-lm` branch to support LLaDA-style diffusion inference. It is intentionally implementation-focused and does not try to summarize model behavior beyond what the code now exposes.

## Public Entry Points

### `mlx_lm/__init__.py`

Added diffusion-facing exports:

- `llada_generate`
- `llada_generate_tokens`
- `llada_batch_generate`
- `make_llada_runtime`

These are now part of the top-level `mlx_lm` import surface.

### `mlx_lm/diffusion_generate.py`

Implemented the main LLaDA generation entry points:

- `llada_generate_tokens(...)`
- `llada_generate(...)`
- `llada_batch_generate(...)`
- `make_llada_runtime(...)`

This file no longer acts as a one-off baseline loop only. It now builds and drives the diffusion runtime subsystem, including:

- runtime construction from LLaDA-oriented arguments
- single-request generation
- batched generation
- transfer-index construction for vectorized commit selection
- compatibility with multiple runtime modes

Current behavior in this branch:

- default runtime mode is `faithful_llada`
- `use_logits_range` is available, but not enabled by default in the main path
- generation is routed through the new diffusion runtime package rather than keeping all logic inline

## Core Model Support

### `mlx_lm/models/llada.py`

Added a dedicated MLX model implementation for LLaDA.

Implemented pieces include:

- `ModelArgs`
- padding-mask helpers
- `LLaDABlock`
- `Transformer`
- `Model`

The model code now supports both full-sequence and block-aware diffusion execution paths.

Implemented model-facing features include:

- standard full-sequence forward for LLaDA logits
- `logits_range=...` support on the transformer/model path
- `supports_logits_range = True`
- block-local forward support
- prefix-state prefill support for block-local execution
- model capability flags used by the runtime:
  - `supports_logits_range`
  - `supports_block_local`
  - context-window cache support where available

Also implemented in this file:

- normalization before slicing in the `logits_range` path
- contiguous materialization of sliced hidden states before the output projection

## Diffusion Runtime Subsystem

### `mlx_lm/diffusion/__init__.py`

Added the diffusion package export surface so the runtime can be used as a small subsystem instead of a single helper file.

### `mlx_lm/diffusion/state.py`

Implemented `DiffusionState`.

This state object now owns:

- token tensor for the current refinement state
- attention mask
- prompt/frozen-token mask
- per-request prompt lengths
- prompt width
- mask token id
- per-run stats
- cached block prefix states
- optional request ids for batched execution

Implemented helpers include:

- `from_prompt_batch(...)`
- `block_tokens(...)`
- `remaining_masks(...)`
- `update_block(...)`
- `unresolved_suffix(...)`

### `mlx_lm/diffusion/scheduler.py`

Implemented scheduler classes and the compiled token-transfer helper:

- `_transfer_tokens(...)`
- `DiffusionSchedulerConfig`
- `BaseDiffusionScheduler`
- `LLaDAFixedScheduler`
- `DynamicBlockDiffusionScheduler`

What is implemented here:

- fixed LLaDA-style step scheduling
- per-block schedule derivation
- adaptive dynamic-block scheduling mode
- shared block-range and stop logic

### `mlx_lm/diffusion/decoder.py`

Implemented decoder/commit-policy logic:

- `DiffusionDecoderConfig`
- `BaseDiffusionDecoder`
- `TopKConfidenceDecoder`
- `ThresholdDecoder`
- `add_gumbel_noise(...)`
- compiled `gather_selected_probs(...)`

What is implemented here:

- token proposal from logits
- low-confidence / top-k style commit policy
- threshold-based commit policy support
- Gumbel-noise sampling helper
- chosen-token confidence gathering

### `mlx_lm/diffusion/step.py`

Implemented step execution types:

- `DiffusionStepContext`
- `DiffusionStepResult`
- `BaseDiffusionStep`
- `FullSequenceDiffusionStep`
- `BlockLocalDiffusionStep`

This file contains the code paths for:

- full-sequence model forward refinement
- block-local refinement execution
- step result packaging used by the runtime

### `mlx_lm/diffusion/cache.py`

Implemented experimental context caching support:

- `ExperimentalCacheConfig`
- `CachedBlockContext`
- `ExperimentalContextCache`

This provides cache infrastructure for block-local runtime modes, including:

- no-cache mode
- windowed context-cache behavior

### `mlx_lm/diffusion/runtime.py`

Implemented the main runtime engine:

- `DiffusionGenerationResult`
- `DiffusionRuntimeConfig`
- `DiffusionRuntime`
- prompt normalization and padding helpers

Runtime functionality added here includes:

- single and batched prompt normalization
- padded prompt-batch construction
- diffusion runtime configuration
- compiled-step caching
- full-sequence and block-local execution dispatch
- runtime stats tracking
- optional dynamic batching of active rows
- integration with decoder, scheduler, step, and cache components

### `mlx_lm/diffusion/batching.py`

Implemented batching support:

- `DiffusionBatchRequest`
- `ContinuousDiffusionBatcher`

What this adds:

- queued diffusion requests
- runtime-compatible request grouping
- request-id tracking
- batched draining into per-request results

### `mlx_lm/diffusion/modes.py`

Implemented named runtime modes:

- `faithful_llada`
- `experimental_block_local`
- `dynamic_block_diffusion`

This file adds:

- `DiffusionRuntimeMode`
- `DiffusionModeConfig`
- `mode_config(...)`

These mode configs bundle scheduler/decoder/runtime defaults for each supported diffusion mode.

### `mlx_lm/diffusion/validation.py`

Implemented validation harness code:

- `ValidationConfig`
- `ValidationCase`
- `validate_quality(...)`

This is used to compare runtime modes and preserve regression coverage while iterating on the diffusion runtime.

## CLI and User-Facing Integration

### `mlx_lm/cli.py`

Added the `diffusion_chat` subcommand to the main CLI dispatcher.

### `mlx_lm/diffusion_chat.py`

Implemented a dedicated chat CLI for diffusion models.

Added pieces include:

- diffusion-specific argument parser
- `_resolve_diffusion_chat_config(...)`
- `DiffusionChatVisualizer`
- `main()`

What this file implements:

- interactive diffusion chat loop
- mode-aware defaults for steps / gen length / block length
- optional live denoising visualization in the terminal
- runtime creation via `make_llada_runtime(...)`

### `mlx_lm/generate.py`

Integrated diffusion generation into the standard generation CLI.

Implemented additions include:

- diffusion CLI arguments:
  - `--diffusion-mode`
  - `--diffusion-steps`
  - `--diffusion-gen-length`
  - `--diffusion-block-length`
  - `--diffusion-compile-steps`
  - `--diffusion-block-local`
  - `--diffusion-dynamic-batching`
  - `--diffusion-warmup`
- routing into `llada_generate(...)` when diffusion generation is requested
- diffusion-aware warmup configuration passed through model loading

### `mlx_lm/utils.py`

Implemented diffusion-aware model warmup support in loading utilities.

Added or changed behavior includes:

- `_warmup_loaded_model(...)`
- `load(..., warmup=False, warmup_config=None, ...)`

The warmup path now supports:

- standard forward warmup
- `logits_range` warmup when supported
- block-local prefix-state prefill and block forward warmup when supported

### `mlx_lm/server.py`

Integrated diffusion generation into the server path.

Implemented pieces include:

- diffusion request configuration on generation args
- diffusion request validation
- direct `llada_generate(...)` server execution path
- diffusion runtime construction inside the server
- continuous diffusion batching support with:
  - `ContinuousDiffusionBatcher`
  - `DiffusionBatchRequest`
- diffusion request bucketing/signature matching
- progress updates during batched diffusion draining

This branch therefore adds both:

- single diffusion request handling
- grouped diffusion request handling in the server loop

## Examples and Benchmarks

Added example scripts:

- `mlx_lm/examples/benchmark_llada.py`
- `mlx_lm/examples/benchmark_llada_runtime.py`
- `mlx_lm/examples/validate_llada_quality.py`

These scripts cover:

- baseline and runtime benchmarking
- mode comparisons
- quality validation across runtime variants

## Tests

### `tests/test_diffusion_generate.py`

Added diffusion-generation tests covering:

- transfer schedule behavior
- vectorized transfer-index construction
- masked suffix filling with a fake diffusion model
- compiled full-step equivalence with eager mode
- batched generation with prompt-length tracking
- runtime mode switching
- optional-step dynamic block diffusion mode
- continuous batcher request-id preservation
- single vs batched invariance
- LLaDA model forward shape and sliced-logit regression coverage
- diffusion-aware warmup smoke coverage

### `tests/test_diffusion_chat.py`

Added diffusion chat tests covering:

- parser defaults
- default config resolution
- dynamic-mode config resolution
- system prompt/history handling
- reset behavior clearing chat history

## Summary of New Implementation Surface

The current branch adds all of the following implementation surface to `mlx-lm`:

- a dedicated MLX LLaDA model implementation
- a diffusion runtime package
- scheduler, decoder, state, step, batching, cache, mode, and validation modules
- LLaDA generation entry points
- batched diffusion generation
- diffusion chat CLI
- diffusion generation CLI flags
- diffusion-aware load warmup
- server-side diffusion handling and batching
- benchmark and validation example scripts
- diffusion test coverage
