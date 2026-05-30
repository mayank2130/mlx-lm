#!/usr/bin/env python3

import argparse
import json
import statistics
import time
from pathlib import Path

from mlx_lm import load, llada_batch_generate, llada_generate


DEFAULT_PROMPT = "Why is the sky blue?"
DEFAULT_BATCH_PROMPTS = [
    "Why is the sky blue?",
    "Why do leaves look green?",
]


def _summarize_stats(stats):
    step_times = stats.get("step_times", []) if stats else []
    return {
        "steps_completed": stats.get("steps_completed") if stats else None,
        "blocks_completed": stats.get("blocks_completed") if stats else None,
        "stopped_early": stats.get("stopped_early") if stats else None,
        "masks_remaining": stats.get("masks_remaining") if stats else None,
        "prompt_lengths": stats.get("prompt_lengths") if stats else None,
        "step_time_mean_s": round(statistics.mean(step_times), 4) if step_times else None,
        "step_time_max_s": round(max(step_times), 4) if step_times else None,
    }


def _build_configs(include_experimental):
    configs = [
        {
            "name": "baseline_single",
            "batch": False,
            "load_kwargs": {
                "lazy": False,
                "warmup": False,
                "tokenizer_config": {"trust_remote_code": True},
            },
            "generate_kwargs": {
                "steps": 2,
                "gen_length": 8,
                "block_length": 8,
                "temperature": 0.0,
                "cfg_scale": 0.0,
                "compile_steps": False,
                "block_local": False,
            },
            "kind": "faithful",
        },
        {
            "name": "compiled_single",
            "batch": False,
            "load_kwargs": {
                "lazy": False,
                "warmup": False,
                "tokenizer_config": {"trust_remote_code": True},
            },
            "generate_kwargs": {
                "steps": 2,
                "gen_length": 8,
                "block_length": 8,
                "temperature": 0.0,
                "cfg_scale": 0.0,
                "compile_steps": True,
                "block_local": False,
            },
            "kind": "faithful",
        },
        {
            "name": "compiled_batch2",
            "batch": True,
            "load_kwargs": {
                "lazy": False,
                "warmup": False,
                "tokenizer_config": {"trust_remote_code": True},
            },
            "generate_kwargs": {
                "steps": 2,
                "gen_length": 8,
                "block_length": 8,
                "temperature": 0.0,
                "cfg_scale": 0.0,
                "compile_steps": True,
                "block_local": False,
            },
            "kind": "faithful",
        },
    ]

    if include_experimental:
        configs.extend(
            [
                {
                    "name": "compiled_block_local_single",
                    "batch": False,
                    "load_kwargs": {
                        "lazy": False,
                        "warmup": True,
                        "warmup_config": {
                            "sequence_length": 16,
                            "block_length": 4,
                            "max_tokens": 16,
                        },
                        "tokenizer_config": {"trust_remote_code": True},
                    },
                    "generate_kwargs": {
                        "steps": 2,
                        "gen_length": 8,
                        "block_length": 4,
                        "temperature": 0.0,
                        "cfg_scale": 0.0,
                        "compile_steps": True,
                        "block_local": True,
                    },
                    "kind": "experimental",
                },
                {
                    "name": "compiled_block_local_batch2",
                    "batch": True,
                    "load_kwargs": {
                        "lazy": False,
                        "warmup": True,
                        "warmup_config": {
                            "sequence_length": 16,
                            "block_length": 4,
                            "max_tokens": 16,
                            "batch_size": 2,
                        },
                        "tokenizer_config": {"trust_remote_code": True},
                    },
                    "generate_kwargs": {
                        "steps": 2,
                        "gen_length": 8,
                        "block_length": 4,
                        "temperature": 0.0,
                        "cfg_scale": 0.0,
                        "compile_steps": True,
                        "block_local": True,
                    },
                    "kind": "experimental",
                },
            ]
        )

    return configs


def _run_config(model_path, prompt, batch_prompts, cfg):
    print(f"\n=== {cfg['name']} ({cfg['kind']}) ===")

    t0 = time.perf_counter()
    model, tokenizer = load(model_path, **cfg["load_kwargs"])
    load_time = time.perf_counter() - t0

    t1 = time.perf_counter()
    if cfg["batch"]:
        result = llada_batch_generate(
            model,
            tokenizer,
            batch_prompts,
            **cfg["generate_kwargs"],
        )
    else:
        result = llada_generate(
            model,
            tokenizer,
            prompt,
            **cfg["generate_kwargs"],
        )
    generation_time = time.perf_counter() - t1

    record = {
        "name": cfg["name"],
        "kind": cfg["kind"],
        "batch": cfg["batch"],
        "load_time_s": round(load_time, 4),
        "generation_time_s": round(generation_time, 4),
        "total_time_s": round(load_time + generation_time, 4),
        "text": result.text,
        "stats": _summarize_stats(result.stats),
    }
    print(json.dumps(record, indent=2, ensure_ascii=False))
    return record


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark LLaDA diffusion inference paths in mlx-lm."
    )
    parser.add_argument("model_path", help="Local MLX model path or repo id.")
    parser.add_argument(
        "--prompt",
        default=DEFAULT_PROMPT,
        help="Single-prompt benchmark input.",
    )
    parser.add_argument(
        "--batch-prompt",
        action="append",
        dest="batch_prompts",
        help="Batch prompt. Repeat this flag to supply multiple prompts.",
    )
    parser.add_argument(
        "--include-experimental",
        action="store_true",
        help="Include block-local benchmark configs. These are faster paths but may not match full-sequence baseline semantics.",
    )
    parser.add_argument(
        "--output",
        default="benchmark_llada_results.json",
        help="Where to save the JSON benchmark report.",
    )
    args = parser.parse_args()

    model_path = Path(args.model_path)
    if not model_path.exists():
        raise FileNotFoundError(model_path)

    batch_prompts = args.batch_prompts or list(DEFAULT_BATCH_PROMPTS)
    configs = _build_configs(args.include_experimental)

    print(f"Model: {model_path}")
    print(f"Prompt: {args.prompt}")
    print(f"Batch prompts: {batch_prompts}")

    results = [
        _run_config(model_path, args.prompt, batch_prompts, cfg) for cfg in configs
    ]
    output_path = Path(args.output)
    output_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\nSaved results to {output_path.resolve()}")


if __name__ == "__main__":
    main()
