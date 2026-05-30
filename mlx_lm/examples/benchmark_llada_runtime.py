#!/usr/bin/env python3

import argparse
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

from mlx_lm import load, make_llada_runtime


DEFAULT_PROMPTS = [
    "Why is the sky blue?",
    "Why do leaves look green?",
    "Explain tides in one sentence.",
    "What causes rainbows?",
    "Why is fire hot?",
    "How do clouds form?",
]


def _token_overlap(a: str, b: str) -> float:
    a_tokens = set(a.split())
    b_tokens = set(b.split())
    if not a_tokens and not b_tokens:
        return 1.0
    if not a_tokens or not b_tokens:
        return 0.0
    return len(a_tokens & b_tokens) / len(a_tokens | b_tokens)


def _summarize_stats(stats):
    if not stats:
        return {}
    step_times = stats.get("step_times", [])
    return {
        "steps_completed": stats.get("steps_completed"),
        "blocks_completed": stats.get("blocks_completed"),
        "stopped_early": stats.get("stopped_early"),
        "model_forwards": stats.get("model_forwards"),
        "compiled_step_hits": stats.get("compiled_step_hits"),
        "compiled_step_misses": stats.get("compiled_step_misses"),
        "step_time_mean_s": round(statistics.mean(step_times), 4) if step_times else None,
        "step_time_max_s": round(max(step_times), 4) if step_times else None,
        "active_rows_history": stats.get("active_rows_history"),
        "logits_span_history": stats.get("logits_span_history"),
    }


def _single_prompt_text(result):
    return result.text[0] if isinstance(result.text, list) else result.text


def _render_prompt_input(tokenizer, prompt, *, prompt_style, system_prompt):
    if prompt_style == "raw":
        return prompt
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    rendered = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    return tokenizer.encode(rendered, add_special_tokens=False)


def _benchmark_single_config(
    model,
    tokenizer,
    prompts,
    cfg,
    baseline_outputs,
    *,
    prompt_style,
    system_prompt,
):
    runtime = make_llada_runtime(
        model,
        mode=cfg["mode"],
        steps=cfg.get("steps"),
        gen_length=cfg["gen_length"],
        block_length=cfg["block_length"],
        compile_steps=cfg.get("compile_steps"),
        block_local=cfg.get("block_local"),
        dynamic_batching=cfg.get("dynamic_batching"),
    )
    outputs = {}
    generations = []
    case_reports = []
    stat_reports = []

    for prompt in prompts:
        rendered_prompt = _render_prompt_input(
            tokenizer,
            prompt,
            prompt_style=prompt_style,
            system_prompt=system_prompt,
        )
        start = time.perf_counter()
        result = runtime.generate(
            tokenizer,
            rendered_prompt,
        )
        generations.append(time.perf_counter() - start)
        text = _single_prompt_text(result)
        outputs[prompt] = text
        stat_reports.append(_summarize_stats(result.stats))

        faithful_text = baseline_outputs[prompt]
        case_reports.append(
            {
                "prompt": prompt,
                "output": text,
                "exact_match": text == faithful_text,
                "nonempty": bool(text.strip()),
                "token_overlap": round(_token_overlap(faithful_text, text), 4),
            }
        )

    return {
        "name": cfg["name"],
        "mode": cfg["mode"],
        "kind": cfg["kind"],
        "steps": cfg.get("steps"),
        "gen_length": cfg["gen_length"],
        "block_length": cfg["block_length"],
        "compile_steps": cfg.get("compile_steps"),
        "block_local": cfg.get("block_local"),
        "dynamic_batching": cfg.get("dynamic_batching"),
        "mean_generation_time_s": round(statistics.mean(generations), 4),
        "max_generation_time_s": round(max(generations), 4),
        "exact_match_rate": round(
            statistics.mean(float(case["exact_match"]) for case in case_reports), 4
        ),
        "nonempty_rate": round(
            statistics.mean(float(case["nonempty"]) for case in case_reports), 4
        ),
        "mean_token_overlap": round(
            statistics.mean(case["token_overlap"] for case in case_reports), 4
        ),
        "cases": case_reports,
        "stats": stat_reports,
        "outputs": outputs,
    }


def _benchmark_batch_config(model, tokenizer, prompts, cfg, *, prompt_style, system_prompt):
    runtime = make_llada_runtime(
        model,
        mode=cfg["mode"],
        steps=cfg.get("steps"),
        gen_length=cfg["gen_length"],
        block_length=cfg["block_length"],
        compile_steps=cfg.get("compile_steps"),
        block_local=cfg.get("block_local"),
        dynamic_batching=cfg.get("dynamic_batching"),
    )
    rendered_prompts = [
        _render_prompt_input(
            tokenizer,
            prompt,
            prompt_style=prompt_style,
            system_prompt=system_prompt,
        )
        for prompt in prompts
    ]
    start = time.perf_counter()
    result = runtime.generate(
        tokenizer,
        rendered_prompts,
    )
    generation_time = time.perf_counter() - start
    texts = result.text if isinstance(result.text, list) else [result.text]
    return {
        "name": cfg["name"],
        "mode": cfg["mode"],
        "kind": cfg["kind"],
        "steps": cfg.get("steps"),
        "gen_length": cfg["gen_length"],
        "block_length": cfg["block_length"],
        "compile_steps": cfg.get("compile_steps"),
        "block_local": cfg.get("block_local"),
        "dynamic_batching": cfg.get("dynamic_batching"),
        "batch_size": len(prompts),
        "generation_time_s": round(generation_time, 4),
        "texts": texts,
        "stats": _summarize_stats(result.stats),
    }


def _run_cli_faithful_parity(
    model_path,
    prompts,
    baseline_cfg,
    baseline_outputs,
    *,
    prompt_style,
    system_prompt,
):
    reports = []
    repo_root = Path(__file__).resolve().parents[2]
    for prompt in prompts:
        cmd = [
            sys.executable,
            "-m",
            "mlx_lm",
            "generate",
            "--model",
            str(model_path),
            "--trust-remote-code",
            "--prompt",
            prompt,
            "--diffusion-mode",
            baseline_cfg["mode"],
            "--diffusion-steps",
            str(baseline_cfg["steps"]),
            "--diffusion-gen-length",
            str(baseline_cfg["gen_length"]),
            "--diffusion-block-length",
            str(baseline_cfg["block_length"]),
            "--diffusion-compile-steps",
            "true" if baseline_cfg.get("compile_steps") else "false",
            "--diffusion-block-local",
            "true" if baseline_cfg.get("block_local") else "false",
            "--diffusion-dynamic-batching",
            "true" if baseline_cfg.get("dynamic_batching") else "false",
            "--verbose",
            "false",
        ]
        if prompt_style == "raw":
            cmd.append("--ignore-chat-template")
        elif system_prompt:
            cmd.extend(["--system-prompt", system_prompt])

        start = time.perf_counter()
        proc = subprocess.run(
            cmd,
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
        elapsed = time.perf_counter() - start
        cli_text = proc.stdout.strip()
        direct_text = baseline_outputs[prompt].strip()
        reports.append(
            {
                "prompt": prompt,
                "cli_output": cli_text,
                "direct_output": direct_text,
                "exact_match": cli_text == direct_text,
                "token_overlap": round(_token_overlap(direct_text, cli_text), 4),
                "generation_time_s": round(elapsed, 4),
            }
        )
    return {
        "mode": baseline_cfg["mode"],
        "prompt_style": prompt_style,
        "exact_match_rate": round(
            statistics.mean(float(report["exact_match"]) for report in reports), 4
        ),
        "mean_token_overlap": round(
            statistics.mean(report["token_overlap"] for report in reports), 4
        ),
        "mean_generation_time_s": round(
            statistics.mean(report["generation_time_s"] for report in reports), 4
        ),
        "cases": reports,
    }


def _default_single_configs(gen_length, block_length):
    return [
        {
            "name": "faithful_eager_single",
            "mode": "faithful_llada",
            "kind": "faithful",
            "steps": min(gen_length, 8),
            "gen_length": gen_length,
            "block_length": block_length,
            "compile_steps": False,
            "block_local": False,
            "dynamic_batching": True,
        },
        {
            "name": "faithful_compiled_single",
            "mode": "faithful_llada",
            "kind": "faithful",
            "steps": min(gen_length, 8),
            "gen_length": gen_length,
            "block_length": block_length,
            "compile_steps": True,
            "block_local": False,
            "dynamic_batching": True,
        },
        {
            "name": "experimental_block_local_single",
            "mode": "experimental_block_local",
            "kind": "experimental",
            "steps": min(gen_length, 8),
            "gen_length": gen_length,
            "block_length": max(1, block_length // 2),
            "compile_steps": True,
            "block_local": True,
            "dynamic_batching": True,
        },
        {
            "name": "dynamic_block_diffusion_single",
            "mode": "dynamic_block_diffusion",
            "kind": "experimental",
            "steps": None,
            "gen_length": gen_length,
            "block_length": max(1, block_length // 2),
            "compile_steps": False,
            "block_local": True,
            "dynamic_batching": True,
        },
    ]


def _default_batch_configs(gen_length, block_length):
    return [
        {
            "name": "faithful_compiled_batch",
            "mode": "faithful_llada",
            "kind": "faithful",
            "steps": min(gen_length, 8),
            "gen_length": gen_length,
            "block_length": block_length,
            "compile_steps": True,
            "block_local": False,
            "dynamic_batching": True,
        },
        {
            "name": "dynamic_block_diffusion_batch",
            "mode": "dynamic_block_diffusion",
            "kind": "experimental",
            "steps": None,
            "gen_length": gen_length,
            "block_length": max(1, block_length // 2),
            "compile_steps": False,
            "block_local": True,
            "dynamic_batching": True,
        },
    ]


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark no-cache LLaDA diffusion runtime variants for speed and quality."
    )
    parser.add_argument("model_path", help="Local MLX model path or repo id.")
    parser.add_argument(
        "--prompt",
        action="append",
        dest="prompts",
        help="Prompt to benchmark. Repeat to add multiple prompts.",
    )
    parser.add_argument(
        "--gen-length",
        type=int,
        default=8,
        help="Static generation length budget.",
    )
    parser.add_argument(
        "--block-length",
        type=int,
        default=8,
        help="Static block length for faithful mode.",
    )
    parser.add_argument(
        "--output",
        default="benchmark_llada_runtime.json",
        help="Where to save the JSON report.",
    )
    parser.add_argument(
        "--prompt-style",
        choices=["raw", "chat"],
        default="raw",
        help="Benchmark raw prompts or chat-templated prompts.",
    )
    parser.add_argument(
        "--system-prompt",
        default="You are a concise helpful assistant.",
        help="System prompt to use when --prompt-style chat is selected.",
    )
    parser.add_argument(
        "--include-cli-parity",
        action="store_true",
        help="Also compare direct faithful output against the mlx_lm generate CLI path.",
    )
    args = parser.parse_args()

    model_path = Path(args.model_path)
    if not model_path.exists():
        raise FileNotFoundError(model_path)

    prompts = args.prompts or list(DEFAULT_PROMPTS)

    print(f"Model: {model_path}")
    print(f"Prompts: {prompts}")
    print(
        f"No-cache benchmark with gen_length={args.gen_length}, block_length={args.block_length}, prompt_style={args.prompt_style}"
    )

    load_start = time.perf_counter()
    model, tokenizer = load(
        str(model_path),
        tokenizer_config={"trust_remote_code": True},
        lazy=False,
    )
    load_time = time.perf_counter() - load_start

    single_configs = _default_single_configs(args.gen_length, args.block_length)
    batch_configs = _default_batch_configs(args.gen_length, args.block_length)

    baseline_cfg = next(cfg for cfg in single_configs if cfg["name"] == "faithful_compiled_single")
    baseline_runtime = make_llada_runtime(
        model,
        mode=baseline_cfg["mode"],
        steps=baseline_cfg["steps"],
        gen_length=baseline_cfg["gen_length"],
        block_length=baseline_cfg["block_length"],
        compile_steps=baseline_cfg["compile_steps"],
        block_local=baseline_cfg["block_local"],
        dynamic_batching=baseline_cfg["dynamic_batching"],
    )
    baseline_outputs = {}
    for prompt in prompts:
        rendered_prompt = _render_prompt_input(
            tokenizer,
            prompt,
            prompt_style=args.prompt_style,
            system_prompt=args.system_prompt,
        )
        baseline_result = baseline_runtime.generate(
            tokenizer,
            rendered_prompt,
        )
        baseline_outputs[prompt] = _single_prompt_text(baseline_result)

    single_reports = [
        _benchmark_single_config(
            model,
            tokenizer,
            prompts,
            cfg,
            baseline_outputs,
            prompt_style=args.prompt_style,
            system_prompt=args.system_prompt,
        )
        for cfg in single_configs
    ]
    batch_reports = [
        _benchmark_batch_config(
            model,
            tokenizer,
            prompts[:2],
            cfg,
            prompt_style=args.prompt_style,
            system_prompt=args.system_prompt,
        )
        for cfg in batch_configs
    ]
    cli_parity = (
        _run_cli_faithful_parity(
            model_path,
            prompts,
            baseline_cfg,
            baseline_outputs,
            prompt_style=args.prompt_style,
            system_prompt=args.system_prompt,
        )
        if args.include_cli_parity
        else None
    )

    report = {
        "model_path": str(model_path),
        "load_time_s": round(load_time, 4),
        "prompts": prompts,
        "gen_length": args.gen_length,
        "block_length": args.block_length,
        "prompt_style": args.prompt_style,
        "system_prompt": args.system_prompt if args.prompt_style == "chat" else None,
        "baseline_outputs": baseline_outputs,
        "single_configs": single_reports,
        "batch_configs": batch_reports,
        "cli_parity": cli_parity,
        "notes": [
            "This benchmark intentionally excludes KV/cache policy work.",
            "Quality is measured against faithful_compiled_single outputs.",
            "dynamic_block_diffusion uses steps=None so refinement count is scheduler-driven.",
        ],
    }

    print(json.dumps(report, indent=2, ensure_ascii=False))
    output_path = Path(args.output)
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\nSaved results to {output_path.resolve()}")


if __name__ == "__main__":
    main()
