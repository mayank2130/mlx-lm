#!/usr/bin/env python3

import argparse
import json
import statistics
import time
from pathlib import Path

from mlx_lm import load, make_llada_runtime


BASE_PROMPTS = [
    "Why is the sky blue?",
    "Why do leaves look green?",
    "Explain tides in one sentence.",
    "What causes rainbows?",
    "Why is fire hot?",
    "How do clouds form?",
    "What is machine learning?",
    "How does Wi-Fi work?",
]

SYNTHETIC_TOPICS = [
    "Bitcoin",
    "blockchain",
    "the solar system",
    "photosynthesis",
    "large language models",
    "Apple Silicon",
    "MLX",
    "the water cycle",
    "electric vehicles",
    "volcanoes",
    "DNA",
    "gravity",
    "cloud computing",
    "the stock market",
    "Python",
    "neural networks",
]

SYNTHETIC_TEMPLATES = [
    "Write a short paragraph about {topic}.",
    "Explain {topic} in simple terms.",
    "Give a concise explanation of {topic}.",
    "What should a beginner know about {topic}?",
    "Summarize {topic} in two sentences.",
    "Why does {topic} matter?",
]

DEFAULT_PROFILES = {
    "fast": {"gen_length": 32, "block_length": 16, "steps": 32},
    "balanced": {"gen_length": 64, "block_length": 16, "steps": 48},
    "quality": {"gen_length": 96, "block_length": 16, "steps": 96},
}


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


def _build_prompt_pool(min_size: int) -> list[str]:
    prompts = list(BASE_PROMPTS)
    topic_idx = 0
    template_idx = 0
    while len(prompts) < min_size:
        topic = SYNTHETIC_TOPICS[topic_idx % len(SYNTHETIC_TOPICS)]
        template = SYNTHETIC_TEMPLATES[template_idx % len(SYNTHETIC_TEMPLATES)]
        prompts.append(template.format(topic=topic))
        topic_idx += 1
        template_idx += 1
    return prompts


def _adjacent_repeat_ratio(text: str) -> float:
    tokens = text.split()
    if len(tokens) < 2:
        return 0.0
    repeats = sum(tokens[i] == tokens[i - 1] for i in range(1, len(tokens)))
    return repeats / (len(tokens) - 1)


def _summarize_stats(stats):
    step_times = stats.get("step_times", []) if stats else []
    tps = stats.get("tps", []) if stats else []
    step_tps = stats.get("step_tps", []) if stats else []
    return {
        "steps_completed": stats.get("steps_completed") if stats else None,
        "blocks_completed": stats.get("blocks_completed") if stats else None,
        "model_forwards": stats.get("model_forwards") if stats else None,
        "compiled_step_hits": stats.get("compiled_step_hits") if stats else None,
        "compiled_step_misses": stats.get("compiled_step_misses") if stats else None,
        "step_time_mean_s": round(statistics.mean(step_times), 4) if step_times else None,
        "step_time_max_s": round(max(step_times), 4) if step_times else None,
        "tps_mean": round(statistics.mean(tps), 4) if tps else None,
        "step_tps_mean": round(statistics.mean(step_tps), 4) if step_tps else None,
    }


def _run_case(model, tokenizer, prompts, *, prompt_style, system_prompt, profile_name, cfg):
    runtime = make_llada_runtime(
        model,
        mode="faithful_llada",
        steps=cfg["steps"],
        gen_length=cfg["gen_length"],
        block_length=cfg["block_length"],
        compile_steps=True,
        block_local=False,
        dynamic_batching=True,
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
    result = runtime.generate(tokenizer, rendered_prompts)
    elapsed = time.perf_counter() - start

    texts = result.text if isinstance(result.text, list) else [result.text]
    text_tokens = [len(text.split()) for text in texts]
    total_output_tokens = sum(text_tokens)

    return {
        "profile": profile_name,
        "batch_size": len(prompts),
        "gen_length": cfg["gen_length"],
        "block_length": cfg["block_length"],
        "steps": cfg["steps"],
        "generation_time_s": round(elapsed, 4),
        "prompts_per_second": round(len(prompts) / elapsed, 4) if elapsed > 0 else None,
        "output_tokens_per_second": round(total_output_tokens / elapsed, 4)
        if elapsed > 0
        else None,
        "mean_output_word_count": round(statistics.mean(text_tokens), 2)
        if text_tokens
        else 0.0,
        "nonempty_rate": round(
            statistics.mean(float(bool(text.strip())) for text in texts), 4
        ),
        "mean_adjacent_repeat_ratio": round(
            statistics.mean(_adjacent_repeat_ratio(text) for text in texts), 4
        )
        if texts
        else 0.0,
        "stats": _summarize_stats(result.stats),
        "sample_outputs": [
            {"prompt": prompt, "text": text}
            for prompt, text in list(zip(prompts, texts))[: min(2, len(prompts))]
        ],
    }


def main():
    parser = argparse.ArgumentParser(
        description="Run faithful LLaDA diffusion sweeps across batch sizes and generation profiles."
    )
    parser.add_argument("model_path", help="Local MLX model path or repo id.")
    parser.add_argument(
        "--batch-sizes",
        default="1,2,4,8,16",
        help="Comma-separated batch sizes to benchmark.",
    )
    parser.add_argument(
        "--profiles",
        default="fast,balanced",
        help=f"Comma-separated profile names from: {', '.join(DEFAULT_PROFILES)}",
    )
    parser.add_argument(
        "--prompt-style",
        choices=["raw", "chat"],
        default="chat",
        help="Use raw prompts or chat-templated prompts.",
    )
    parser.add_argument(
        "--system-prompt",
        default="You are a concise helpful assistant.",
        help="System prompt for chat-style sweeps.",
    )
    parser.add_argument(
        "--prompt-count",
        type=int,
        default=None,
        help="Optional explicit prompt pool size. Defaults to the max batch size.",
    )
    parser.add_argument(
        "--output",
        default="faithful_llada_sweep.json",
        help="Where to save the JSON report.",
    )
    args = parser.parse_args()

    model_path = Path(args.model_path)
    if not model_path.exists():
        raise FileNotFoundError(model_path)

    batch_sizes = [int(part.strip()) for part in args.batch_sizes.split(",") if part.strip()]
    profiles = [part.strip() for part in args.profiles.split(",") if part.strip()]
    invalid_profiles = [profile for profile in profiles if profile not in DEFAULT_PROFILES]
    if invalid_profiles:
        raise ValueError(f"Unknown profiles: {invalid_profiles}")

    prompt_pool = _build_prompt_pool(args.prompt_count or max(batch_sizes))

    print(f"Model: {model_path}")
    print(f"Prompt style: {args.prompt_style}")
    print(f"Batch sizes: {batch_sizes}")
    print(f"Profiles: {profiles}")
    print(f"Prompt pool size: {len(prompt_pool)}")

    load_start = time.perf_counter()
    model, tokenizer = load(
        str(model_path),
        tokenizer_config={"trust_remote_code": True},
        lazy=False,
    )
    load_time = time.perf_counter() - load_start

    runs = []
    for profile_name in profiles:
        cfg = DEFAULT_PROFILES[profile_name]
        for batch_size in batch_sizes:
            prompts = prompt_pool[:batch_size]
            print(
                f"[sweep] profile={profile_name} batch_size={batch_size} "
                f"gen_length={cfg['gen_length']} block_length={cfg['block_length']} steps={cfg['steps']}"
            )
            runs.append(
                _run_case(
                    model,
                    tokenizer,
                    prompts,
                    prompt_style=args.prompt_style,
                    system_prompt=args.system_prompt,
                    profile_name=profile_name,
                    cfg=cfg,
                )
            )

    report = {
        "model_path": str(model_path),
        "load_time_s": round(load_time, 4),
        "prompt_style": args.prompt_style,
        "system_prompt": args.system_prompt if args.prompt_style == "chat" else None,
        "batch_sizes": batch_sizes,
        "profiles": {name: DEFAULT_PROFILES[name] for name in profiles},
        "prompt_pool": prompt_pool,
        "runs": runs,
        "notes": [
            "This sweep focuses on faithful_llada with compile_steps=True.",
            "Experimental block-local and dynamic diffusion modes are intentionally excluded.",
            "Prompt pool mixes built-in prompts with synthetic prompts to support larger batch sizes.",
        ],
    }

    print(json.dumps(report, indent=2, ensure_ascii=False))
    output_path = Path(args.output)
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\nSaved sweep results to {output_path.resolve()}")


if __name__ == "__main__":
    main()
