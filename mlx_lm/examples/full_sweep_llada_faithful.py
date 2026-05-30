#!/usr/bin/env python3

import argparse
import json
import statistics
import time
from pathlib import Path

import mlx.core as mx

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
    "Saturn",
    "Mars",
    "climate change",
    "wind energy",
    "quantization",
    "diffusion language models",
    "memory bandwidth",
    "computer vision",
    "compilers",
    "databases",
    "antibiotics",
    "vaccines",
    "eclipses",
    "plate tectonics",
    "evolution",
    "the internet",
]

SYNTHETIC_TEMPLATES = [
    "Write a short paragraph about {topic}.",
    "Explain {topic} in simple terms.",
    "Give a concise explanation of {topic}.",
    "What should a beginner know about {topic}?",
    "Summarize {topic} in two sentences.",
    "Why does {topic} matter?",
    "List the key ideas behind {topic}.",
    "Describe {topic} for a high-school student.",
]

CHAT_HISTORY = [
    ("What is MLX?", "MLX is a machine learning framework designed for Apple silicon."),
    ("Why is batching useful?", "Batching can improve throughput by processing multiple requests together."),
    ("What does quantization do?", "Quantization reduces model memory and compute costs by using lower-precision weights."),
    ("What is a transformer?", "A transformer is a neural network architecture built around attention mechanisms."),
    ("Why is latency important?", "Latency shapes user experience, especially in interactive applications."),
    ("What is a tokenizer?", "A tokenizer breaks text into model-friendly token units."),
]

PROFILES = {
    "short_fast": {"gen_length": 32, "block_length": 16, "steps": 32},
    "short_dense": {"gen_length": 32, "block_length": 8, "steps": 48},
    "medium_fast": {"gen_length": 64, "block_length": 32, "steps": 48},
    "medium_balanced": {"gen_length": 64, "block_length": 16, "steps": 64},
    "medium_dense": {"gen_length": 64, "block_length": 8, "steps": 96},
    "long_balanced": {"gen_length": 96, "block_length": 16, "steps": 96},
}


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


def _unique_word_ratio(text: str) -> float:
    tokens = text.split()
    if not tokens:
        return 0.0
    return len(set(tokens)) / len(tokens)


def _suspicious_start(text: str) -> bool:
    stripped = text.lstrip()
    if not stripped:
        return True
    first = stripped[0]
    if first.islower():
        return True
    if first in ",.;:)]}":
        return True
    return False


def _ends_cleanly(text: str) -> bool:
    stripped = text.rstrip()
    return bool(stripped) and stripped[-1] in ".!?"


def _render_prompt_input(
    tokenizer,
    prompt: str,
    *,
    prompt_style: str,
    system_prompt: str | None,
    chat_history_depth: int,
):
    if prompt_style == "raw":
        return prompt
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    for user_text, assistant_text in CHAT_HISTORY[:chat_history_depth]:
        messages.append({"role": "user", "content": user_text})
        messages.append({"role": "assistant", "content": assistant_text})
    messages.append({"role": "user", "content": prompt})
    rendered = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    return tokenizer.encode(rendered, add_special_tokens=False)


def _summarize_runtime_stats(stats: dict) -> dict:
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


def _run_case(
    model,
    tokenizer,
    prompts: list[str],
    *,
    prompt_style: str,
    system_prompt: str | None,
    chat_history_depth: int,
    profile_name: str,
    profile_cfg: dict,
):
    runtime = make_llada_runtime(
        model,
        mode="faithful_llada",
        steps=profile_cfg["steps"],
        gen_length=profile_cfg["gen_length"],
        block_length=profile_cfg["block_length"],
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
            chat_history_depth=chat_history_depth,
        )
        for prompt in prompts
    ]

    start = time.perf_counter()
    result = runtime.generate(tokenizer, rendered_prompts)
    elapsed = time.perf_counter() - start

    texts = result.text if isinstance(result.text, list) else [result.text]
    token_ids = result.token_ids
    if isinstance(token_ids, mx.array) and token_ids.ndim == 2:
        total_generated_tokens = int(token_ids.shape[0] * token_ids.shape[1])
    elif isinstance(token_ids, mx.array) and token_ids.ndim == 1:
        total_generated_tokens = int(token_ids.shape[0])
    else:
        total_generated_tokens = 0

    word_counts = [len(text.split()) for text in texts]
    repeat_ratios = [_adjacent_repeat_ratio(text) for text in texts]
    unique_ratios = [_unique_word_ratio(text) for text in texts]
    suspicious_starts = [float(_suspicious_start(text)) for text in texts]
    clean_endings = [float(_ends_cleanly(text)) for text in texts]
    nonempty = [float(bool(text.strip())) for text in texts]

    return {
        "profile": profile_name,
        "batch_size": len(prompts),
        "prompt_style": prompt_style,
        "chat_history_depth": chat_history_depth,
        "gen_length": profile_cfg["gen_length"],
        "block_length": profile_cfg["block_length"],
        "steps": profile_cfg["steps"],
        "generation_time_s": round(elapsed, 4),
        "prompts_per_second": round(len(prompts) / elapsed, 4) if elapsed > 0 else None,
        "generated_tokens_per_second": round(total_generated_tokens / elapsed, 4)
        if elapsed > 0
        else None,
        "mean_output_word_count": round(statistics.mean(word_counts), 2)
        if word_counts
        else 0.0,
        "nonempty_rate": round(statistics.mean(nonempty), 4) if nonempty else 0.0,
        "mean_adjacent_repeat_ratio": round(statistics.mean(repeat_ratios), 4)
        if repeat_ratios
        else 0.0,
        "mean_unique_word_ratio": round(statistics.mean(unique_ratios), 4)
        if unique_ratios
        else 0.0,
        "suspicious_start_rate": round(statistics.mean(suspicious_starts), 4)
        if suspicious_starts
        else 0.0,
        "clean_ending_rate": round(statistics.mean(clean_endings), 4)
        if clean_endings
        else 0.0,
        "stats": _summarize_runtime_stats(result.stats),
        "sample_outputs": [
            {"prompt": prompt, "text": text}
            for prompt, text in list(zip(prompts, texts))[: min(2, len(prompts))]
        ],
    }


def _parse_int_list(raw: str) -> list[int]:
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def _parse_str_list(raw: str) -> list[str]:
    return [part.strip() for part in raw.split(",") if part.strip()]


def main():
    parser = argparse.ArgumentParser(
        description="Run a broad faithful LLaDA sweep across batch size, generation shape, prompt style, and chat depth."
    )
    parser.add_argument("model_path", help="Local MLX model path or repo id.")
    parser.add_argument(
        "--batch-sizes",
        default="1,2,4,8,16,32,64",
        help="Comma-separated batch sizes for the main shape sweep.",
    )
    parser.add_argument(
        "--profiles",
        default="short_fast,short_dense,medium_fast,medium_balanced,medium_dense,long_balanced",
        help=f"Comma-separated profiles from: {', '.join(PROFILES)}",
    )
    parser.add_argument(
        "--style-batch-sizes",
        default="1,8,32",
        help="Batch sizes for the raw-vs-chat prompt-style sweep.",
    )
    parser.add_argument(
        "--depth-batch-sizes",
        default="1,8",
        help="Batch sizes for the chat-depth sweep.",
    )
    parser.add_argument(
        "--chat-history-depths",
        default="0,2,4",
        help="Comma-separated counts of prior user/assistant exchanges for chat-depth sweeps.",
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
        help="Optional prompt pool size. Defaults to the max requested batch size.",
    )
    parser.add_argument(
        "--output",
        default="full_faithful_llada_sweep.json",
        help="Where to save the JSON report.",
    )
    args = parser.parse_args()

    model_path = Path(args.model_path)
    if not model_path.exists():
        raise FileNotFoundError(model_path)

    batch_sizes = _parse_int_list(args.batch_sizes)
    style_batch_sizes = _parse_int_list(args.style_batch_sizes)
    depth_batch_sizes = _parse_int_list(args.depth_batch_sizes)
    chat_history_depths = _parse_int_list(args.chat_history_depths)
    profiles = _parse_str_list(args.profiles)
    invalid_profiles = [profile for profile in profiles if profile not in PROFILES]
    if invalid_profiles:
        raise ValueError(f"Unknown profiles: {invalid_profiles}")

    prompt_pool = _build_prompt_pool(
        args.prompt_count
        or max(batch_sizes + style_batch_sizes + depth_batch_sizes)
    )

    print(f"Model: {model_path}")
    print(f"Profiles: {profiles}")
    print(f"Main batch sizes: {batch_sizes}")
    print(f"Style sweep batch sizes: {style_batch_sizes}")
    print(f"Depth sweep batch sizes: {depth_batch_sizes}")
    print(f"Chat history depths: {chat_history_depths}")
    print(f"Prompt pool size: {len(prompt_pool)}")

    load_start = time.perf_counter()
    model, tokenizer = load(
        str(model_path),
        tokenizer_config={"trust_remote_code": True},
        lazy=False,
    )
    load_time = time.perf_counter() - load_start

    shape_runs = []
    for profile_name in profiles:
        cfg = PROFILES[profile_name]
        for batch_size in batch_sizes:
            print(
                f"[shape] profile={profile_name} batch={batch_size} "
                f"gen={cfg['gen_length']} block={cfg['block_length']} steps={cfg['steps']}"
            )
            shape_runs.append(
                _run_case(
                    model,
                    tokenizer,
                    prompt_pool[:batch_size],
                    prompt_style="chat",
                    system_prompt=args.system_prompt,
                    chat_history_depth=0,
                    profile_name=profile_name,
                    profile_cfg=cfg,
                )
            )

    style_profiles = ["short_fast", "medium_balanced", "long_balanced"]
    style_runs = []
    for profile_name in style_profiles:
        cfg = PROFILES[profile_name]
        for prompt_style in ("raw", "chat"):
            for batch_size in style_batch_sizes:
                print(
                    f"[style] profile={profile_name} style={prompt_style} batch={batch_size}"
                )
                style_runs.append(
                    _run_case(
                        model,
                        tokenizer,
                        prompt_pool[:batch_size],
                        prompt_style=prompt_style,
                        system_prompt=args.system_prompt,
                        chat_history_depth=0,
                        profile_name=profile_name,
                        profile_cfg=cfg,
                    )
                )

    depth_profiles = ["short_fast", "medium_balanced"]
    depth_runs = []
    for profile_name in depth_profiles:
        cfg = PROFILES[profile_name]
        for history_depth in chat_history_depths:
            for batch_size in depth_batch_sizes:
                print(
                    f"[depth] profile={profile_name} history_depth={history_depth} batch={batch_size}"
                )
                depth_runs.append(
                    _run_case(
                        model,
                        tokenizer,
                        prompt_pool[:batch_size],
                        prompt_style="chat",
                        system_prompt=args.system_prompt,
                        chat_history_depth=history_depth,
                        profile_name=profile_name,
                        profile_cfg=cfg,
                    )
                )

    report = {
        "model_path": str(model_path),
        "load_time_s": round(load_time, 4),
        "profiles": {name: PROFILES[name] for name in profiles},
        "batch_sizes": batch_sizes,
        "style_batch_sizes": style_batch_sizes,
        "depth_batch_sizes": depth_batch_sizes,
        "chat_history_depths": chat_history_depths,
        "system_prompt": args.system_prompt,
        "prompt_pool": prompt_pool,
        "shape_runs": shape_runs,
        "style_runs": style_runs,
        "depth_runs": depth_runs,
        "notes": [
            "This full sweep focuses on faithful_llada with compile_steps=True.",
            "Experimental block-local and dynamic diffusion modes are intentionally excluded.",
            "Shape sweep uses chat-style prompts with zero prior history.",
            "Style sweep compares raw vs chat rendering.",
            "Depth sweep measures chat-quality drift as prior conversation history grows.",
        ],
    }

    print(json.dumps(report, indent=2, ensure_ascii=False))
    output_path = Path(args.output)
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\nSaved full sweep results to {output_path.resolve()}")


if __name__ == "__main__":
    main()
