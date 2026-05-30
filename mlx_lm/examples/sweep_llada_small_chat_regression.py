#!/usr/bin/env python3

import argparse
import json
import sys
import time
from pathlib import Path

from mlx_lm import load
from mlx_lm.diffusion_chat import DiffusionChatVisualizer
from mlx_lm.diffusion_generate import encode_diffusion_chat_prompt, llada_generate


DEFAULT_CONFIGS = [
    {"gen_length": 64, "block_length": 64, "steps": 256},
    {"gen_length": 128, "block_length": 32, "steps": 128},
    {"gen_length": 256, "block_length": 32, "steps": 256},
]

DEFAULT_MODES = [
    {"mode": "faithful_llada", "extra": {}},
    {
        "mode": "fast_dllm_v1",
        "extra": {"threshold": 0.95, "remasking": "low_confidence"},
    },
    {
        "mode": "fast_dllm_v1_dynamic",
        "extra": {"factor": 0.5, "remasking": "low_confidence"},
    },
]


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Run a small chat-templated diffusion regression sweep that mirrors "
            "the prompt formatting used by diffusion_chat."
        )
    )
    parser.add_argument("model_path", help="Local MLX model path or repo id.")
    parser.add_argument(
        "--prompts",
        required=True,
        help="Path to a JSON file containing a list of user prompts.",
    )
    parser.add_argument(
        "--output",
        default="fastdllm_small_chat_regression.json",
        help="Where to save the incremental JSON report.",
    )
    parser.add_argument(
        "--show-progress",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Render the live denoising visualization in the terminal for each run. "
            "Uses the same progress callback path as diffusion_chat."
        ),
    )
    args = parser.parse_args()

    prompts_path = Path(args.prompts)
    if not prompts_path.exists():
        raise FileNotFoundError(prompts_path)
    prompts = json.loads(prompts_path.read_text())

    output_path = Path(args.output)
    if output_path.exists():
        results = json.loads(output_path.read_text())
    else:
        results = []

    done = {
        (r["mode"], r["gen_length"], r["block_length"], r["steps"], r["prompt_index"])
        for r in results
    }

    model, tokenizer = load(
        str(args.model_path),
        tokenizer_config={"trust_remote_code": True},
    )

    for cfg in DEFAULT_CONFIGS:
        for mode_cfg in DEFAULT_MODES:
            mode = mode_cfg["mode"]
            extra = mode_cfg["extra"]

            for prompt_index, user_prompt in enumerate(prompts):
                key = (
                    mode,
                    cfg["gen_length"],
                    cfg["block_length"],
                    cfg["steps"],
                    prompt_index,
                )
                if key in done:
                    print("SKIP", key)
                    continue

                prompt_ids = encode_diffusion_chat_prompt(
                    tokenizer,
                    [{"role": "user", "content": user_prompt}],
                )

                print(
                    "\nRUN",
                    {
                        "mode": mode,
                        **cfg,
                        "prompt_index": prompt_index,
                    },
                )
                print("USER PROMPT:", user_prompt)

                visualizer = DiffusionChatVisualizer(
                    tokenizer,
                    enabled=bool(args.show_progress and sys.stdout.isatty()),
                )
                t0 = time.perf_counter()
                try:
                    result = llada_generate(
                        model,
                        tokenizer,
                        prompt_ids,
                        mode=mode,
                        steps=cfg["steps"],
                        gen_length=cfg["gen_length"],
                        block_length=cfg["block_length"],
                        temperature=0.0,
                        progress_callback=visualizer.update,
                        verbose=False,
                        **extra,
                    )
                finally:
                    visualizer.finish()
                dt = time.perf_counter() - t0

                token_ids = (
                    result.token_ids[0].tolist()
                    if result.token_ids.ndim == 2
                    else result.token_ids.tolist()
                )
                raw_text = tokenizer.decode(
                    token_ids,
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=False,
                )
                clean_text = result.text[0] if isinstance(result.text, list) else result.text
                stats = result.stats or {}

                print("TOKEN_IDS_HEAD:", token_ids[:40])
                print("RAW_OUTPUT_REPR:", repr(raw_text[:500]))
                print("CLEAN_OUTPUT:")
                print(clean_text)
                print(f"TIME: {dt:.2f}s")
                print(
                    "FINAL_MASKS:",
                    int(stats["masks_remaining"][-1])
                    if stats.get("masks_remaining")
                    else "n/a",
                )

                row = {
                    "mode": mode,
                    "gen_length": cfg["gen_length"],
                    "block_length": cfg["block_length"],
                    "steps": cfg["steps"],
                    "prompt_index": prompt_index,
                    "user_prompt": user_prompt,
                    "wall_time_s": dt,
                    "elapsed_time_s": float(stats.get("elapsed_time", dt)),
                    "generated_tokens": int(stats.get("generated_tokens", len(token_ids))),
                    "tps": float(
                        stats.get(
                            "tps",
                            (len(token_ids) / dt) if dt > 0 else 0.0,
                        )
                    ),
                    "token_ids_head": token_ids[:40],
                    "raw_output": raw_text,
                    "clean_output": clean_text,
                    "final_masks_remaining": (
                        int(stats["masks_remaining"][-1])
                        if stats.get("masks_remaining")
                        else None
                    ),
                }
                results.append(row)
                output_path.write_text(json.dumps(results, indent=2))
                print("SAVED", key)

    print(f"\nDone. Results saved to {output_path}")


if __name__ == "__main__":
    main()
