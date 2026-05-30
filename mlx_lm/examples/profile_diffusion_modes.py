#!/usr/bin/env python3

import argparse
import json
import statistics
import time
from collections import defaultdict
from pathlib import Path

from mlx_lm import load
from mlx_lm.diffusion_generate import encode_diffusion_chat_prompt, llada_generate


DEFAULT_MODES = [
    "faithful_llada",
    "fast_dllm_v1",
    "fast_dllm_v1_dynamic",
    "fast_dllm_v2",
]

DEFAULT_CONFIGS = [
    (64, 64, 256),
    (128, 32, 128),
    (256, 32, 256),
]

MODE_EXTRAS = {
    "faithful_llada": {},
    "fast_dllm_v1": {"threshold": 0.95, "remasking": "low_confidence"},
    "fast_dllm_v1_dynamic": {"factor": 0.5, "remasking": "low_confidence"},
    "fast_dllm_v2": {},
}


def _parse_config(value: str) -> tuple[int, int, int]:
    parts = [p.strip() for p in value.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            f"Invalid config '{value}'. Expected gen_length,block_length,steps"
        )
    try:
        gen_length, block_length, steps = (int(p) for p in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Invalid config '{value}'. All values must be integers."
        ) from exc
    return gen_length, block_length, steps


def _safe_mean(values):
    values = list(values)
    return statistics.mean(values) if values else None


def _safe_sum(values):
    values = list(values)
    return sum(values) if values else 0


def _flatten_numeric(rows, key):
    out = []
    for row in rows:
        vals = row.get(key) or []
        out.extend(vals)
    return out


def _ratio(num, den):
    return float(num) / float(den) if den else 0.0


def _summarize_group(rows):
    elapsed = [row["elapsed_time_s"] for row in rows]
    wall = [row["wall_time_s"] for row in rows]
    generated_tokens = [row["generated_tokens"] for row in rows]
    model_forwards = [row["model_forwards"] for row in rows]
    steps_completed = [row["steps_completed"] for row in rows]
    compiled_hits = [row["compiled_step_hits"] for row in rows]
    compiled_misses = [row["compiled_step_misses"] for row in rows]
    final_masks = [row["final_masks_remaining"] for row in rows]
    context_spans = _flatten_numeric(rows, "context_span_history")
    logits_spans = _flatten_numeric(rows, "logits_span_history")
    active_rows = _flatten_numeric(rows, "active_rows_history")
    tokens_committed = _flatten_numeric(rows, "tokens_committed_history")

    total_generated_tokens = _safe_sum(generated_tokens)
    total_elapsed = _safe_sum(elapsed)
    total_model_forwards = _safe_sum(model_forwards)
    total_steps_completed = _safe_sum(steps_completed)
    total_context_work = _safe_sum(context_spans)
    total_logits_work = _safe_sum(logits_spans)

    total_compile_events = _safe_sum(compiled_hits) + _safe_sum(compiled_misses)

    return {
        "num_cases": len(rows),
        "mean_elapsed_time_s": round(_safe_mean(elapsed) or 0.0, 4),
        "mean_wall_time_s": round(_safe_mean(wall) or 0.0, 4),
        "mean_generated_tokens": round(_safe_mean(generated_tokens) or 0.0, 2),
        "mean_tps": round(_safe_mean(row["tps"] for row in rows) or 0.0, 4),
        "aggregate_tps": round(_ratio(total_generated_tokens, total_elapsed), 4),
        "mean_model_forwards": round(_safe_mean(model_forwards) or 0.0, 2),
        "forwards_per_generated_token": round(
            _ratio(total_model_forwards, total_generated_tokens), 4
        ),
        "mean_steps_completed": round(_safe_mean(steps_completed) or 0.0, 2),
        "steps_per_generated_token": round(
            _ratio(total_steps_completed, total_generated_tokens), 4
        ),
        "compiled_hit_rate": round(
            _ratio(_safe_sum(compiled_hits), total_compile_events), 4
        ),
        "mean_context_span": round(_safe_mean(context_spans) or 0.0, 2),
        "mean_logits_span": round(_safe_mean(logits_spans) or 0.0, 2),
        "mean_active_rows": round(_safe_mean(active_rows) or 0.0, 2),
        "context_tokens_per_generated_token": round(
            _ratio(total_context_work, total_generated_tokens), 4
        ),
        "logits_tokens_per_generated_token": round(
            _ratio(total_logits_work, total_generated_tokens), 4
        ),
        "mean_tokens_committed_per_step": round(_safe_mean(tokens_committed) or 0.0, 4),
        "mean_final_masks_remaining": round(_safe_mean(final_masks) or 0.0, 2),
        "stopped_early_rate": round(
            _safe_mean(float(bool(row["stopped_early"])) for row in rows) or 0.0, 4
        ),
    }


def _print_summary(key, summary):
    mode, gen_length, block_length, steps = key
    print(
        f"mode={mode:>20} gen={gen_length:>3} block={block_length:>3} steps={steps:>3} "
        f"time={summary['mean_elapsed_time_s']:.2f}s "
        f"tps={summary['aggregate_tps']:.2f} "
        f"fwds/gen_tok={summary['forwards_per_generated_token']:.3f} "
        f"steps/gen_tok={summary['steps_per_generated_token']:.3f} "
        f"ctx/gen_tok={summary['context_tokens_per_generated_token']:.2f} "
        f"logits/gen_tok={summary['logits_tokens_per_generated_token']:.2f} "
        f"compile_hit={summary['compiled_hit_rate']:.2%}"
    )


def _build_report(
    *,
    model_path,
    prompts_path,
    modes,
    configs,
    rows,
):
    grouped = defaultdict(list)
    for row in rows:
        key = (row["mode"], row["gen_length"], row["block_length"], row["steps"])
        grouped[key].append(row)

    summaries = {}
    for key in sorted(grouped):
        summaries["|".join(str(v) for v in key)] = _summarize_group(grouped[key])

    return {
        "model_path": str(model_path),
        "prompts_path": str(prompts_path),
        "modes": list(modes),
        "configs": [
            {"gen_length": g, "block_length": b, "steps": s} for g, b, s in configs
        ],
        "rows": rows,
        "summaries": summaries,
    }


def _save_report(
    output_path: Path,
    *,
    model_path,
    prompts_path,
    modes,
    configs,
    rows,
):
    report = _build_report(
        model_path=model_path,
        prompts_path=prompts_path,
        modes=modes,
        configs=configs,
        rows=rows,
    )
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Profile diffusion runtime behavior across modes/configs so we can "
            "see where wall time is really going."
        )
    )
    parser.add_argument("model_path", help="Local MLX model path or repo id.")
    parser.add_argument(
        "--prompts",
        required=True,
        help="Path to a JSON file containing a list of user prompts.",
    )
    parser.add_argument(
        "--mode",
        action="append",
        choices=DEFAULT_MODES,
        dest="modes",
        help="Mode to profile. Repeat to add more modes.",
    )
    parser.add_argument(
        "--config",
        type=_parse_config,
        action="append",
        dest="configs",
        help="Config triple as gen_length,block_length,steps. Repeat to add more.",
    )
    parser.add_argument(
        "--output",
        default="diffusion_profile_report.json",
        help="Where to save the JSON profiling report.",
    )
    args = parser.parse_args()

    prompts_path = Path(args.prompts)
    if not prompts_path.exists():
        raise FileNotFoundError(prompts_path)
    prompts = json.loads(prompts_path.read_text())

    modes = args.modes or list(DEFAULT_MODES)
    configs = args.configs or list(DEFAULT_CONFIGS)
    output_path = Path(args.output)

    rows = []
    done = set()
    if output_path.exists():
        existing = json.loads(output_path.read_text())
        rows = list(existing.get("rows", []))
        done = {
            (
                row["mode"],
                row["gen_length"],
                row["block_length"],
                row["steps"],
                row["prompt_index"],
            )
            for row in rows
        }
        print(f"[INFO] Resuming from {output_path} with {len(rows)} completed cases.")

    model, tokenizer = load(
        str(args.model_path),
        tokenizer_config={"trust_remote_code": True},
    )

    for gen_length, block_length, steps in configs:
        for mode in modes:
            extra = MODE_EXTRAS.get(mode, {})
            for prompt_index, user_prompt in enumerate(prompts):
                case_key = (mode, gen_length, block_length, steps, prompt_index)
                if case_key in done:
                    print("SKIP", case_key)
                    continue

                prompt_ids = encode_diffusion_chat_prompt(
                    tokenizer,
                    [{"role": "user", "content": user_prompt}],
                )

                print(
                    "\nRUN",
                    {
                        "mode": mode,
                        "gen_length": gen_length,
                        "block_length": block_length,
                        "steps": steps,
                        "prompt_index": prompt_index,
                    },
                )
                print("USER PROMPT:", user_prompt)

                t0 = time.perf_counter()
                result = llada_generate(
                    model,
                    tokenizer,
                    prompt_ids,
                    mode=mode,
                    steps=steps,
                    gen_length=gen_length,
                    block_length=block_length,
                    temperature=0.0,
                    verbose=False,
                    **extra,
                )
                wall_time_s = time.perf_counter() - t0

                stats = result.stats or {}
                token_ids = (
                    result.token_ids[0].tolist()
                    if getattr(result.token_ids, "ndim", 1) == 2
                    else result.token_ids.tolist()
                )
                clean_text = result.text[0] if isinstance(result.text, list) else result.text

                row = {
                    "mode": mode,
                    "gen_length": gen_length,
                    "block_length": block_length,
                    "steps": steps,
                    "prompt_index": prompt_index,
                    "user_prompt": user_prompt,
                    "wall_time_s": float(wall_time_s),
                    "elapsed_time_s": float(stats.get("elapsed_time", wall_time_s)),
                    "generated_tokens": int(stats.get("generated_tokens", len(token_ids))),
                    "tps": float(
                        stats.get(
                            "tps",
                            (len(token_ids) / wall_time_s) if wall_time_s > 0 else 0.0,
                        )
                    ),
                    "model_forwards": int(stats.get("model_forwards", 0)),
                    "steps_completed": int(stats.get("steps_completed", 0)),
                    "blocks_completed": int(stats.get("blocks_completed", 0)),
                    "compiled_step_hits": int(stats.get("compiled_step_hits", 0)),
                    "compiled_step_misses": int(stats.get("compiled_step_misses", 0)),
                    "final_masks_remaining": (
                        int(stats["masks_remaining"][-1])
                        if stats.get("masks_remaining")
                        else None
                    ),
                    "stopped_early": bool(stats.get("stopped_early", False)),
                    "active_rows_history": list(stats.get("active_rows_history", [])),
                    "logits_span_history": list(stats.get("logits_span_history", [])),
                    "context_span_history": list(stats.get("context_span_history", [])),
                    "tokens_committed_history": list(stats.get("tokens_committed", [])),
                    "clean_output": clean_text,
                }
                rows.append(row)
                done.add(case_key)

                print(
                    "RESULT",
                    {
                        "elapsed_s": round(row["elapsed_time_s"], 2),
                        "wall_s": round(row["wall_time_s"], 2),
                        "tps": round(row["tps"], 2),
                        "generated_tokens": row["generated_tokens"],
                        "model_forwards": row["model_forwards"],
                        "steps_completed": row["steps_completed"],
                        "compiled_hits": row["compiled_step_hits"],
                        "compiled_misses": row["compiled_step_misses"],
                    },
                )
                _save_report(
                    output_path,
                    model_path=args.model_path,
                    prompts_path=prompts_path,
                    modes=modes,
                    configs=configs,
                    rows=rows,
                )
                print("CHECKPOINTED", case_key)

    report = _build_report(
        model_path=args.model_path,
        prompts_path=prompts_path,
        modes=modes,
        configs=configs,
        rows=rows,
    )
    print("\n=== SUMMARY ===")
    for summary_key, summary in sorted(report["summaries"].items()):
        mode, gen_length, block_length, steps = summary_key.split("|")
        _print_summary(
            (mode, int(gen_length), int(block_length), int(steps)),
            summary,
        )

    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\nSaved profiling report to {output_path.resolve()}")


if __name__ == "__main__":
    main()
