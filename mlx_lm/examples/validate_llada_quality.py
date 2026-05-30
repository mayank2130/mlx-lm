#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

from mlx_lm.diffusion.validation import validate_quality


def main():
    parser = argparse.ArgumentParser(
        description="Compare faithful and experimental LLaDA diffusion paths over a prompt set."
    )
    parser.add_argument("model_path", help="Local MLX model path or repo id.")
    parser.add_argument(
        "--prompt",
        action="append",
        dest="prompts",
        help="Prompt to validate. Repeat this flag to add multiple prompts.",
    )
    parser.add_argument(
        "--output",
        default="llada_quality_validation.json",
        help="Where to save the JSON validation report.",
    )
    args = parser.parse_args()

    prompts = args.prompts or [
        "Why is the sky blue?",
        "Why do leaves look green?",
        "Explain tides in one sentence.",
        "What causes rainbows?",
        "Why is fire hot?",
    ]
    report = validate_quality(args.model_path, prompts)
    output_path = Path(args.output)
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\nSaved results to {output_path.resolve()}")


if __name__ == "__main__":
    main()
