# Copyright © 2025 Apple Inc.

import argparse
import re
import shutil
import sys

import mlx.core as mx

from .diffusion_generate import make_llada_runtime
from .utils import load, sharded_load

DEFAULT_SEED = 0
DEFAULT_MAX_TOKENS = 32
DEFAULT_MODEL = "GSAI-ML/LLaDA-8B-Base"

RESET = "\033[0m"
TOKEN_STYLE = "\033[30;47m"
MASK_STYLE = "\033[30;48;5;224m"
CHANGED_STYLE = "\033[30;48;5;194m"
ACTIVE_STYLE = "\033[30;48;5;229m"
ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;]*m")
MASK_NOISE_CHARS = ".+*?^$|/\\~=<>:#%&@"


def setup_arg_parser():
    parser = argparse.ArgumentParser(description="Chat with a diffusion language model")
    parser.add_argument(
        "--model",
        type=str,
        help="The path to the local model directory or Hugging Face repo.",
        default=DEFAULT_MODEL,
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Enable trusting remote code for tokenizer",
    )
    parser.add_argument(
        "--adapter-path",
        type=str,
        help="Optional path for the trained adapter weights and config.",
    )
    parser.add_argument(
        "--temp",
        type=float,
        default=0.0,
        help="Sampling temperature for diffusion decoding.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="PRNG seed",
    )
    parser.add_argument(
        "--max-tokens",
        "-m",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help="Maximum number of suffix tokens to generate per turn.",
    )
    parser.add_argument(
        "--system-prompt",
        default=None,
        help="System prompt to be used for the chat template",
    )
    parser.add_argument(
        "--pipeline",
        action="store_true",
        help="Use pipelining instead of tensor parallelism",
    )
    parser.add_argument(
        "--diffusion-mode",
        type=str,
        default="faithful_llada",
        choices=[
            "faithful_llada",
            "experimental_block_local",
            "dynamic_block_diffusion",
            "fast_dllm_v1",
            "fast_dllm_v1_dynamic",
            "fast_dllm_v2",
        ],
        help="Diffusion runtime mode.",
    )
    parser.add_argument(
        "--diffusion-steps",
        type=int,
        default=None,
        help="Number of diffusion refinement steps per turn. Defaults to a chat-sized mode-aware value.",
    )
    parser.add_argument(
        "--diffusion-gen-length",
        type=int,
        default=None,
        help="Generated suffix length for diffusion generation. Defaults to --max-tokens.",
    )
    parser.add_argument(
        "--diffusion-block-length",
        type=int,
        default=None,
        help="Block length for diffusion generation. Defaults to gen length.",
    )
    parser.add_argument(
        "--diffusion-compile-steps",
        type=lambda x: x.lower() in {"1", "true", "yes", "y"},
        default=True,
        help="Compile the diffusion denoise-update step.",
    )
    parser.add_argument(
        "--diffusion-dynamic-batching",
        type=lambda x: x.lower() in {"1", "true", "yes", "y"},
        default=True,
        help="Only process active rows during diffusion refinement.",
    )
    parser.add_argument(
        "--diffusion-ui",
        type=lambda x: x.lower() in {"1", "true", "yes", "y"},
        default=True,
        help="Render a live denoising visualization in the terminal.",
    )
    parser.add_argument(
        "--logits-eos-inf",
        type=lambda x: x.lower() in {"1", "true", "yes", "y"},
        default=True,
        help="Suppress EOS in diffusion token selection.",
    )
    parser.add_argument(
        "--confidence-eos-eot-inf",
        type=lambda x: x.lower() in {"1", "true", "yes", "y"},
        default=True,
        help="Suppress EOS/EOT when scoring candidate diffusion tokens.",
    )
    parser.add_argument(
        "--eot-token-id",
        type=int,
        default=126348,
        help="EOT token id used by LLaDA-family chat models.",
    )
    parser.add_argument(
        "--diffusion-remasking",
        type=str,
        choices=["low_confidence", "random"],
        default="low_confidence",
        help="Confidence scoring mode for diffusion token transfers.",
    )
    parser.add_argument(
        "--diffusion-threshold",
        type=float,
        default=None,
        help="Confidence threshold for threshold-based parallel decoding.",
    )
    parser.add_argument(
        "--diffusion-factor",
        type=float,
        default=None,
        help="Dynamic transfer factor used by Fast-dLLM-style adaptive decoding.",
    )
    return parser


def _resolve_diffusion_chat_config(args):
    gen_length = (
        args.diffusion_gen_length
        if args.diffusion_gen_length is not None
        else args.max_tokens
    )

    if args.diffusion_block_length is not None:
        block_length = args.diffusion_block_length
    elif args.diffusion_mode == "faithful_llada":
        block_length = gen_length
    else:
        block_length = min(gen_length, 32)

    if args.diffusion_steps is not None:
        steps = args.diffusion_steps
    elif args.diffusion_mode == "faithful_llada":
        # Keep chat more responsive and reduce repetitive long-tail generations.
        steps = min(gen_length, 16)
    elif args.diffusion_mode in {"fast_dllm_v1", "fast_dllm_v1_dynamic", "fast_dllm_v2"}:
        steps = gen_length
    else:
        steps = None

    return {
        "steps": steps,
        "gen_length": gen_length,
        "block_length": block_length,
    }


class DiffusionChatVisualizer:
    def __init__(self, tokenizer, *, enabled: bool):
        self.tokenizer = tokenizer
        self.enabled = enabled and sys.stdout.isatty()
        self._rendered_lines = 0
        self._previous_suffix = None
        self._cursor_hidden = False
        self._frame_index = 0

    def _mask_noise_text(self, token_index: int) -> str:
        seed = ((self._frame_index + 1) * 1103515245 + (token_index + 1) * 12345) & (
            0xFFFFFFFF
        )
        noise_len = 3 + (seed % 5)
        chars = []
        for _ in range(noise_len):
            seed = (seed * 1664525 + 1013904223) & 0xFFFFFFFF
            chars.append(MASK_NOISE_CHARS[seed % len(MASK_NOISE_CHARS)])
        return "".join(chars)

    def _token_text(self, token_id: int, mask_id: int, token_index: int) -> str:
        if token_id == mask_id:
            return self._mask_noise_text(token_index)
        token = self.tokenizer.decode(
            [int(token_id)],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        token = (
            token.replace("\n", "\\n")
            .replace("\r", "")
            .replace("\t", "\\t")
            .strip()
        )
        if not token:
            raw = self.tokenizer.convert_ids_to_tokens(int(token_id))
            token = raw[0] if isinstance(raw, list) else raw
        token = "".join(ch for ch in token if ch.isprintable())
        return token[:20] if token else "·"

    def _styled_chip(self, text: str, *, style: str) -> tuple[str, int]:
        visible = len(text) + 2
        return f"{style} {text} {RESET}", visible

    def _visible_len(self, text: str) -> int:
        return len(ANSI_ESCAPE_RE.sub("", text))

    def _pad_visible(self, text: str, width: int) -> str:
        visible = self._visible_len(text)
        if visible >= width:
            return text
        return text + (" " * (width - visible))

    def _wrap_chips(self, chips, width: int):
        lines = []
        current = []
        current_width = 0
        for styled, visible in chips:
            extra = visible if not current else visible + 1
            if current and current_width + extra > width:
                lines.append(" ".join(current))
                current = [styled]
                current_width = visible
            else:
                current.append(styled)
                current_width += extra
        if current:
            lines.append(" ".join(current))
        return lines or [""]

    def _frame_lines(self, title: str, subtitle: str, content_lines):
        terminal_width = shutil.get_terminal_size((120, 30)).columns
        inner_width = max(min(terminal_width - 4, 140), 40)
        top = f"┌─ {title} " + "─" * max(inner_width - len(title) - 1, 1) + "┐"
        framed = [
            top,
            f"│ {self._pad_visible(subtitle, inner_width)} │",
            f"├{'─' * (inner_width + 2)}┤",
        ]
        for line in content_lines:
            framed.append(f"│ {self._pad_visible(line, inner_width)} │")
        framed.append(f"└{'─' * (inner_width + 2)}┘")
        return framed

    def update(self, update):
        if not self.enabled:
            return

        token_rows = update.get("token_ids") or []
        if not token_rows:
            return
        token_ids = token_rows[0]
        prompt_length = update["prompt_length"]
        mask_id = update["mask_id"]
        suffix_ids = token_ids[prompt_length:]
        block_start = max(update["block_start"] - prompt_length, 0)
        block_end = max(update["block_end"] - prompt_length, 0)
        self._frame_index += 1

        chips = []
        for idx, token_id in enumerate(suffix_ids):
            text = self._token_text(token_id, mask_id, idx)
            changed = (
                self._previous_suffix is not None
                and idx < len(self._previous_suffix)
                and self._previous_suffix[idx] != token_id
            )
            in_active_block = block_start <= idx < block_end
            if token_id == mask_id:
                style = MASK_STYLE
            elif changed:
                style = CHANGED_STYLE
            elif in_active_block:
                style = ACTIVE_STYLE
            else:
                style = TOKEN_STYLE
            chips.append(self._styled_chip(text, style=style))

        chip_lines = self._wrap_chips(
            chips,
            max(min(shutil.get_terminal_size((120, 30)).columns - 6, 136), 36),
        )
        step_limit = update.get("step_limit")
        step_label = (
            f"{update['step_index'] + 1}/{step_limit}"
            if step_limit is not None
            else f"{update['step_index'] + 1}/dynamic"
        )
        subtitle = (
            f"mode={update['mode']}  block={update['block_index'] + 1}/{update['num_blocks']}  "
            f"step={step_label}  masked={update['masks_remaining']}  "
            f"committed={update.get('tokens_committed', 0)}  "
            f"dt={update['step_time']:.2f}s"
        )
        frame = self._frame_lines(
            "Denoising Process Visualization",
            subtitle,
            chip_lines,
        )

        if not self._cursor_hidden:
            sys.stdout.write("\033[?25l")
            self._cursor_hidden = True
        if self._rendered_lines:
            sys.stdout.write("\033[F" * self._rendered_lines)
            for _ in range(self._rendered_lines):
                sys.stdout.write("\033[2K\033[1E")
            sys.stdout.write("\033[F" * self._rendered_lines)
        sys.stdout.write("\n".join(frame) + "\n")
        sys.stdout.flush()
        self._rendered_lines = len(frame)
        self._previous_suffix = list(suffix_ids)

    def finish(self):
        if not self.enabled:
            return
        if self._cursor_hidden:
            sys.stdout.write("\033[?25h")
            sys.stdout.flush()
            self._cursor_hidden = False
        if self._rendered_lines:
            sys.stdout.write("\033[2K")
            sys.stdout.write("\n")
            sys.stdout.flush()
        self._rendered_lines = 0
        self._previous_suffix = None
        self._frame_index = 0


def main():
    parser = setup_arg_parser()
    args = parser.parse_args()

    group = mx.distributed.init()
    rank = group.rank()
    pipeline_group = group if args.pipeline else None
    tensor_group = group if not args.pipeline else None

    def rprint(*args, **kwargs):
        if rank == 0:
            print(*args, **kwargs)

    mx.random.seed(args.seed)
    resolved = _resolve_diffusion_chat_config(args)

    rprint(f"[INFO] Loading diffusion chat model from {args.model}...")

    if group.size() > 1:
        if args.adapter_path:
            parser.error("Adapters not supported in distributed mode")
        model, tokenizer = sharded_load(args.model, pipeline_group, tensor_group)
    else:
        model, tokenizer = load(
            args.model,
            adapter_path=args.adapter_path,
            tokenizer_config={
                "trust_remote_code": True if args.trust_remote_code else None
            },
        )

    def print_help():
        rprint("The command list:")
        rprint("- 'q' to exit")
        rprint("- 'r' to reset the chat")
        rprint("- 'h' to display these commands")

    rprint(f"[INFO] Starting diffusion chat session with {args.model}.")
    rprint(
        "[INFO] Diffusion config: "
        f"mode={args.diffusion_mode} "
        f"steps={'dynamic' if resolved['steps'] is None else resolved['steps']} "
        f"gen_length={resolved['gen_length']} "
        f"block_length={resolved['block_length']}"
    )
    print_help()

    runtime = make_llada_runtime(
        model,
        mode=args.diffusion_mode,
        steps=resolved["steps"],
        gen_length=resolved["gen_length"],
        block_length=resolved["block_length"],
        temperature=args.temp,
        remasking=args.diffusion_remasking,
        threshold=args.diffusion_threshold,
        factor=args.diffusion_factor,
        logits_eos_inf=args.logits_eos_inf,
        confidence_eos_eot_inf=args.confidence_eos_eot_inf,
        eos_token_id=tokenizer.eos_token_id,
        eot_token_id=args.eot_token_id,
        compile_steps=args.diffusion_compile_steps,
        dynamic_batching=args.diffusion_dynamic_batching,
    )
    visualizer = DiffusionChatVisualizer(tokenizer, enabled=args.diffusion_ui)

    messages = []
    if args.system_prompt is not None:
        messages.append({"role": "system", "content": args.system_prompt})

    while True:
        try:
            query = input(">> " if rank == 0 else "")
        except EOFError:
            break
        query = query.strip()
        if not query:
            continue
        if query == "q":
            break
        if query == "r":
            messages = []
            if args.system_prompt is not None:
                messages.append({"role": "system", "content": args.system_prompt})
            rprint("[INFO] Chat history cleared.")
            continue
        if query == "h":
            print_help()
            continue

        messages.append({"role": "user", "content": query})
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        prompt = tokenizer.encode(prompt, add_special_tokens=False)
        result = runtime.generate(tokenizer, prompt, progress_callback=visualizer.update)
        visualizer.finish()
        response_text = result.text[0] if isinstance(result.text, list) else result.text
        rprint(response_text)
        if isinstance(result.stats, dict):
            rprint(
                "[INFO] Diffusion generation: "
                f"tokens={result.stats.get('generated_tokens', 0)} "
                f"time={result.stats.get('elapsed_time', 0.0):.2f}s "
                f"tps={result.stats.get('tps', 0.0):.1f}"
            )
        messages.append({"role": "assistant", "content": response_text})


if __name__ == "__main__":
    print(
        "Calling `python -m mlx_lm.diffusion_chat...` directly is deprecated."
        " Use `mlx_lm.diffusion_chat...` or `python -m mlx_lm diffusion_chat ...` instead."
    )
    main()
