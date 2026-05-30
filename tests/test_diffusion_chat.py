import os
import unittest
from copy import deepcopy
from unittest.mock import MagicMock, patch

from mlx_lm.diffusion_chat import (
    ACTIVE_STYLE,
    DiffusionChatVisualizer,
    RESET,
    _resolve_diffusion_chat_config,
    setup_arg_parser,
)


class TestDiffusionChat(unittest.TestCase):
    def test_setup_arg_parser_defaults(self):
        parser = setup_arg_parser()
        args = parser.parse_args([])
        self.assertEqual(args.diffusion_mode, "faithful_llada")
        self.assertIsNone(args.diffusion_steps)
        self.assertIsNone(args.diffusion_gen_length)
        self.assertIsNone(args.diffusion_block_length)
        self.assertTrue(args.diffusion_compile_steps)
        self.assertTrue(args.diffusion_dynamic_batching)
        self.assertTrue(args.diffusion_ui)
        self.assertTrue(args.logits_eos_inf)
        self.assertTrue(args.confidence_eos_eot_inf)
        self.assertEqual(args.eot_token_id, 126348)

    def test_resolve_diffusion_chat_defaults(self):
        parser = setup_arg_parser()
        args = parser.parse_args([])
        resolved = _resolve_diffusion_chat_config(args)
        self.assertEqual(resolved["gen_length"], 32)
        self.assertEqual(resolved["block_length"], 32)
        self.assertEqual(resolved["steps"], 16)

    def test_resolve_diffusion_chat_dynamic_mode_defaults(self):
        parser = setup_arg_parser()
        args = parser.parse_args(["--diffusion-mode", "dynamic_block_diffusion"])
        resolved = _resolve_diffusion_chat_config(args)
        self.assertEqual(resolved["gen_length"], 32)
        self.assertEqual(resolved["block_length"], 32)
        self.assertIsNone(resolved["steps"])

    @patch("mlx_lm.diffusion_chat.load")
    @patch("mlx_lm.diffusion_chat.make_llada_runtime")
    @patch("builtins.input")
    @patch("builtins.print")
    def test_system_prompt_and_history_are_used(
        self,
        mock_print,
        mock_input,
        mock_make_runtime,
        mock_load,
    ):
        from mlx_lm.diffusion_chat import main

        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        captured_messages = []

        def _apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False, tools=None
        ):
            captured_messages.append(deepcopy(messages))
            return f"prompt_{len(captured_messages)}"

        mock_tokenizer.apply_chat_template.side_effect = _apply_chat_template
        mock_load.return_value = (mock_model, mock_tokenizer)

        runtime = MagicMock()
        first_result = MagicMock()
        first_result.text = "First answer"
        second_result = MagicMock()
        second_result.text = "Second answer"
        runtime.generate.side_effect = [first_result, second_result]
        mock_make_runtime.return_value = runtime

        mock_input.side_effect = ["Hello", "Tell me more", "q"]

        with patch(
            "sys.argv",
            [
                "diffusion_chat.py",
                "--system-prompt",
                "You are a helpful diffusion assistant.",
            ],
        ):
            try:
                main()
            except SystemExit:
                pass

        first_messages = captured_messages[0]
        second_messages = captured_messages[1]

        self.assertEqual(first_messages[0]["role"], "system")
        self.assertEqual(first_messages[1], {"role": "user", "content": "Hello"})
        self.assertEqual(
            second_messages,
            [
                {"role": "system", "content": "You are a helpful diffusion assistant."},
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "First answer"},
                {"role": "user", "content": "Tell me more"},
            ],
        )
        first_call = mock_make_runtime.call_args.kwargs
        self.assertEqual(first_call["steps"], 16)
        self.assertEqual(first_call["gen_length"], 32)
        self.assertEqual(first_call["block_length"], 32)
        self.assertTrue(first_call["logits_eos_inf"])
        self.assertTrue(first_call["confidence_eos_eot_inf"])
        self.assertEqual(first_call["eot_token_id"], 126348)

    @patch("mlx_lm.diffusion_chat.load")
    @patch("mlx_lm.diffusion_chat.make_llada_runtime")
    @patch("builtins.input")
    @patch("builtins.print")
    def test_reset_clears_history(
        self,
        mock_print,
        mock_input,
        mock_make_runtime,
        mock_load,
    ):
        from mlx_lm.diffusion_chat import main

        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        captured_messages = []

        def _apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False, tools=None
        ):
            captured_messages.append(deepcopy(messages))
            return f"prompt_{len(captured_messages)}"

        mock_tokenizer.apply_chat_template.side_effect = _apply_chat_template
        mock_load.return_value = (mock_model, mock_tokenizer)

        runtime = MagicMock()
        first_result = MagicMock()
        first_result.text = "First answer"
        second_result = MagicMock()
        second_result.text = "Second answer"
        runtime.generate.side_effect = [first_result, second_result]
        mock_make_runtime.return_value = runtime

        mock_input.side_effect = ["Hello", "r", "New topic", "q"]

        with patch("sys.argv", ["diffusion_chat.py"]):
            try:
                main()
            except SystemExit:
                pass

        first_messages = captured_messages[0]
        second_messages = captured_messages[1]

        self.assertEqual(first_messages, [{"role": "user", "content": "Hello"}])
        self.assertEqual(second_messages, [{"role": "user", "content": "New topic"}])

    def test_frame_lines_do_not_slice_ansi_sequences(self):
        tokenizer = MagicMock()
        visualizer = DiffusionChatVisualizer(tokenizer, enabled=False)
        styled = f"{ACTIVE_STYLE} hello {RESET}"

        with patch("shutil.get_terminal_size", return_value=os.terminal_size((40, 30))):
            frame = visualizer._frame_lines(
                "Denoising Process Visualization",
                "subtitle",
                [styled],
            )

        joined = "\n".join(frame)
        self.assertNotIn("│30;48;5;229m", joined)
        self.assertNotIn("\n30;48;5;229m", joined)
        self.assertIn(ACTIVE_STYLE, joined)
        self.assertIn(RESET, joined)

    def test_visualizer_header_includes_commit_speed(self):
        tokenizer = MagicMock()
        visualizer = DiffusionChatVisualizer(tokenizer, enabled=False)

        with patch("shutil.get_terminal_size", return_value=os.terminal_size((120, 30))):
            frame = visualizer._frame_lines(
                "Denoising Process Visualization",
                "mode=faithful_llada  block=1/1  step=1/4  masked=12  committed=4  dt=0.22s",
                ["hello"],
            )

        joined = "\n".join(frame)
        self.assertIn("committed=4", joined)
        self.assertIn("dt=0.22s", joined)

    def test_visualizer_header_omits_tps(self):
        tokenizer = MagicMock()
        visualizer = DiffusionChatVisualizer(tokenizer, enabled=False)
        visualizer.enabled = True
        visualizer._previous_suffix = None

        update = {
            "mode": "faithful_llada",
            "block_index": 0,
            "num_blocks": 1,
            "step_index": 0,
            "step_limit": 4,
            "masks_remaining": 12,
            "tokens_committed": 4,
            "step_tps": 40.0,
            "step_time": 0.10,
            "prompt_length": 0,
            "mask_id": 0,
            "block_start": 0,
            "block_end": 4,
            "token_ids": [[1, 2, 3, 4]],
        }

        with patch.object(visualizer, "_frame_lines", wraps=visualizer._frame_lines) as frame_lines:
            with patch("sys.stdout.write"), patch("sys.stdout.flush"):
                visualizer.update(update)

        subtitle = frame_lines.call_args[0][1]
        self.assertNotIn("tps=", subtitle)

    def test_mask_noise_changes_across_frames(self):
        tokenizer = MagicMock()
        visualizer = DiffusionChatVisualizer(tokenizer, enabled=False)

        visualizer._frame_index = 0
        first = visualizer._token_text(0, 0, 3)
        visualizer._frame_index = 1
        second = visualizer._token_text(0, 0, 3)

        self.assertNotEqual(first, second)
        self.assertTrue(all(ch in ".+*?^$|/\\~=<>:#%&@" for ch in first))
        self.assertTrue(all(ch in ".+*?^$|/\\~=<>:#%&@" for ch in second))

if __name__ == "__main__":
    unittest.main()
