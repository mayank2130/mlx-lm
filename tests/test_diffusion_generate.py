# Copyright © 2024 Apple Inc.

import unittest
from typing import NamedTuple

import mlx.core as mx

from mlx_lm.diffusion import ContinuousDiffusionBatcher, DiffusionBatchRequest
from mlx_lm.diffusion_generate import (
    _build_transfer_index,
    get_num_transfer_tokens,
    llada_batch_generate,
    llada_generate_tokens,
)
from mlx_lm.models.llada import Model, ModelArgs, _replace_cache_slice
from mlx_lm.utils import _warmup_loaded_model


class FakeDiffusionModel:
    supports_logits_range = True
    supports_block_local = True
    supports_dual_cache = False
    supports_block_cache = False

    def __call__(
        self,
        x,
        attention_mask=None,
        logits_range=None,
        block_range=None,
        prefix_states=None,
        use_cache=False,
        use_block_cache=False,
        past_key_values=None,
        block_past_key_values=None,
        block_cache_range=None,
        update_past_key_values=True,
        replace_position=None,
    ):
        batch, length = x.shape
        vocab_size = 64
        if block_range is not None:
            start, end = block_range
        elif logits_range is not None:
            start, end = logits_range
        else:
            start, end = 0, length
        logits = mx.full((batch, end - start, vocab_size), -1000.0)
        for rel_pos, pos in enumerate(range(start, end)):
            token_id = (pos % 16) + 1
            logits[:, rel_pos, token_id] = float(length - pos)
        return logits


class FakeCacheOutput(NamedTuple):
    logits: mx.array
    attn_key_values: list[tuple[mx.array, mx.array]]
    block_attn_key_values: list[tuple[mx.array, mx.array]] | None = None


class FakeDualCacheDiffusionModel(FakeDiffusionModel):
    supports_dual_cache = True
    supports_block_cache = True

    def __call__(
        self,
        x,
        attention_mask=None,
        logits_range=None,
        block_range=None,
        prefix_states=None,
        use_cache=False,
        use_block_cache=False,
        past_key_values=None,
        block_past_key_values=None,
        block_cache_range=None,
        update_past_key_values=True,
        replace_position=None,
    ):
        logits = super().__call__(
            x,
            attention_mask=attention_mask,
            logits_range=logits_range,
            block_range=block_range,
            prefix_states=prefix_states,
        )
        if not use_cache:
            return logits
        if past_key_values is None:
            cache_length = x.shape[1]
        else:
            cache_length = past_key_values[0][0].shape[2]
        cache = [
            (
                mx.zeros((x.shape[0], 1, cache_length, 4), dtype=mx.float32),
                mx.zeros((x.shape[0], 1, cache_length, 4), dtype=mx.float32),
            )
        ]
        block_cache = [
            (
                mx.zeros((x.shape[0], 1, x.shape[1], 4), dtype=mx.float32),
                mx.zeros((x.shape[0], 1, x.shape[1], 4), dtype=mx.float32),
            )
        ]
        return FakeCacheOutput(
            logits=logits,
            attn_key_values=cache,
            block_attn_key_values=block_cache if use_block_cache else None,
        )


class FakeWindowDiffusionModel(FakeDiffusionModel):
    supports_context_window_cache = True


class FakeTokenizer:
    bos_token = None
    eos_token_id = 63
    pad_token_id = 0
    clean_up_tokenization_spaces = False
    chat_template = None

    def decode(self, tokens, skip_special_tokens=True):
        return " ".join(str(token) for token in tokens)

    def get_vocab(self):
        return {str(i): i for i in range(64)}


class TestDiffusionGenerate(unittest.TestCase):
    def test_transfer_schedule(self):
        mask_index = mx.array([[True, True, True, True, False]])
        schedule = get_num_transfer_tokens(mask_index, steps=3)
        self.assertEqual(schedule.tolist(), [[2, 1, 1]])

    def test_transfer_index_is_vectorized_per_row(self):
        confidence = mx.array(
            [
                [1.0, 5.0, 3.0, 4.0],
                [9.0, 2.0, 8.0, 1.0],
            ]
        )
        k_per_batch = mx.array([2, 1], dtype=mx.int32)
        transfer_index = _build_transfer_index(confidence, k_per_batch)
        self.assertEqual(
            transfer_index.tolist(),
            [
                [False, True, False, True],
                [True, False, False, False],
            ],
        )

    def test_llada_generate_tokens_fills_masked_suffix(self):
        model = FakeDiffusionModel()
        prompt = mx.array([7, 8], dtype=mx.int32)
        out = llada_generate_tokens(
            model,
            prompt,
            steps=4,
            gen_length=4,
            block_length=4,
            mask_id=0,
        )
        self.assertEqual(out.tolist(), [[7, 8, 3, 4, 5, 6]])

    def test_compiled_full_step_matches_eager(self):
        model = FakeDiffusionModel()
        prompt = mx.array([7, 8], dtype=mx.int32)
        eager = llada_generate_tokens(
            model,
            prompt,
            mode="faithful_llada",
            steps=4,
            gen_length=4,
            block_length=4,
            mask_id=0,
            compile_steps=False,
        )
        compiled = llada_generate_tokens(
            model,
            prompt,
            mode="faithful_llada",
            steps=4,
            gen_length=4,
            block_length=4,
            mask_id=0,
            compile_steps=True,
        )
        self.assertEqual(compiled.tolist(), eager.tolist())

    def test_llada_batch_generate_pads_and_tracks_prompt_lengths(self):
        model = FakeDiffusionModel()
        tokenizer = FakeTokenizer()
        result = llada_batch_generate(
            model,
            tokenizer,
            [
                mx.array([7, 8], dtype=mx.int32),
                mx.array([4], dtype=mx.int32),
            ],
            steps=4,
            gen_length=4,
            block_length=4,
            mask_id=0,
            compile_steps=False,
        )
        self.assertEqual(result.text, ["3 4 5 6", "3 4 5 6"])
        self.assertEqual(result.stats["prompt_lengths"], [2, 1])
        self.assertIn("model_forwards", result.stats)
        self.assertIn("compiled_step_hits", result.stats)
        self.assertIn("active_rows_history", result.stats)

    def test_runtime_mode_switches_to_experimental_block_local(self):
        model = FakeDiffusionModel()
        prompt = mx.array([7, 8], dtype=mx.int32)
        out = llada_generate_tokens(
            model,
            prompt,
            mode="experimental_block_local",
            steps=4,
            gen_length=4,
            block_length=2,
            mask_id=0,
            compile_steps=False,
        )
        self.assertEqual(out.tolist(), [[7, 8, 3, 4, 5, 6]])

    def test_dynamic_block_diffusion_accepts_optional_steps(self):
        model = FakeDiffusionModel()
        prompt = mx.array([7, 8], dtype=mx.int32)
        out = llada_generate_tokens(
            model,
            prompt,
            mode="dynamic_block_diffusion",
            steps=None,
            gen_length=4,
            block_length=2,
            mask_id=0,
            compile_steps=False,
        )
        self.assertEqual(out.tolist(), [[7, 8, 3, 4, 5, 6]])

    def test_fast_dllm_v1_mode_runs_with_dual_cache_model(self):
        model = FakeDualCacheDiffusionModel()
        prompt = mx.array([7, 8], dtype=mx.int32)
        out, stats = llada_generate_tokens(
            model,
            prompt,
            mode="fast_dllm_v1",
            steps=4,
            gen_length=4,
            block_length=2,
            mask_id=0,
            compile_steps=True,
            return_stats=True,
        )
        self.assertEqual(out.tolist(), [[7, 8, 3, 4, 5, 6]])
        self.assertGreaterEqual(stats["model_forwards"], 2)
        self.assertEqual(stats["runtime_mode"], "fast_dllm_v1")
        self.assertGreaterEqual(stats["compiled_step_hits"], 1)

    def test_fast_dllm_v1_dynamic_mode_runs_with_factor_policy(self):
        model = FakeDualCacheDiffusionModel()
        prompt = mx.array([7, 8], dtype=mx.int32)
        out, stats = llada_generate_tokens(
            model,
            prompt,
            mode="fast_dllm_v1_dynamic",
            steps=4,
            gen_length=4,
            block_length=2,
            factor=1.0,
            mask_id=0,
            compile_steps=False,
            return_stats=True,
        )
        self.assertEqual(out.tolist(), [[7, 8, 3, 4, 5, 6]])
        self.assertEqual(stats["runtime_mode"], "fast_dllm_v1_dynamic")

    def test_fast_dllm_v2_mode_runs_with_block_cache(self):
        model = FakeDualCacheDiffusionModel()
        prompt = mx.array([7, 8], dtype=mx.int32)
        out, stats = llada_generate_tokens(
            model,
            prompt,
            mode="fast_dllm_v2",
            steps=4,
            gen_length=4,
            block_length=2,
            mask_id=0,
            compile_steps=True,
            return_stats=True,
        )
        self.assertEqual(out.tolist(), [[7, 8, 3, 4, 5, 6]])
        self.assertEqual(stats["runtime_mode"], "fast_dllm_v2")
        self.assertTrue(stats["quality_gate_passed"])
        self.assertGreaterEqual(stats["compiled_step_hits"], 1)

    def test_replace_cache_slice_supports_batch_specific_spans(self):
        cached = mx.arange(2 * 1 * 4 * 2).reshape(2, 1, 4, 2)
        current = mx.array(
            [
                [[[100, 101], [102, 103]]],
                [[[200, 201], [202, 203]]],
            ]
        )
        replace_position = mx.array(
            [
                [False, True, True, False],
                [True, True, False, False],
            ]
        )
        updated = _replace_cache_slice(cached, current, replace_position)
        self.assertEqual(
            updated.tolist(),
            [
                [[[0, 1], [100, 101], [102, 103], [6, 7]]],
                [[[200, 201], [202, 203], [12, 13], [14, 15]]],
            ],
        )

    def test_replace_cache_slice_vectorizes_uniform_spans(self):
        cached = mx.arange(2 * 1 * 4 * 2).reshape(2, 1, 4, 2)
        current = mx.array(
            [
                [[[100, 101], [102, 103]]],
                [[[200, 201], [202, 203]]],
            ]
        )
        replace_position = mx.array(
            [
                [False, True, True, False],
                [False, True, True, False],
            ]
        )
        updated = _replace_cache_slice(cached, current, replace_position)
        self.assertEqual(
            updated.tolist(),
            [
                [[[0, 1], [100, 101], [102, 103], [6, 7]]],
                [[[8, 9], [200, 201], [202, 203], [14, 15]]],
            ],
        )

    def test_llada_model_supports_cache_and_replace_position(self):
        args = ModelArgs(
            model_type="llada",
            d_model=16,
            n_heads=4,
            n_layers=2,
            mlp_hidden_size=32,
            vocab_size=32,
            embedding_size=32,
            max_sequence_length=16,
            rms_norm_eps=1e-5,
            mask_token_id=0,
        )
        model = Model(args)
        input_ids = mx.array([[1, 2, 0, 0]], dtype=mx.int32)
        attention_mask = mx.array([[1, 1, 1, 1]], dtype=mx.int32)

        warm = model(input_ids, attention_mask=attention_mask, use_cache=True)
        self.assertEqual(warm.logits.shape, (1, 4, 32))
        self.assertEqual(len(warm.attn_key_values), args.n_layers)
        self.assertEqual(warm.attn_key_values[0][0].shape[2], 4)

        replace_position = mx.array([[False, False, True, True]])
        block = model(
            input_ids[:, 2:4],
            attention_mask=attention_mask,
            past_key_values=warm.attn_key_values,
            use_cache=True,
            replace_position=replace_position,
        )
        self.assertEqual(block.logits.shape, (1, 2, 32))
        self.assertEqual(block.attn_key_values[0][0].shape[2], 4)

        block_warm = model(
            input_ids[:, 2:4],
            attention_mask=attention_mask,
            past_key_values=warm.attn_key_values,
            use_cache=True,
            use_block_cache=True,
            block_cache_range=(2, 4),
            replace_position=replace_position,
            update_past_key_values=False,
        )
        self.assertEqual(block_warm.logits.shape, (1, 2, 32))
        self.assertEqual(len(block_warm.block_attn_key_values), args.n_layers)
        self.assertEqual(block_warm.block_attn_key_values[0][0].shape[2], 2)

    def test_progress_callback_reports_commit_speed(self):
        model = FakeDiffusionModel()
        prompt = mx.array([7, 8], dtype=mx.int32)
        updates = []
        _, stats = llada_generate_tokens(
            model,
            prompt,
            mode="faithful_llada",
            steps=4,
            gen_length=4,
            block_length=4,
            mask_id=0,
            compile_steps=False,
            progress_callback=updates.append,
            return_stats=True,
        )
        self.assertTrue(updates)
        self.assertIn("tokens_committed", updates[0])
        self.assertIn("step_tps", updates[0])
        self.assertGreaterEqual(updates[0]["tokens_committed"], 0)
        self.assertGreaterEqual(updates[0]["step_tps"], 0.0)
        self.assertEqual(len(stats["tokens_committed"]), stats["steps_completed"])
        self.assertEqual(len(stats["step_tps"]), stats["steps_completed"])
        self.assertGreaterEqual(stats["generated_tokens"], 4)
        self.assertGreaterEqual(stats["tps"], 0.0)

    def test_continuous_batcher_drain_preserves_request_ids(self):
        model = FakeDiffusionModel()
        tokenizer = FakeTokenizer()
        runtime_result = llada_batch_generate(
            model,
            tokenizer,
            [
                mx.array([7, 8], dtype=mx.int32),
                mx.array([4], dtype=mx.int32),
            ],
            steps=4,
            gen_length=4,
            block_length=4,
            mask_id=0,
            compile_steps=False,
        )
        self.assertEqual(runtime_result.text, ["3 4 5 6", "3 4 5 6"])

        from mlx_lm.diffusion_generate import _make_runtime

        runtime = _make_runtime(
            model,
            mode="faithful_llada",
            steps=4,
            gen_length=4,
            block_length=4,
            temperature=0.0,
            cfg_scale=0.0,
            remasking="low_confidence",
            mask_id=0,
            logits_eos_inf=False,
            confidence_eos_eot_inf=False,
            eos_token_id=None,
            eot_token_id=None,
            use_logits_range=None,
            compile_steps=False,
            block_local=None,
            dynamic_batching=True,
        )
        batcher = ContinuousDiffusionBatcher(runtime)
        batcher.enqueue(
            DiffusionBatchRequest(
                prompt=mx.array([7, 8], dtype=mx.int32),
                request_id="first",
            )
        )
        batcher.enqueue(
            DiffusionBatchRequest(
                prompt=mx.array([4], dtype=mx.int32),
                request_id="second",
            )
        )
        drained = batcher.drain(tokenizer)
        self.assertEqual(sorted(drained.keys()), ["first", "second"])
        self.assertEqual(drained["first"]["text"], "3 4 5 6")
        self.assertEqual(drained["second"]["text"], "3 4 5 6")

    def test_single_vs_batched_invariance_for_faithful_mode(self):
        model = FakeDiffusionModel()
        tokenizer = FakeTokenizer()
        single = llada_batch_generate(
            model,
            tokenizer,
            [mx.array([7, 8], dtype=mx.int32)],
            mode="faithful_llada",
            steps=4,
            gen_length=4,
            block_length=4,
            mask_id=0,
            compile_steps=False,
        )
        batched = llada_batch_generate(
            model,
            tokenizer,
            [
                mx.array([7, 8], dtype=mx.int32),
                mx.array([4], dtype=mx.int32),
            ],
            mode="faithful_llada",
            steps=4,
            gen_length=4,
            block_length=4,
            mask_id=0,
            compile_steps=False,
        )
        self.assertEqual(single.text[0], batched.text[0])

    def test_runtime_stats_capture_compilation_and_block_iterations(self):
        model = FakeDiffusionModel()
        prompt = mx.array([7, 8], dtype=mx.int32)
        _, stats = llada_generate_tokens(
            model,
            prompt,
            mode="faithful_llada",
            steps=4,
            gen_length=4,
            block_length=4,
            mask_id=0,
            compile_steps=True,
            return_stats=True,
        )
        self.assertGreaterEqual(stats["compiled_step_hits"], 1)
        self.assertEqual(stats["compiled_step_hits"] + stats["compiled_step_misses"], stats["steps_completed"])
        self.assertEqual(len(stats["active_rows_history"]), stats["steps_completed"])
        self.assertEqual(len(stats["logits_span_history"]), stats["steps_completed"])
        self.assertEqual(len(stats["context_span_history"]), stats["steps_completed"])
        self.assertEqual(sum(stats["block_iterations"].values()), stats["steps_completed"])

    def test_experimental_modes_report_window_cache_context(self):
        model = FakeWindowDiffusionModel()
        prompt = mx.array([7, 8], dtype=mx.int32)
        _, stats = llada_generate_tokens(
            model,
            prompt,
            mode="experimental_block_local",
            steps=4,
            gen_length=4,
            block_length=2,
            mask_id=0,
            compile_steps=False,
            return_stats=True,
        )
        self.assertEqual(stats["cache_strategy"], "window")
        self.assertGreater(max(stats["context_span_history"]), max(stats["logits_span_history"]))

    def test_llada_model_forward_shape(self):
        args = ModelArgs(
            model_type="llada",
            d_model=32,
            n_heads=4,
            n_layers=2,
            mlp_hidden_size=64,
            vocab_size=128,
            embedding_size=128,
            max_sequence_length=32,
            rms_norm_eps=1e-5,
            rope=True,
            weight_tying=False,
        )
        model = Model(args)
        x = mx.array([[1, 2, 3, 4]], dtype=mx.int32)
        logits = model(x)
        self.assertEqual(logits.shape, (1, 4, 128))
        sliced_logits = model(x, logits_range=(1, 3))
        self.assertEqual(sliced_logits.shape, (1, 2, 128))
        full_slice = logits[:, 1:3, :]
        mx.eval(full_slice, sliced_logits)
        self.assertTrue(mx.allclose(full_slice, sliced_logits).item())

        prefix_states = model.transformer.prefill_prefix_states(
            x,
            prefix_length=2,
        )
        block_logits = model(
            x,
            block_range=(2, 4),
            prefix_states=prefix_states,
        )
        self.assertEqual(block_logits.shape, (1, 2, 128))

    def test_warmup_loaded_model_runs_for_diffusion_model(self):
        args = ModelArgs(
            model_type="llada",
            d_model=32,
            n_heads=4,
            n_layers=2,
            mlp_hidden_size=64,
            vocab_size=128,
            embedding_size=128,
            max_sequence_length=32,
            rms_norm_eps=1e-5,
            rope=True,
            weight_tying=False,
        )
        model = Model(args)
        _warmup_loaded_model(
            model,
            {"max_sequence_length": 32, "vocab_size": 128},
            warmup_config={"sequence_length": 8, "block_length": 4},
        )
