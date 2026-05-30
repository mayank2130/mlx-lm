#!/usr/bin/env python3

from dataclasses import asdict, dataclass
from statistics import mean
from typing import Optional, Sequence

from ..diffusion_generate import llada_generate
from ..utils import load


@dataclass(frozen=True)
class ValidationConfig:
    name: str
    mode: str
    steps: int
    gen_length: int
    block_length: int
    compile_steps: Optional[bool] = None
    block_local: Optional[bool] = None
    dynamic_batching: Optional[bool] = None


@dataclass
class ValidationCase:
    prompt: str
    config_name: str
    mode: str
    faithful_text: str
    candidate_text: str
    exact_match: bool
    nonempty: bool
    token_overlap: float
    faithful_length: int
    candidate_length: int


def _token_overlap(a: str, b: str) -> float:
    a_tokens = set(a.split())
    b_tokens = set(b.split())
    if not a_tokens and not b_tokens:
        return 1.0
    if not a_tokens or not b_tokens:
        return 0.0
    return len(a_tokens & b_tokens) / len(a_tokens | b_tokens)


def _aggregate_cases(name: str, mode: str, cases: Sequence[ValidationCase]) -> dict:
    return {
        "name": name,
        "mode": mode,
        "prompt_count": len(cases),
        "exact_match_rate": mean(float(case.exact_match) for case in cases),
        "nonempty_rate": mean(float(case.nonempty) for case in cases),
        "mean_token_overlap": mean(case.token_overlap for case in cases),
        "mean_candidate_length": mean(case.candidate_length for case in cases),
        "cases": [asdict(case) for case in cases],
    }


def validate_quality(
    model_path: str,
    prompts: Sequence[str],
    *,
    faithful_steps: int = 2,
    faithful_gen_length: int = 8,
    candidate_configs: Optional[Sequence[ValidationConfig]] = None,
):
    model, tokenizer = load(
        model_path,
        tokenizer_config={"trust_remote_code": True},
        lazy=False,
    )

    faithful_config = ValidationConfig(
        name="faithful_baseline",
        mode="faithful_llada",
        steps=faithful_steps,
        gen_length=faithful_gen_length,
        block_length=faithful_gen_length,
        compile_steps=True,
        block_local=False,
        dynamic_batching=True,
    )
    candidate_configs = candidate_configs or [
        ValidationConfig(
            name="experimental_block_local_block4",
            mode="experimental_block_local",
            steps=faithful_steps,
            gen_length=faithful_gen_length,
            block_length=max(1, faithful_gen_length // 2),
            compile_steps=True,
            block_local=True,
            dynamic_batching=True,
        ),
        ValidationConfig(
            name="experimental_block_local_block8",
            mode="experimental_block_local",
            steps=faithful_steps,
            gen_length=faithful_gen_length,
            block_length=faithful_gen_length,
            compile_steps=True,
            block_local=True,
            dynamic_batching=True,
        ),
        ValidationConfig(
            name="dynamic_block_diffusion_block4",
            mode="dynamic_block_diffusion",
            steps=faithful_steps,
            gen_length=faithful_gen_length,
            block_length=max(1, faithful_gen_length // 2),
            compile_steps=False,
            block_local=True,
            dynamic_batching=True,
        ),
        ValidationConfig(
            name="dynamic_block_diffusion_block8",
            mode="dynamic_block_diffusion",
            steps=faithful_steps,
            gen_length=faithful_gen_length,
            block_length=faithful_gen_length,
            compile_steps=False,
            block_local=True,
            dynamic_batching=True,
        ),
    ]

    faithful_outputs = {}
    for prompt in prompts:
        faithful = llada_generate(
            model,
            tokenizer,
            prompt,
            mode=faithful_config.mode,
            steps=faithful_config.steps,
            gen_length=faithful_config.gen_length,
            block_length=faithful_config.block_length,
            compile_steps=faithful_config.compile_steps,
            block_local=faithful_config.block_local,
            dynamic_batching=faithful_config.dynamic_batching,
        )
        faithful_outputs[prompt] = (
            faithful.text[0] if isinstance(faithful.text, list) else faithful.text
        )

    candidate_reports = []
    for config in candidate_configs:
        cases = []
        for prompt in prompts:
            candidate = llada_generate(
                model,
                tokenizer,
                prompt,
                mode=config.mode,
                steps=config.steps,
                gen_length=config.gen_length,
                block_length=config.block_length,
                compile_steps=config.compile_steps,
                block_local=config.block_local,
                dynamic_batching=config.dynamic_batching,
            )
            candidate_text = (
                candidate.text[0] if isinstance(candidate.text, list) else candidate.text
            )
            faithful_text = faithful_outputs[prompt]
            cases.append(
                ValidationCase(
                    prompt=prompt,
                    config_name=config.name,
                    mode=config.mode,
                    faithful_text=faithful_text,
                    candidate_text=candidate_text,
                    exact_match=faithful_text == candidate_text,
                    nonempty=bool(candidate_text.strip()),
                    token_overlap=_token_overlap(faithful_text, candidate_text),
                    faithful_length=len(faithful_text),
                    candidate_length=len(candidate_text),
                )
            )
        candidate_reports.append(_aggregate_cases(config.name, config.mode, cases))

    best_candidate = max(
        candidate_reports,
        key=lambda report: (
            report["exact_match_rate"],
            report["mean_token_overlap"],
            report["nonempty_rate"],
        ),
    )

    return {
        "model_path": model_path,
        "prompt_count": len(prompts),
        "faithful_config": asdict(faithful_config),
        "candidate_configs": [asdict(config) for config in candidate_configs],
        "best_candidate": best_candidate["name"],
        "faithful_outputs": faithful_outputs,
        "candidates": candidate_reports,
    }
