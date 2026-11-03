"""Command-line interface for local masked diffusion checkpoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from . import approximate, ctmc, dream
from .cache import PromptOnlyTargetCacheModel

LLADA_METHODS = {"direct", "spec", "tau", "trapezoidal", "cached-direct", "cached-spec"}
DREAM_METHODS = {"direct", "spec"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run DirectCTMC, SpecCTMC, or a fixed-step approximate CTMC solver."
    )
    parser.add_argument("--model-family", choices=("llada", "dream"), required=True)
    parser.add_argument(
        "--method",
        choices=("direct", "spec", "tau", "trapezoidal", "cached-direct", "cached-spec"),
        required=True,
    )
    parser.add_argument("--model-path", required=True)
    prompt_group = parser.add_mutually_exclusive_group(required=True)
    prompt_group.add_argument("--prompt")
    prompt_group.add_argument("--prompts-file")
    parser.add_argument("--output")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="bfloat16")
    parser.add_argument("--generation-length", type=int, default=128)
    parser.add_argument("--block-length", type=int)
    parser.add_argument("--speculative-depth", type=int, default=4)
    parser.add_argument("--target-batch-size", type=int)
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--refresh-interval", type=int, choices=(8, 16, 32))
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--epsilon", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--chat-template", action="store_true")
    parser.add_argument("--include-trace", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _validate_arguments(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    allowed = LLADA_METHODS if args.model_family == "llada" else DREAM_METHODS
    if args.method not in allowed:
        parser.error(f"method {args.method} is not supported for {args.model_family}")
    if args.method.startswith("cached-"):
        if args.refresh_interval is None:
            parser.error("cached methods require --refresh-interval")
        if args.target_batch_size is None:
            parser.error("cached methods require --target-batch-size")
    if args.generation_length <= 0:
        parser.error("--generation-length must be positive")
    if args.block_length is not None and (
        args.block_length <= 0 or args.generation_length % args.block_length
    ):
        parser.error("--block-length must be positive and divide --generation-length")
    if args.speculative_depth <= 0:
        parser.error("--speculative-depth must be positive")
    if args.target_batch_size is not None and args.target_batch_size <= 0:
        parser.error("--target-batch-size must be positive")
    if args.steps <= 0:
        parser.error("--steps must be positive")
    if args.temperature <= 0:
        parser.error("--temperature must be positive")
    if not 0.0 < args.epsilon < 1.0:
        parser.error("--epsilon must lie strictly between zero and one")
    if args.prompts_file and args.output is None and not args.dry_run:
        parser.error("--prompts-file requires --output")


def _dtype(name: str) -> torch.dtype:
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[name]


def _load_llada(path: Path, args: argparse.Namespace):
    from transformers import AutoTokenizer

    from .modeling_llada_cache import LLaDAConfig, LLaDAModelLM

    config = LLaDAConfig.from_pretrained(path, local_files_only=True)
    model = LLaDAModelLM.from_pretrained(
        path,
        config=config,
        local_files_only=True,
        torch_dtype=_dtype(args.dtype),
        device_map={"": args.device},
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    return model, tokenizer


def _load_dream(path: Path, args: argparse.Namespace):
    from transformers import AutoModel, AutoTokenizer

    model = AutoModel.from_pretrained(
        path,
        trust_remote_code=True,
        local_files_only=True,
        torch_dtype=_dtype(args.dtype),
        device_map={"": args.device},
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(
        path, trust_remote_code=True, local_files_only=True
    )
    return model, tokenizer


def _load_model(args: argparse.Namespace):
    path = Path(args.model_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"local model path does not exist: {path}")
    if args.model_family == "llada":
        return _load_llada(path, args)
    return _load_dream(path, args)


def _prompt_records(args: argparse.Namespace) -> list[dict[str, str]]:
    if args.prompt is not None:
        return [{"id": "prompt", "prompt": args.prompt}]
    path = Path(args.prompts_file).expanduser().resolve()
    records = []
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        item = json.loads(line)
        if not isinstance(item, dict) or not isinstance(item.get("prompt"), str):
            raise ValueError(f"line {line_number} must contain an object with a string prompt")
        records.append({"id": str(item.get("id", line_number)), "prompt": item["prompt"]})
    if not records:
        raise ValueError("prompts file contains no prompts")
    return records


def _encode(tokenizer: Any, text: str, chat_template: bool, device: torch.device) -> torch.Tensor:
    if chat_template:
        tokens = tokenizer.apply_chat_template(
            [{"role": "user", "content": text}],
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
        )
    else:
        tokens = tokenizer(text, return_tensors="pt").input_ids
    return tokens.to(device)


def _method_target_batch_size(args: argparse.Namespace) -> int:
    if args.target_batch_size is not None:
        return args.target_batch_size
    return args.speculative_depth if args.method == "spec" else 1


def _common(args: argparse.Namespace, model: Any, prompt: torch.Tensor) -> dict[str, Any]:
    mask_id = ctmc.MASK_ID if args.model_family == "llada" else dream.DREAM_MASK_ID
    vocab_size = ctmc.VOCAB_SIZE if args.model_family == "llada" else dream.DREAM_VOCAB_SIZE
    return {
        "model": model,
        "prompt": prompt,
        "batch_size": 1,
        "gen_length": args.generation_length,
        "block_length": args.block_length,
        "mask_id": mask_id,
        "vocab_size": vocab_size,
        "temperature": args.temperature,
        "eps": args.epsilon,
        "seed": args.seed,
        "target_evaluation_batch_size": _method_target_batch_size(args),
        "return_trace": True,
    }


def _generate(args: argparse.Namespace, model: Any, prompt: torch.Tensor):
    common = _common(args, model, prompt)
    runtime = ctmc if args.model_family == "llada" else dream
    if args.method == "direct":
        return runtime.generate_ctmc_exact(**common)
    if args.method == "spec":
        return runtime.generate_ctmc_speculative(
            speculative_depth=args.speculative_depth,
            draft_kind="stale_logits",
            bootstrap_kind="uniform",
            **common,
        )
    if args.method == "tau":
        return approximate.generate_tau_leaping(
            rate_function=ctmc.llada_qrate,
            steps=args.steps,
            **common,
        )
    if args.method == "trapezoidal":
        return approximate.generate_trapezoidal(
            rate_function=ctmc.llada_qrate,
            steps=args.steps,
            **common,
        )
    cached = PromptOnlyTargetCacheModel(model, args.refresh_interval)
    common["model"] = cached
    if args.method == "cached-direct":
        return ctmc.generate_ctmc_exact(**common)
    return ctmc.generate_ctmc_speculative(
        speculative_depth=args.speculative_depth,
        draft_kind="stale_logits",
        bootstrap_kind="uniform",
        **common,
    )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _validate_arguments(args, parser)
    if args.dry_run:
        print(
            json.dumps(
                {
                    "dry_run": True,
                    "method": args.method,
                    "model_family": args.model_family,
                    "model_path": args.model_path,
                },
                sort_keys=True,
            )
        )
        return 0
    model, tokenizer = _load_model(args)
    records = []
    for item in _prompt_records(args):
        prompt = _encode(tokenizer, item["prompt"], args.chat_template, model.device)
        output, trace = _generate(args, model, prompt)
        generated = output[0, prompt.shape[1] :]
        record: dict[str, Any] = {
            "id": item["id"],
            "method": args.method,
            "text": tokenizer.decode(generated, skip_special_tokens=True),
        }
        if args.include_trace:
            record["trace"] = trace
        records.append(record)
    payload = "".join(json.dumps(record, sort_keys=True) + "\n" for record in records)
    if args.output:
        Path(args.output).expanduser().resolve().write_text(payload)
    else:
        print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
