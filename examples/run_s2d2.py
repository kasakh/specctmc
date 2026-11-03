"""Run the exact S2D2 proposal with a user-supplied local model loader."""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
from typing import Callable

import torch

from specctmc.s2d2_adapter import generate_direct_natural_event, generate_exact_s2d2_proposal


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the natural-event reference or exact correction of an S2D2 proposal."
    )
    parser.add_argument("--method", choices=("direct", "s2d2-spec"), required=True)
    parser.add_argument("--model-factory", required=True, help="Python module and function as module:function")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--generation-length", type=int, default=128)
    parser.add_argument("--block-length", type=int, default=32)
    parser.add_argument("--speculative-depth", type=int, default=4)
    parser.add_argument("--mask-id", type=int, required=True)
    parser.add_argument("--eos-id", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def load_factory(specification: str) -> Callable:
    module_name, separator, function_name = specification.partition(":")
    if not separator or not module_name or not function_name:
        raise ValueError("model factory must have the form module:function")
    factory = getattr(importlib.import_module(module_name), function_name)
    if not callable(factory):
        raise TypeError("model factory must be callable")
    return factory


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.generation_length <= 0 or args.block_length <= 0:
        parser.error("generation and block lengths must be positive")
    if args.speculative_depth <= 0:
        parser.error("speculative depth must be positive")
    if args.dry_run:
        print(json.dumps({"dry_run": True, "method": args.method}, sort_keys=True))
        return 0
    path = Path(args.model_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"local model path does not exist: {path}")
    model, tokenizer = load_factory(args.model_factory)(path)
    prompt = tokenizer(args.prompt, return_tensors="pt").input_ids.to(model.device)
    common = {
        "model": model,
        "input_ids": prompt,
        "block_length": args.block_length,
        "gen_length": args.generation_length,
        "eos_early_stop": False,
        "mask_id": args.mask_id,
        "eos_id": args.eos_id,
        "seed": args.seed,
    }
    with torch.no_grad():
        if args.method == "direct":
            output, statistics = generate_direct_natural_event(**common)
        else:
            output, statistics = generate_exact_s2d2_proposal(
                depth=args.speculative_depth,
                record_full_proposal=False,
                **common,
            )
    generated = output[0, prompt.shape[1] :]
    print(tokenizer.decode(generated, skip_special_tokens=True))
    print(json.dumps(statistics, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
