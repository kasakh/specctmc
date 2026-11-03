# Exact Speculative CTMC

This repository contains the main inference code for exact and approximate simulation of absorbing-mask continuous-time Markov chains used by masked diffusion language models. Users provide their own local model checkpoints and prompts. The repository contains no checkpoints, datasets, generated answers, paper results, or experiment logs.

## Authors

Siva Rajesh Kasa, Yasong Dai, Sumit Negi, and Hongdong Li.

## Methods

**DirectCTMC** is the event-by-event exact reference. It samples each unmasking event from the current model state and advances the continuous-time event clock.

**SpecCTMC** proposes several unmasking events using previously computed target predictions, verifies each proposal against the current target event distribution, and samples from the positive residual after the first rejection. This correction preserves the DirectCTMC path law.

**Tau leaping** evaluates rates on a fixed time grid and samples several transitions within one step. The **second-order trapezoidal solver** combines rate evaluations from both ends of a step. Both fixed-step solvers approximate a different output law from DirectCTMC.

The **prompt-cache composition** reuses prompt key and value tensors while recomputing the generated suffix. The cache defines an approximate target process because prompt representations can change under bidirectional attention. Cached SpecCTMC applies exact correction to the matching cached sequential process.

The **S2D2 composition** uses one block-diffusion forward pass as a proposal and applies maximal-coupling correction to a separately evaluated natural-event target. It is an adapter for a user-supplied LLaDA2.1 model loader. It does not reproduce the routing or greedy refinement law of standalone S2D2.

## Environment

Create the provided Conda environment:

```bash
conda env create -f environment.yml
conda activate exact-speculative-ctmc
pip install -e .
```

An existing environment can instead install the package directly:

```bash
pip install -e .
```

The model commands read local files only. Place the checkpoint, configuration, tokenizer, and any model-specific Python files in a local directory. No command downloads a model or dataset.

## Input

Pass one prompt with `--prompt`, or pass a JSON Lines file with `--prompts-file`. Each nonempty line must be an object containing a string field named `prompt`. An optional `id` field identifies the output record. Use `--output` when reading a prompts file.

For instruction checkpoints, add `--chat-template` when the tokenizer provides a chat template.

## DirectCTMC

LLaDA:

```bash
python -m specctmc.run \
  --model-family llada \
  --method direct \
  --model-path /path/to/local/llada \
  --prompts-file /path/to/prompts.jsonl \
  --output /path/to/output.jsonl
```

Dream:

```bash
python -m specctmc.run \
  --model-family dream \
  --method direct \
  --model-path /path/to/local/dream \
  --prompt "Your prompt"
```

## SpecCTMC

```bash
python -m specctmc.run \
  --model-family llada \
  --method spec \
  --model-path /path/to/local/llada \
  --prompt "Your prompt"
```

Use the same command with `--model-family dream` for a compatible local Dream checkpoint.

## Approximate CTMC solvers

Tau leaping:

```bash
python -m specctmc.run \
  --model-family llada \
  --method tau \
  --steps <number-of-steps> \
  --model-path /path/to/local/llada \
  --prompt "Your prompt"
```

Second-order trapezoidal solver:

```bash
python -m specctmc.run \
  --model-family llada \
  --method trapezoidal \
  --steps <number-of-steps> \
  --model-path /path/to/local/llada \
  --prompt "Your prompt"
```

## Prompt-cache composition

Use the same `--refresh-interval` and `--target-batch-size` in both commands so that cached sequential CTMC and cached SpecCTMC evaluate the same cached target process.

Cached sequential CTMC:

```bash
python -m specctmc.run \
  --model-family llada \
  --method cached-direct \
  --refresh-interval <refresh-interval> \
  --target-batch-size <target-batch-size> \
  --model-path /path/to/local/llada \
  --prompt "Your prompt"
```

Cached SpecCTMC:

```bash
python -m specctmc.run \
  --model-family llada \
  --method cached-spec \
  --refresh-interval <refresh-interval> \
  --target-batch-size <target-batch-size> \
  --model-path /path/to/local/llada \
  --prompt "Your prompt"
```

## S2D2 exact-correction composition

The model factory is a Python function that accepts the local model path and returns `(model, tokenizer)`. Put the factory module on `PYTHONPATH`, then run:

```bash
python examples/run_s2d2.py \
  --method s2d2-spec \
  --model-factory your_loader:load_model_and_tokenizer \
  --model-path /path/to/local/llada2 \
  --mask-id <mask-token-id> \
  --eos-id <end-token-id> \
  --prompt "Your prompt"
```

Change `--method` to `direct` for the matching natural-event reference.

## Tests

The tests use small synthetic models on CPU. They do not load checkpoints or access datasets.

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

Every public command also supports `--dry-run`, which validates its arguments without opening the model path.

## License and attribution

The repository is licensed under Apache License 2.0. The cache-capable LLaDA model files retain their original copyright, license, and source headers. `THIRD_PARTY_NOTICES.md` records external method and implementation sources.
