# Third-party notices

## Cache-capable LLaDA model files

`src/specctmc/modeling_llada_cache/configuration_llada.py` and `src/specctmc/modeling_llada_cache/modeling_llada.py` retain the copyright and Apache License 2.0 headers from their source. Their headers identify NVIDIA Corporation and affiliates and state that the files were modified from the LLaDA repositories at https://github.com/ML-GSAI/LLaDA.

The LLaDA model card identifies its model repository as https://huggingface.co/GSAI-ML/LLaDA-8B-Instruct and declares the model repository license as MIT. Model weights are not included here.

## S2D2 method

The exact-correction adapter is informed by S2D2, "Fast Decoding for Diffusion LLMs via Training-Free Self-Speculation," available at https://github.com/phymhan/S2D2 and https://arxiv.org/abs/2603.25702.

No source file from the S2D2 repository is copied here. The packaged adapter defines a different natural-event target and applies exact maximal-coupling correction to an S2D2-inspired block proposal. Users must obtain any model implementation and checkpoint separately under the terms supplied by their respective authors.
