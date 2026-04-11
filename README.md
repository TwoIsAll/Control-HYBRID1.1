# ControlH1: Hybrid Byte-Level Language Model

This is an experimental hybrid language model that mixes Mamba-2 state space models, Transformer attention, and Byte Latent Transformer ideas. It works directly on bytes not tokens and uses entropy-based dynamic patching to figure out where to spend computation.

Current config is around 2.3M parameters.

## Architecture

Three main parts:

**Frontend**: Takes raw bytes, embeds them, uses an entropy predictor (small transformer with attention) to segment into patches. High-entropy areas get more granular processing, low-entropy repetitive areas get compressed. The segmentation is greedy - starts new patch when entropy exceeds threshold or patch hits max length.

**Middle**: Hybrid backbone with Transformer and Mamba-2 layers. Pattern string controls layer types - "tmtm" means Transformer, Mamba, Transformer, Mamba. You can mess with different patterns to balance speed vs accuracy.

Transformer blocks use RoPE for positional encoding, SwiGLU activation, LayerScale, RMSNorm. Mamba-2 blocks use selective SSM with learnable discretization, SSD dual mixer, depthwise convolution, and gating.

**Backend**: Converts patches back to byte-level predictions using local transformer. Projects patch representations to byte dimension, scatters to original positions, processes to produce logits.

## Why This

Working on bytes means no tokenization artifacts and true multilingual support. Model doesn't care what language or script. Dynamic patching lets the model decide where to focus compute instead of fixed vocab.

Trade-off is model has to learn subword structure from data, might need more training than tokenized models with built-in subword knowledge.

Mamba layers are linear-time (good for long sequences) while Transformer layers are quadratic-time (better for precise attention). Mixing them gives benefits of both. Pattern string is simple way to experiment without code changes.

## Configuration

Config in config.py. Key stuff:

- vocab_size: 256 (per byte)
- context_len: 2048 bytes
- d_model: 64
- n_layers: 4
- n_heads: 4
- d_ff: 192
- hybrid_pattern: "tmtm"
- max_patch_len: 16
- min_patch_len: 1
- patch_entropy_threshold: 2.2
- mamba_state_dim: 16
- mamba_expand: 2
- ssd_rank: 16

There's also TINY_HYBRID_CONFIG for ~240K params if you want something even smaller.

## Running

Run model to see param count:
```
python model.py
```

Other commands:
- `python model.py info` - detailed info
- `python model.py dryrun` - test forward pass
- `python model.py generate` - generate text
- `python model.py bench` - benchmark
- `python model.py smoke` - smoke tests
- `python model.py export` - save to .hwcf
- `python model.py validate` - validate .hwcf file

Train:
```
python train.py pretrain --data input.txt
python train.py finetune --data tiny.jsonl
```

## Checkpoint Format

.hwcf is simple binary format with magic bytes, version, config JSON, tensor index, and tensor data aligned to 16-byte boundaries. Includes optimizer state and training metadata for resumption.

## Notes

This code is experimental and untested. Might have bugs or numerical issues. Use at your own risk. Mainly for research and exploration not production.

Model needs substantial data to learn byte-level patterns. Small datasets won't see much benefit over tokenization since model learns subword structure from scratch.

Finding optimal hybrid pattern requires experimentation. No theoretical guidance on what pattern works best for what tasks, so gotta try different ones and see.

## References

- Mamba-2: "Transformers are SSMs" (Dao & Gu, 2024)
- BLT: "Byte Latent Transformer" (Pagnoni et al., 2024)
- Transformer: "Attention Is All You Need" (Vaswani et al., 2017)
- RoPE: "RoFormer: Enhanced Transformer with Rotary Position Embedding" (Su et al., 2021)
- SwiGLU: "GLU Variants Improve Transformer" (Shazeer, 2020)
- RMSNorm: "Root Mean Square Layer Normalization" (Zhang & Sennrich, 2019)
