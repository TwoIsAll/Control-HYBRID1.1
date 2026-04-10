# ControlH1: Hybrid Byte-Level Language Model

This is an experimental implementation of a hybrid language model that combines Mamba-2 state space models, Transformer attention, and Byte Latent Transformer principles. The model works directly on bytes instead of tokens and uses entropy-based dynamic patching to decide where to spend computation.

The current configuration targets about 5M parameters.

## Architecture

The model has three main parts:

**Frontend**: Takes raw bytes, embeds them, and uses an entropy predictor to segment them into patches. High-entropy (complex) regions get more granular processing, low-entropy (repetitive) regions get compressed. The entropy predictor is a small transformer with multi-head self-attention that's trained end-to-end with the rest of the model. The transformer uses custom MultiHeadAttention, FeedForward, and TransformerEncoderLayer classes with descriptive variable names for better readability.

The segmentation algorithm is greedy - it starts a new patch when entropy exceeds a threshold or the patch reaches max length. This prevents pathological cases like single-byte patches or patches that grow without bound. Patches can be aggregated using mean, max, sum, or attention-weighted pooling.

**Middle**: A hybrid backbone with Transformer and Mamba-2 layers. The pattern string controls the layer types - for example "tmtmtm" means Transformer, Mamba, Transformer, Mamba, Transformer, Mamba. You can experiment with different patterns to balance speed and accuracy.

Transformer blocks use RoPE for positional encoding, SwiGLU activation, LayerScale for stability, and RMSNorm for normalization. These are standard modern transformer components.

Mamba-2 blocks use selective SSM with learnable discretization, SSD dual mixer (linear attention with ELU), depthwise 1D convolution for local context, and gating. The selective aspect means SSM parameters can adapt based on input.

**Backend**: Converts patches back to byte-level predictions using a local transformer. The patch representations are projected back to byte dimension and scattered to their original positions, then processed to produce byte-level logits.

## Why This Approach

Working on bytes eliminates tokenization artifacts and gives true multilingual support. The model doesn't care what language or script the text is in. Dynamic patching lets the model decide where to focus computation instead of using a fixed vocabulary.

The trade-off is that the model has to learn subword structure from data, which might require more training than a tokenized model that already has built-in subword knowledge.

Mamba layers are linear-time (efficient for long sequences) while Transformer layers are quadratic-time (better for precise attention). Mixing them lets you get the benefits of both. The pattern string is a simple way to experiment with different arrangements without changing code.

## SSM Scan Modes

The SelectiveSSMCore supports three scan modes:

- parallel: Full parallel scan using cumulative products, efficient for short sequences
- chunk: Chunked scan with state passing between chunks, balanced approach
- recurrent: Pure recurrent scan, useful for very long sequences or debugging

The discretization uses softplus for the time step parameter (ensures positivity) and learns state decay in log-space for numerical stability. The selective aspect means discretization parameters can vary based on input.

## Configuration

The config is in config.py. Key settings:

- vocab_size: 256 (one per byte)
- context_len: 2048 bytes
- d_model: 96
- n_layers: 6
- n_heads: 4
- d_ff: 256
- hybrid_pattern: "tmtmtm"
- max_patch_len: 16
- min_patch_len: 1
- patch_entropy_threshold: 2.2
- local_encoder_layers: 1
- local_decoder_layers: 1
- mamba_state_dim: 16
- mamba_conv_kernel: 3
- mamba_expand: 2
- ssd_rank: 16
- entropy_predictor_hidden: 64
- patch_embed_dim: 96
- patch_agg: "mean"
- ssm_scan_mode: "auto"

You can modify these to experiment with different sizes and architectures. The model.py file also has a tiny_2m_context2k() method for an even smaller config.

## Running

Run the model directly to see parameter count:
```
python model.py
```

Other commands:
- `python model.py info` - detailed model info
- `python model.py dryrun` - test forward pass
- `python model.py generate` - generate text
- `python model.py bench` - benchmark scan modes
- `python model.py ablate` - run ablation suite
- `python model.py smoke` - run smoke tests
- `python model.py export` - save model to .hwcf format
- `python model.py validate` - validate a .hwcf file

## Checkpoint Format

The .hwcf format is a simple binary format that stores magic bytes, version info, model config as JSON, tensor index (name, dtype, shape, offset, size), and tensor data aligned to 16-byte boundaries. It also includes optimizer state and training metadata for checkpoint resumption.

## Observations

At about 5M parameters, this is much smaller than modern LLMs (billions of parameters). The hybrid architecture helps make the most of limited parameters by using SSM layers for efficient long-range context without quadratic attention cost everywhere.

During training, you can observe the model learning to place patch boundaries at sensible locations. For English text, patches often align with word boundaries. For code, they might align with tokens or statements.

The choice of hybrid pattern affects performance and efficiency. More Transformer layers give better retrieval but slower inference. More Mamba layers are faster but might lose some precision for tasks requiring exact attention.

## Important Notes

This code is experimental and untested. It may contain bugs or numerical issues. Use at your own risk. This is primarily for research and architectural exploration, not production use.

The model needs substantial data to learn meaningful byte-level patterns. Small datasets won't see much benefit over tokenization since the model must learn subword structure from scratch.

The entropy predictor is a simple MLP which looks at each byte position independently. This might not capture complex boundary patterns where the decision to split depends on context from neighboring positions. A transformer-based predictor would use attention to look at relationships between different byte positions when deciding where to place patch boundaries, which could capture more complex patterns but would add more parameters.

Finding the optimal hybrid pattern for a given task requires experimentation. There's no theoretical guidance on what pattern works best for which tasks, so you need to try different patterns and evaluate.

## References

- Mamba-2: "Transformers are SSMs" (Dao & Gu, 2024)
- BLT: "Byte Latent Transformer" (Pagnoni et al., 2024)
- Transformer: "Attention Is All You Need" (Vaswani et al., 2017)
- RoPE: "RoFormer: Enhanced Transformer with Rotary Position Embedding" (Su et al., 2021)
- SwiGLU: "GLU Variants Improve Transformer" (Shazeer, 2020)
- RMSNorm: "Root Mean Square Layer Normalization" (Zhang & Sennrich, 2019)
- LayerScale: "On Layer Normalization in the Transformer Architecture" (Touvron et al., 2021)

## Future Work

Potential improvements include scaling up to larger models, systematic pattern search for optimal layer arrangements, learned aggregation instead of simple pooling, quantization support for faster inference, CUDA kernels for SSM scans, and JAX or Triton implementations for better performance.
