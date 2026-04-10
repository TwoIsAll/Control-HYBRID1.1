# ControlH1: Architecture Documentation

This document explores the design and implementation of a hybrid language model architecture that combines three distinct approaches to sequence modeling: **Mamba-2 state space models**, **Transformer attention**, and **Byte Latent Transformer (BLT)** principles for byte-level processing.

The implementation is experimental and serves as a testbed for understanding how these components can be integrated. The model operates at approximately 2M parameters and processes raw bytes directly, using entropy-based dynamic patching to determine where to allocate computational resources.

## Motivation

Most modern language models rely on tokenization schemes like BPE or WordPiece to convert text into discrete tokens before processing. This works well but introduces artifacts - token boundaries don't always align with semantic units, and the vocabulary is fixed at training time, limiting the model's ability to handle novel text patterns.

The BLT paper (Pagnoni et al., 2024) proposed an alternative: work directly on bytes and dynamically group them into "patches" based on their predictability. High-entropy (unpredictable) regions get more granular processing, while low-entropy (repetitive) regions are compressed. This is appealing because it lets the model decide where to spend computation rather than using a fixed vocabulary.

Meanwhile, Mamba-2 (Dao & Gu, 2024) showed that state space models can achieve performance competitive with Transformers while having linear-time complexity. The "structured state space duality" framework unifies SSMs and attention, suggesting they're two sides of the same coin.

This implementation explores what happens when you combine these ideas: a byte-level frontend with entropy-based patching, a hybrid backbone mixing Transformer and Mamba-2 layers, and a decoder that converts patches back to byte-level predictions.

**Important**: This code is experimental and untested. The implementation may contain bugs, numerical instability issues, or architectural errors. Use at your own risk. This is primarily a research exploration rather than production-ready software.

## Architecture Overview

The model consists of three stages:

### Frontend: Byte-to-Patch Conversion

Raw UTF-8 bytes are embedded directly (256-dimensional vocabulary) and passed through a small local transformer network. A lightweight MLP predictor estimates the entropy at each byte position. Based on these entropy values, the byte sequence is segmented into variable-length patches:

- If entropy exceeds a threshold, start a new patch
- Enforce minimum and maximum patch length constraints
- Patches are aggregated (mean, max, sum, or attention-weighted)

The intuition is that the model learns to identify "interesting" boundaries. Repetitive sequences (low entropy) get compressed into fewer patches, while complex or unpredictable content (high entropy) gets more granular processing.

### Middle: Hybrid Backbone

The patch representations pass through a configurable sequence of Transformer and Mamba-2 blocks. A pattern string determines the layer types - for example, `"tmttmtmtt"` means Transformer, Mamba, Transformer, Transformer, Mamba, etc.

**Transformer blocks** use:
- RoPE (rotary positional embeddings)
- SwiGLU activation in the feedforward network
- LayerScale for training stability
- RMSNorm for normalization

**Mamba-2 blocks** use:
- Selective SSM with learnable discretization parameters
- SSD dual mixer (essentially linear attention with ELU)
- Depthwise 1D convolution for local context
- Gating mechanism

The hybrid pattern is interesting because it lets you explore different trade-offs. Mamba layers are O(n) in sequence length, while Transformer layers are O(n²). The pattern lets you place attention where you think it matters most - perhaps at the beginning and end of the network, with SSMs in the middle.

### Backend: Patch-to-Byte Conversion

The processed patch representations are projected back to the byte dimension and scattered to their original positions. A local transformer network processes the byte-level representations, and a final head produces byte-level logits.

## Design Decisions

### Why Bytes?

Working directly on bytes eliminates tokenization artifacts and gives the model true multilingual support - it doesn't care what language or script the text is in. The trade-off is that the model has to learn subword structure from data, which might require more training data than a tokenized model.

### Why Entropy-Based Patching?

Fixed patch sizes would be simple but don't adapt to input complexity. Entropy-based patching lets the model dynamically decide where to spend computation. The entropy predictor is small (3-layer MLP) and trained end-to-end with the rest of the model.

The segmentation algorithm ensures patches respect minimum and maximum length constraints, which prevents pathological cases (e.g., single-byte patches or patches that grow without bound).

### Why Hybrid Layers?

The theoretical framework from Mamba-2 suggests that SSMs and attention are dual formulations of the same underlying operation. In practice, they have different computational characteristics - SSMs are linear-time but might lose some precision, while attention is quadratic but exact.

Mixing them lets you get the best of both worlds: SSMs for efficient long-range modeling, attention for precise retrieval when needed. The pattern string is a simple way to experiment with different arrangements without changing the code.

### Custom Checkpoint Format

The `.hwcf` format is a simple binary format that stores:
- Magic bytes for identification
- Version information
- Model configuration as JSON
- Tensor index (name, dtype, shape, offset, size)
- Tensor data (16-byte aligned)

This is inspired by formats like GGUF but simplified for this use case. It includes optimizer state and training metadata, enabling checkpoint resumption.

## Configuration

The default configuration targets ~2M parameters:

```python
{
    "vocab_size": 256,              # Byte vocabulary
    "context_len": 2048,            # Context window
    "d_model": 192,                 # Model dimension
    "n_layers": 12,                 # Total layers
    "n_heads": 6,                   # Attention heads
    "d_ff": 512,                    # FFN dimension
    "hybrid_pattern": "tmttmtmttmtt",  # Layer pattern
    "max_patch_len": 16,            # Maximum bytes per patch
    "min_patch_len": 1,             # Minimum bytes per patch
    "patch_entropy_threshold": 2.2,  # Entropy threshold
    "local_encoder_layers": 2,      # BLT encoder layers
    "local_decoder_layers": 2,      # BLT decoder layers
    "mamba_state_dim": 64,          # SSM state dimension
    "ssd_rank": 32,                 # SSD mixer rank
}
```

A smaller preset (`HybridConfig.tiny_2m_context2k()`) uses d_model=160, n_layers=10, and pattern `"tmtmtmtmtt"` for a tighter parameter budget.

## Implementation Notes

### Mamba-2 SSM

The SelectiveSSMCore implements the selective state space model with three scan modes:
- `parallel`: Full parallel scan using cumulative products (efficient for short sequences)
- `chunk`: Chunked scan with state passing between chunks (balanced)
- `recurrent`: Pure recurrent scan (for very long sequences or debugging)

The discretization uses the softplus function for the time step parameter dt, which ensures positivity. The state decay parameter A is learned in log-space for numerical stability.

### SSD Dual Mixer

The SSDDualMixer implements the "structured state space duality" formulation. It's essentially linear attention with ELU activation:

```
q = ELU(W_q x)
k = ELU(W_k x)
v = W_v x
KV = cumsum(k ⊗ v)
K = cumsum(k)
output = (q ⊗ KV) / K
```

This is mathematically equivalent to a certain class of SSMs, which is why it's called "dual" to SSMs.

### BLT Patching

The entropy predictor is a simple MLP: Linear → SiLU → Linear → SiLU → Linear → softplus. The softplus ensures the output is positive (entropy is non-negative).

The segmentation algorithm is greedy - it starts a new patch when either entropy exceeds threshold or the patch reaches maximum length. This is simple and works well in practice, though more sophisticated approaches could be explored.

### Training

The training script supports:
- Pretraining on raw text (`.txt`, `.jsonl`, `.parquet`)
- Finetuning on conversation data with role tagging
- AdamW optimizer with cosine decay and warmup
- Gradient clipping and mixed precision
- Per-epoch checkpointing with validation evaluation

Conversation data is tagged with role markers (`<|user|>`, `<|assistant|>`, `<|end|>`) for finetuning. The tagging is applied automatically when the input format includes role information.

### Generation

Autoregressive generation uses a sliding context window. At each step:
1. Truncate input to context_len
2. Run forward pass
3. Sample next byte (temperature + top-k)
4. Append and repeat

Dynamic patching can be disabled during generation (`force_uniform_patch=True`) for deterministic behavior, though this typically hurts quality since the model was trained with adaptive patching.

## Observations

### Parameter Efficiency

At ~2M parameters, the model is much smaller than modern LLMs (which are typically billions of parameters). The hybrid architecture helps make the most of limited parameters - SSM layers provide efficient long-range context without the quadratic cost of attention everywhere.

### Entropy Patterning

During training, you can observe the model learning to place patch boundaries at sensible locations. For English text, patches often align with word boundaries. For code, they might align with tokens or statements. The entropy threshold becomes a learned hyperparameter that balances compression vs. granularity.

### Layer Pattern Sensitivity

The choice of hybrid pattern affects both performance and efficiency. More Transformer layers (pattern with more 't's) give better retrieval but slower inference. More Mamba layers (more 'm's) are faster but might lose some precision. The optimal pattern likely depends on the task and data distribution.

## Limitations

- **Scale**: At 2M parameters, the model is too small for serious language modeling tasks. It's primarily useful for architectural exploration.
- **Byte-level overhead**: Processing bytes directly can be less efficient than tokenization for very long repetitive sequences, though dynamic patching mitigates this.
- **Training data**: The model needs substantial data to learn meaningful byte-level patterns. Small datasets won't see much benefit over tokenization.
- **Entropy predictor**: The simple MLP might not capture complex boundary patterns. A more sophisticated predictor (e.g., a small transformer) could improve patching.

## Future Directions

Things worth exploring:

- **Scale up**: Larger models with more parameters to see if the architecture scales effectively
- **Pattern search**: Systematic exploration of optimal layer patterns for different tasks
- **Better patching**: Learned aggregation (attention-weighted instead of simple pooling), more sophisticated entropy models
- **Quantization**: The custom checkpoint format could support quantized weights for faster inference
- **Efficient scans**: CUDA kernels for the SSM scan operations
- **Alternative backends**: JAX or Triton implementations for better performance

## References

- Mamba-2: "Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality" (Dao & Gu, 2024)
- BLT: "Byte Latent Transformer: Patches Scale Better Than Tokens" (Pagnoni et al., 2024)
- Transformer: "Attention Is All You Need" (Vaswani et al., 2017)
- RoPE: "RoFormer: Enhanced Transformer with Rotary Position Embedding" (Su et al., 2021)
- SwiGLU: "GLU Variants Improve Transformer" (Shazeer, 2020)
