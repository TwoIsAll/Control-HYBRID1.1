# ControlH1: Hybrid Byte-Level Language Model

Experimental hybrid model mixing Mamba-2, Transformer, and Byte Latent Transformer. Works on raw bytes with entropy-based dynamic patching.

Current config: ~34M parameters.

## Architecture

**Frontend**: Takes raw bytes, embeds them, entropy predictor segments into patches. High entropy = more granular, low entropy = compressed. Entropy is precomputed before training for speed.

**Middle**: Hybrid backbone with Transformer and Mamba-2 layers. Pattern "mmmt" = 3 Mamba + 1 Transformer. More Mamba = faster, more Transformer = better attention.

**Backend**: Local transformer converts patches back to byte-level predictions.

## Why bytes

No tokenization artifacts, true multilingual. Model learns subword structure from data. Dynamic patching lets model decide where to spend compute.

## Future upgrades

Optimizing and bug fixing.
Hybrid pattern improvements (e.g. "mmmtt")

## Config (config.py)

- vocab_size: 256
- context_len: 512 bytes
- d_model: 128
- n_layers: 9
- n_heads: 8
- hybrid_pattern: "mmmt"
- max_patch_len: 16
- gradient_checkpointing: True

## Running

Check params:
```
python model.py
```

Train:
```
python train.py pretrain --data train-00000-of-00001.parquet --epochs 5 --batch-size 32 --stride 512
python train.py finetune --data tiny.jsonl --epochs 15 --batch-size 8
```

Resume from checkpoint:
```
python train.py pretrain --data train-00000-of-00001.parquet --epochs 5 --batch-size 32 --stride 512 --resume checkpoints/epoch_XXXX_step_XXXXXXXX.hwcf
```

Entropy is precomputed automatically. Checkpoints include optimizer state and entropy data for resuming.

## Notes

Experimental code, might have bugs. Needs substantial data to learn byte patterns. Try different hybrid patterns to find what works.

## References

- Mamba-2: Dao & Gu 2024
- BLT: Pagnoni et al 2024
- Transformer: Vaswani et al 2017
