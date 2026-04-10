# Control-H1

A hybrid architecture mixing ideas from BLT (Byte Latent Transformer), Mamba-2 (SSD), and the original Transformer.

Works directly on raw bytes instead of tokens, with dynamic patching and a mixed attention/state-space core.

Still experimental. Built to try out different architecture ideas, not for production.

## What’s inside

- Byte-level input (no tokenizer)
- Entropy-based patching
- Latent compression stage
- Hybrid stack (Transformer + Mamba-style blocks)
- Custom checkpoint format

## Files

- model.py – model definition
- train.py – training loop
- config.py – config

## Usage

Train:
```bash
python train.py pretrain --data data.txt
```

Finetune:
```bash
python train.py finetune --data data.jsonl
```

Sample:
```bash
python train.py sample --checkpoint model.hwcf
```

## Notes

This is a research project. Expect rough edges.
