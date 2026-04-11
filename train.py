from __future__ import annotations
import argparse
import json
import math
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import pandas as pd
from config import DEFAULT_HYBRID_CONFIG
from model import (
    HybridConfig,
    ControlH1Model,
    append_tagged_turn,
    build_optimizer_param_groups,
    bytes_from_text,
    causal_lm_targets,
    format_param_report,
)

def read_text_file(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return f.read()

def read_jsonl(path: str) -> List[Dict[str, object]]:
    rows = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

def read_parquet_records(path: str) -> List[Dict[str, object]]:
    return pd.read_parquet(path).to_dict(orient="records")

def conversation_record_to_bytes(rec: Dict[str, object]) -> List[int]:
    out: List[int] = []
    if isinstance(rec.get("messages"), list):
        for msg in rec["messages"]:
            if isinstance(msg, dict):
                append_tagged_turn(out, str(msg.get("role", "user")), str(msg.get("content", "")), True)
        return out
    if "prompt" in rec and "response" in rec:
        append_tagged_turn(out, "user", str(rec.get("prompt", "")), False)
        out.extend(bytes_from_text("\n"))
        append_tagged_turn(out, "assistant", str(rec.get("response", "")), True)
        return out
    append_tagged_turn(out, "user", str(rec.get("text", "")), True)
    return out

def pretrain_record_to_bytes(rec: Dict[str, object]) -> List[int]:
    return bytes_from_text(str(rec.get("text", json.dumps(rec, ensure_ascii=False))))

def load_data_as_byte_stream(paths: Sequence[str], mode: str) -> List[int]:
    data: List[int] = []
    for p in paths:
        ext = Path(p).suffix.lower()
        if ext == ".txt":
            data.extend(bytes_from_text(read_text_file(p)))
        elif ext == ".jsonl":
            rows = read_jsonl(p)
            fn = conversation_record_to_bytes if mode == "finetune" else pretrain_record_to_bytes
            for r in rows:
                data.extend(fn(r))
        elif ext in (".parquet", ".pq"):
            for r in read_parquet_records(p):
                data.extend(pretrain_record_to_bytes(r))
        else:
            raise ValueError(f"Unsupported input extension: {ext} for file {p}")
        data.extend(bytes_from_text("\n"))
    return data

class ByteSequenceDataset(Dataset):

    def __init__(self, byte_stream: Sequence[int], sequence_length: int, stride: Optional[int] = None, random_offset: bool = False, precomputed_entropy: Optional[torch.Tensor] = None) -> None:
        self.sequence_length = sequence_length
        self.stride = stride or sequence_length
        self.random_offset = random_offset
        self.data = torch.tensor(byte_stream, dtype=torch.long)
        self.precomputed_entropy = precomputed_entropy
        self.starts = list(range(0, self.data.numel() - sequence_length - 1, self.stride))

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        s = self.starts[idx]
        if self.random_offset:
            s = min(s + random.randint(0, max(0, self.stride - 1)), self.data.numel() - self.sequence_length - 1)
        x = self.data[s:s + self.sequence_length]
        output = {"input_ids": x, "labels": causal_lm_targets(x)}
        if self.precomputed_entropy is not None:
            output["entropy"] = self.precomputed_entropy[s:s + self.sequence_length]
        return output

def make_dataloader(dataset, batch_size, shuffle, num_workers, pin_memory):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=True, num_workers=num_workers, pin_memory=pin_memory)

@dataclass
class TrainState:
    epoch: int = 0
    global_step: int = 0
    best_loss: float = 1e9

def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def build_optimizer(model, lr, wd, betas):
    return torch.optim.AdamW(build_optimizer_param_groups(model, weight_decay=wd), lr=lr, betas=betas, eps=1e-8)

def build_scheduler(optimizer, warmup_steps, total_steps, min_lr_ratio=0.1):
    def f(step):
        if step < warmup_steps:
            return (step + 1) / max(1.0, warmup_steps)
        p = min(max((step - warmup_steps) / max(1.0, total_steps - warmup_steps), 0.0), 1.0)
        return min_lr_ratio + (1.0 - min_lr_ratio) * (0.5 * (1.0 + math.cos(math.pi * p)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, f)

def maybe_autocast(device, enabled):
    if enabled and device.type == "cuda":
        return torch.cuda.amp.autocast(dtype=torch.bfloat16)
    class D:
        def __enter__(self): return None
        def __exit__(self, *a): return False
    return D()

def evaluate(model, loader, device, max_batches=64, amp=True):
    model.eval()
    losses = []
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= max_batches:
                break
            x = batch["input_ids"].to(device)
            y = batch["labels"].to(device)
            with maybe_autocast(device, amp):
                loss = model(x, labels=y, return_aux=True)["loss"]
            losses.append(loss.item())
    model.train()
    if not losses:
        return {"loss": 0.0, "ppl": 0.0}
    avg = sum(losses) / len(losses)
    return {"loss": avg, "ppl": float(math.exp(min(20, avg)))}

def save_epoch_checkpoint(model, optimizer, scheduler, output_dir, state, tag="epoch", entropy_data=None):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"{tag}_{state.epoch:04d}_step_{state.global_step:08d}.hwcf")
    model.save_hwcf(path, optimizer_state={"optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "train_state": vars(state)}, extra_meta={k: str(v) for k, v in vars(state).items()}, entropy_data=entropy_data)
    return path

def load_checkpoint_if_any(model, optimizer, scheduler, resume_path, device):
    state = TrainState()
    if not resume_path:
        return state, None
    loaded_model, opt_state, meta, entropy_data = ControlH1Model.load_hwcf(resume_path, map_location=device, load_optimizer=True)
    model.load_state_dict(loaded_model.state_dict(), strict=True)
    if opt_state:
        if "optimizer" in opt_state:
            optimizer.load_state_dict(opt_state["optimizer"])
        if "scheduler" in opt_state:
            scheduler.load_state_dict(opt_state["scheduler"])
        ts = opt_state.get("train_state", {})
        state.epoch = int(ts.get("epoch", meta.get("epoch", 0)))
        state.global_step = int(ts.get("global_step", meta.get("global_step", 0)))
        state.best_loss = float(ts.get("best_loss", meta.get("best_loss", 1e9)))
    return state, entropy_data

def precompute_entropy(data: Sequence[int], config: HybridConfig, device: torch.device, batch_size: int = 32) -> torch.Tensor:
    from model import BLTFrontEnd, ByteEmbedding
    byte_emb = ByteEmbedding(config.d_model, dropout=config.byte_dropout).to(device)
    entropy_pred = BLTFrontEnd(config).entropy_pred.to(device)
    byte_emb.eval()
    entropy_pred.eval()
    data_tensor = torch.tensor(data, dtype=torch.long, device=device)
    entropy_values = []
    with torch.no_grad():
        for i in tqdm(range(0, len(data), config.context_len), desc="Precomputing entropy"):
            chunk = data_tensor[i:i + config.context_len].unsqueeze(0)
            h = byte_emb(chunk)
            entropy = entropy_pred(h).squeeze(0)
            entropy_values.append(entropy.cpu())
    return torch.cat(entropy_values, dim=0)[:len(data)]

def train_loop(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    config = HybridConfig.from_dict(json.load(open(args.config_json))) if args.config_json else HybridConfig.from_dict(DEFAULT_HYBRID_CONFIG)
    data = load_data_as_byte_stream(args.data, args.mode)
    if args.max_data_bytes > 0:
        data = data[:args.max_data_bytes]
    split = int(len(data) * (1.0 - args.val_ratio))
    train_bytes = data[:split]
    val_bytes = data[split:] if split < len(data) else data[-config.context_len * 4:]
    if getattr(args, 'precompute_entropy', True):
        print("Precomputing entropy...")
        train_entropy = precompute_entropy(train_bytes, config, device)
        val_entropy = precompute_entropy(val_bytes, config, device)
        entropy_data = {"train": train_entropy, "val": val_entropy}
    else:
        train_entropy = None
        val_entropy = None
        entropy_data = None
    train_loader = make_dataloader(ByteSequenceDataset(train_bytes, config.context_len, args.stride, args.random_offset, train_entropy), args.batch_size, True, args.num_workers, device.type == "cuda")
    val_loader = make_dataloader(ByteSequenceDataset(val_bytes, config.context_len, max(1, config.context_len // 2), False, val_entropy), args.batch_size, False, args.num_workers, device.type == "cuda")
    model = ControlH1Model(config).to(device)
    print(format_param_report(model))
    optimizer = build_optimizer(model, args.lr, args.weight_decay, (args.beta1, args.beta2))
    scheduler = build_scheduler(optimizer, args.warmup_steps, max(1, args.epochs * len(train_loader)))
    scaler = torch.amp.GradScaler('cuda', enabled=(args.amp and device.type == "cuda"))
    state, _ = load_checkpoint_if_any(model, optimizer, scheduler, args.resume, device)
    model.train()
    os.makedirs(args.output_dir, exist_ok=True)
    for epoch in range(state.epoch, args.epochs):
        state.epoch = epoch + 1
        loss_acc = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {state.epoch}/{args.epochs}")
        for batch in pbar:
            x = batch["input_ids"].to(device)
            y = batch["labels"].to(device)
            entropy = batch["entropy"].to(device) if "entropy" in batch else None
            optimizer.zero_grad(set_to_none=True)
            with maybe_autocast(device, args.amp):
                out = model(x, labels=y, precomputed_entropy=entropy, return_aux=True)
                loss = out["loss"]
                loss_scaled = loss / args.grad_accum_steps
            if scaler.is_enabled():
                scaler.scale(loss_scaled).backward()
            else:
                loss_scaled.backward()
            if (state.global_step + 1) % args.grad_accum_steps == 0:
                if args.grad_clip > 0:
                    if scaler.is_enabled():
                        scaler.unscale_(optimizer)
                    clip_grad_norm_(model.parameters(), args.grad_clip)
                if scaler.is_enabled():
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                scheduler.step()
            state.global_step += 1
            loss_acc += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "avg_loss": f"{loss_acc/(state.global_step - (epoch-1)*len(train_loader)):.4f}"})
        val = evaluate(model, val_loader, device, args.eval_batches, args.amp)
        print(f"Epoch {state.epoch}: train_loss={loss_acc/len(train_loader):.5f} val_loss={val['loss']:.5f} val_ppl={val['ppl']:.3f}")
        save_epoch_checkpoint(model, optimizer, scheduler, args.output_dir, state, entropy_data=entropy_data)
        if val["loss"] < state.best_loss:
            state.best_loss = val["loss"]
            save_epoch_checkpoint(model, optimizer, scheduler, args.output_dir, state, "best", entropy_data=entropy_data)

def add_common_model_args(p):
    p.add_argument("--config-json", type=str, default="")
    p.add_argument("--context-len", type=int, default=2048)
    p.add_argument("--d-model", type=int, default=160)
    p.add_argument("--latent-dim", type=int, default=160)
    p.add_argument("--n-layers", type=int, default=10)
    p.add_argument("--n-heads", type=int, default=5)
    p.add_argument("--d-ff", type=int, default=416)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--hybrid-pattern", type=str, default="tmtmtmtmtt")
    p.add_argument("--local-encoder-layers", type=int, default=2)
    p.add_argument("--local-decoder-layers", type=int, default=2)
    p.add_argument("--max-patch-len", type=int, default=12)
    p.add_argument("--min-patch-len", type=int, default=1)
    p.add_argument("--patch-entropy-threshold", type=float, default=2.2)
    p.add_argument("--mamba-state-dim", type=int, default=48)
    p.add_argument("--mamba-expand", type=int, default=2)
    p.add_argument("--mamba-conv-kernel", type=int, default=3)

def add_common_train_args(p):
    p.add_argument("--output-dir", type=str, default="checkpoints")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--stride", type=int, default=1024)
    p.add_argument("--random-offset", action="store_true")
    p.add_argument("--precompute-entropy", action="store_true", default=True)
    p.add_argument("--val-ratio", type=float, default=0.02)
    p.add_argument("--eval-batches", type=int, default=64)
    p.add_argument("--max-data-bytes", type=int, default=0)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--beta1", type=float, default=0.9)
    p.add_argument("--beta2", type=float, default=0.95)
    p.add_argument("--warmup-steps", type=int, default=100)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--grad-accum-steps", type=int, default=1)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--resume", type=str, default="")

def build_parser():
    parser = argparse.ArgumentParser(prog="train.py")
    sub = parser.add_subparsers(dest="command", required=True)
    pre = sub.add_parser("pretrain")
    pre.add_argument("--data", nargs="+", required=True)
    add_common_model_args(pre)
    add_common_train_args(pre)
    pre.set_defaults(mode="pretrain")
    ft = sub.add_parser("finetune")
    ft.add_argument("--data", nargs="+", required=True)
    add_common_model_args(ft)
    add_common_train_args(ft)
    ft.set_defaults(mode="finetune")
    infer = sub.add_parser("sample")
    infer.add_argument("--checkpoint", type=str, required=True)
    infer.add_argument("--prompt", type=str, default="<|user|>\nHello\n<|end|>\n<|assistant|>\n")
    infer.add_argument("--max-new-tokens", type=int, default=256)
    infer.add_argument("--temperature", type=float, default=0.9)
    infer.add_argument("--top-k", type=int, default=50)
    infer.add_argument("--cpu", action="store_true")
    return parser

def run_sample(args):
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    model, _, meta, _ = ControlH1Model.load_hwcf(args.checkpoint, map_location=device, load_optimizer=False)
    model.eval()
    x = torch.tensor([bytes_from_text(args.prompt)], dtype=torch.long, device=device)
    with torch.no_grad():
        y = model.generate(x, max_new_tokens=args.max_new_tokens, temperature=args.temperature, top_k=args.top_k)
    print(meta)
    print(bytes(y[0].tolist()).decode("utf-8", errors="replace"))

def main():
    args = build_parser().parse_args()
    if args.command in ("pretrain", "finetune"):
        train_loop(args)
    elif args.command == "sample":
        run_sample(args)
    else:
        raise ValueError(args.command)

if __name__ == "__main__":
    main()
