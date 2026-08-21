"""Fine-tune a small transformer on the Yelp fake/real review task.

Port of notebook Phase 2 (RoBERTa fine-tuning), with distilbert-base-uncased and
a plain-PyTorch loop. Entrypoint for `make train` / `make smoke`.

    python -m training.train [--sample N] [--epochs N] [--max_len N]
                             [--device cpu|mps] [--with-shap]
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from . import config as tc
from .data import clean_dataset, load_yelp, make_splits, save_splits


class ReviewDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels": self.labels[idx],
        }


def build_loader(texts, labels, tokenizer, max_length, batch_size, shuffle=False):
    ds = ReviewDataset(texts, labels, tokenizer, max_length)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def run_epoch(model, loader, device, optimizer=None, scheduler=None, scaler=None,
              use_amp=False, desc="Train"):
    """One training or evaluation pass. Returns (avg_loss, f1, preds, labels)."""
    training = optimizer is not None
    model.train() if training else model.eval()
    total_loss, all_preds, all_labels = 0.0, [], []
    ctx = torch.enable_grad() if training else torch.no_grad()
    n = len(loader)

    with ctx:
        for step, batch in enumerate(loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            if use_amp:
                with torch.autocast(device_type=device.type, dtype=torch.float16):
                    outputs = model(
                        input_ids=input_ids, attention_mask=attention_mask, labels=labels
                    )
                    loss = outputs.loss
            else:
                outputs = model(
                    input_ids=input_ids, attention_mask=attention_mask, labels=labels
                )
                loss = outputs.loss

            if training:
                optimizer.zero_grad()
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    # No AMP (CPU): plain backward + step.
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                scheduler.step()

            total_loss += loss.item()
            preds = outputs.logits.argmax(dim=-1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

            if training and (step + 1) % 50 == 0:
                print(f"    {desc} step {step + 1}/{n}  loss={loss.item():.4f}")

    avg_loss = total_loss / max(1, n)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    return avg_loss, f1, all_preds, all_labels


@torch.no_grad()
def predict_probs(model, tokenizer, device, texts, max_length, batch_size) -> np.ndarray:
    """P(Fake) for each text (used for the test-set probabilities)."""
    model.eval()
    probs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i: i + batch_size]
        enc = tokenizer(
            batch,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        logits = model(**enc).logits
        p = torch.softmax(logits.float(), dim=-1)[:, 1].cpu().numpy()
        probs.append(p)
    return np.concatenate(probs)


def train(args: argparse.Namespace):
    t0 = time.time()
    print(f"== Loading Yelp data (sample_n={args.sample}) ==")
    df = load_yelp(args.sample)
    df = clean_dataset(df)
    splits = make_splits(df)
    X_train, X_val, X_test, y_train, y_val, y_test = splits
    save_splits(*splits)
    print(f"Splits: train={len(X_train)} val={len(X_val)} test={len(X_test)}")

    device = torch.device(args.device)
    if device.type == "cpu":
        torch.set_num_threads(tc.NUM_THREADS)

    print(f"== Loading base model: {args.base_model} ==")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    model = AutoModelForSequenceClassification.from_pretrained(args.base_model, num_labels=2)
    model.to(device)

    train_loader = build_loader(X_train, y_train, tokenizer, args.max_len, args.batch, shuffle=True)
    val_loader = build_loader(X_val, y_val, tokenizer, args.max_len, args.batch)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=tc.WEIGHT_DECAY)
    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(total_steps * tc.WARMUP_RATIO)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )
    use_amp = device.type in ("cuda", "mps")
    scaler = torch.amp.GradScaler(device.type) if use_amp else None

    print(f"== Training on {device} ({args.epochs} epochs, {total_steps} steps) ==")
    best_f1 = 0.0
    for epoch in range(1, args.epochs + 1):
        train_loss, train_f1, _, _ = run_epoch(
            model, train_loader, device, optimizer, scheduler, scaler, use_amp, "Train"
        )
        val_loss, val_f1, _, _ = run_epoch(model, val_loader, device, desc="Val")
        print(
            f"Epoch {epoch}/{args.epochs} | Train loss {train_loss:.4f} F1 {train_f1:.4f} "
            f"| Val loss {val_loss:.4f} F1 {val_f1:.4f}"
        )
        if val_f1 > best_f1:
            best_f1 = val_f1
            tc.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(tc.OUTPUT_DIR)
            tokenizer.save_pretrained(tc.OUTPUT_DIR)
            print(f"  ✓ best model saved (val F1={best_f1:.4f})")

    # Test-set evaluation + probabilities (needed by HITL pool / examples).
    test_loader = build_loader(X_test, y_test, tokenizer, args.max_len, args.batch)
    test_loss, test_f1, all_preds, all_labels = run_epoch(model, test_loader, device, desc="Test")
    all_probs = predict_probs(model, tokenizer, device, X_test, args.max_len, args.batch)
    print(f"Test loss {test_loss:.4f} | Test F1 {test_f1:.4f}")

    tc.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    with open(tc.MODEL_DIR / "label_map.json", "w", encoding="utf-8") as f:
        json.dump({"fake": 1, "real": 0}, f, indent=2)
    with open(tc.MODEL_DIR / "training_meta.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "base_model": args.base_model,
                "sample_n": args.sample,
                "epochs": args.epochs,
                "max_length": args.max_len,
                "batch_size": args.batch,
                "device": args.device,
                "best_val_f1": best_f1,
                "test_f1": test_f1,
                "splits": {"train": len(X_train), "val": len(X_val), "test": len(X_test)},
                "preprocessing": "clean_text + remove_stopwords (canonical app.preprocess)",
                "elapsed_seconds": round(time.time() - t0, 1),
            },
            f,
            indent=2,
        )

    return model, tokenizer, device, (X_test, y_test), (all_preds, all_labels), all_probs


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Fine-tune the frede detector.")
    parser.add_argument("--sample", type=int, default=tc.SAMPLE_N)
    parser.add_argument("--epochs", type=int, default=tc.NUM_EPOCHS)
    parser.add_argument("--max_len", type=int, default=tc.MAX_LENGTH)
    parser.add_argument("--batch", type=int, default=tc.BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=tc.LEARNING_RATE)
    parser.add_argument("--device", default=tc.DEVICE, choices=["cpu", "mps"])
    parser.add_argument("--base-model", default=tc.BASE_MODEL_NAME)
    parser.add_argument("--skip-extras", action="store_true",
                        help="skip HITL pool + examples (training only)")
    parser.add_argument("--with-shap", action="store_true",
                        help="precompute SHAP for the Examples page (slow on CPU)")
    args = parser.parse_args(argv)

    model, tokenizer, device, (X_test, y_test), (all_preds, all_labels), all_probs = train(args)

    if not args.skip_extras:
        from . import examples, hitl_pool

        hitl_pool.build_hitl_pool(
            model, tokenizer, device, X_test, y_test, all_preds, all_probs
        )
        examples.build_examples(
            model, tokenizer, device, X_test, y_test,
            all_preds, all_probs, all_labels,
            with_shap=args.with_shap,
        )

    print("== training complete ==")


if __name__ == "__main__":
    main()
