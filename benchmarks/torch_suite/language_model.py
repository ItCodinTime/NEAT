"""Fine-tune a small Hugging Face language model on SST-2 or Alpaca."""

from __future__ import annotations

import argparse
import math
import time

import numpy as np

from benchmarks.torch_suite.common import (
    EpochMetrics,
    ExperimentLogger,
    build_optimizer,
    require_torch,
    run_name,
    seed_everything,
    select_device,
)


def parse_args(argv=None):
    """Parse one sequence-classification or causal-LM fine-tuning trial."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("sst2", "alpaca"), default="sst2")
    parser.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--optimizer", choices=("neat", "adamw"), default="neat")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-eval-samples", type=int, default=None)
    parser.add_argument("--target", type=float, default=None)
    parser.add_argument("--lora", action="store_true")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", default="benchmark-runs")
    parser.add_argument("--amp", action="store_true")
    return parser.parse_args(argv)


def _alpaca_text(example: dict[str, str]) -> str:
    """Render an Alpaca record into a stable instruction/response template."""
    context = f"\nInput: {example['input']}" if example.get("input") else ""
    return (
        f"Instruction: {example['instruction']}{context}\n"
        f"Response: {example['output']}"
    )


def prepare(args):
    """Load, tokenize, and batch the chosen Hugging Face task lazily."""
    torch = require_torch()
    try:
        from datasets import load_dataset
        from transformers import (
            AutoModelForCausalLM,
            AutoModelForSequenceClassification,
            AutoTokenizer,
        )
    except ImportError as exc:
        raise SystemExit(
            "Install transformers and datasets for the LLM suite."
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # SST-2 exposes a public classification label. Alpaca instead trains every
    # non-padding token with the standard causal language-model objective.
    if args.task == "sst2":
        raw = load_dataset("glue", "sst2")
        model = AutoModelForSequenceClassification.from_pretrained(
            args.model, num_labels=2
        )
        model.config.pad_token_id = tokenizer.pad_token_id

        def tokenize(batch):
            values = tokenizer(
                batch["sentence"],
                truncation=True,
                padding="max_length",
                max_length=args.max_length,
            )
            values["labels"] = batch["label"]
            return values

        train = raw["train"]
        valid = raw["validation"]
    else:
        raw = load_dataset("tatsu-lab/alpaca", split="train")
        split = raw.train_test_split(test_size=0.05, seed=args.seed)
        model = AutoModelForCausalLM.from_pretrained(args.model)
        model.config.pad_token_id = tokenizer.pad_token_id

        def tokenize(batch):
            examples = [
                dict(zip(batch, values, strict=True))
                for values in zip(*batch.values(), strict=True)
            ]
            encoded = tokenizer(
                [_alpaca_text(example) for example in examples],
                truncation=True,
                padding="max_length",
                max_length=args.max_length,
            )
            encoded["labels"] = [
                [
                    token if mask else -100
                    for token, mask in zip(row, attention, strict=True)
                ]
                for row, attention in zip(
                    encoded["input_ids"], encoded["attention_mask"], strict=True
                )
            ]
            return encoded

        train, valid = split["train"], split["test"]
    if args.max_train_samples:
        train = train.select(range(min(args.max_train_samples, len(train))))
    if args.max_eval_samples:
        valid = valid.select(range(min(args.max_eval_samples, len(valid))))
    remove_train = train.column_names
    remove_valid = valid.column_names
    train = train.map(tokenize, batched=True, remove_columns=remove_train)
    valid = valid.map(tokenize, batched=True, remove_columns=remove_valid)
    train.set_format("torch")
    valid.set_format("torch")
    if args.lora:
        try:
            from peft import LoraConfig, TaskType, get_peft_model
        except ImportError as exc:
            raise SystemExit("Install peft to use --lora.") from exc
        task_type = TaskType.SEQ_CLS if args.task == "sst2" else TaskType.CAUSAL_LM
        model = get_peft_model(
            model,
            LoraConfig(
                task_type=task_type, r=8, lora_alpha=16, lora_dropout=0.05
            ),
        )

    def collate(rows):
        return {key: torch.stack([row[key] for row in rows]) for key in rows[0]}

    loaders = (
        torch.utils.data.DataLoader(
            train, batch_size=args.batch_size, shuffle=True, collate_fn=collate
        ),
        torch.utils.data.DataLoader(
            valid, batch_size=args.batch_size, shuffle=False, collate_fn=collate
        ),
    )
    return model, loaders


def evaluate(model, loader, device, task: str) -> float:
    """Return accuracy for SST-2 or token perplexity for Alpaca."""
    torch = require_torch()
    model.eval()
    losses, correct, total = [], 0, 0
    with torch.inference_mode():
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            output = model(**batch)
            losses.append(float(output.loss))
            if task == "sst2":
                predictions = output.logits.argmax(dim=-1)
                correct += int((predictions == batch["labels"]).sum())
                total += predictions.numel()
    if task == "sst2":
        return correct / total
    return math.exp(min(20.0, float(np.mean(losses))))


def main(argv=None):
    """Fine-tune one model/optimizer/seed combination and record convergence."""
    args = parse_args(argv)
    torch = require_torch()
    seed_everything(args.seed)
    device = select_device(args.device)
    model, (train_loader, valid_loader) = prepare(args)
    model.to(device)
    trainable = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    optimizer = build_optimizer(args.optimizer, trainable, args.lr, args.weight_decay)
    logger = ExperimentLogger(
        args.output_dir,
        run_name(f"lm-{args.task}", args.optimizer, args.seed),
        vars(args),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")
    for epoch in range(1, args.epochs + 1):
        started = time.perf_counter()
        model.train()
        optimizer.zero_grad(set_to_none=True)
        losses = []
        # Gradient accumulation changes effective batch size without changing
        # the memory footprint of an individual forward pass.
        for step, batch in enumerate(train_loader, 1):
            batch = {key: value.to(device) for key, value in batch.items()}
            with torch.autocast(
                device_type=device.type,
                enabled=args.amp and device.type == "cuda",
            ):
                loss = model(**batch).loss / args.gradient_accumulation
            scaler.scale(loss).backward()
            if step % args.gradient_accumulation == 0 or step == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            losses.append(float(loss.detach()) * args.gradient_accumulation)
        metric = evaluate(model, valid_loader, device, args.task)
        logger.log(
            EpochMetrics(
                epoch,
                float(np.mean(losses)),
                metric,
                time.perf_counter() - started,
                float(np.var(losses)),
                optimizer.param_groups[0]["lr"],
            )
        )
    logger.finish(
        device=device,
        target=args.target,
        higher_is_better=args.task == "sst2",
        optimizer=optimizer,
        extra={
            "metric": "accuracy" if args.task == "sst2" else "perplexity",
            "trainable_parameters": sum(p.numel() for p in trainable),
        },
    )


if __name__ == "__main__":
    main()
