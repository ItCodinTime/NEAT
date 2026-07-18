"""Run one reproducible Multi-MNIST or Multi-Fashion-MNIST trial."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import resource
import sys
import time
from pathlib import Path

import numpy as np

from benchmarks.manifest import reproducibility_metadata, seed_libraries, write_json
from benchmarks.sts import DATASETS, METHODS, PAPER_SEEDS
from benchmarks.sts.gradients import (
    assign_gradients,
    cosine_between,
    gradient_geometry,
    mean_gradients,
    mgda_two_task,
    pcgrad_two_task,
    task_gradients,
)

SETTINGS = ("multi", "single_task_1", "single_task_2")
NEAT_VARIANTS = (
    "base",
    "no_correction",
    "adaptive_alpha",
    "adaptive_preconditioning",
    "opponent_previous_gradient",
    "opponent_gradient_ema",
    "conflict_threshold",
    "correction_warmup",
    "lion",
    "lookahead",
    "gradient_centralization",
    "nesterov",
)
DATASET_SPEC_VERSION = "fixed-overlap-v1"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=DATASETS, default="multi_mnist")
    parser.add_argument("--method", choices=METHODS, default="neat")
    parser.add_argument("--setting", choices=SETTINGS, default="multi")
    parser.add_argument("--neat-variant", choices=NEAT_VARIANTS, default="base")
    parser.add_argument("--seed", type=int, default=PAPER_SEEDS[0])
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--eval-limit", type=int, default=None)
    parser.add_argument("--data-dir", default="data/sts")
    parser.add_argument("--output-dir", default="benchmark-runs/sts")
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--deterministic", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--skip-test", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def _require_torch():
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Install PyTorch and torchvision to run STS benchmarks."
        ) from exc
    return torch


class PairedVisionDataset:
    """Overlay two fixed-order source images on a 36x36 canvas."""

    def __init__(self, base, indices, *, pairing_seed: int, limit: int | None = None):
        torch = _require_torch()
        self.base = base
        self.indices = torch.as_tensor(indices, dtype=torch.int64)
        if limit is not None:
            self.indices = self.indices[:limit]
        generator = torch.Generator().manual_seed(pairing_seed)
        self.partner_order = torch.randperm(len(self.indices), generator=generator)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int):
        torch = _require_torch()
        first_index = int(self.indices[item])
        second_index = int(self.indices[int(self.partner_order[item])])
        first_image, first_label = self.base[first_index]
        second_image, second_label = self.base[second_index]
        canvas = torch.zeros((1, 36, 36), dtype=first_image.dtype)
        canvas[:, :28, :28] = first_image
        canvas[:, 8:36, 8:36] = torch.maximum(canvas[:, 8:36, 8:36], second_image)
        return canvas, torch.tensor(first_label), torch.tensor(second_label)


def dataset_spec_hash(dataset: str) -> str:
    payload = {
        "dataset": dataset,
        "spec_version": DATASET_SPEC_VERSION,
        "canvas": [36, 36],
        "first_origin": [0, 0],
        "second_origin": [8, 8],
        "composition": "pixelwise_max",
        "split_seed": 20260718,
        "train_pairing_seed": 314159,
        "validation_pairing_seed": 271828,
        "test_pairing_seed": 161803,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def make_datasets(args):
    torch = _require_torch()
    try:
        from torchvision import datasets, transforms
    except ImportError as exc:
        raise SystemExit("Install torchvision to run STS benchmarks.") from exc
    dataset_type = (
        datasets.MNIST if args.dataset == "multi_mnist" else datasets.FashionMNIST
    )
    transform = transforms.ToTensor()
    base_train = dataset_type(
        args.data_dir, train=True, download=True, transform=transform
    )
    base_test = dataset_type(
        args.data_dir, train=False, download=True, transform=transform
    )
    split_generator = torch.Generator().manual_seed(20260718)
    order = torch.randperm(len(base_train), generator=split_generator)
    train_indices, valid_indices = order[:50000], order[50000:]
    train = PairedVisionDataset(
        base_train,
        train_indices,
        pairing_seed=314159,
        limit=args.train_limit,
    )
    valid = PairedVisionDataset(
        base_train,
        valid_indices,
        pairing_seed=271828,
        limit=args.eval_limit,
    )
    test = PairedVisionDataset(
        base_test,
        torch.arange(len(base_test)),
        pairing_seed=161803,
        limit=args.eval_limit,
    )
    return train, valid, test


def make_loaders(args):
    torch = _require_torch()
    train, valid, test = make_datasets(args)
    generator = torch.Generator().manual_seed(args.seed)
    common = {
        "batch_size": args.batch_size,
        "num_workers": args.workers,
        "persistent_workers": args.workers > 0,
    }
    train_loader = torch.utils.data.DataLoader(
        train, shuffle=True, generator=generator, **common
    )
    valid_loader = torch.utils.data.DataLoader(valid, shuffle=False, **common)
    test_loader = torch.utils.data.DataLoader(test, shuffle=False, **common)
    return train_loader, valid_loader, test_loader


def make_model():
    torch = _require_torch()

    class MultiHeadCNN(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.features = torch.nn.Sequential(
                torch.nn.Conv2d(1, 32, 3, padding=1),
                torch.nn.ReLU(),
                torch.nn.MaxPool2d(2),
                torch.nn.Conv2d(32, 64, 3, padding=1),
                torch.nn.ReLU(),
                torch.nn.MaxPool2d(2),
                torch.nn.Conv2d(64, 64, 3, padding=1),
                torch.nn.ReLU(),
                torch.nn.AdaptiveAvgPool2d((4, 4)),
            )
            self.shared = torch.nn.Sequential(
                torch.nn.Flatten(),
                torch.nn.Linear(64 * 4 * 4, 128),
                torch.nn.ReLU(),
            )
            self.task_1 = torch.nn.Linear(128, 10)
            self.task_2 = torch.nn.Linear(128, 10)

        def forward(self, images):
            representation = self.shared(self.features(images))
            return self.task_1(representation), self.task_2(representation)

    return MultiHeadCNN()


def select_device(name: str):
    torch = _require_torch()
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def neat_overrides(variant: str, *, steps_per_epoch: int) -> dict[str, object]:
    overrides: dict[str, object] = {}
    if variant == "no_correction":
        overrides["nce_mode"] = "off"
    elif variant == "adaptive_alpha":
        overrides["adaptive_alpha"] = True
    elif variant == "adaptive_preconditioning":
        overrides["adaptive_preconditioning"] = True
    elif variant == "opponent_previous_gradient":
        overrides["opponent_source"] = "previous_gradient"
    elif variant == "opponent_gradient_ema":
        overrides["opponent_source"] = "gradient_ema"
    elif variant == "conflict_threshold":
        overrides["conflict_threshold"] = 0.05
    elif variant == "correction_warmup":
        overrides["correction_warmup_steps"] = steps_per_epoch
    elif variant == "lion":
        overrides["update_mode"] = "lion"
    elif variant == "lookahead":
        overrides["lookahead_k"] = 5
    elif variant == "gradient_centralization":
        overrides["gradient_centralization"] = True
    elif variant == "nesterov":
        overrides["nesterov"] = True
    return overrides


def build_optimizer(args, model, *, steps_per_epoch: int):
    torch = _require_torch()
    if args.method == "neat":
        from neat_optim import TorchNEAT

        overrides = neat_overrides(args.neat_variant, steps_per_epoch=steps_per_epoch)
        return TorchNEAT(
            model.parameters(),
            learning_rate=args.lr,
            weight_decay=args.weight_decay,
            diagnostic_interval=1,
            alpha=0.25,
            beta=0.9,
            opponent_source=overrides.pop("opponent_source", "momentum"),
            **overrides,
        )
    return torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )


def active_losses(setting: str, outputs, labels):
    torch = _require_torch()
    losses = (
        torch.nn.functional.cross_entropy(outputs[0], labels[0]),
        torch.nn.functional.cross_entropy(outputs[1], labels[1]),
    )
    if setting == "single_task_1":
        return (losses[0],)
    if setting == "single_task_2":
        return (losses[1],)
    return losses


def train_epoch(args, model, loader, optimizer, device):
    model.train()
    parameters = tuple(model.parameters())
    losses_seen = []
    correction_ratios = []
    alignments = []
    if hasattr(optimizer, "reset_diagnostics"):
        optimizer.reset_diagnostics()
    for images, first, second in loader:
        images = images.to(device)
        labels = (first.to(device), second.to(device))
        optimizer.zero_grad(set_to_none=True)
        losses = active_losses(args.setting, model(images), labels)
        if args.method in {"pcgrad", "mgda"}:
            if args.setting != "multi":
                raise ValueError(f"{args.method} requires --setting multi")
            rows = task_gradients(losses, parameters)
            raw = mean_gradients(rows)
            combined = (
                pcgrad_two_task(rows)
                if args.method == "pcgrad"
                else mgda_two_task(rows)
            )
            ratio, alignment = gradient_geometry(raw, combined)
            correction_ratios.append(float(ratio.detach()))
            alignments.append(float(alignment.detach()))
            assign_gradients(parameters, combined)
        else:
            sum(losses).div(len(losses)).backward()
        optimizer.step()
        losses_seen.append(float(sum(loss.detach() for loss in losses) / len(losses)))
    if args.method == "neat":
        diagnostics = optimizer.diagnostic_snapshot()
        correction_ratio = diagnostics.get("mean_correction_ratio", 0.0)
        alignment = diagnostics.get("mean_update_alignment", 1.0)
    elif correction_ratios:
        diagnostics = {}
        correction_ratio = float(np.mean(correction_ratios))
        alignment = float(np.mean(alignments))
    else:
        diagnostics = {}
        correction_ratio, alignment = 0.0, 1.0
    return {
        "train_loss": float(np.mean(losses_seen)),
        "train_loss_variance": float(np.var(losses_seen)),
        "correction_ratio": correction_ratio,
        "gradient_update_alignment": alignment,
        **diagnostics,
    }


def evaluate(model, loader, device) -> dict[str, float]:
    torch = _require_torch()
    model.eval()
    correct = [0, 0]
    total = 0
    loss_sum = 0.0
    with torch.inference_mode():
        for images, first, second in loader:
            images = images.to(device)
            labels = (first.to(device), second.to(device))
            outputs = model(images)
            loss_sum += float(
                sum(
                    torch.nn.functional.cross_entropy(output, label, reduction="sum")
                    for output, label in zip(outputs, labels, strict=True)
                )
            )
            for index in range(2):
                correct[index] += int((outputs[index].argmax(1) == labels[index]).sum())
            total += images.shape[0]
    first_accuracy, second_accuracy = (value / total for value in correct)
    return {
        "loss": loss_sum / (2 * total),
        "task_1_accuracy": first_accuracy,
        "task_2_accuracy": second_accuracy,
        "average_accuracy": (first_accuracy + second_accuracy) / 2,
    }


def task_gradient_cosine(model, loader, device) -> float:
    model.train()
    parameters = tuple(model.parameters())
    images, first, second = next(iter(loader))
    images = images.to(device)
    labels = (first.to(device), second.to(device))
    rows = task_gradients(active_losses("multi", model(images), labels), parameters)
    cosine = float(cosine_between(rows).detach())
    model.zero_grad(set_to_none=True)
    return cosine


def peak_memory_mb(device) -> float:
    torch = _require_torch()
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated(device) / (1024**2)
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if platform.system() == "Darwin":
        return peak / (1024**2)
    return peak / 1024


def result_path(args) -> Path:
    variant = args.neat_variant if args.method == "neat" else "base"
    return (
        Path(args.output_dir)
        / "runs"
        / args.dataset
        / args.setting
        / args.method
        / variant
        / f"seed_{args.seed}"
        / "result.json"
    )


def main(argv=None) -> None:
    args = parse_args(argv)
    if args.method != "neat" and args.neat_variant != "base":
        raise SystemExit("--neat-variant applies only to --method neat")
    destination = result_path(args)
    if destination.exists() and not args.overwrite:
        print(f"complete result exists, skipping: {destination}")
        return
    torch = _require_torch()
    seed_libraries(args.seed)
    if args.deterministic:
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
    device = select_device(args.device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    train_loader, valid_loader, test_loader = make_loaders(args)
    model = make_model().to(device)
    optimizer = build_optimizer(args, model, steps_per_epoch=len(train_loader))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    destination.parent.mkdir(parents=True, exist_ok=True)
    config = {
        **vars(args),
        "dataset_spec_version": DATASET_SPEC_VERSION,
        "dataset_spec_sha256": dataset_spec_hash(args.dataset),
        "train_examples": len(train_loader.dataset),
        "validation_examples": len(valid_loader.dataset),
        "test_examples": len(test_loader.dataset),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "optimizer_config": (
            optimizer.config.as_dict()
            if hasattr(optimizer, "config")
            else {
                key: value
                for key, value in optimizer.defaults.items()
                if isinstance(value, (int, float, bool, str, type(None)))
            }
        ),
    }
    write_json(destination.parent / "config.json", config)
    metrics_path = destination.parent / "metrics.jsonl"
    metrics_path.unlink(missing_ok=True)
    started_run = time.perf_counter()
    history = []
    for epoch in range(1, args.epochs + 1):
        started_epoch = time.perf_counter()
        train_metrics = train_epoch(args, model, train_loader, optimizer, device)
        training_seconds = time.perf_counter() - started_epoch
        started_evaluation = time.perf_counter()
        validation = evaluate(model, valid_loader, device)
        evaluation_seconds = time.perf_counter() - started_evaluation
        started_diagnostic = time.perf_counter()
        diagnostic_cosine = (
            task_gradient_cosine(model, valid_loader, device)
            if args.setting == "multi"
            else None
        )
        diagnostic_seconds = time.perf_counter() - started_diagnostic
        row = {
            "epoch": epoch,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "seconds": time.perf_counter() - started_epoch,
            "training_seconds": training_seconds,
            "evaluation_seconds": evaluation_seconds,
            "diagnostic_seconds": diagnostic_seconds,
            "task_gradient_cosine": diagnostic_cosine,
            **train_metrics,
            **{f"validation_{key}": value for key, value in validation.items()},
        }
        history.append(row)
        with metrics_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
        print(json.dumps(row, sort_keys=True), flush=True)
        scheduler.step()
    test = None if args.skip_test else evaluate(model, test_loader, device)
    wall_time = time.perf_counter() - started_run
    total_training_seconds = sum(row["training_seconds"] for row in history)
    provenance = reproducibility_metadata(
        seed=args.seed, device=device, invocation=list(sys.argv)
    )
    provenance["torch_num_threads"] = torch.get_num_threads()
    result = {
        "schema_version": 1,
        "study": "regeneron_sts_multitask_vision",
        "config": config,
        "history": history,
        "test": test,
        "wall_time_seconds": wall_time,
        "training_time_seconds": total_training_seconds,
        "mean_step_time_seconds": total_training_seconds
        / (args.epochs * len(train_loader)),
        "peak_memory_mb": peak_memory_mb(device),
        "diagnostic_semantics": {
            "adam": "identity correction before Adam transport",
            "pcgrad": "PCGrad-combined gradient relative to raw task mean",
            "mgda": "MGDA-combined gradient relative to raw task mean",
            "neat": (
                "Nash correction relative to the raw gradient before momentum transport"
            ),
        }[args.method],
        "reproducibility": provenance,
        "preliminary": args.train_limit is not None or args.eval_limit is not None,
    }
    write_json(destination, result)
    print(f"wrote {destination}")


if __name__ == "__main__":
    main()
