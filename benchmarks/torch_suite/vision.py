"""CIFAR and ImageNet-folder classification with ResNet, ViT, or DeiT."""

from __future__ import annotations

import argparse
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
    """Parse a single vision trial configuration."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        choices=("cifar10", "cifar100", "imagefolder"),
        default="cifar10",
    )
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument(
        "--model",
        choices=("resnet18", "resnet34", "vit_small", "deit_small"),
        default="resnet18",
    )
    parser.add_argument("--optimizer", choices=("neat", "adamw", "sgd"), default="neat")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--target-accuracy", type=float, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", default="benchmark-runs")
    parser.add_argument("--amp", action="store_true")
    return parser.parse_args(argv)


def make_data(args):
    """Create matched augmented training and deterministic evaluation loaders."""
    torch = require_torch()
    try:
        from torchvision import datasets, transforms
    except ImportError as exc:
        raise SystemExit("Install torchvision to run vision benchmarks.") from exc
    image_size = 224 if args.dataset == "imagefolder" else 32
    normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(), transforms.ToTensor(), normalize,
    ])
    eval_transform = transforms.Compose([
        transforms.Resize(image_size), transforms.CenterCrop(image_size),
        transforms.ToTensor(), normalize,
    ])
    # CIFAR is downloaded through torchvision. ImageNet-style experiments use
    # ImageFolder so licensed datasets never need repository-specific code.
    if args.dataset.startswith("cifar"):
        dataset_type = (
            datasets.CIFAR10 if args.dataset == "cifar10" else datasets.CIFAR100
        )
        train = dataset_type(
            args.data_dir, train=True, download=True, transform=train_transform
        )
        valid = dataset_type(
            args.data_dir, train=False, download=True, transform=eval_transform
        )
    else:
        train = datasets.ImageFolder(f"{args.data_dir}/train", train_transform)
        valid = datasets.ImageFolder(f"{args.data_dir}/val", eval_transform)
    loader_args = dict(
        batch_size=args.batch_size, num_workers=args.workers, pin_memory=True
    )
    return (
        torch.utils.data.DataLoader(train, shuffle=True, **loader_args),
        torch.utils.data.DataLoader(valid, shuffle=False, **loader_args),
        len(train.classes),
    )


def make_model(name: str, classes: int, image_size: int):
    """Construct an untrained timm model with the requested classifier head."""
    try:
        import timm
    except ImportError as exc:
        raise SystemExit("Install timm to run modern vision benchmarks.") from exc
    names = {
        "resnet18": "resnet18",
        "resnet34": "resnet34",
        "vit_small": "vit_small_patch16_224",
        "deit_small": "deit_small_patch16_224",
    }
    try:
        return timm.create_model(
            names[name],
            pretrained=False,
            num_classes=classes,
            img_size=image_size,
        )
    except TypeError:
        return timm.create_model(names[name], pretrained=False, num_classes=classes)


def evaluate(model, loader, device) -> float:
    """Compute exact Top-1 accuracy without retaining activation graphs."""
    torch = require_torch()
    model.eval()
    correct = total = 0
    with torch.inference_mode():
        for images, labels in loader:
            predictions = model(images.to(device)).argmax(dim=1)
            labels = labels.to(device)
            correct += int((predictions == labels).sum())
            total += labels.numel()
    return correct / total


def main(argv=None):
    """Run one optimizer/seed trial and persist its full metric history."""
    args = parse_args(argv)
    torch = require_torch()
    seed_everything(args.seed)
    device = select_device(args.device)
    train_loader, valid_loader, classes = make_data(args)
    image_size = 224 if args.dataset == "imagefolder" else 32
    model = make_model(args.model, classes, image_size).to(device)
    optimizer = build_optimizer(
        args.optimizer, model.parameters(), args.lr, args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")
    config = vars(args)
    logger = ExperimentLogger(
        args.output_dir,
        run_name(f"{args.dataset}-{args.model}", args.optimizer, args.seed),
        config,
    )
    # The loop is deliberately optimizer-agnostic: data, schedule, precision,
    # and model code remain identical for every comparator.
    for epoch in range(1, args.epochs + 1):
        started = time.perf_counter()
        model.train()
        losses = []
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                enabled=args.amp and device.type == "cuda",
            ):
                loss = torch.nn.functional.cross_entropy(model(images), labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach()))
        accuracy = evaluate(model, valid_loader, device)
        logger.log(
            EpochMetrics(
                epoch,
                float(np.mean(losses)),
                accuracy,
                time.perf_counter() - started,
                float(np.var(losses)),
                optimizer.param_groups[0]["lr"],
            )
        )
        scheduler.step()
    logger.finish(
        device=device,
        target=args.target_accuracy,
        higher_is_better=True,
        optimizer=optimizer,
        extra={"metric": "top1_accuracy"},
    )


if __name__ == "__main__":
    main()
