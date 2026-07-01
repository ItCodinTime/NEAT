"""Train a compact unconditional DDPM on MNIST with NEAT or AdamW."""

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
    """Parse one compact DDPM training trial."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--optimizer", choices=("neat", "adamw"), default="neat")
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--diffusion-steps", type=int, default=1000)
    parser.add_argument("--target-loss", type=float, default=None)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", default="benchmark-runs")
    parser.add_argument("--amp", action="store_true")
    return parser.parse_args(argv)


def make_data(args):
    """Create a deterministic 55k/5k MNIST train/validation split."""
    torch = require_torch()
    try:
        from torchvision import datasets, transforms
    except ImportError as exc:
        raise SystemExit("Install torchvision for the diffusion suite.") from exc
    dataset = datasets.MNIST(
        args.data_dir,
        train=True,
        download=True,
        transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Lambda(lambda x: x * 2 - 1)]
        ),
    )
    train, valid = torch.utils.data.random_split(
        dataset, [55_000, 5_000], generator=torch.Generator().manual_seed(args.seed)
    )
    options = dict(
        batch_size=args.batch_size, num_workers=args.workers, pin_memory=True
    )
    return (
        torch.utils.data.DataLoader(train, shuffle=True, **options),
        torch.utils.data.DataLoader(valid, shuffle=False, **options),
    )


def diffusion_loss(model, scheduler, images, device):
    """Evaluate the standard random-timestep noise-prediction objective."""
    torch = require_torch()
    noise = torch.randn_like(images)
    timesteps = torch.randint(
        0,
        scheduler.config.num_train_timesteps,
        (images.shape[0],),
        device=device,
    ).long()
    noisy = scheduler.add_noise(images, noise, timesteps)
    prediction = model(noisy, timesteps).sample
    return torch.nn.functional.mse_loss(prediction, noise)


def evaluate(model, scheduler, loader, device) -> float:
    """Average validation noise MSE with gradients disabled."""
    model.eval()
    losses = []
    torch = require_torch()
    with torch.inference_mode():
        for images, _ in loader:
            loss = diffusion_loss(model, scheduler, images.to(device), device)
            losses.append(float(loss))
    return float(np.mean(losses))


def main(argv=None):
    """Train one unconditional MNIST DDPM and log optimizer behavior."""
    args = parse_args(argv)
    torch = require_torch()
    try:
        from diffusers import DDPMScheduler, UNet2DModel
    except ImportError as exc:
        raise SystemExit("Install diffusers for the diffusion suite.") from exc
    seed_everything(args.seed)
    device = select_device(args.device)
    train_loader, valid_loader = make_data(args)
    model = UNet2DModel(
        sample_size=28,
        in_channels=1,
        out_channels=1,
        layers_per_block=2,
        block_out_channels=(32, 64, 64),
        down_block_types=("DownBlock2D", "AttnDownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "AttnUpBlock2D", "UpBlock2D"),
        norm_num_groups=8,
    ).to(device)
    scheduler = DDPMScheduler(num_train_timesteps=args.diffusion_steps)
    optimizer = build_optimizer(
        args.optimizer, model.parameters(), args.lr, args.weight_decay
    )
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")
    logger = ExperimentLogger(
        args.output_dir,
        run_name("diffusion-mnist", args.optimizer, args.seed),
        vars(args),
    )
    for epoch in range(1, args.epochs + 1):
        started = time.perf_counter()
        model.train()
        losses = []
        for images, _ in train_loader:
            optimizer.zero_grad(set_to_none=True)
            images = images.to(device)
            with torch.autocast(
                device_type=device.type,
                enabled=args.amp and device.type == "cuda",
            ):
                loss = diffusion_loss(model, scheduler, images, device)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach()))
        validation_loss = evaluate(model, scheduler, valid_loader, device)
        logger.log(
            EpochMetrics(
                epoch,
                float(np.mean(losses)),
                validation_loss,
                time.perf_counter() - started,
                float(np.var(losses)),
                optimizer.param_groups[0]["lr"],
            )
        )
    logger.finish(
        device=device,
        target=args.target_loss,
        higher_is_better=False,
        optimizer=optimizer,
        extra={"metric": "validation_noise_mse"},
    )


if __name__ == "__main__":
    main()
