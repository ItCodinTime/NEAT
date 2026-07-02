"""Exercise standard first-order optimizers on a small deterministic quadratic."""

from __future__ import annotations

import argparse
import time

from benchmarks.torch_suite.common import (
    STANDARD_OPTIMIZERS,
    EpochMetrics,
    ExperimentLogger,
    add_seed_arguments,
    build_optimizer,
    orchestrate_seeds,
    require_torch,
    run_name,
    seed_everything,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--optimizer", choices=STANDARD_OPTIMIZERS, default="neat")
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--dimensions", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--output-dir", default="benchmark-runs/optimizer-smoke")
    add_seed_arguments(parser)
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    if orchestrate_seeds("benchmarks.torch_suite.optimizer_smoke", args, argv):
        return
    torch = require_torch()
    seed_everything(args.seed)
    parameter = torch.nn.Parameter(torch.zeros(args.dimensions))
    target = torch.linspace(-1.0, 1.0, args.dimensions)
    optimizer = build_optimizer(args.optimizer, [parameter], args.lr, args.weight_decay)
    logger = ExperimentLogger(
        args.output_dir,
        run_name("optimizer-smoke", args.optimizer, args.seed),
        {**vars(args), "task": "optimizer_smoke"},
    )
    for step in range(1, args.steps + 1):
        started = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        loss = 0.5 * torch.mean((parameter - target) ** 2)
        loss.backward()
        optimizer.step()
        value = float(loss.detach())
        logger.log(
            EpochMetrics(
                epoch=step,
                train_loss=value,
                eval_metric=value,
                seconds=time.perf_counter() - started,
                loss_variance=0.0,
                learning_rate=optimizer.param_groups[0]["lr"],
            )
        )
    logger.finish(
        device=torch.device("cpu"),
        target=None,
        higher_is_better=False,
        optimizer=optimizer,
        extra={"metric": "quadratic_loss"},
    )


if __name__ == "__main__":
    main()
