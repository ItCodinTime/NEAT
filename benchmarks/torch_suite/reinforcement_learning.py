"""Compare NEAT and AdamW sample efficiency on MuJoCo control tasks."""

from __future__ import annotations

import argparse
import time

from benchmarks.torch_suite.common import (
    EpochMetrics,
    ExperimentLogger,
    require_torch,
    run_name,
    seed_everything,
    select_device,
)


def parse_args(argv=None):
    """Parse a single SAC/MuJoCo comparison trial."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--env",
        choices=("HalfCheetah-v5", "Hopper-v5", "Walker2d-v5"),
        default="HalfCheetah-v5",
    )
    parser.add_argument("--optimizer", choices=("neat", "adamw"), default="neat")
    parser.add_argument("--timesteps", type=int, default=1_000_000)
    parser.add_argument("--eval-frequency", type=int, default=50_000)
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--target-reward", type=float, default=None)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", default="benchmark-runs")
    return parser.parse_args(argv)


def main(argv=None):
    """Train SAC in fixed timestep chunks and measure sample efficiency."""
    args = parse_args(argv)
    torch = require_torch()
    try:
        import gymnasium as gym
        from stable_baselines3 import SAC
        from stable_baselines3.common.evaluation import evaluate_policy
    except ImportError as exc:
        raise SystemExit(
            "Install the benchmark extra and MuJoCo runtime for RL."
        ) from exc
    seed_everything(args.seed)
    device = select_device(args.device)
    if args.optimizer == "neat":
        from neat_optim import TorchNEAT

        optimizer_class = TorchNEAT
        optimizer_kwargs = {
            "weight_decay": args.weight_decay,
            "alpha": 0.25,
            "opponent_source": "previous_gradient",
            "adaptive_alpha": True,
            "adaptive_preconditioning": True,
            "bias_correction": True,
        }
    else:
        optimizer_class = torch.optim.AdamW
        optimizer_kwargs = {"weight_decay": args.weight_decay}
    env = gym.make(args.env)
    eval_env = gym.make(args.env)
    model = SAC(
        "MlpPolicy",
        env,
        learning_rate=args.lr,
        seed=args.seed,
        device=device,
        policy_kwargs={
            "optimizer_class": optimizer_class,
            "optimizer_kwargs": optimizer_kwargs,
        },
        verbose=0,
    )
    logger = ExperimentLogger(
        args.output_dir,
        run_name(f"rl-{args.env}", args.optimizer, args.seed),
        vars(args),
    )
    checkpoints = max(1, args.timesteps // args.eval_frequency)
    rewards = []
    # Evaluation is checkpoint-based rather than episode-based because SAC's
    # training episodes have environment-dependent lengths.
    for checkpoint in range(1, checkpoints + 1):
        started = time.perf_counter()
        model.learn(args.eval_frequency, reset_num_timesteps=False)
        mean_reward, std_reward = evaluate_policy(
            model, eval_env, n_eval_episodes=args.eval_episodes, deterministic=True
        )
        rewards.append(float(mean_reward))
        logger.log(
            EpochMetrics(
                checkpoint,
                -float(mean_reward),
                float(mean_reward),
                time.perf_counter() - started,
                float(std_reward) ** 2,
                args.lr,
            )
        )
    actor_optimizer = model.actor.optimizer
    timesteps_to_target = next(
        (
            index * args.eval_frequency
            for index, value in enumerate(rewards, 1)
            if args.target_reward is not None and value >= args.target_reward
        ),
        None,
    )
    result = logger.finish(
        device=device,
        target=args.target_reward,
        higher_is_better=True,
        optimizer=actor_optimizer,
        extra={
            "metric": "mean_episode_reward",
            "checkpoint_timesteps": args.eval_frequency,
            "timesteps_to_target": timesteps_to_target,
        },
    )
    env.close()
    eval_env.close()
    return result


if __name__ == "__main__":
    main()
