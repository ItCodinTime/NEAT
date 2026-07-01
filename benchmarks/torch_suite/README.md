# NEAT PyTorch Benchmark Suite

A reproducible, multi-domain benchmark suite comparing **NEAT** against AdamW (and SGD where
applicable) across six workload categories: image classification, language-model fine-tuning,
vision transformers, reinforcement learning, diffusion models, and large-batch training stability.

Every run writes a self-contained `result.json` manifest that includes the full config,
per-epoch metrics, wall-time, environment metadata, and NEAT diagnostic snapshots.

---

## Quick start

```bash
# install benchmark dependencies
pip install -e ".[benchmarks]"

# single CIFAR-10 run (ResNet-18, NEAT, ~15 min on MPS/GPU)
python -m benchmarks.torch_suite.vision \
    --dataset cifar10 --model resnet18 --optimizer neat --epochs 100

# head-to-head suite (NEAT vs AdamW, all categories, quick smoke-test)
python -m benchmarks.torch_suite.compare_all --quick

# full suite (long — use a GPU server)
python -m benchmarks.torch_suite.compare_all
```

---

## Benchmark categories

| # | Category | Script | Models | Datasets | Key metric |
|---|----------|--------|--------|----------|------------|
| 1 | Image classification | `vision.py` | ResNet-18/34 | CIFAR-10, CIFAR-100, ImageFolder | Top-1 accuracy |
| 2 | LLM fine-tuning | `language_model.py` | TinyLlama-1.1B, Phi-2, Gemma-2B | GLUE SST-2, Alpaca | Accuracy / perplexity |
| 3 | Vision transformer | `vision.py` | ViT-S/16, DeiT-S/16 | CIFAR-100, ImageFolder | Top-1 accuracy |
| 4 | Reinforcement learning | `reinforcement_learning.py` | SAC MLP policy | HalfCheetah-v5, Hopper-v5, Walker2d-v5 | Mean episode reward |
| 5 | Diffusion models | `diffusion.py` | UNet-2D DDPM | MNIST | Noise-prediction MSE |
| 6 | Large-batch stability | `large_batch.py` | ViT-S or ResNet-18 | CIFAR-100 | Loss variance × batch size |

---

## Tier 1 — Image Classification

### CIFAR-10 with ResNet-18

```bash
# NEAT
python -m benchmarks.torch_suite.vision \
    --dataset cifar10 --model resnet18 --optimizer neat \
    --epochs 100 --batch-size 128 --lr 1e-3 --weight-decay 5e-4 \
    --target-accuracy 0.92 --seed 7

# AdamW baseline
python -m benchmarks.torch_suite.vision \
    --dataset cifar10 --model resnet18 --optimizer adamw \
    --epochs 100 --batch-size 128 --lr 1e-3 --weight-decay 5e-4 \
    --target-accuracy 0.92 --seed 7
```

### CIFAR-100 with ResNet-34

```bash
python -m benchmarks.torch_suite.vision \
    --dataset cifar100 --model resnet34 --optimizer neat \
    --epochs 200 --batch-size 128 --lr 1e-3 --weight-decay 5e-4 \
    --target-accuracy 0.70

python -m benchmarks.torch_suite.vision \
    --dataset cifar100 --model resnet34 --optimizer adamw \
    --epochs 200 --batch-size 128 --lr 1e-3 --weight-decay 5e-4 \
    --target-accuracy 0.70
```

### ImageNet-100 / ImageNet-1K (ImageFolder format)

```bash
# data-dir must contain train/<class>/ and val/<class>/ sub-directories
python -m benchmarks.torch_suite.vision \
    --dataset imagefolder --data-dir /data/imagenet100 \
    --model resnet34 --optimizer neat \
    --epochs 90 --batch-size 256 --lr 1e-3 --amp \
    --target-accuracy 0.78
```

### Arguments for `vision.py`

| Flag | Default | Notes |
|------|---------|-------|
| `--dataset` | `cifar10` | `cifar10`, `cifar100`, `imagefolder` |
| `--model` | `resnet18` | `resnet18`, `resnet34`, `vit_small`, `deit_small` |
| `--optimizer` | `neat` | `neat`, `adamw`, `sgd` |
| `--epochs` | `100` | Increase to 200 for full convergence studies |
| `--batch-size` | `128` | |
| `--lr` | `3e-4` | Tune independently per optimizer |
| `--weight-decay` | `5e-4` | |
| `--amp` | off | Enable mixed-precision on CUDA |
| `--target-accuracy` | none | Records `epochs_to_target` in manifest |
| `--seed` | `7` | Use 7, 11, 19 for variance estimates |
| `--output-dir` | `benchmark-runs` | All artifacts written here |
| `--device` | `auto` | `cuda`, `mps`, or `cpu` |

**Metrics recorded**: Top-1 accuracy, training loss, per-epoch loss variance (stability proxy),
learning rate, wall time per epoch.

---

## Tier 1 — Language Model Fine-Tuning

Fine-tunes a small LLM on **GLUE SST-2** (sentiment classification) or the
**Alpaca** instruction dataset. Reports accuracy (SST-2) or perplexity (Alpaca).

```bash
# TinyLlama on SST-2 — NEAT
python -m benchmarks.torch_suite.language_model \
    --task sst2 --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --optimizer neat --epochs 3 --lr 2e-5 --batch-size 16

# TinyLlama on SST-2 — AdamW baseline
python -m benchmarks.torch_suite.language_model \
    --task sst2 --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --optimizer adamw --epochs 3 --lr 2e-5 --batch-size 16

# Phi-2 on Alpaca with LoRA, NEAT
python -m benchmarks.torch_suite.language_model \
    --task alpaca --model microsoft/phi-2 \
    --optimizer neat --lora --epochs 1 --lr 5e-5 \
    --max-train-samples 10000 --max-eval-samples 500

# Gemma-2B on Alpaca (requires HF auth)
python -m benchmarks.torch_suite.language_model \
    --task alpaca --model google/gemma-2b \
    --optimizer neat --lora --epochs 1 --lr 2e-5 \
    --batch-size 4 --gradient-accumulation 8 --amp
```

> **Memory tip**: For machines with <24 GB VRAM, use `--lora --batch-size 4
> --gradient-accumulation 8 --amp --max-train-samples 5000`.

### Arguments for `language_model.py`

| Flag | Default | Notes |
|------|---------|-------|
| `--task` | `sst2` | `sst2` (classification) or `alpaca` (causal LM) |
| `--model` | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` | Any HuggingFace seq-cls / causal model |
| `--optimizer` | `neat` | `neat`, `adamw` |
| `--lora` | off | LoRA adapter via `peft` |
| `--epochs` | `3` | |
| `--batch-size` | `8` | |
| `--gradient-accumulation` | `1` | Effective batch = batch-size × accum |
| `--max-length` | `256` | Truncation length |
| `--max-train-samples` | all | Subsample for quick experiments |
| `--target` | none | Records `epochs_to_target` in manifest |

**Metrics recorded**: SST-2 accuracy or Alpaca perplexity, training loss, loss variance.

---

## Tier 1 — Vision Transformers (ViT / DeiT)

Same `vision.py` entrypoint with `--model vit_small` or `--model deit_small`.
Attention-heavy models stress-test NEAT's adaptive preconditioning differently from ConvNets.

```bash
# ViT-small on CIFAR-100
python -m benchmarks.torch_suite.vision \
    --dataset cifar100 --model vit_small --optimizer neat \
    --epochs 200 --batch-size 128 --lr 5e-4 --weight-decay 0.05

python -m benchmarks.torch_suite.vision \
    --dataset cifar100 --model vit_small --optimizer adamw \
    --epochs 200 --batch-size 128 --lr 5e-4 --weight-decay 0.05

# DeiT-small on ImageNet-100
python -m benchmarks.torch_suite.vision \
    --dataset imagefolder --data-dir /data/imagenet100 \
    --model deit_small --optimizer neat \
    --epochs 90 --batch-size 256 --lr 5e-4 --amp \
    --target-accuracy 0.80
```

---

## Tier 2 — Reinforcement Learning (MuJoCo)

Uses **Stable-Baselines3 SAC** with NEAT or AdamW as the policy optimizer.
Measures mean episode reward every `--eval-frequency` timesteps.

```bash
# HalfCheetah — NEAT vs AdamW sample efficiency
python -m benchmarks.torch_suite.reinforcement_learning \
    --env HalfCheetah-v5 --optimizer neat \
    --timesteps 1000000 --target-reward 8000

python -m benchmarks.torch_suite.reinforcement_learning \
    --env HalfCheetah-v5 --optimizer adamw \
    --timesteps 1000000 --target-reward 8000

# Hopper and Walker2d
python -m benchmarks.torch_suite.reinforcement_learning \
    --env Hopper-v5 --optimizer neat --timesteps 500000 --target-reward 2500

python -m benchmarks.torch_suite.reinforcement_learning \
    --env Walker2d-v5 --optimizer neat --timesteps 1000000 --target-reward 3000
```

> **Requirements**: `gymnasium[mujoco]>=0.29`, `stable-baselines3>=2.3`, and a
> working MuJoCo native installation.

### Arguments for `reinforcement_learning.py`

| Flag | Default | Notes |
|------|---------|-------|
| `--env` | `HalfCheetah-v5` | `HalfCheetah-v5`, `Hopper-v5`, `Walker2d-v5` |
| `--timesteps` | `1_000_000` | Total environment steps |
| `--eval-frequency` | `50_000` | Steps between evaluations |
| `--eval-episodes` | `10` | Episodes per checkpoint |
| `--target-reward` | none | Records `timesteps_to_target` in manifest |

**Metrics recorded**: Mean episode reward per checkpoint, reward variance, wall time.

---

## Tier 2 — Diffusion Models (DDPM on MNIST)

Trains a compact **UNet-2D DDPM** on MNIST. Lower noise-prediction MSE is better.

```bash
python -m benchmarks.torch_suite.diffusion \
    --optimizer neat --epochs 50 --lr 1e-4 --batch-size 128 \
    --target-loss 0.015

python -m benchmarks.torch_suite.diffusion \
    --optimizer adamw --epochs 50 --lr 1e-4 --batch-size 128 \
    --target-loss 0.015
```

> **Requirements**: `diffusers>=0.30`.

### Arguments for `diffusion.py`

| Flag | Default | Notes |
|------|---------|-------|
| `--epochs` | `20` | Increase to 50–100 for full convergence |
| `--batch-size` | `128` | |
| `--diffusion-steps` | `1000` | DDPM noise schedule steps |
| `--target-loss` | none | Records `epochs_to_target` in manifest |

**Metrics recorded**: Validation noise-prediction MSE, training loss, loss variance.

---

## Tier 2 — Large-Batch Training Stability

Sweeps three batch sizes (128 → 512 → 2048) with linear-scaled learning rates for
both NEAT and AdamW. The core question: does NEAT maintain lower **loss variance**
as the batch size grows?

```bash
# Default sweep: CIFAR-100, ViT-small, batch sizes 128/512/2048
python -m benchmarks.torch_suite.large_batch \
    --dataset cifar100 --model vit_small \
    --batch-sizes 128 512 2048 --base-lr 3e-4 --epochs 100

# ResNet-18 on CIFAR-10
python -m benchmarks.torch_suite.large_batch \
    --dataset cifar10 --model resnet18 \
    --batch-sizes 256 1024 4096 --base-lr 1e-3 --epochs 50
```

The sweep runs one `vision.py` subprocess per (batch-size × optimizer) combination.
Each run writes its own `result.json`; aggregate with the report tool.

---

## Aggregating and visualising results

```bash
# Summarise all runs under benchmark-runs/ into CSV + convergence PNG
python -m benchmarks.torch_suite.report benchmark-runs --output-dir benchmark-runs/report

# Specific sub-directories only
python -m benchmarks.torch_suite.report \
    benchmark-runs/cifar10-study \
    benchmark-runs/cifar100-study \
    --output-dir benchmark-runs/combined-report

# View in TensorBoard
tensorboard --logdir benchmark-runs
```

The report tool produces:
- `summary.csv` — one row per run: final metric, best metric, epochs to target,
  mean loss variance, wall time
- `convergence.png` — eval metric and training loss curves per optimizer/seed

---

## Running multiple seeds for statistical rigour

```bash
for SEED in 7 11 19; do
  python -m benchmarks.torch_suite.vision \
      --dataset cifar10 --model resnet18 --optimizer neat \
      --epochs 100 --seed $SEED --output-dir benchmark-runs/cifar10-multiseed
  python -m benchmarks.torch_suite.vision \
      --dataset cifar10 --model resnet18 --optimizer adamw \
      --epochs 100 --seed $SEED --output-dir benchmark-runs/cifar10-multiseed
done
python -m benchmarks.torch_suite.report benchmark-runs/cifar10-multiseed
```

Report mean ± std across seeds. Tune learning rate independently for each optimizer
from the **same candidate grid** and fix all other hyperparameters.

---

## Interpreting `result.json`

Each run writes `<output-dir>/<run-name>/result.json`:

```json
{
  "config": { "dataset": "cifar10", "model": "resnet18", "optimizer": "neat", ... },
  "environment": { "torch": "2.2.0", "device": "mps", "platform": "...", ... },
  "history": [
    { "epoch": 1, "train_loss": 1.82, "eval_metric": 0.451, "loss_variance": 0.031, ... },
    ...
  ],
  "epochs_to_target": 67,
  "wall_time_seconds": 3120,
  "optimizer_diagnostics": {
    "mean_conflict_ratio": 0.041,
    "mean_correction_ratio": 0.0013,
    "mean_effective_alpha": 0.26,
    "correction_active_fraction": 0.213
  }
}
```

| Field | Meaning |
|-------|---------|
| `history[*].eval_metric` | Validation accuracy, reward, or perplexity |
| `history[*].loss_variance` | Per-epoch minibatch-loss variance — lower is more stable |
| `epochs_to_target` | First epoch hitting the target metric — convergence speed |
| `optimizer_diagnostics.mean_conflict_ratio` | Average gradient conflict detected by NEAT |
| `optimizer_diagnostics.correction_active_fraction` | Fraction of steps with NCE correction |
| `optimizer_diagnostics.mean_effective_alpha` | Adaptive correction strength used |

---

## Completed studies

| Study | Hardware | Notes | Location |
|-------|----------|-------|----------|
| CIFAR-10 ResNet-18 MPS, 3 seeds, 3 epochs | Apple M-series MPS | Parity with AdamW; 2.2× slower on MPS | `docs/research/cifar10-resnet18-mps-2026-06-29.md` |
| Raw manifests (multiepoch, seeds 7/11/19) | Apple MPS | 3-epoch runs | `benchmark-runs/multiepoch/` |
