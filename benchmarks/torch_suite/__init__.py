"""PyTorch benchmark runners for modern NEAT workloads.

Modules
-------
vision              CIFAR-10/100 and ImageFolder with ResNet, ViT, DeiT
language_model      LLM fine-tuning on GLUE SST-2 or Alpaca
reinforcement_learning  MuJoCo SAC with NEAT or AdamW policy optimizer
diffusion           Compact DDPM on MNIST
large_batch         Controlled batch-size / stability sweep
compare_all         Head-to-head orchestration across all categories
common              Shared utilities (device, seeding, logging)
report              Aggregate result manifests into CSV and plots
"""

