# Algorithm

Standard NEAT is built around a single idea: if the current gradient is moving
against an opponent proxy, add a bounded correction term before momentum and
parameter updates.

Design choices for the first release:

- projection-based correction is the default
- cosine-only correction is retained as a simpler ablation mode
- the correction term is clipped relative to the gradient norm
- decoupled weight decay is the default behavior
- the correction can be delayed with `correction_warmup_steps`
- the correction can be gated with `conflict_threshold`

Opponent proxy options for the standard optimizer:

- `momentum`: previous momentum vector
- `previous_gradient`: previous raw gradient
- `gradient_ema`: exponential moving average of recent gradients

The benchmark diagnostics now report mean conflict ratio, mean correction
ratio, mean update alignment, mean opponent norm, and the fraction of steps
where the correction was active. On the current digits benchmark, those
numbers show that the correction is small and rarely changes the update
direction materially, which explains why NEAT is still trailing Adam and SGD
there.

The repository also includes a separate player-aware mode for explicit
per-example or per-task gradients. That path treats each player as a distinct
gradient contributor and forms a leave-one-out opponent proxy from the
remaining players before aggregation.
