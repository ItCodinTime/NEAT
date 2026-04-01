# Math Specification

Spec version: `nce_spec_v1`

## State

Per trainable tensor:

- `m_t`: momentum-like running state
- `nce_t`: last Nash Correction Estimate
- `step`: step counter

## Inputs

- parameter tensor `theta_t`
- raw gradient `g_t`
- prior momentum state `m_{t-1}`
- hyperparameters `lr`, `alpha`, `beta`, `eps`, `weight_decay`

## Update Rule

```text
conflict_t = relu(-cos(g_t, m_{t-1}))
proj_t     = proj_{m_{t-1}}(g_t)
nce_t      = -alpha * conflict_t * proj_t
u_t        = g_t + nce_t
m_t        = beta * m_{t-1} + (1 - beta) * u_t
theta'     = (1 - lr * wd) * theta_t
theta_t+1  = theta' - lr * m_t
```

Where:

```text
proj_v(g) = ((g · v) / (v · v + eps)) * v
cos(g, v) = (g · v) / (||g|| ||v|| + eps)
```

## Invariants

- all norms are computed in `float32` or higher precision
- if `||m_{t-1}||` is effectively zero, the NCE term is zero
- the NCE vector is clipped to `nce_clip_ratio * ||g_t||`
- `nce_mode="off"` disables the correction term entirely

## Worked Example

Given:

- `theta_t = [1.0, 2.0]`
- `g_t = [1.0, 0.0]`
- `m_{t-1} = [-1.0, 0.0]`
- `alpha = 0.5`
- `beta = 0.0`
- `lr = 0.1`

Then:

```text
conflict_t = 1.0
proj_t     = [1.0, 0.0]
nce_t      = [-0.5, 0.0]
u_t        = [0.5, 0.0]
m_t        = [0.5, 0.0]
theta_t+1  = [0.95, 2.0]
```
