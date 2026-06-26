# Math Specification

Spec version: `nce_spec_v2`

## State

Per trainable tensor:

- `m_t`: momentum-like running state
- `nce_t`: last Nash Correction Estimate
- `v_t`: optional diagonal second-moment / Fisher-style accumulator
- `s_t`: optional lookahead slow parameter
- `rho_t`: EMA of measured conflict
- `eta_t`: EMA of gradient noise
- `a_t`: EMA of gradient alignment
- `step`: step counter

## Inputs

- parameter tensor `theta_t`
- raw gradient `g_t`
- prior momentum state `m_{t-1}`
- hyperparameters `lr`, `alpha`, `beta`, `eps`, `weight_decay`

If `gradient_centralization=True` and `g_t` is matrix-like:

```text
g_t = g_t - mean(g_t, axes=all_axes_except_last)
```

## Update Rule

```text
o_t        = opponent_proxy(g_t, m_{t-1})
conflict_t = relu(-cos(g_t, o_t))
alpha_t    = adaptive_alpha(alpha, rho_t, eta_t, a_t)
proj_t     = proj_{o_t}(g_t)
nce_t      = -alpha_t * conflict_t * proj_t
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

When `adaptive_preconditioning=True`, the optimizer maintains:

```text
v_t = beta2 * v_{t-1} + (1 - beta2) * g_t^2
p_t = sqrt(v_t) + eps
```

The step update is divided by `p_t`. If `precondition_nce=True`, the conflict
correction is computed in the preconditioned space and mapped back before the
momentum update. This is a cheap diagonal curvature approximation; it is not a
full Hessian method.

When `adaptive_alpha=True`:

```text
noise_t   = ||g_t - ema(g)_t|| / (||g_t|| + ||ema(g)_t|| + eps)
alpha_t   = clip(alpha * (1 + rho_t + eta_t - 0.5 * max(a_t, 0)),
                 adaptive_alpha_min,
                 adaptive_alpha_max)
```

This makes the conflict strength tensor-local and data-dependent. Conflict and
noise increase the correction budget; stable positive alignment reduces it.

When `update_mode="lion"`, the final parameter update uses `sign(step_update)`.
This is a Lion-style sign update applied after the NEAT correction and optional
preconditioning.

When `nesterov=True`, the step update uses:

```text
step_update_t = beta * m_t + (1 - beta) * u_t
```

When `lookahead_k > 0`, every `k` steps the slow parameter is updated as:

```text
s_t       = s_{t-k} + lookahead_alpha * (theta_t - s_{t-k})
theta_t   = s_t
```

## Invariants

- all norms are computed in `float32` or higher precision
- if `||m_{t-1}||` is effectively zero, the NCE term is zero
- the NCE vector is clipped to `nce_clip_ratio * ||g_t||`
- `nce_mode="off"` disables the correction term entirely
- all new research modes default to off, preserving the original update

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
