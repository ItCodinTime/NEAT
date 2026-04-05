# Player-Aware Mode

NEAT now includes an explicit player-aware stepping mode for cases where you
want each example or task in a batch to contribute its own gradient.

## What Counts As A Player

In this mode, a player is one element along the leading dimension of
`player_grads`:

- one training example
- one task loss
- one objective head
- any other source that yields an individual gradient tensor

## How The Opponent Signal Is Built

For each player gradient `g_i`, the engine constructs an opponent proxy `o_i`:

- default: mean of all other players, excluding `g_i`
- optional: mean of the full batch, including `g_i`

The correction then uses the same NEAT ingredients as the base optimizer:

```text
c_i   = relu(-cos(g_i, o_i))
nce_i = -alpha * c_i * proj_{o_i}(g_i)
u_t   = reduce_i(g_i + nce_i)
```

This is a Nash-inspired approximation for batch conflict handling. It is still
not an exact Nash-equilibrium solver for arbitrary multi-agent games.

## Lightweight Objective

The player-aware path also supports lightweight-model pressure through two
controls:

- `sparsity_l1`: soft-threshold shrinkage after the update
- `prune_threshold`: hard-prune small-magnitude parameters to zero

These controls encourage sparse weights. They do not automatically redesign the
network architecture or remove layers.
