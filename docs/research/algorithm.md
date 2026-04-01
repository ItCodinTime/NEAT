# Algorithm

The first NEAT release treats the previous momentum vector as the opponent
signal. This keeps the optimizer state minimal while still making the
correction term sensitive to directional conflict.

Design choices for the first release:

- projection-based correction is the default
- cosine-only correction is retained as a simpler ablation mode
- the correction term is clipped relative to the gradient norm
- decoupled weight decay is the default behavior
