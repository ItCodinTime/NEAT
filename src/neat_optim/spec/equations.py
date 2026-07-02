"""Versioned mathematical specification for the NEAT release line."""

NCE_SPEC_V2 = "nce_spec_v2"
NCE_SPEC_V3 = "nce_spec_v3"
# Public compatibility alias: v2 remains the default specification.
SPEC_VERSION = NCE_SPEC_V2
PLAYER_SPEC_VERSION = "nce_players_v1"

EQUATION_SUMMARY = """
o_t       = opponent_proxy(g_t, m_{t-1})
c_t       = relu(-cos(g_t, o_t))
alpha_t   = adaptive_alpha(alpha, conflict_ema, noise_ema, alignment_ema)
p_t       = proj_{o_t}(g_t)
nce_t     = -alpha_t * c_t * p_t
u_t       = g_t + nce_t
m_t       = beta * m_{t-1} + (1 - beta) * u_t
theta_t+1 = (1 - lr * wd) * theta_t - lr * m_t
""".strip()

V3_EQUATION_SUMMARY = """
base_update = nce_spec_v2_update
optional_v3_transform = selected_default_off_transform(base_update)

All v3 transforms remain default-off until their reference semantics, adapter
parity, and author decisions are complete.
""".strip()

PLAYER_EQUATION_SUMMARY = """
o_i       = mean_{j != i}(g_j)
c_i       = relu(-cos(g_i, o_i))
p_i       = proj_{o_i}(g_i)
nce_i     = -alpha * c_i * p_i
u_t       = reduce_i(g_i + nce_i)
m_t       = beta * m_{t-1} + (1 - beta) * u_t
theta_t+1 = sparsify((1 - lr * wd) * theta_t - lr * m_t)
""".strip()
