"""Versioned mathematical specification for the NEAT release line."""

SPEC_VERSION = "nce_spec_v2"
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

PLAYER_EQUATION_SUMMARY = """
o_i       = mean_{j != i}(g_j)
c_i       = relu(-cos(g_i, o_i))
p_i       = proj_{o_i}(g_i)
nce_i     = -alpha * c_i * p_i
u_t       = reduce_i(g_i + nce_i)
m_t       = beta * m_{t-1} + (1 - beta) * u_t
theta_t+1 = sparsify((1 - lr * wd) * theta_t - lr * m_t)
""".strip()
