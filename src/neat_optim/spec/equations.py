"""Versioned mathematical specification for the first NEAT release line."""

SPEC_VERSION = "nce_spec_v1"
PLAYER_SPEC_VERSION = "nce_players_v1"

EQUATION_SUMMARY = """
c_t       = relu(-cos(g_t, m_{t-1}))
p_t       = proj_{m_{t-1}}(g_t)
nce_t     = -alpha * c_t * p_t
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
