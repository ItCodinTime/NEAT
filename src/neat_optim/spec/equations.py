"""Versioned mathematical specification for the first NEAT release line."""

SPEC_VERSION = "nce_spec_v1"

EQUATION_SUMMARY = """
c_t       = relu(-cos(g_t, m_{t-1}))
p_t       = proj_{m_{t-1}}(g_t)
nce_t     = -alpha * c_t * p_t
u_t       = g_t + nce_t
m_t       = beta * m_{t-1} + (1 - beta) * u_t
theta_t+1 = (1 - lr * wd) * theta_t - lr * m_t
""".strip()
