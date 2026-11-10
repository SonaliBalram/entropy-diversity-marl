import numpy as np


def to_force(actions):
    actions = np.asarray(actions, dtype=np.float64)

    u_x = actions[..., 2] - actions[..., 1]
    u_y = actions[..., 4] - actions[..., 3]

    return np.stack([u_x, u_y], axis=-1)


def safe_cosine(u, v, eps=1e-8):
    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)

    nu = np.linalg.norm(u)
    nv = np.linalg.norm(v)

    if nu < eps or nv < eps:
        return np.nan

    return float(np.clip(np.dot(u, v) / (nu * nv), -1.0, 1.0))
