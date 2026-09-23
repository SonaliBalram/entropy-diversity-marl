"""
action_utils.py

Shared conversion from the raw MPE action vector to the
PHYSICAL force the environment actually applies.

WHY THIS FILE EXISTS
--------------------
simple_spread_v3 with continuous_actions=True gives each
agent a 5-vector a = [no_op, left, right, down, up] in
(0,1)^5. MPE's `_set_action` then computes (verbatim from
mpe2/_mpe_utils/simple_env.py):

        agent.action.u[0] += action[0][2] - action[0][1]
        agent.action.u[1] += action[0][4] - action[0][3]
        agent.action.u    *= sensitivity          # 5.0

i.e.

        u_x = a[2] - a[1]
        u_y = a[4] - a[3]

and a[0] is NEVER READ.

NOTE ON SIGN: an earlier version of this file had these
subtractions reversed, giving exactly -u. Every metric here
is invariant to negating both vectors --
cos(-u,-v) = cos(u,v), ||(-u)-(-v)|| = ||u-v||, and
corr(-x,-y) = corr(x,y) -- so no similarity number was
affected. The sign only matters if a force vector is plotted
or a direction is named ("agent moved left"), so it is
corrected here.

Any behavioural metric computed on the raw 5-vector is
therefore measuring a representation artifact, not
behaviour:

  * all five components are positive, so the action vectors
    live in the positive orthant and pairwise cosine
    similarity is confined to roughly [0.1, 1.0] -- it
    cannot express opposition at all. Two agents driving in
    EXACTLY OPPOSITE directions score +0.59, and uniformly
    random agents average +0.77.

  * a[0] contributes to Euclidean distance while having zero
    effect on the world (corr between raw 5-D distance and
    true physical distance is only 0.56).

Projecting to u = (u_x, u_y) first fixes both: cosine on u
spans the full [-1, 1] and means what it says.
"""

import numpy as np


def to_force(actions):
    """
    actions : [..., 5]   raw MPE actions
    returns : [..., 2]   physical force (u_x, u_y)
    """

    actions = np.asarray(actions, dtype=np.float64)

    u_x = actions[..., 2] - actions[..., 1]
    u_y = actions[..., 4] - actions[..., 3]

    return np.stack([u_x, u_y], axis=-1)


def safe_cosine(u, v, eps=1e-8):
    """
    Cosine similarity, returning NaN (not 0.0) when either
    vector is degenerate.

    Returning 0.0 for a zero vector -- as the original code
    did -- silently encodes "undefined" as "orthogonal",
    which drags the episode mean toward 0 and makes a
    stalled agent look behaviourally DIVERSE. NaN plus
    np.nanmean keeps undefined timesteps out of the average
    instead of biasing it.
    """

    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)

    nu = np.linalg.norm(u)
    nv = np.linalg.norm(v)

    if nu < eps or nv < eps:
        return np.nan

    return float(np.clip(np.dot(u, v) / (nu * nv), -1.0, 1.0))
