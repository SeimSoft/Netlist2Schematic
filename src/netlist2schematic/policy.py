"""MLX policy: GNN context over the netlist + candidate scoring MLP.

Per decision:
1. GNN: node features of all parts (type, pins, ground/supply pins, placed?,
   current?, position) are embedded and refined by ``GNN_LAYERS`` rounds of
   message passing over the netlist graph (mean over neighbours):
       h <- tanh(h W_self + (A h) W_nb + b)
   The context is [embedding of the current part, mean of placed parts, mean of
   unplaced parts]. It tells the scorer what structure the part sits in (e.g. one
   leg of an H-bridge) and what is still to come.
2. Scorer: every candidate (anchor, orientation) is scored by a shared MLP over its
   features, modulated by the context:
       score = tanh(x W1 + context Wc + b1) W2 + b2 + x . prior
   A linear prior over the candidate features encodes the obvious conventions so
   proposals are usable before training. It is only the starting value: training
   adjusts it, so a convention the examples contradict can be unlearned.
3. Relations: for every candidate and every placed neighbour of the current part
   (up to MAX_RELATIONS), a small MLP scores the exact offset between them (Fourier
   features, so one cell makes a difference), both orientations and both parts'
   GNN embeddings ("C_res sits 3 cells left of M_1"). The mean over the neighbours is
   added to the score. This is what lets the policy learn a concrete drawing style
   instead of only "close to the neighbours".
4. Masked softmax over the valid candidates gives the policy.

Training uses one loss for both regimes:  loss = -sum_t c_t * sum_a q_t(a) log pi(a | s_t)
- behaviour cloning on manual placements: c_t = 1, q_t = soft target around the
  human's position (neighbouring cells with the same orientation get partial credit,
  so a slightly different spacing is not punished like a wrong placement)
- REINFORCE on the environment reward:    c_t = reward - baseline, q_t = one-hot action
"""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx
import mlx.optimizers as optim
import numpy as np

from . import env as E
from .env import N_FEATURES, N_NODE_FEATURES

EMBED = 16
GNN_LAYERS = 3
HIDDEN = 32
REL_HIDDEN = 64
PERIODS = (2.0, 4.0, 8.0, 16.0, 32.0, 64.0)  # cells; Fourier features of relative offsets
ORIENT_EMBED = 8

_PRIOR = np.zeros(N_FEATURES, dtype=np.float32)
_PRIOR[E.F_DIST] = -6.0
_PRIOR[E.F_ALIGN_ROW] = 0.8
_PRIOR[E.F_ALIGN_COL] = 0.8
_PRIOR[E.F_CONGESTION] = -2.0
_PRIOR[E.F_GROUND_DOWN] = 2.0
_PRIOR[E.F_GROUND_LOW] = 1.0
_PRIOR[E.F_SUPPLY_UP] = 2.0
_PRIOR[E.F_SUPPLY_HIGH] = 1.0
_PRIOR[E.F_FIRST_PORT] = -3.0
_PRIOR[E.F_LAST_PORT] = -3.0
_PRIOR[E.F_CENTRE] = -1.0
_PRIOR[E.F_PIN_FACING] = 1.5
_PRIOR[E.F_STAY] = 1.0  # inertia: move a placed part only when clearly better
_PRIOR[E.F_WIRE] = -2.0
_PRIOR[E.F_CROSS] = -1.5
PRIOR = mx.array(_PRIOR)

# A step: (candidate features, valid mask, target index, graph inputs[, soft target | info])
# with graph inputs = (node features (n, N_NODE_FEATURES), adjacency (n, n), current part)
# and the optional soft target a distribution over the candidates (default: one-hot).
Step = tuple[np.ndarray, np.ndarray, int, tuple[np.ndarray, np.ndarray, int]]


def init_params(seed: int = 0, n_in: int = N_FEATURES, n_hidden: int = HIDDEN, n_node: int = N_NODE_FEATURES) -> dict[str, mx.array]:
    keys = mx.random.split(mx.random.key(seed), 12 + 2 * GNN_LAYERS)
    params = {
        "w1": mx.random.normal((n_in, n_hidden), key=keys[0]) * (1.0 / np.sqrt(n_in)),
        "b1": mx.zeros((n_hidden,)),
        "w2": mx.random.normal((n_hidden, 1), key=keys[1]) * (0.1 / np.sqrt(n_hidden)),
        "b2": mx.zeros((1,)),
        "wc": mx.random.normal((3 * EMBED, n_hidden), key=keys[2]) * (0.5 / np.sqrt(3 * EMBED)),
        "g_in": mx.random.normal((n_node, EMBED), key=keys[3]) * (1.0 / np.sqrt(n_node)),
        "prior": mx.array(_PRIOR) if n_in == N_FEATURES else mx.zeros((n_in,)),
        # Relation scorer
        "r_f": mx.random.normal((4 * len(PERIODS), REL_HIDDEN), key=keys[4 + 2 * GNN_LAYERS]) * (1.0 / np.sqrt(4 * len(PERIODS))),
        "r_oc": mx.random.normal((8, ORIENT_EMBED), key=keys[5 + 2 * GNN_LAYERS]) * 0.5,
        "r_on": mx.random.normal((8, ORIENT_EMBED), key=keys[6 + 2 * GNN_LAYERS]) * 0.5,
        "r_wc": mx.random.normal((ORIENT_EMBED + EMBED, REL_HIDDEN), key=keys[7 + 2 * GNN_LAYERS]) * (1.0 / np.sqrt(ORIENT_EMBED + EMBED)),
        "r_wn": mx.random.normal((ORIENT_EMBED + EMBED, REL_HIDDEN), key=keys[8 + 2 * GNN_LAYERS]) * (1.0 / np.sqrt(ORIENT_EMBED + EMBED)),
        "r_b1": mx.zeros((REL_HIDDEN,)),
        "r_w2": mx.random.normal((REL_HIDDEN, 1), key=keys[9 + 2 * GNN_LAYERS]) * (0.1 / np.sqrt(REL_HIDDEN)),
    }
    for layer in range(GNN_LAYERS):
        params[f"g_self{layer}"] = mx.random.normal((EMBED, EMBED), key=keys[4 + 2 * layer]) * (1.0 / np.sqrt(EMBED))
        params[f"g_nb{layer}"] = mx.random.normal((EMBED, EMBED), key=keys[5 + 2 * layer]) * (1.0 / np.sqrt(EMBED))
        params[f"g_b{layer}"] = mx.zeros((EMBED,))
    return params


def context(params: dict[str, mx.array], nodes: mx.array, adj: mx.array, current: mx.array, node_mask: mx.array, embeddings: bool = False):
    """(steps, 3 * EMBED) context from the netlist graph; nodes (S, M, F), adj (S, M, M).

    With ``embeddings`` also returns the node embeddings (S, M, EMBED) and the current
    part's embedding (S, EMBED)."""
    h = mx.tanh(nodes @ params["g_in"])
    for layer in range(GNN_LAYERS):
        h = mx.tanh(h @ params[f"g_self{layer}"] + (adj @ h) @ params[f"g_nb{layer}"] + params[f"g_b{layer}"])
    h = h * node_mask[..., None]
    current_h = mx.take_along_axis(h, current[:, None, None], axis=1)[:, 0, :]
    placed = nodes[..., N_NODE_FEATURES - 5] * node_mask  # state column 0: placed
    unplaced = (1.0 - placed) * node_mask

    def mean(weights: mx.array) -> mx.array:
        total = mx.maximum(weights.sum(axis=1, keepdims=True), 1.0)
        return (h * weights[..., None]).sum(axis=1) / total

    ctx = mx.concatenate([current_h, mean(placed), mean(unplaced)], axis=1)
    return (ctx, h, current_h) if embeddings else ctx


def relations(params, cand: mx.array, nbrs: mx.array, nb_mask: mx.array, h: mx.array, current_h: mx.array) -> mx.array:
    """(S, N) mean pairwise score of each candidate against the placed neighbours.

    cand (S, N, 3) = candidate x, y, orientation; nbrs (S, K, 4) = neighbour x, y,
    orientation, node index; nb_mask (S, K); h (S, M, EMBED) node embeddings."""
    dx = (cand[:, :, None, 0] - nbrs[:, None, :, 0]).astype(mx.float32)
    dy = (cand[:, :, None, 1] - nbrs[:, None, :, 1]).astype(mx.float32)
    parts = []
    for period in PERIODS:
        w = 2 * np.pi / period
        parts += [mx.sin(dx * w), mx.cos(dx * w), mx.sin(dy * w), mx.cos(dy * w)]
    fourier = mx.stack(parts, axis=-1)  # (S, N, K, 4P)
    pre = fourier @ params["r_f"]
    cand_part = mx.concatenate([params["r_oc"][cand[:, :, 2]], mx.broadcast_to(current_h[:, None, :], (*cand.shape[:2], current_h.shape[-1]))], axis=-1)
    nb_emb = mx.take_along_axis(h, nbrs[:, :, 3:4].astype(mx.int32), axis=1) if nbrs.shape[1] else mx.zeros((h.shape[0], 0, h.shape[-1]))
    nb_part = mx.concatenate([params["r_on"][nbrs[:, :, 2]], nb_emb], axis=-1)
    pre = pre + (cand_part @ params["r_wc"])[:, :, None, :] + (nb_part @ params["r_wn"])[:, None, :, :] + params["r_b1"]
    pair = (mx.tanh(pre) @ params["r_w2"])[..., 0]  # (S, N, K)
    weight = nb_mask[:, None, :]
    return (pair * weight).sum(axis=2) / mx.maximum(weight.sum(axis=2), 1.0)


def scores(params: dict[str, mx.array], features: mx.array, ctx: mx.array | None = None) -> mx.array:
    """Score per candidate. ``features`` (S, N, F); ``ctx`` (S, 3 * EMBED) or None (no graph)."""
    pre = features @ params["w1"] + params["b1"]
    if ctx is not None and "wc" in params:
        pre = pre + (ctx @ params["wc"])[:, None, :]
    learned = (mx.tanh(pre) @ params["w2"] + params["b2"])[..., 0]
    prior = params.get("prior")
    if prior is None:
        prior = PRIOR if features.shape[-1] == N_FEATURES else mx.zeros((features.shape[-1],))
    return learned + features @ prior


def masked_log_softmax(z: mx.array, mask: mx.array) -> mx.array:
    z = mx.where(mask, z, -1e9)
    return z - mx.logsumexp(z, axis=-1, keepdims=True)


def batch_scores(params: dict[str, mx.array], batch: dict[str, mx.array]) -> mx.array:
    if "nodes" not in batch:
        return scores(params, batch["features"])
    ctx, h, current_h = context(params, batch["nodes"], batch["adj"], batch["current"], batch["node_mask"], embeddings=True)
    z = scores(params, batch["features"], ctx)
    if "cand" in batch and "r_w2" in params:
        z = z + relations(params, batch["cand"], batch["nbrs"], batch["nb_mask"], h, current_h)
    return z


def batch_loss(params: dict[str, mx.array], batch: dict[str, mx.array]) -> mx.array:
    logp = masked_log_softmax(batch_scores(params, batch), batch["masks"])
    picked = (batch["q"] * mx.where(batch["masks"], logp, 0.0)).sum(axis=1)
    return -mx.sum(batch["coefs"] * picked) / batch["features"].shape[0]


_loss_and_grad = mx.value_and_grad(batch_loss)


def loss_and_grads(params: dict[str, mx.array], batch: dict[str, mx.array]):
    return _loss_and_grad(params, batch)


def make_batch(steps: list[Step], coefs: np.ndarray | None = None) -> dict[str, mx.array]:
    """Stack steps of different grid and circuit sizes; padding is masked out."""
    n = max(s[0].shape[0] for s in steps)
    m = max(s[3][0].shape[0] for s in steps)
    k = max([s[3][4].shape[0] for s in steps if len(s[3]) > 4] + [1])
    count = len(steps)
    cand = np.zeros((count, n, 3), dtype=np.int32)
    nbrs = np.zeros((count, k, 4), dtype=np.int32)
    nb_mask = np.zeros((count, k), dtype=np.float32)
    features = np.zeros((count, n, steps[0][0].shape[1]), dtype=np.float32)
    masks = np.zeros((count, n), dtype=bool)
    nodes = np.zeros((count, m, N_NODE_FEATURES), dtype=np.float32)
    adj = np.zeros((count, m, m), dtype=np.float32)
    node_mask = np.zeros((count, m), dtype=np.float32)
    current = np.zeros(count, dtype=np.int32)
    targets = np.zeros(count, dtype=np.int32)
    q = np.zeros((count, n), dtype=np.float32)
    for i, step in enumerate(steps):
        f, valid, target, graph = step[:4]
        nf, a, cur = graph[:3]
        if len(graph) > 4:
            cand[i, : len(graph[3])] = graph[3]
            nbrs[i, : len(graph[4])] = graph[4]
            nb_mask[i, : len(graph[4])] = 1.0
        soft = step[4] if len(step) > 4 and isinstance(step[4], np.ndarray) else None
        features[i, : len(f)] = f
        masks[i, : len(valid)] = valid
        if soft is None:
            q[i, target] = 1.0
        else:
            q[i, : len(soft)] = soft
        nodes[i, : len(nf)] = nf
        adj[i, : len(nf), : len(nf)] = a
        node_mask[i, : len(nf)] = 1.0
        current[i] = cur
        targets[i] = target
    return {
        "features": mx.array(features),
        "masks": mx.array(masks),
        "targets": mx.array(targets),
        "q": mx.array(q),
        "coefs": mx.array(np.ones(count, dtype=np.float32) if coefs is None else coefs.astype(np.float32)),
        "nodes": mx.array(nodes),
        "adj": mx.array(adj),
        "node_mask": mx.array(node_mask),
        "current": mx.array(current),
        "cand": mx.array(cand),
        "nbrs": mx.array(nbrs),
        "nb_mask": mx.array(nb_mask),
    }


def make_optimizer(lr: float) -> optim.Adam:
    return optim.Adam(learning_rate=lr)


def step_probabilities(params: dict[str, mx.array], features: np.ndarray, valid: np.ndarray, graph) -> np.ndarray:
    batch = make_batch([(features, valid, 0, graph)])
    logp = masked_log_softmax(batch_scores(params, batch), batch["masks"])[0]
    mx.eval(logp)
    return np.exp(np.array(logp))


def greedy_action(params: dict[str, mx.array], features: np.ndarray, valid: np.ndarray, graph) -> int:
    probs = step_probabilities(params, features, valid, graph)
    return int(np.argmax(np.where(valid, probs, -1.0)))


def sample_action(params: dict[str, mx.array], features: np.ndarray, valid: np.ndarray, graph, rng: np.random.Generator) -> int:
    probs = step_probabilities(params, features, valid, graph).astype(np.float64)
    probs = np.where(valid, probs, 0.0)
    probs /= probs.sum()
    return int(rng.choice(len(probs), p=probs))


def save(params: dict[str, mx.array], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mx.savez(str(path), **params)


def load(path: Path) -> dict[str, mx.array]:
    return dict(mx.load(str(path)))


def load_or_init(path: Path | None, seed: int = 0) -> dict[str, mx.array]:
    """Stored weights if they exist and match the current layout, else fresh ones."""
    if path is not None and path.exists():
        params = load(path)
        fresh = init_params(seed)
        if params.keys() == fresh.keys() and all(params[k].shape == fresh[k].shape for k in fresh):
            return params
    return init_params(seed)
