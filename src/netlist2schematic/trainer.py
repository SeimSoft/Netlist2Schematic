"""Training and inference for the placement policy.

Stage 1, behaviour cloning: every manually approved placement in the database is
replayed in the environment and the network is trained to pick the human's
candidate at each step. Each placement is augmented (see ``augment``) with
connected sub-circuits and shifted copies. Per step the loss uses the human's
candidate plus a random sample of the other valid candidates (sampled softmax),
which keeps memory bounded on large grids.

Stage 1b, DAgger: the trained policy places the examples itself; in every state it
reaches, an automatic oracle gives the right answer (the human's relative layout,
anchored at the neighbours as they actually stand). These states are added and
training continues, so the policy learns to continue consistently after its own
small deviations instead of drifting off (compounding errors of plain cloning).

Stage 2, REINFORCE (optional): episodes are sampled on the circuits in the
database and reinforced by the readability reward minus a running baseline.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Callable

import mlx.core as mx
import numpy as np

from . import groups as G
from . import policy
from .diagram import Diagram, PinMaps, build_diagram
from .augment import variants
from .env import N_NODE_FEATURES, ORIENTATIONS, PlacementEnv
from .placement import Placement, extent, fit_to_grid, repair
from .netlist import parse_netlist
from .training_db import Example, TrainingDB

Progress = Callable[[str, int, int, str], None]


@dataclass
class TrainConfig:
    bc_epochs: int = 200  # minibatch epochs (each epoch = one pass over all steps)
    bc_lr: float = 1e-2
    rl_episodes: int = 0
    rl_lr: float = 5e-4
    reward_scale: float = 5.0
    seed: int = 0
    augment_subsets: int = 8  # connected sub-circuits per example
    augment_shifts: int = 2  # shifted copies per example
    candidate_samples: int = 1024  # candidates per step in the BC loss (target included)
    newest_per_netlist: bool = True  # several drawings of the same netlist: train on the newest only
    revisit_perturb: float = 0.5  # share of revisions trained from a displaced position (learn to correct)
    dagger_rounds: int = 2  # policy rollouts labelled by the relative oracle
    dagger_epochs: int = 60  # training epochs after each round
    dagger_oracle_share: float = 0.3  # rollouts follow the oracle in this share of decisions


@dataclass
class HistoryStep:
    component: int
    position: tuple[int, int, int]  # in the caller's coordinates
    heat: np.ndarray | None  # (w, h) policy probability per anchor (max over orientations), None for given parts
    offset: tuple[int, int] = (0, 0)  # heat[x + offset_x, y + offset_y] belongs to caller cell (x, y)
    kind: str = "place"  # place: new part | move: policy revision | manual: hand correction | given: kept
    previous: tuple[int, int, int] | None = None  # position before a move / correction


@dataclass
class Suggestion:
    placement: dict[str, tuple[int, int, int]]
    reward: float
    details: dict[str, float]
    history: list[HistoryStep] = field(default_factory=list)
    offset: tuple[int, int] = (0, 0)  # grid coordinates = caller coordinates + offset
    regions: dict[str, tuple[int, int, int, int]] = field(default_factory=dict)  # group floorplan (caller coordinates)


def default_model_path(data_dir: Path) -> Path:
    return data_dir / "policy.npz"


def env_for(netlist_text: str, circuit: str, pin_maps: PinMaps | None = None) -> PlacementEnv:
    diagram: Diagram = build_diagram(parse_netlist(netlist_text), circuit, pin_maps)
    return PlacementEnv(diagram)


def fitted_env(diagram: Diagram, named: dict[str, tuple[int, int, int]]) -> tuple[PlacementEnv, dict[str, tuple[int, int, int]]]:
    """Environment whose grid holds ``named`` (shifted into the grid if needed), and the shifted placement.

    Entries for parts the diagram no longer has (e.g. a dropped port symbol) are ignored, a
    combined symbol whose members were placed singly starts at their centre;
    orientations the symbol no longer allows fall back to its first allowed one; parts that
    overlap under the current symbol rules are moved to the nearest free spot.
    """
    names = {c.name for c in diagram.components}
    placement = {diagram.component_index(n): pos for n, pos in named.items() if n in names}
    for ci, comp in enumerate(diagram.components):
        # A combined symbol (e.g. a bridge rectifier) drawn as single parts before: centre of those.
        old = [named[m] for m in comp.members if m in named]
        if ci not in placement and old:
            placement[ci] = (round(sum(p[0] for p in old) / len(old)), round(sum(p[1] for p in old) / len(old)), 0)
    for ci, (x, y, o) in placement.items():
        # An orientation the symbol no longer allows (e.g. a mirrored IC whose pins are now
        # grouped by function): its first allowed one.
        if o not in diagram.components[ci].rotations:
            placement[ci] = (x, y, diagram.components[ci].rotations[0])
    placement = repair(replace(diagram, bounded=False), placement)
    fitted, shifted = fit_to_grid(diagram, placement)
    return PlacementEnv(fitted), {fitted.components[ci].name: pos for ci, pos in shifted.items()}


SOFT_SIGMA = 0.5  # cells: spread of the soft target around the human position
SOFT_RADIUS = 2  # cells: beyond this a candidate gets no credit


def _first_decision(graph) -> bool:
    """No part is placed yet (graph state column "placed" is all zero)."""
    return float(graph[0][:, N_NODE_FEATURES - 5].sum()) == 0.0


def _sample_candidates(step, rng: np.random.Generator, k: int, h: int):
    """Reduce a step to the target plus at most ``k - 1`` other valid candidates and
    attach a soft target (the human's cell gets the most weight, cells within
    SOFT_RADIUS with the same orientation get exp(-d^2 / 2 sigma^2)).

    Orientations that merely look alike (a resistor turned by 180 degrees) are NOT
    treated as equivalent: their pins are swapped, so the parts connected to them
    belong on the other side.

    A quarter of the sample are the candidates closest to the target (hard
    negatives); the rest are drawn at random. ``h`` is the grid height (candidate
    index = (x * h + y) * 8 + orientation).
    """
    features, valid, target, graph = step[:4]

    def decode(idx):
        idx = np.asarray(idx)
        return idx // (h * ORIENTATIONS), (idx // ORIENTATIONS) % h, idx % ORIENTATIONS

    tx, ty, to = decode(target)
    others = np.flatnonzero(valid)
    others = others[others != target]
    if len(others) > k - 1:
        ox, oy, oo = decode(others)
        distance = np.maximum(np.abs(ox - tx), np.abs(oy - ty)) + 0.5 * (oo != to)
        order = np.argsort(distance, kind="stable")
        hard = others[order[: k // 4]]
        rest = others[order[k // 4 :]]
        others = np.concatenate([hard, rng.choice(rest, k - 1 - len(hard), replace=False)])
    keep = np.concatenate([[target], others]).astype(np.int64)
    kx, ky, ko = decode(keep)
    d2 = (kx - tx) ** 2 + (ky - ty) ** 2
    if _first_decision(graph):
        # Nothing placed yet: the position is arbitrary, only the orientation matters
        # (a mirrored anchor mirrors the whole drawing).
        soft = (ko == to).astype(float)
    else:
        soft = np.where((ko == to) & (np.maximum(np.abs(kx - tx), np.abs(ky - ty)) <= SOFT_RADIUS), np.exp(-d2 / (2 * SOFT_SIGMA**2)), 0.0)
    soft = (soft / soft.sum()).astype(np.float32)
    if len(graph) > 3:
        graph = (*graph[:3], graph[3][keep], *graph[4:])  # candidate coordinates follow the sample
    return features[keep].astype(np.float32), np.ones(len(keep), dtype=bool), 0, graph, soft


def netlist_key(ex: Example) -> str:
    """Identity of an example's circuit: subcircuit name plus its element lines."""
    sub = parse_netlist(ex.netlist).find(ex.circuit)
    lines = sorted((e.name, e.nodes, e.model or "", e.refs) for e in sub.elements) if sub else []
    return hashlib.sha1(repr((ex.circuit, lines)).encode()).hexdigest()


def superseded(examples: list[Example]) -> dict[int, int]:
    """Example id -> id of the newer example with the same netlist that replaces it."""
    newest: dict[str, Example] = {}
    for ex in examples:
        key = netlist_key(ex)
        if key not in newest or ex.id > newest[key].id:
            newest[key] = ex
    return {ex.id: newest[netlist_key(ex)].id for ex in examples if newest[netlist_key(ex)].id != ex.id}


def training_examples(examples: list[Example], cfg: TrainConfig) -> list[Example]:
    """The examples training uses (the newest drawing per netlist unless configured otherwise)."""
    if not cfg.newest_per_netlist:
        return examples
    replaced = superseded(examples)
    return [ex for ex in examples if ex.id not in replaced]


def example_variants(ex: Example, cfg: TrainConfig) -> tuple[PlacementEnv, dict[str, tuple[int, int, int]], list]:
    """(environment, placement, augmented variants) of one example.

    Each example has its own random stream (seed, example id), so the variants of an
    example do not depend on the other examples and the training data dialog shows
    exactly what training uses. Raises KeyError/ValueError for unusable examples.
    """
    diagram = build_diagram(parse_netlist(ex.netlist), ex.circuit, ex.pin_maps)
    env, named = fitted_env(diagram, ex.placement)
    rng = np.random.default_rng([cfg.seed, ex.id])
    placement = {env.diagram.component_index(n): pos for n, pos in named.items()}
    return env, named, variants(env.diagram, placement, rng, cfg.augment_subsets, cfg.augment_shifts)


def demo_steps(
    examples: list[Example],
    cfg: TrainConfig | None = None,
) -> tuple[list[tuple[np.ndarray, np.ndarray, int]], list[str]]:
    """Training steps of all valid examples and their augmented variants, plus warnings.

    Without ``cfg`` only the original examples are replayed, with all candidates.
    """
    steps = []
    messages = []
    variant_count = 0
    if cfg is not None:
        used = training_examples(examples, cfg)
        if len(used) < len(examples):
            messages.append(f"{len(examples) - len(used)} ältere Zeichnung(en) derselben Netzliste ersetzt, nicht trainiert")
        examples = used
    for ex in examples:
        try:
            if cfg is None:
                diagram = build_diagram(parse_netlist(ex.netlist), ex.circuit, ex.pin_maps)
                env, named = fitted_env(diagram, ex.placement)
                found = []
            else:
                env, named, found = example_variants(ex, cfg)
            perturb_rng = np.random.default_rng([cfg.seed, ex.id, 2]) if cfg is not None else None
            steps_ex = env.demo_steps(named, perturb_rng, cfg.revisit_perturb if cfg is not None else 0.0)
        except (KeyError, ValueError) as exc:
            messages.append(f"Beispiel {ex.id} ({ex.circuit}) übersprungen: {exc}")
            continue
        if cfg is None:
            steps.extend(steps_ex)
            continue
        # Sample right away so only the reduced candidate sets are kept in memory.
        sample_rng = np.random.default_rng([cfg.seed, ex.id, 1])
        steps.extend(_sample_candidates(step, sample_rng, cfg.candidate_samples, env.h) for step in steps_ex)
        for sub, sub_placement in found:
            sub_named = {sub.components[ci].name: pos for ci, pos in sub_placement.items()}
            try:
                sub_env = PlacementEnv(sub)
                sub_steps = sub_env.demo_steps(sub_named, perturb_rng, cfg.revisit_perturb)
            except ValueError:
                continue  # not expected: variants keep the approved positions
            steps.extend(_sample_candidates(step, sample_rng, cfg.candidate_samples, sub_env.h) for step in sub_steps)
            variant_count += 1
    if cfg is not None:
        messages.append(f"{len(examples)} Beispiele + {variant_count} Varianten = {len(steps)} Schritte")
    return steps, messages


MINIBATCH = 64  # steps per update (the relation scorer needs memory per candidate and neighbour)


def _fit(params, steps, epochs: int, opt, progress: Progress | None, label: str, history: list[float], seed: int = 0):
    """Minibatch training; the partition into batches is fixed, their order shuffled per epoch."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(steps))
    batches = [policy.make_batch([steps[i] for i in order[k : k + MINIBATCH]]) for k in range(0, len(steps), MINIBATCH)]
    for epoch in range(epochs):
        total = 0.0
        for b in rng.permutation(len(batches)):
            loss, grads = policy.loss_and_grads(params, batches[b])
            params = opt.update(params, grads) or params
            mx.eval(params, opt.state)
            total += float(loss.item()) * batches[b]["features"].shape[0]
        history.append(total / len(steps))
        if progress and (epoch % 10 == 0 or epoch == epochs - 1):
            progress(label, epoch + 1, epochs, f"loss {history[-1]:.4f}")
    return params


def oracle_target(env: PlacementEnv, human: Placement, valid: np.ndarray) -> int | None:
    """The human's position for the current part, shifted to fit where its placed
    neighbours actually stand (most common neighbour offset, weighted by
    connections). Falls back to the nearest valid cell (same orientation first)."""
    ci = env.current
    shifts: dict[tuple[int, int], float] = {}
    base = env._base()
    for other, links in env._neighbours[ci].items():
        if other in base and other in human:
            key = (base[other][0] - human[other][0], base[other][1] - human[other][1])
            shifts[key] = shifts.get(key, 0.0) + links
    if not shifts:
        for other, pos in base.items():
            if other in human:
                key = (pos[0] - human[other][0], pos[1] - human[other][1])
                shifts[key] = shifts.get(key, 0.0) + 1.0
    sx, sy = max(shifts, key=shifts.get) if shifts else (0, 0)
    hx, hy, ho = human[ci]
    tx, ty = hx + sx, hy + sy
    for orientations in ((ho,), env.diagram.components[ci].rotations):
        for r in range(4):
            for dx in range(-r, r + 1):
                for dy in range(-r, r + 1):
                    if max(abs(dx), abs(dy)) != r:
                        continue
                    x, y = tx + dx, ty + dy
                    if not (0 <= x < env.w and 0 <= y < env.h):
                        continue
                    for o in orientations:
                        index = env.index_of(x, y, o)
                        if valid[index]:
                            return index
    return None


def dagger_steps(params, examples: list[Example], cfg: TrainConfig, round_index: int) -> list:
    """States reached by the policy on the examples, labelled by ``oracle_target``."""
    steps = []
    for ex in training_examples(examples, cfg):
        try:
            env, named = fitted_env(build_diagram(parse_netlist(ex.netlist), ex.circuit, ex.pin_maps), ex.placement)
        except (KeyError, ValueError):
            continue
        human = {env.diagram.component_index(n): pos for n, pos in named.items()}
        rng = np.random.default_rng([cfg.seed, ex.id, 3, round_index])
        env.reset()
        while not env.done:
            features, valid = env.observe()
            graph = env.graph_inputs()
            target = oracle_target(env, human, valid)
            if target is None:
                action = policy.greedy_action(params, features, valid, graph)
            else:
                steps.append(_sample_candidates((features, valid, target, graph), rng, cfg.candidate_samples, env.h))
                follow = rng.random() < cfg.dagger_oracle_share
                action = target if follow else policy.greedy_action(params, features, valid, graph)
            env.step(action)
    return steps


def behaviour_clone(
    params: dict[str, mx.array],
    examples: list[Example],
    cfg: TrainConfig,
    progress: Progress | None = None,
) -> tuple[dict[str, mx.array], list[float]]:
    steps, skipped = demo_steps(examples, cfg)
    if progress:
        for message in skipped:
            progress("behaviour cloning", 0, cfg.bc_epochs, message)
    if not steps:
        return params, []
    opt = policy.make_optimizer(cfg.bc_lr)
    history: list[float] = []
    params = _fit(params, steps, cfg.bc_epochs, opt, progress, "behaviour cloning", history)
    for round_index in range(cfg.dagger_rounds):
        new = dagger_steps(params, examples, cfg, round_index)
        steps = steps + new
        if progress:
            progress(f"DAgger {round_index + 1}", 0, cfg.dagger_epochs, f"{len(new)} neue Zustände, {len(steps)} gesamt")
        params = _fit(params, steps, cfg.dagger_epochs, opt, progress, f"DAgger {round_index + 1}", history)
    return params, history


def reinforce(
    params: dict[str, mx.array],
    examples: list[Example],
    cfg: TrainConfig,
    progress: Progress | None = None,
) -> tuple[dict[str, mx.array], list[float]]:
    if not examples or cfg.rl_episodes <= 0:
        return params, []
    rng = np.random.default_rng(cfg.seed)
    envs = [env_for(ex.netlist, ex.circuit, ex.pin_maps) for ex in examples]
    opt = policy.make_optimizer(cfg.rl_lr)
    baseline: float | None = None
    rewards = []
    for episode in range(cfg.rl_episodes):
        env = envs[episode % len(envs)]
        env.reset()
        trajectory = []
        while not env.done:
            features, valid = env.observe()
            graph = env.graph_inputs()
            action = policy.sample_action(params, features, valid, graph, rng)
            trajectory.append((features, valid, action, graph))
            env.step(action)
        reward, _, _ = env.evaluate()
        rewards.append(reward)
        baseline = reward if baseline is None else 0.9 * baseline + 0.1 * reward
        advantage = (reward - baseline) / cfg.reward_scale
        batch = policy.make_batch(trajectory, np.full(len(trajectory), advantage))
        _, grads = policy.loss_and_grads(params, batch)
        params = opt.update(params, grads) or params
        mx.eval(params, opt.state)
        if progress and (episode % 10 == 0 or episode == cfg.rl_episodes - 1):
            progress("reinforce", episode + 1, cfg.rl_episodes, f"reward {reward:.2f}")
    return params, rewards


def train(
    db: TrainingDB,
    model_path: Path,
    cfg: TrainConfig | None = None,
    progress: Progress | None = None,
) -> dict[str, list[float]]:
    """Train on the whole database and store the weights. Fresh weights if no model exists yet."""
    cfg = cfg or TrainConfig()
    examples = db.examples()
    params = policy.load_or_init(model_path, cfg.seed)
    rl_examples = training_examples(examples, cfg)
    params, bc_history = behaviour_clone(params, examples, cfg, progress)
    params, rl_rewards = reinforce(params, rl_examples, cfg, progress)
    policy.save(params, model_path)
    return {"bc_loss": bc_history, "rl_reward": rl_rewards}


def example_history(ex: Example, model_path: Path | None) -> tuple[Diagram, list[HistoryStep]]:
    """An example's placement as a history (the order training replays it), with the
    current policy's probability map at every placement for comparison."""
    diagram = build_diagram(parse_netlist(ex.netlist), ex.circuit, ex.pin_maps)
    env, named = fitted_env(diagram, ex.placement)
    params = policy.load_or_init(model_path)
    name, (x0, y0, _) = next(iter(named.items()))
    ox, oy, _ = ex.placement[name]
    offset = (x0 - ox, y0 - oy)  # grid coordinates = example coordinates + offset
    history: list[HistoryStep] = []
    env.reset()
    while not env.done:
        ci = env.current
        revision = env.revisiting
        features, valid = env.observe()
        target = named[env.diagram.components[ci].name]
        if not revision:
            probs = policy.step_probabilities(params, features, valid, env.graph_inputs())
            heat = probs.reshape(env.w, env.h, -1).max(axis=2)
            x, y, o = target
            history.append(HistoryStep(ci, (x - offset[0], y - offset[1], o), (heat / max(heat.max(), 1e-12)).astype(np.float32), offset, kind="example"))
        env.step(env.index_of(*target))
    return diagram, history


def _anchor_choices(env: PlacementEnv, probs: np.ndarray, valid: np.ndarray, width: int) -> list[int]:
    """First part: the most probable orientations (summed over all cells), each at the
    valid cell closest to the grid centre (its absolute position does not matter)."""
    per_orientation = np.where(valid, probs, 0.0).reshape(env.w, env.h, ORIENTATIONS).sum(axis=(0, 1))
    xs, ys = env.anchors[:, 0], env.anchors[:, 1]
    centre = np.abs(xs - env.w / 2) + np.abs(ys - env.h / 2)
    choices = []
    for o in np.argsort(per_orientation)[::-1][:width]:
        if per_orientation[o] <= 0:
            continue
        candidates = np.flatnonzero(valid & (env.anchors[:, 2] == o))
        choices.append(int(candidates[np.argmin(centre[candidates])]))
    return choices


SUGGEST_GRID_SCALE = 2.0  # proposals get this much room per dimension (features do not depend on it)


def _start_env(diagram: Diagram, prefix: Placement | None) -> tuple[PlacementEnv, Placement, tuple[int, int]]:
    """Environment for proposals: generous grid, ``prefix`` (caller coordinates) shifted in.

    With functional groups the coarse placement comes first: ``groups.floorplan`` gives
    every group its region (aligned to the parts already placed, if any), and the
    environment keeps each part in its region (``env.regions``, grid coordinates).

    Returns (environment, prefix in grid coordinates, offset); grid = caller + offset."""
    # Generous grid: drawings with more spacing than the default grid allows must fit.
    bounded = replace(diagram, bounded=True, w=int(diagram.w * SUGGEST_GRID_SCALE), h=int(diagram.h * SUGGEST_GRID_SCALE))
    regions = G.floorplan(diagram, origin=(2, 2))
    if prefix:
        fitted, shifted = fit_to_grid(bounded, prefix)
        # Leave room around the given parts for the rest.
        fitted = replace(fitted, w=max(fitted.w, bounded.w), h=max(fitted.h, bounded.h))
        x0, y0, _, _ = extent(bounded, prefix)
        offset = (1 - x0 if x0 < 1 else 0, 1 - y0 if y0 < 1 else 0)
        if regions:
            regions = G.aligned(diagram, regions, shifted)
            # Regions that would leave the grid on the low side are clipped there.
            regions = {g: (max(0, a), max(0, b), max(0, c), max(0, d)) for g, (a, b, c, d) in regions.items()}
    else:
        fitted, shifted, offset = bounded, {}, (0, 0)
    if regions:
        _, _, rx1, ry1 = G.extent(regions)
        fitted = replace(fitted, w=max(fitted.w, rx1 + 3), h=max(fitted.h, ry1 + 3))
    env = PlacementEnv(fitted, regions=regions)
    env.reset(shifted)
    return env, shifted, offset


def caller_regions(env: PlacementEnv, offset: tuple[int, int]) -> dict[str, tuple[int, int, int, int]]:
    """The environment's group regions in the caller's coordinates."""
    return G.shifted(env.regions, -offset[0], -offset[1])


class Rollout:
    """Step-by-step proposals from a given state (for the editor's "next step").

    ``next()`` returns the policy's next decision as a history step (a placement or
    an actual move; kept parts are skipped), or None when everything is placed."""

    def __init__(self, diagram: Diagram, params: dict[str, mx.array], state: Placement):
        self.params = params
        self.env, _, self.offset = _start_env(diagram, state)
        self.regions = caller_regions(self.env, self.offset)  # coarse group placement, shown while building

    @property
    def done(self) -> bool:
        return self.env.done

    def next(self) -> HistoryStep | None:
        env = self.env
        while not env.done:
            features, valid = env.observe()
            if not valid.any():
                raise RuntimeError(f"no valid position left for {env.diagram.components[env.current].name}")
            probs = policy.step_probabilities(self.params, features, valid, env.graph_inputs())
            heat = probs.reshape(env.w, env.h, -1).max(axis=2)
            if not env.placement:
                action = _anchor_choices(env, probs, valid, 1)[0]
            else:
                action = int(np.argmax(np.where(valid, probs, -1.0)))
            ci = env.current
            revision = env.revisiting
            before = env.placement.get(ci)
            env.step(action)
            if revision and env.placement[ci] == before:
                continue  # kept where it was: not a step
            ox, oy = self.offset
            x, y, o = env.placement[ci]
            previous = (before[0] - ox, before[1] - oy, before[2]) if revision else None
            return HistoryStep(
                ci,
                (x - ox, y - oy, o),
                (heat / max(heat.max(), 1e-12)).astype(np.float32),
                self.offset,
                kind="move" if revision else "place",
                previous=previous,
            )
        return None


def suggest(diagram: Diagram, model_path: Path | None, prefix: Placement | None = None, beam: int = 1) -> Suggestion:
    """Rollout with the stored policy (or fresh weights when no model exists).

    ``beam`` > 1 keeps that many partial placements (the most probable decision
    sequences) instead of committing to the single best decision at every step,
    so one early slip does not decide the whole drawing.

    ``prefix`` keeps already placed parts (any coordinates, e.g. from the unbounded
    editor); the policy places the rest and may move placed parts (revisions).
    The result is in the caller's coordinates; ``history`` starts with the given
    parts, then lists every placement and every actual move with the policy's
    probability map. Replaying it step by step reproduces the result.
    """
    params = policy.load_or_init(model_path)
    start_env, shifted, offset = _start_env(diagram, prefix)

    def back(pos: tuple[int, int, int]) -> tuple[int, int, int]:
        return pos[0] - offset[0], pos[1] - offset[1], pos[2]

    given = [HistoryStep(ci, back(pos), None, kind="given") for ci, pos in shifted.items()]
    # A beam entry: (log probability, environment, history).
    beams = [(0.0, start_env, given)]
    width = max(1, beam)
    while not all(env.done for _, env, _ in beams):
        expanded = []
        for logp, env, history in beams:
            if env.done:
                expanded.append((logp, env, history))
                continue
            features, valid = env.observe()
            if not valid.any():
                raise RuntimeError(f"no valid position left for {env.diagram.components[env.current].name}")
            probs = policy.step_probabilities(params, features, valid, env.graph_inputs())
            heat = probs.reshape(env.w, env.h, -1).max(axis=2)
            heat = (heat / max(heat.max(), 1e-12)).astype(np.float32)
            masked = np.where(valid, probs, -1.0)
            if not env.placement:
                choices = _anchor_choices(env, probs, valid, width)
            else:
                choices = np.argsort(masked)[::-1][:width]
            for action in choices:
                if masked[action] <= 0 and action != choices[0]:
                    continue
                child = env.clone() if len(choices) > 1 else env
                ci = child.current
                revision = child.revisiting
                before = child.placement.get(ci)
                child.step(int(action))
                steps = history
                if not (revision and child.placement[ci] == before):  # a kept part is not a step
                    steps = history + [
                        HistoryStep(
                            ci,
                            back(child.placement[ci]),
                            heat,
                            offset,
                            kind="move" if revision else "place",
                            previous=back(before) if revision else None,
                        )
                    ]
                expanded.append((logp + float(np.log(max(probs[action], 1e-12))), child, steps))
        expanded.sort(key=lambda b: b[0], reverse=True)
        beams = expanded[:width]
    _, env, history = beams[0]
    reward, details, _ = env.evaluate()
    placement = {env.diagram.components[ci].name: back(pos) for ci, pos in env.placement.items()}
    return Suggestion(
        placement=placement, reward=reward, details=details, history=history, offset=offset,
        regions=caller_regions(start_env, offset),
    )
