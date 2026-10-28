from pathlib import Path

import mlx.core as mx
import numpy as np
from circuits import HUMAN, RC

from netlist2schematic import policy
from netlist2schematic.diagram import build_diagram
from netlist2schematic.netlist import parse_netlist
from netlist2schematic.trainer import TrainConfig, behaviour_clone, fitted_env, suggest, train
from netlist2schematic.training_db import TrainingDB


def rc():
    return build_diagram(parse_netlist(RC), "rc")


def _bump(array, index, delta):
    flat = np.array(array)
    flat[index] += delta
    return mx.array(flat)


def _random_steps(rng, count=3, candidates=6, nodes=5):
    steps = []
    for i in range(count):
        features = rng.normal(size=(candidates, policy.N_FEATURES)).astype(np.float32)
        valid = np.ones(candidates, dtype=bool)
        valid[2] = False
        node_feats = rng.normal(size=(nodes, policy.N_NODE_FEATURES)).astype(np.float32)
        adj = rng.random((nodes, nodes)).astype(np.float32)
        adj /= adj.sum(axis=1, keepdims=True)
        cand = np.stack([rng.integers(0, 20, candidates), rng.integers(0, 20, candidates), rng.integers(0, 8, candidates)], axis=1).astype(np.int32)
        neighbours = np.array([[5, 7, 2, (i + 1) % nodes], [12, 3, 5, (i + 2) % nodes]], dtype=np.int32)
        steps.append((features, valid, [0, 3, 5][i % 3], (node_feats, adj, i % nodes, cand, neighbours)))
    return steps


def test_loss_gradient_matches_finite_differences():
    rng = np.random.default_rng(0)
    params = policy.init_params(1)
    batch = policy.make_batch(_random_steps(rng), np.array([1.0, -0.5, 2.0]))
    _, grads = policy.loss_and_grads(params, batch)
    eps = 1e-2
    for name, index in [
        ("w1", (0, 0)),
        ("w2", (2, 0)),
        ("wc", (1, 3)),
        ("g_in", (2, 1)),
        ("g_nb0", (0, 4)),
        ("g_self2", (3, 3)),
        ("r_f", (3, 5)),
        ("r_wn", (2, 7)),
        ("r_w2", (4, 0)),
        ("prior", (3,)),
    ]:
        plus = {**params, name: _bump(params[name], index, eps)}
        minus = {**params, name: _bump(params[name], index, -eps)}
        numeric = float((policy.batch_loss(plus, batch) - policy.batch_loss(minus, batch)).item()) / (2 * eps)
        analytic = float(grads[name][index].item())
        assert abs(numeric - analytic) < 2e-2 * max(1.0, abs(numeric)), (name, numeric, analytic)


def test_context_depends_on_graph():
    rng = np.random.default_rng(3)
    params = policy.init_params(2)
    steps = _random_steps(rng, count=1)
    features, valid, target, (nodes, adj, cur, cand, nbrs) = steps[0]
    a = policy.step_probabilities(params, features, valid, (nodes, adj, cur, cand, nbrs))
    b = policy.step_probabilities(params, features, valid, (nodes, np.eye(len(nodes), dtype=np.float32), cur, cand, nbrs))
    assert not np.allclose(a, b)


def test_masked_softmax_ignores_invalid_candidates():
    z = mx.array(np.array([0.0, 10.0, 0.0], dtype=np.float32))
    mask = mx.array(np.array([True, False, True]))
    probs = np.exp(np.array(policy.masked_log_softmax(z, mask)))
    assert probs[1] < 1e-6
    assert abs(probs[0] - 0.5) < 1e-5


def test_old_model_with_other_feature_count_is_replaced(tmp_path):
    path = tmp_path / "old.npz"
    old = policy.init_params(0, n_in=21)
    del old["wc"]  # an older model without the GNN
    policy.save(old, path)
    fresh = policy.load_or_init(path)
    assert fresh["w1"].shape[0] == policy.N_FEATURES and "wc" in fresh


def test_suggestion_history_and_heat():
    d = rc()
    result = suggest(d, Path("/nonexistent"))
    assert len([s for s in result.history if s.kind == "place"]) == len(d.components)
    assert {step.component for step in result.history} == set(range(len(d.components)))
    from netlist2schematic.trainer import SUGGEST_GRID_SCALE

    grid = (int(d.w * SUGGEST_GRID_SCALE), int(d.h * SUGGEST_GRID_SCALE))
    for step in result.history:
        assert step.heat is not None and step.heat.shape == grid
        assert abs(step.heat.max() - 1.0) < 1e-6
    fitted_env(d, result.placement)[0].demo_steps(result.placement)


def test_revisions_can_move_placed_parts():
    from netlist2schematic.env import MAX_REVISITS_PER_PART, PlacementEnv

    d = rc()
    env = PlacementEnv(d)
    first = env.current
    while not env.revisiting and not env.done:
        features, valid = env.observe()
        env.step(int(np.flatnonzero(valid)[len(np.flatnonzero(valid)) // 3]))
    assert env.revisiting
    part = env.current
    assert part != first  # the anchor (first part) is never re-decided
    features, valid = env.observe()
    here = env.index_of(*env.placement[part])
    assert features[here, policy.E.F_STAY] == 1.0 and valid[here]  # staying is always allowed
    target = int(np.flatnonzero(valid)[len(np.flatnonzero(valid)) // 2])
    env.step(target)
    assert env.placement[part] == tuple(int(v) for v in env.anchors[target])
    assert env.revisit_count[part] <= MAX_REVISITS_PER_PART


def test_history_replay_reproduces_suggestion():
    d = rc()
    result = suggest(d, Path("/nonexistent"))
    state = {}
    for step in result.history:
        if step.kind == "move":
            assert state[step.component] == step.previous
        state[step.component] = step.position
    assert {d.components[ci].name: pos for ci, pos in state.items()} == result.placement


def test_continuation_keeps_prefix():
    d = rc()
    r1 = d.component_index("R1")
    prefix = {r1: (-40, -20, 4)}  # far outside the grid, mirrored
    result = suggest(d, Path("/nonexistent"), prefix)
    assert result.placement["R1"] == (-40, -20, 4)
    assert result.history[0].component == r1 and result.history[0].kind == "given"
    assert len(result.placement) == len(d.components)


def test_behaviour_cloning_reproduces_human_placement(tmp_path):
    db = TrainingDB(tmp_path / "t.sqlite")
    db.add("rc", RC, HUMAN)
    params, history = behaviour_clone(policy.init_params(0), db.examples(), TrainConfig(bc_epochs=300, bc_lr=1e-2))
    assert history[-1] < history[0]
    model = tmp_path / "policy.npz"
    policy.save(params, model)
    # The relative arrangement is what makes a schematic readable: compare up to a common shift.
    proposed = suggest(rc(), model).placement
    dx = proposed["R1"][0] - HUMAN["R1"][0]
    dy = proposed["R1"][1] - HUMAN["R1"][1]
    assert proposed == {n: (x + dx, y + dy, o) for n, (x, y, o) in HUMAN.items()}


def test_train_skips_invalid_examples_and_saves_model(tmp_path):
    db = TrainingDB(tmp_path / "t.sqlite")
    db.add("rc", RC, HUMAN)
    db.add("rc", RC, {"R1": (1, 1, 0)})  # incomplete -> skipped
    model = tmp_path / "policy.npz"
    messages = []
    history = train(db, model, TrainConfig(bc_epochs=10, rl_episodes=2), progress=lambda *a: messages.append(a[3]))
    assert model.exists()
    assert len(history["rl_reward"]) == 2
    assert any("übersprungen" in m for m in messages)


def test_database_delete(tmp_path):
    db = TrainingDB(tmp_path / "t.sqlite")
    ids = [db.add("rc", RC, HUMAN) for _ in range(3)]
    assert db.delete(ids[:2]) == 2
    assert [e.id for e in db.examples()] == ids[2:]


def test_database_round_trip_with_pin_maps(tmp_path):
    db = TrainingDB(tmp_path / "t.sqlite")
    db.add("rc", RC, HUMAN, pin_maps={"chip": {"in": 1}})
    ex = db.examples("rc")[0]
    assert ex.placement == HUMAN
    assert ex.pin_maps == {"chip": {"in": 1}}


def test_sub_diagram_keeps_human_positions_valid():
    from circuits import TWO_ICS

    from netlist2schematic.augment import connected_subset, sub_diagram, variants

    d = build_diagram(parse_netlist(TWO_ICS), "top")
    rng = np.random.default_rng(1)
    keep = connected_subset(d, rng, 3)
    assert len(keep) == 3
    sub, remap = sub_diagram(d, keep)
    assert len(sub.components) == 3
    assert all(c < 3 for pins in sub.nets.values() for c, _ in pins)
    # Every variant of a valid placement replays without errors.
    env, named = fitted_env(rc(), HUMAN)
    placement = {env.diagram.component_index(n): pos for n, pos in named.items()}
    found = variants(env.diagram, placement, rng, subsets=4, shifts=2)
    assert found
    from netlist2schematic.env import PlacementEnv

    for sub, sub_placement in found:
        PlacementEnv(sub).demo_steps({sub.components[ci].name: pos for ci, pos in sub_placement.items()})


def test_augmented_steps_are_sampled_and_include_target(tmp_path):
    from netlist2schematic.trainer import demo_steps

    db = TrainingDB(tmp_path / "t.sqlite")
    db.add("rc", RC, HUMAN)
    plain, _ = demo_steps(db.examples())
    augmented, messages = demo_steps(db.examples(), TrainConfig(candidate_samples=64))
    assert len(augmented) > len(plain)
    assert all(f.shape[0] <= 64 and t == 0 and m.all() for f, m, t, _, _ in augmented)
    for *_, soft in augmented:
        assert abs(soft.sum() - 1.0) < 1e-5 and soft[0] == soft.max()  # soft target peaks at the human cell
    assert "Varianten" in messages[-1]


def test_variant_count_matches_training(tmp_path):
    from netlist2schematic.trainer import demo_steps, example_variants, superseded, training_examples

    db = TrainingDB(tmp_path / "t.sqlite")
    first_id = db.add("rc", RC, HUMAN)
    second_id = db.add("rc", RC, HUMAN)
    cfg = TrainConfig()
    # Same netlist twice: the newer drawing replaces the older one in training.
    assert superseded(db.examples()) == {first_id: second_id}
    assert [e.id for e in training_examples(db.examples(), cfg)] == [second_id]
    shown = sum(len(example_variants(ex, cfg)[2]) for ex in training_examples(db.examples(), cfg))
    _, messages = demo_steps(db.examples(), cfg)
    assert f"+ {shown} Varianten" in messages[-1]
    # Per-example streams: an example's variants do not depend on the others.
    first = db.examples()[0]
    alone = len(example_variants(first, cfg)[2])
    assert alone == len(example_variants(db.examples()[0], cfg)[2])


def test_perturbed_revisions_teach_correction():
    from netlist2schematic.env import F_STAY, PlacementEnv

    d = rc()
    env, named = fitted_env(d, HUMAN)
    steps = PlacementEnv(env.diagram).demo_steps(named, np.random.default_rng(0), perturb=1.0)
    revisions = [s for s in steps if s[0][:, policy.E.F_REVISIT].max() == 1.0]
    assert revisions
    # The "current position" marker no longer sits on the target: the policy must move the part back.
    assert any(s[0][s[2], F_STAY] == 0.0 and s[0][:, F_STAY].max() == 1.0 for s in revisions)


def test_oracle_follows_shifted_neighbours():
    from netlist2schematic.env import PlacementEnv
    from netlist2schematic.trainer import oracle_target

    d = rc()
    from dataclasses import replace

    env, named = fitted_env(d, HUMAN)
    human = {env.diagram.component_index(n): pos for n, pos in named.items()}
    env = PlacementEnv(replace(env.diagram, w=env.w + 10, h=env.h + 10))  # room for the shifted layout
    first = env.current
    x, y, o = human[first]
    env.step(env.index_of(x + 3, y + 1, o))  # first part 3 right, 1 down of the human position
    _, valid = env.observe()
    target = oracle_target(env, human, valid)
    hx, hy, ho = human[env.current]
    assert tuple(int(v) for v in env.anchors[target]) == (hx + 3, hy + 1, ho)


def test_relation_scorer_learns_exact_offsets(tmp_path):
    # Trained on one example, the policy reproduces it exactly in a free run.
    db = TrainingDB(tmp_path / "t.sqlite")
    db.add("rc", RC, HUMAN)
    model = tmp_path / "m.npz"
    train(db, model, TrainConfig(bc_epochs=120, dagger_rounds=0))
    proposed = suggest(rc(), model).placement
    dx = proposed["R1"][0] - HUMAN["R1"][0]
    dy = proposed["R1"][1] - HUMAN["R1"][1]
    assert proposed == {n: (x + dx, y + dy, o) for n, (x, y, o) in HUMAN.items()}
