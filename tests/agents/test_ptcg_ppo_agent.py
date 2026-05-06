import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from rlcard.agents.ptcg_ppo_agent import PtcgPPOAgent, compute_gae, legal_mask_from_state
from rlcard.agents.ptcg_raw_tiebreaker import PtcgRawActionTieBreaker
from rlcard.utils.ptcg_deck_pool import load_deck_pool, sample_deck_pair


def make_state():
    return {
        "obs": np.zeros(6, dtype=np.float32),
        "legal_actions": {1: None, 3: None},
        "raw_legal_actions": [
            {"id": "play-basic:hand:0:bench:0", "template_id": 1, "kind": "play_basic", "target": "bench-0"},
            {"id": "attack:0", "template_id": 3, "kind": "attack", "target": None},
        ],
        "raw_obs": {"raw_public_state": {"self": {"active": {}, "bench": []}}},
    }


def test_ppo_masked_policy_samples_only_legal_templates():
    agent = PtcgPPOAgent(num_actions=5, state_shape=[6], hidden_layers_sizes=[8], device="cpu")
    agent.set_raw_action_tie_breaker(None)
    state = make_state()
    for _ in range(50):
        _, info = agent.sample_action(state, deterministic=False)
        assert info["action"] in (1, 3)


def test_ppo_step_returns_legal_raw_action_when_tiebreaker_enabled():
    agent = PtcgPPOAgent(num_actions=5, state_shape=[6], hidden_layers_sizes=[8], device="cpu")
    agent.set_raw_action_tie_breaker(PtcgRawActionTieBreaker())
    action = agent.step(make_state())
    assert action in {"play-basic:hand:0:bench:0", "attack:0"}
    assert agent.use_raw is True


def test_ppo_checkpoint_roundtrip_preserves_output_shapes(tmp_path):
    agent = PtcgPPOAgent(num_actions=5, state_shape=[6], hidden_layers_sizes=[8], device="cpu")
    save_dir = tmp_path / "ppo"
    agent.save_checkpoint(save_dir)

    restored = PtcgPPOAgent(num_actions=5, state_shape=[6], hidden_layers_sizes=[8], device="cpu")
    restored.load_checkpoint(save_dir / "checkpoint_ppo.pt")

    obs = torch.zeros((2, 6), dtype=torch.float32)
    logits, values = restored.network(obs)
    assert logits.shape == (2, 5)
    assert values.shape == (2,)


def test_compute_gae_sparse_terminal_reward_shapes():
    advantages, returns = compute_gae(
        rewards=[0.0, 0.0, 1.0],
        values=[0.1, 0.2, 0.3],
        gamma=0.99,
        gae_lambda=0.95,
    )
    assert advantages.shape == (3,)
    assert returns.shape == (3,)
    assert returns[-1] == pytest.approx(1.0)


def test_legal_mask_from_state_has_only_legal_actions():
    mask = legal_mask_from_state(make_state(), num_actions=5)
    assert mask.tolist() == [False, True, False, True, False]


def test_deck_pool_load_and_sample(tmp_path):
    pool_path = tmp_path / "pool.json"
    pool_path.write_text(json.dumps({
        "decks": [
            {"name": "a1", "archetype": "a", "split": "train", "path": "/tmp/a1.txt"},
            {"name": "b1", "archetype": "b", "split": "validation", "path": "/tmp/b1.txt"},
        ]
    }), encoding="utf-8")
    train_decks = load_deck_pool(pool_path, split="train", validate_paths=False)
    assert [deck["name"] for deck in train_decks] == ["a1"]

    rng = np.random.default_rng(1)

    class RngAdapter:
        def choice(self, values):
            return values[int(rng.integers(0, len(values)))]

    deck_a, deck_b = sample_deck_pair(train_decks, rng=RngAdapter())
    assert deck_a["name"] == "a1"
    assert deck_b["name"] == "a1"
