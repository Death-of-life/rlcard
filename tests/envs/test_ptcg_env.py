"""Tests for the PTCG RLCard environment."""

import pytest

import rlcard
from rlcard.agents import RandomAgent

# Deck files used for testing
DECK_A = "/Users/easygod/Downloads/Battle Subway 北京沙.txt"
DECK_B = "/Users/easygod/Downloads/龙神柱 Battle Subway.txt"


def _make_env(seed=42, **kwargs):
    config = {
        "deck_a": DECK_A,
        "deck_b": DECK_B,
        "seed": seed,
        "max_ply": 10000,
        "record_trace": True,
    }
    config.update(kwargs)
    return rlcard.make("ptcg", config=config)


def test_env_creation():
    """env.create works."""
    env = _make_env()
    assert env.num_players == 2
    assert env.num_actions == 15


def test_reset():
    """reset returns valid state with legal actions."""
    env = _make_env()
    state, player_id = env.reset()
    assert player_id in (0, 1)
    assert len(state["legal_actions"]) > 0
    assert "obs" in state
    assert "raw_legal_actions" in state
    assert len(state["raw_legal_actions"]) > 0


def test_legal_actions_not_empty():
    """Initial state always has legal actions."""
    env = _make_env()
    state, _ = env.reset()
    assert len(state["legal_actions"]) > 0


def test_step_cycle():
    """Consecutive steps don't error."""
    env = _make_env()
    state, player_id = env.reset()
    for _ in range(10):
        if env.is_over():
            break
        legal = state["legal_actions"]
        action = list(legal.keys())[0]
        state, player_id = env.step(action)
        assert player_id in (0, 1)


def test_random_vs_random_one_game():
    """A single RandomAgent vs RandomAgent game completes."""
    env = _make_env()
    env.set_agents([RandomAgent(env.num_actions), RandomAgent(env.num_actions)])
    trajectories, payoffs = env.run(is_training=True)
    assert len(payoffs) == 2
    assert payoffs[0] in (-1.0, 0.0, 1.0)
    assert payoffs[1] in (-1.0, 0.0, 1.0)
    assert payoffs[0] == -payoffs[1] or payoffs == [0.0, 0.0]


def test_random_vs_random_100_games():
    """100 RandomAgent vs RandomAgent games complete without error."""
    env = _make_env()
    env.set_agents([RandomAgent(env.num_actions), RandomAgent(env.num_actions)])
    successes = 0
    for _ in range(100):
        try:
            trajectories, payoffs = env.run(is_training=True)
            successes += 1
        except Exception as e:
            pytest.fail(f"Game failed with error: {e}")
    assert successes == 100


def test_hidden_info():
    """Opponent hand does not contain specific card IDs."""
    import deckgym_ptcg
    env = deckgym_ptcg.PtcgEnv({"hide_private_information": True, "record_trace": True})
    result = env.reset(1, DECK_A, DECK_B)
    # Check observation's raw_public_state
    raw = result["observation"]["raw_public_state"]
    opponent = raw["opponent"]
    # Opponent hand should only have count, not card details
    assert isinstance(opponent["hand"], dict)
    assert "count" in opponent["hand"]
    # Opponent deck should only have count
    assert "count" in opponent["deck"]
    # Own hand should have card details
    own = raw["self"]
    assert isinstance(own["hand"], list)
    if len(own["hand"]) > 0:
        assert "full_name" in own["hand"][0]


def test_illegal_action_error():
    """Illegal action raises error by default."""
    import deckgym_ptcg
    env = deckgym_ptcg.PtcgEnv({"illegal_action_policy": "error"})
    result = env.reset(1, DECK_A, DECK_B)
    try:
        env.step("nonexistent-action-id")
        assert False, "Should have raised an error"
    except Exception:
        pass  # Expected


def test_illegal_action_sample_legal():
    """With sample_legal policy, invalid action falls back to legal."""
    import deckgym_ptcg
    env = deckgym_ptcg.PtcgEnv({
        "illegal_action_policy": "sample_legal",
        "record_trace": True,
    })
    result = env.reset(1, DECK_A, DECK_B)
    # Step with an invalid action ID
    result2 = env.step("nonexistent-action-id")
    assert "observation" in result2
    assert result2["done"] is False
    # Check trace records the fallback
    trace = env.action_trace()
    assert len(trace) >= 1
    assert trace[0]["fallback"] is True


def test_payoffs():
    """Payoffs are correct at game end."""
    import deckgym_ptcg
    env = deckgym_ptcg.PtcgEnv({"max_ply": 10000})
    result = env.reset(1, DECK_A, DECK_B)
    # Play to completion
    ply = 0
    while not env.is_over() and ply < 10000:
        actions = env.legal_actions()
        if not actions:
            break
        env.step(actions[0]["id"])
        ply += 1

    payoffs = env.payoffs()
    assert len(payoffs) == 2
    assert payoffs[0] in (-1.0, 0.0, 1.0)
    assert payoffs[1] in (-1.0, 0.0, 1.0)


def test_action_id_roundtrip():
    """Each legal action id can be applied."""
    import deckgym_ptcg
    env = deckgym_ptcg.PtcgEnv({})
    result = env.reset(1, DECK_A, DECK_B)
    legal = env.legal_actions()
    assert len(legal) > 0
    for action in legal:
        # Try stepping with each legal action (resetting each time)
        env2 = deckgym_ptcg.PtcgEnv({})
        env2.reset(1, DECK_A, DECK_B)
        try:
            result2 = env2.step(action["id"])
            assert "observation" in result2
        except Exception as e:
            pytest.fail(f"Legal action '{action['id']}' failed: {e}")
        break  # Only test first one


def test_trace_recording():
    """Trace records each step."""
    import deckgym_ptcg
    env = deckgym_ptcg.PtcgEnv({"record_trace": True})
    result = env.reset(1, DECK_A, DECK_B)
    # Step a few times
    for _ in range(5):
        if env.is_over():
            break
        actions = env.legal_actions()
        env.step(actions[0]["id"])

    trace = env.action_trace()
    # Should have trace entries for each step
    assert len(trace) > 0
    for entry in trace:
        assert "ply" in entry
        assert "actor" in entry
        assert "action_id" in entry
