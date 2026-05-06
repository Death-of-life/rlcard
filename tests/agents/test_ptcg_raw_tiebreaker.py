from collections import OrderedDict

import numpy as np

from rlcard.agents.dqn_agent import DQNAgent
from rlcard.agents.nfsp_agent import NFSPAgent
from rlcard.agents.ptcg_raw_tiebreaker import PtcgRawActionTieBreaker


def _slot(hp=100, current_hp=100, energy_count=0, stage="BASIC"):
    return {
        "hp": hp,
        "current_hp": current_hp,
        "energy_count": energy_count,
        "stage": stage,
    }


def _state(raw_actions):
    template_ids = sorted({action["template_id"] for action in raw_actions})
    return {
        "obs": np.zeros(2, dtype=np.float32),
        "legal_actions": OrderedDict((template_id, None) for template_id in template_ids),
        "raw_legal_actions": raw_actions,
        "raw_obs": {
            "raw_public_state": {
                "self": {
                    "active": _slot(hp=100, current_hp=50, energy_count=1),
                    "bench": [
                        _slot(hp=100, current_hp=20, energy_count=0),
                        _slot(hp=120, current_hp=120, energy_count=2, stage="STAGE1"),
                    ],
                }
            }
        },
    }


def _action(action_id, template_id, kind, target=None, hand_index=None):
    payload = None if hand_index is None else {"hand_index": hand_index}
    return {
        "id": action_id,
        "template_id": template_id,
        "kind": kind,
        "target": target,
        "payload": payload,
    }


def test_tie_breaker_uses_only_matching_template():
    tie_breaker = PtcgRawActionTieBreaker()
    state = _state([
        _action("play-basic:hand:0:bench:1", 1, "play_basic", "bench:1"),
        _action("evolve:hand:0:target:active", 2, "evolve", "active"),
    ])

    assert tie_breaker.choose(1, state) == "play-basic:hand:0:bench:1"


def test_tie_breaker_falls_back_to_first_legal_raw_action():
    tie_breaker = PtcgRawActionTieBreaker()
    state = _state([
        _action("pass-turn", 0, "pass_turn"),
        _action("play-basic:hand:0:bench:0", 1, "play_basic", "bench:0"),
    ])

    assert tie_breaker.choose(14, state) == "pass-turn"


def test_play_basic_prefers_low_bench_index():
    tie_breaker = PtcgRawActionTieBreaker()
    state = _state([
        _action("play-basic:hand:0:bench:1", 1, "play_basic", "bench:1"),
        _action("play-basic:hand:1:bench:0", 1, "play_basic", "bench:0"),
    ])

    assert tie_breaker.choose(1, state) == "play-basic:hand:1:bench:0"


def test_attach_energy_bench_prefers_better_target_slot():
    tie_breaker = PtcgRawActionTieBreaker()
    state = _state([
        _action("attach-energy:hand:0:target:bench-0", 4, "attach_energy_bench", "bench-0"),
        _action("attach-energy:hand:0:target:bench-1", 4, "attach_energy_bench", "bench-1"),
    ])

    assert tie_breaker.choose(4, state) == "attach-energy:hand:0:target:bench-1"


def test_retreat_prefers_better_bench_target():
    tie_breaker = PtcgRawActionTieBreaker()
    state = _state([
        _action("retreat:bench:0", 9, "retreat", "bench:0"),
        _action("retreat:bench:1", 9, "retreat", "bench:1"),
    ])

    assert tie_breaker.choose(9, state) == "retreat:bench:1"


def test_dqn_feed_maps_raw_action_to_template_id():
    agent = DQNAgent(num_actions=15, state_shape=[2], mlp_layers=[8], replay_memory_init_size=100)
    state = _state([
        _action("attach-energy:hand:0:target:bench-1", 4, "attach_energy_bench", "bench-1"),
    ])
    next_state = _state([
        _action("pass-turn", 0, "pass_turn"),
    ])

    agent.feed([state, "attach-energy:hand:0:target:bench-1", 0.0, next_state, False])

    assert agent.memory.memory[-1].action == 4


def test_dqn_returns_raw_action_when_tie_breaker_enabled():
    agent = DQNAgent(num_actions=15, state_shape=[2], mlp_layers=[8], replay_memory_init_size=100)
    agent.set_raw_action_tie_breaker(PtcgRawActionTieBreaker())
    state = _state([
        _action("attach-energy:hand:0:target:bench-0", 4, "attach_energy_bench", "bench-0"),
        _action("attach-energy:hand:0:target:bench-1", 4, "attach_energy_bench", "bench-1"),
    ])

    assert agent.use_raw is True
    assert agent.step(state) == "attach-energy:hand:0:target:bench-1"


def test_nfsp_returns_raw_action_when_tie_breaker_enabled():
    agent = NFSPAgent(
        num_actions=15,
        state_shape=[2],
        hidden_layers_sizes=[8],
        q_mlp_layers=[8],
    )
    agent._mode = "average_policy"
    agent.set_raw_action_tie_breaker(PtcgRawActionTieBreaker())
    state = _state([
        _action("attach-energy:hand:0:target:bench-0", 4, "attach_energy_bench", "bench-0"),
        _action("attach-energy:hand:0:target:bench-1", 4, "attach_energy_bench", "bench-1"),
    ])

    assert agent.use_raw is True
    assert agent.step(state) == "attach-energy:hand:0:target:bench-1"
