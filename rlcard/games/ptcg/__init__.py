"""Thin RLCard game facade over deckgym_ptcg.PtcgEnv.

This module does NOT contain any PTCG game rules. All rules live in the
Rust engine (deckgym-core). This class only bridges the PyO3 PtcgEnv to
RLCard's expected game interface.
"""

import deckgym_ptcg


class PtcgGame:
    """RLCard-compatible game wrapper around the Rust PTCG engine."""

    def __init__(self, allow_step_back=False):
        self.allow_step_back = allow_step_back
        self.np_random = None  # Set by Env base class
        self._ptcg_env: deckgym_ptcg.PtcgEnv | None = None
        self._config = {}
        self._seed = 0
        self._deck_a = None
        self._deck_b = None
        self._current_player = 0

    def configure(self, config: dict):
        """Store game-specific config (called by Env base)."""
        self._config.update(config)

    def init_game(self):
        """Start a new game. Returns (state, player_id) tuple."""
        import deckgym_ptcg

        if self._ptcg_env is None:
            self._ptcg_env = deckgym_ptcg.PtcgEnv({
                "max_ply": self._config.get("max_ply", 10000),
                "hide_private_information": self._config.get("hide_private_information", True),
                "record_trace": self._config.get("record_trace", False),
                "illegal_action_policy": self._config.get("illegal_action_policy", "error"),
            })

        if self._deck_a is None or self._deck_b is None:
            raise RuntimeError(
                "deck_a and deck_b must be set via config. "
                "Example: rlcard.make('ptcg', config={'deck_a': '/path/a.txt', 'deck_b': '/path/b.txt'})"
            )

        result = self._ptcg_env.reset(self._seed, self._deck_a, self._deck_b)
        self._current_player = result["current_player"]
        state = self._ptcg_state_to_dict(result)
        return state, self._current_player

    def step(self, action):
        """Apply action (string action_id). Returns (state, player_id)."""
        result = self._ptcg_env.step(action)
        self._current_player = result["current_player"]
        state = self._ptcg_state_to_dict(result)
        return state, self._current_player

    def get_state(self, player_id):
        """Get the observation for a given player as an RLCard state dict."""
        obs = self._ptcg_env.observe(player_id)
        legal = self._ptcg_env.legal_actions()
        return self._build_rlcard_state(obs, legal)

    def get_player_id(self):
        return self._current_player

    def is_over(self):
        return self._ptcg_env.is_over() if self._ptcg_env else False

    def get_num_players(self):
        return 2

    def get_num_actions(self):
        return 15  # Fixed number of coarse action templates

    def get_payoffs(self):
        if self._ptcg_env and self._ptcg_env.is_over():
            return self._ptcg_env.payoffs()
        return [0.0, 0.0]

    def get_legal_actions(self):
        """Return list of template IDs currently legal."""
        legal = self._ptcg_env.legal_actions()
        seen = set()
        template_ids = []
        for a in legal:
            tid = a["template_id"]
            if tid not in seen:
                seen.add(tid)
                template_ids.append(tid)
        return template_ids

    def get_action_record(self):
        return self._action_record if hasattr(self, '_action_record') else []

    def set_seed(self, seed):
        self._seed = seed

    def set_decks(self, deck_a, deck_b):
        self._deck_a = deck_a
        self._deck_b = deck_b

    def _ptcg_state_to_dict(self, result):
        """Convert a PyO3 StepResult to an RLCard-compatible state dict."""
        obs = result["observation"]
        legal = result["legal_actions"]
        return self._build_rlcard_state(obs, legal)

    def _build_rlcard_state(self, obs, legal_actions):
        """Build RLCard-compatible state dict from observation and legal actions."""
        import numpy as np
        from collections import OrderedDict

        # Features: concatenate public + private features
        pub = obs.get("public_features", [])
        priv = obs.get("private_features", [])
        action_mask = obs.get("action_mask", [False] * 15)

        feature_vec = np.array(pub + priv + list(action_mask), dtype=np.float32)

        # Legal actions as OrderedDict
        seen = set()
        legal = OrderedDict()
        for a in legal_actions:
            tid = a["template_id"]
            if tid not in seen:
                seen.add(tid)
                legal[tid] = None

        return {
            "obs": feature_vec,
            "legal_actions": legal,
            "raw_obs": obs,
            "raw_legal_actions": legal_actions,
            "action_record": obs.get("action_record", []),
            "action_mask": np.array(action_mask, dtype=bool),
        }
