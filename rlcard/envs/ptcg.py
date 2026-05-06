"""PTCG RLCard Environment.

Wraps the Rust deckgym_ptcg.PtcgEnv as an RLCard-compatible Env.
"""

from collections import OrderedDict

import numpy as np

from rlcard.envs.env import Env
from rlcard.games.ptcg import PtcgGame


class PtcgEnv(Env):
    """RLCard PTCG environment.

    Config keys:
        deck_a: Path to player A's deck file (required).
        deck_b: Path to player B's deck file (required).
        seed: Random seed (default: None, uses random).
        max_ply: Maximum ply per game (default: 10000).
        hide_private_information: If True, hide opponent hidden info (default: True).
        record_trace: If True, record per-step trace (default: False).
        illegal_action_policy: "error" or "sample_legal" (default: "error").
        allow_step_back: Not supported for PTCG (default: False).
    """

    def __init__(self, config):
        self.name = "ptcg"
        self.default_game_config = {
            "max_ply": 10000,
            "hide_private_information": True,
            "record_trace": False,
            "illegal_action_policy": "error",
        }

        # Handle required deck paths
        deck_a = config.get("deck_a")
        deck_b = config.get("deck_b")
        if not deck_a or not deck_b:
            raise ValueError(
                "config must contain 'deck_a' and 'deck_b' (paths to deck files). "
                "Example: rlcard.make('ptcg', config={'deck_a': '/path/a.txt', 'deck_b': '/path/b.txt'})"
            )

        # Build the game facade
        self.game = PtcgGame(allow_step_back=False)
        self.game.set_seed(config.get("seed", 0))
        self.game.set_decks(deck_a, deck_b)

        # Pass PtcgGame-specific configs
        game_config = self.default_game_config.copy()
        for key in game_config:
            if key in config:
                game_config[key] = config[key]
        self.game.configure(game_config)

        # Ensure base class has defaults it needs
        if "allow_step_back" not in config:
            config = config.copy()
            config["allow_step_back"] = False
        if "seed" not in config:
            config = config.copy()
            config["seed"] = config.get("seed", 0)

        super().__init__(config)

        # Auto-detect state_shape from the actual observation
        self.state_shape = self._detect_state_shape()

    def _detect_state_shape(self):
        """Compute the observation vector dimension from a sample state."""
        state, _ = self.reset()
        return [len(state["obs"])]

    # ---- RLCard Env abstract methods ----

    def _extract_state(self, state):
        """Convert a state dict from the game into the RLCard state format.

        The state dict already contains all needed fields from PtcgGame.
        This method ensures the format is correct.
        """
        if state is None:
            return {
                "obs": np.zeros(1, dtype=np.float32),
                "legal_actions": OrderedDict(),
                "raw_obs": {},
                "raw_legal_actions": [],
                "action_record": state.get("action_record", []) if state else [],
                "action_mask": np.zeros(15, dtype=bool),
            }
        return state

    def _decode_action(self, action_id):
        """Decode an integer template_id into a concrete action_id string.

        Picks the first legal action matching the template.
        """
        legal = self.game._ptcg_env.legal_actions()
        for a in legal:
            if a["template_id"] == action_id:
                return a["id"]
        # Fallback: return first legal action
        if legal:
            return legal[0]["id"]
        return "pass-turn"

    def _get_legal_actions(self):
        """Return the list of legal template IDs for the current state."""
        return self.game.get_legal_actions()

    def get_payoffs(self):
        """Return payoffs: [+1, -1] for win/loss, [0, 0] for tie."""
        return self.game.get_payoffs()

    def get_perfect_information(self):
        """Return the full state (no hidden info protection).

        Note: This exposes player 0's observation by default since
        we need a specific player view. For true perfect information,
        use the Rust engine's replay mode.
        """
        if self.game._ptcg_env:
            obs0 = self.game._ptcg_env.observe(0)
            obs1 = self.game._ptcg_env.observe(1)
            return {
                "player_0": obs0,
                "player_1": obs1,
            }
        return {}
