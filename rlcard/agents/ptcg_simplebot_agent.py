"""SimpleBot heuristic agent for PTCG.

Based on Battle-Subway simplebot scoring concepts. Uses priority-based
action selection among raw action_ids.
"""

import random


class PtcgSimpleBotAgent:
    """Heuristic agent with priority-based action selection.

    Template priority order (highest first):
        attack > attach_energy > evolve > play_supporter > play_item
        > play_stadium > retreat > use_ability > use_stadium
        > use_trainer_in_play > pass_turn
    """

    TEMPLATE_PRIORITY = [
        "attack",
        "attach_energy_active",
        "attach_energy_bench",
        "evolve",
        "play_supporter",
        "play_item",
        "play_stadium",
        "retreat",
        "use_ability",
        "use_stadium",
        "use_trainer_in_play",
        "play_basic",
        "resolve_choice",
        "pass_turn",
    ]

    def __init__(self, seed=None):
        self.use_raw = True  # Use raw action_id strings, not template_ids
        self.rng = random.Random(seed)

    def step(self, state):
        """Select an action based on priority."""
        return self._choose_action(state)

    def eval_step(self, state):
        """Evaluation step (same as training for heuristic agent)."""
        return self.step(state), {}

    def _choose_action(self, state):
        """Pick the highest-priority legal action."""
        raw_legal = state.get("raw_legal_actions", [])
        legal_ids = state.get("legal_actions", {})

        if not raw_legal:
            return "pass-turn"

        # Build a lookup: template kind → list of action views
        by_kind = {}
        for action in raw_legal:
            kind = action.get("kind", "")
            by_kind.setdefault(kind, []).append(action)

        # Try each priority kind
        for kind in self.TEMPLATE_PRIORITY:
            candidates = by_kind.get(kind)
            if candidates:
                return candidates[0]["id"]

        # Fallback: first legal action
        if raw_legal:
            return raw_legal[0]["id"]
        return "pass-turn"
