"""Heuristic raw action tie-breaker for PTCG template agents.

The neural agents still choose a coarse template id. This helper chooses a
concrete Rust action id among legal raw actions with that template.
"""

import re


class PtcgRawActionTieBreaker:
    """Select a concrete legal raw action for a chosen template id."""

    def choose(self, template_id, state):
        raw_legal = state.get("raw_legal_actions", []) or []
        candidates = [
            action for action in raw_legal
            if isinstance(action, dict) and action.get("template_id") == template_id
        ]
        if not candidates:
            return self._fallback(raw_legal)

        best = sorted(candidates, key=lambda action: (-self._score(action, state), self._stable_id(action)))[0]
        return best.get("id", self._fallback(raw_legal))

    def _score(self, action, state):
        kind = action.get("kind", "")
        if kind == "play_basic":
            return -self._target_index(action.get("target"), default=99)
        if kind in ("evolve", "attach_energy_active", "attach_energy_bench", "attach_tool"):
            return self._slot_score(state, action.get("target"), active_bonus=True)
        if kind == "retreat":
            return self._slot_score(state, action.get("target"), active_bonus=False)
        if kind == "attack":
            return self._attack_index(action)
        if kind in ("play_item", "play_supporter", "play_stadium"):
            return -self._hand_index(action)
        if kind == "resolve_choice":
            return self._choice_score(action)
        return -self._hand_index(action)

    def _fallback(self, raw_legal):
        for action in raw_legal:
            if isinstance(action, dict) and action.get("id"):
                return action["id"]
            if isinstance(action, str):
                return action
        return "pass-turn"

    def _stable_id(self, action):
        return action.get("id", "")

    def _hand_index(self, action):
        payload = action.get("payload") if isinstance(action, dict) else {}
        if isinstance(payload, dict) and "hand_index" in payload:
            try:
                return int(payload["hand_index"])
            except (TypeError, ValueError):
                pass
        action_id = action.get("id", "")
        match = re.search(r":hand:(\d+)", action_id)
        if match:
            return int(match.group(1))
        return 999

    def _attack_index(self, action):
        action_id = action.get("id", "")
        match = re.search(r"^attack:(\d+)$", action_id)
        if match:
            return int(match.group(1))
        return 0

    def _choice_score(self, action):
        action_id = action.get("id", "")
        if "none" in action_id:
            return -1
        match = re.search(r"index:(\d+)", action_id)
        if match:
            return int(match.group(1))
        if "bool:true" in action_id:
            return 1
        if "bool:false" in action_id:
            return 0
        return 0

    def _slot_score(self, state, target, active_bonus):
        slot = self._slot_from_target(state, target)
        if not slot:
            return 0.0
        score = 0.0
        if active_bonus and self._is_active_target(target):
            score += 2.0
        score += float(slot.get("energy_count", 0)) * 0.5
        score += self._hp_ratio(slot)
        score += self._stage_score(slot.get("stage")) * 0.25
        return score

    def _slot_from_target(self, state, target):
        raw_state = self._raw_public_state(state)
        own = raw_state.get("self", {}) if isinstance(raw_state, dict) else {}
        if self._is_active_target(target):
            return own.get("active")

        bench_index = self._target_index(target, default=None)
        if bench_index is None:
            return None
        bench = own.get("bench", [])
        if not isinstance(bench, list) or bench_index >= len(bench):
            return None
        return bench[bench_index]

    def _raw_public_state(self, state):
        raw_obs = state.get("raw_obs", {}) if isinstance(state, dict) else {}
        if isinstance(raw_obs, dict):
            return raw_obs.get("raw_public_state", {})
        return {}

    def _target_index(self, target, default=999):
        if target is None:
            return default
        match = re.search(r"bench[:-](\d+)", str(target))
        if match:
            return int(match.group(1))
        return default

    def _is_active_target(self, target):
        return target == "active"

    def _hp_ratio(self, slot):
        hp = self._as_float(slot.get("hp"), default=0.0)
        if hp <= 0:
            return 0.0
        if "current_hp" in slot:
            current_hp = self._as_float(slot.get("current_hp"), default=0.0)
        else:
            current_hp = hp - self._as_float(slot.get("damage"), default=0.0)
        return max(0.0, min(1.0, current_hp / hp))

    def _stage_score(self, stage):
        if stage is None:
            return 0.0
        normalized = str(stage).lower()
        if normalized in ("stage2", "stage_2", "2"):
            return 2.0
        if normalized in ("stage1", "stage_1", "1"):
            return 1.0
        return 0.0

    def _as_float(self, value, default=0.0):
        try:
            return float(value)
        except (TypeError, ValueError):
            return default
