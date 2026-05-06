"""PPO agent for PTCG template-action policies."""

import os

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical


def legal_mask_from_state(state, num_actions):
    """Return a dense bool mask for legal template ids."""
    mask = np.zeros(num_actions, dtype=bool)
    for action in state.get("legal_actions", {}).keys():
        try:
            action_id = int(action)
        except (TypeError, ValueError):
            continue
        if 0 <= action_id < num_actions:
            mask[action_id] = True
    if not mask.any():
        mask[0] = True
    return mask


def compute_gae(rewards, values, gamma=0.99, gae_lambda=0.95):
    """Compute GAE returns for one player's sparse terminal-reward trajectory."""
    rewards = np.asarray(rewards, dtype=np.float32)
    values = np.asarray(values, dtype=np.float32)
    advantages = np.zeros_like(rewards, dtype=np.float32)
    last_gae = 0.0
    for index in reversed(range(len(rewards))):
        if index == len(rewards) - 1:
            next_nonterminal = 0.0
            next_value = 0.0
        else:
            next_nonterminal = 1.0
            next_value = values[index + 1]
        delta = rewards[index] + gamma * next_value * next_nonterminal - values[index]
        last_gae = delta + gamma * gae_lambda * next_nonterminal * last_gae
        advantages[index] = last_gae
    returns = advantages + values
    return advantages, returns


class PtcgPPONetwork(nn.Module):
    """Shared trunk with policy and value heads."""

    def __init__(self, num_actions, state_shape, hidden_layers_sizes):
        super().__init__()
        self.num_actions = num_actions
        self.state_shape = state_shape
        self.hidden_layers_sizes = hidden_layers_sizes

        input_dim = int(np.prod(state_shape))
        layers = []
        last_dim = input_dim
        for hidden_dim in hidden_layers_sizes:
            layers.append(nn.Linear(last_dim, hidden_dim))
            layers.append(nn.ReLU())
            last_dim = hidden_dim
        self.trunk = nn.Sequential(*layers)
        self.policy_head = nn.Linear(last_dim, num_actions)
        self.value_head = nn.Linear(last_dim, 1)

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, obs):
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        obs = obs.view(obs.shape[0], -1)
        features = self.trunk(obs)
        logits = self.policy_head(features)
        values = self.value_head(features).squeeze(-1)
        return logits, values


class PtcgPPOAgent:
    """Masked PPO actor-critic over PTCG template ids.

    The policy learns a 15-way template action. A raw action tie-breaker can
    convert the selected template into a concrete Rust action id.
    """

    def __init__(
        self,
        num_actions,
        state_shape,
        hidden_layers_sizes=None,
        learning_rate=3e-4,
        device=None,
        raw_action_tie_breaker="heuristic",
    ):
        self.num_actions = num_actions
        self.state_shape = state_shape
        self.hidden_layers_sizes = hidden_layers_sizes or [256, 256, 128]
        self.learning_rate = learning_rate
        self.device = self._resolve_device(device)
        self.total_t = 0
        self.train_t = 0
        self.training_state = {}

        self.network = PtcgPPONetwork(
            num_actions=self.num_actions,
            state_shape=self.state_shape,
            hidden_layers_sizes=self.hidden_layers_sizes,
        ).to(self.device)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=self.learning_rate)
        self.raw_action_tie_breaker = self._make_tie_breaker(raw_action_tie_breaker)
        self.use_raw = self.raw_action_tie_breaker is not None
        self.network.eval()

    def set_raw_action_tie_breaker(self, tie_breaker):
        self.raw_action_tie_breaker = tie_breaker
        self.use_raw = tie_breaker is not None

    def step(self, state):
        action, _ = self.sample_action(state, deterministic=False)
        return action

    def eval_step(self, state):
        action, info = self.sample_action(state, deterministic=True)
        return action, info

    def sample_action(self, state, deterministic=False):
        template_id, info = self.select_template_action(state, deterministic=deterministic)
        return self._select_action_output(template_id, state), info

    def select_template_action(self, state, deterministic=False):
        obs = self._obs_tensor(state)
        legal_mask = legal_mask_from_state(state, self.num_actions)
        legal_mask_tensor = torch.as_tensor(legal_mask, dtype=torch.bool, device=self.device).unsqueeze(0)

        self.network.eval()
        with torch.no_grad():
            logits, values = self.network(obs)
            masked_logits = self._mask_logits(logits, legal_mask_tensor)
            distribution = Categorical(logits=masked_logits)
            if deterministic:
                action_tensor = torch.argmax(masked_logits, dim=-1)
            else:
                action_tensor = distribution.sample()
            log_prob = distribution.log_prob(action_tensor)
            entropy = distribution.entropy()
            probs = distribution.probs

        template_id = int(action_tensor.item())
        info = {
            "action": template_id,
            "log_prob": float(log_prob.item()),
            "value": float(values.squeeze(0).item()),
            "entropy": float(entropy.item()),
            "legal_mask": legal_mask.astype(np.float32),
            "template_probs": probs.squeeze(0).detach().cpu().numpy(),
            "probs": self._raw_action_probs(state, probs.squeeze(0).detach().cpu().numpy()),
        }
        self.total_t += 1
        return template_id, info

    def evaluate_actions(self, obs_batch, action_batch, legal_mask_batch):
        logits, values = self.network(obs_batch)
        masked_logits = self._mask_logits(logits, legal_mask_batch)
        distribution = Categorical(logits=masked_logits)
        log_probs = distribution.log_prob(action_batch)
        entropy = distribution.entropy()
        return log_probs, entropy, values

    def update(
        self,
        rollout,
        clip_ratio=0.2,
        vf_coef=0.5,
        ent_coef=0.01,
        update_epochs=4,
        minibatch_size=2048,
        max_grad_norm=0.5,
        target_kl=None,
    ):
        obs = np.asarray(rollout["obs"], dtype=np.float32)
        if len(obs) == 0:
            return {
                "policy_loss": 0.0,
                "value_loss": 0.0,
                "entropy": 0.0,
                "approx_kl": 0.0,
                "clip_fraction": 0.0,
                "epochs_trained": 0.0,
            }

        actions = np.asarray(rollout["actions"], dtype=np.int64)
        old_log_probs = np.asarray(rollout["log_probs"], dtype=np.float32)
        returns = np.asarray(rollout["returns"], dtype=np.float32)
        advantages = np.asarray(rollout["advantages"], dtype=np.float32)
        legal_masks = np.asarray(rollout["legal_masks"], dtype=bool)

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        action_tensor = torch.as_tensor(actions, dtype=torch.long, device=self.device)
        old_log_prob_tensor = torch.as_tensor(old_log_probs, dtype=torch.float32, device=self.device)
        return_tensor = torch.as_tensor(returns, dtype=torch.float32, device=self.device)
        advantage_tensor = torch.as_tensor(advantages, dtype=torch.float32, device=self.device)
        legal_mask_tensor = torch.as_tensor(legal_masks, dtype=torch.bool, device=self.device)

        batch_size = len(obs)
        minibatch_size = max(1, min(minibatch_size, batch_size))
        metrics = {
            "policy_loss": [],
            "value_loss": [],
            "entropy": [],
            "approx_kl": [],
            "clip_fraction": [],
            "epochs_trained": [],
        }

        self.network.train()
        stop_early = False
        for epoch_index in range(update_epochs):
            indices = np.random.permutation(batch_size)
            for start in range(0, batch_size, minibatch_size):
                mb_idx = torch.as_tensor(indices[start:start + minibatch_size], dtype=torch.long, device=self.device)
                new_log_probs, entropy, values = self.evaluate_actions(
                    obs_tensor[mb_idx],
                    action_tensor[mb_idx],
                    legal_mask_tensor[mb_idx],
                )
                ratio = torch.exp(new_log_probs - old_log_prob_tensor[mb_idx])
                unclipped = ratio * advantage_tensor[mb_idx]
                clipped = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * advantage_tensor[mb_idx]
                policy_loss = -torch.min(unclipped, clipped).mean()
                value_loss = 0.5 * (return_tensor[mb_idx] - values).pow(2).mean()
                entropy_mean = entropy.mean()
                loss = policy_loss + vf_coef * value_loss - ent_coef * entropy_mean

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), max_grad_norm)
                self.optimizer.step()

                with torch.no_grad():
                    approx_kl = (old_log_prob_tensor[mb_idx] - new_log_probs).mean()
                    clip_fraction = ((ratio - 1.0).abs() > clip_ratio).float().mean()
                metrics["policy_loss"].append(float(policy_loss.item()))
                metrics["value_loss"].append(float(value_loss.item()))
                metrics["entropy"].append(float(entropy_mean.item()))
                metrics["approx_kl"].append(float(approx_kl.item()))
                metrics["clip_fraction"].append(float(clip_fraction.item()))
                if target_kl is not None and float(approx_kl.item()) > float(target_kl):
                    stop_early = True
                    break
            metrics["epochs_trained"].append(float(epoch_index + 1))
            if stop_early:
                break

        self.train_t += 1
        self.network.eval()
        return {key: float(np.mean(values)) if values else 0.0 for key, values in metrics.items()}

    def save_checkpoint(self, path, filename="checkpoint_ppo.pt"):
        os.makedirs(path, exist_ok=True)
        torch.save(self.checkpoint_attributes(), os.path.join(path, filename))

    def load_checkpoint(self, path):
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        restored = self.__class__.from_checkpoint(checkpoint, device=self.device)
        self.__dict__.update(restored.__dict__)

    def checkpoint_attributes(self):
        return {
            "agent_type": "PtcgPPOAgent",
            "num_actions": self.num_actions,
            "state_shape": self.state_shape,
            "hidden_layers_sizes": self.hidden_layers_sizes,
            "learning_rate": self.learning_rate,
            "device": str(self.device),
            "network": self.network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "total_t": self.total_t,
            "train_t": self.train_t,
            "training_state": self.training_state,
        }

    @classmethod
    def from_checkpoint(cls, checkpoint, device=None):
        agent = cls(
            num_actions=checkpoint["num_actions"],
            state_shape=checkpoint["state_shape"],
            hidden_layers_sizes=checkpoint["hidden_layers_sizes"],
            learning_rate=checkpoint["learning_rate"],
            device=device or checkpoint.get("device"),
            raw_action_tie_breaker="heuristic",
        )
        agent.network.load_state_dict(checkpoint["network"])
        agent.optimizer.load_state_dict(checkpoint["optimizer"])
        agent._move_optimizer_state_to_device()
        agent.total_t = checkpoint.get("total_t", 0)
        agent.train_t = checkpoint.get("train_t", 0)
        agent.training_state = checkpoint.get("training_state", {})
        agent.network.eval()
        return agent

    def set_device(self, device):
        self.device = self._resolve_device(device)
        self.network.to(self.device)
        self._move_optimizer_state_to_device()

    def _select_action_output(self, template_id, state):
        if self.raw_action_tie_breaker is None:
            return int(template_id)
        return self.raw_action_tie_breaker.choose(int(template_id), state)

    def _obs_tensor(self, state):
        obs = np.asarray(state["obs"], dtype=np.float32).ravel()
        return torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)

    def _mask_logits(self, logits, legal_mask):
        return logits.masked_fill(~legal_mask, -1e9)

    def _raw_action_probs(self, state, template_probs):
        raw_probs = {}
        for raw_action in state.get("raw_legal_actions", []) or []:
            if not isinstance(raw_action, dict):
                continue
            action_id = raw_action.get("id")
            template_id = raw_action.get("template_id")
            if action_id is None or template_id is None:
                continue
            try:
                raw_probs[action_id] = float(template_probs[int(template_id)])
            except (IndexError, TypeError, ValueError):
                continue
        return raw_probs

    def _make_tie_breaker(self, mode):
        if mode in (None, "first"):
            return None
        if mode == "heuristic":
            from rlcard.agents.ptcg_raw_tiebreaker import PtcgRawActionTieBreaker
            return PtcgRawActionTieBreaker()
        return mode

    def _resolve_device(self, device):
        if device is None:
            if torch.backends.mps.is_available():
                return torch.device("mps:0")
            if torch.cuda.is_available():
                return torch.device("cuda:0")
            return torch.device("cpu")
        device = torch.device(device)
        if device.type == "mps" and not torch.backends.mps.is_available():
            return torch.device("cpu")
        if device.type == "cuda" and not torch.cuda.is_available():
            return torch.device("cpu")
        return device

    def _move_optimizer_state_to_device(self):
        for state in self.optimizer.state.values():
            for key, value in state.items():
                if torch.is_tensor(value):
                    state[key] = value.to(self.device)
