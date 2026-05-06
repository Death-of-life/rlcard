#!/usr/bin/env python3
"""Evaluation script for PTCG RL agents.

Usage:
    python examples/evaluate_ptcg.py \\
        --deck-a /path/to/deck_a.txt \\
        --deck-b /path/to/deck_b.txt \\
        --num-games 200 \\
        --opponent random|simplebot|dqn|nfsp|ppo \\
        [--checkpoint /path/to/checkpoint]
"""

import argparse
import os
import random
import sys

import numpy as np
import rlcard
from rlcard.agents import RandomAgent
from rlcard.utils import get_device


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)


def make_env(deck_a, deck_b, seed):
    return rlcard.make("ptcg", config={
        "deck_a": deck_a,
        "deck_b": deck_b,
        "seed": seed,
        "max_ply": 10000,
    })


def resolve_checkpoint_file(checkpoint, agent_type=None):
    """Resolve a checkpoint directory or file to the actual .pt file."""
    if not checkpoint:
        return None

    if os.path.isfile(checkpoint):
        return checkpoint

    if not os.path.isdir(checkpoint):
        raise FileNotFoundError(f"Checkpoint path does not exist: {checkpoint}")

    candidates = []
    if agent_type in (None, "dqn"):
        candidates.append(("dqn", os.path.join(checkpoint, "checkpoint_dqn.pt")))
    if agent_type in (None, "nfsp"):
        candidates.append(("nfsp", os.path.join(checkpoint, "checkpoint_nfsp.pt")))
    if agent_type in (None, "ppo"):
        candidates.append(("ppo", os.path.join(checkpoint, "checkpoint_ppo.pt")))

    existing = [(kind, path) for kind, path in candidates if os.path.isfile(path)]
    if len(existing) == 1:
        return existing[0][1]
    if len(existing) > 1:
        found = ", ".join(path for _, path in existing)
        raise ValueError(f"Multiple checkpoint files found; specify --agent. Found: {found}")
    raise FileNotFoundError(f"No checkpoint_dqn.pt, checkpoint_nfsp.pt, or checkpoint_ppo.pt in {checkpoint}")


def infer_checkpoint_agent_type(checkpoint_file):
    """Infer the agent type from a checkpoint filename or metadata."""
    name = os.path.basename(checkpoint_file).lower()
    if "dqn" in name:
        return "dqn"
    if "nfsp" in name:
        return "nfsp"
    if "ppo" in name:
        return "ppo"

    try:
        import torch
    except ImportError as exc:
        raise ImportError("PyTorch is required to inspect checkpoint metadata.") from exc

    checkpoint = torch.load(checkpoint_file, map_location="cpu", weights_only=False)
    agent_type = checkpoint.get("agent_type", "").lower()
    if "dqn" in agent_type:
        return "dqn"
    if "nfsp" in agent_type:
        return "nfsp"
    if "ppo" in agent_type:
        return "ppo"
    raise ValueError(f"Cannot infer checkpoint type from {checkpoint_file}")


def apply_raw_action_tie_breaker(agent, mode):
    if not hasattr(agent, "set_raw_action_tie_breaker"):
        return agent
    if mode == "heuristic":
        from rlcard.agents.ptcg_raw_tiebreaker import PtcgRawActionTieBreaker
        agent.set_raw_action_tie_breaker(PtcgRawActionTieBreaker())
    elif mode == "first":
        agent.set_raw_action_tie_breaker(None)
    else:
        raise ValueError(f"Unknown raw action tie-breaker: {mode}")
    return agent


def create_ptcg_agent(agent_type, env, device, checkpoint=None, seed=None, raw_action_tie_breaker="heuristic"):
    """Create a PTCG-compatible evaluation agent."""
    checkpoint_agent_type = agent_type if agent_type in ("dqn", "nfsp", "ppo") else None
    checkpoint_file = resolve_checkpoint_file(checkpoint, checkpoint_agent_type)
    if checkpoint_file:
        agent_type = infer_checkpoint_agent_type(checkpoint_file)

    if agent_type == "random":
        return RandomAgent(env.num_actions)

    if agent_type == "simplebot":
        from rlcard.agents.ptcg_simplebot_agent import PtcgSimpleBotAgent
        return PtcgSimpleBotAgent(seed=seed)

    if agent_type == "dqn":
        try:
            from rlcard.agents import DQNAgent
        except ImportError as exc:
            raise ImportError("DQN requires PyTorch. Install torch and retry.") from exc
        agent = DQNAgent(
            num_actions=env.num_actions,
            state_shape=env.state_shape,
            mlp_layers=[256, 256, 128],
            device=device,
        )
        if not checkpoint_file:
            raise ValueError("--checkpoint is required for DQN evaluation")
        agent.load_checkpoint(checkpoint_file)
        apply_raw_action_tie_breaker(agent, raw_action_tie_breaker)
        print(f"Loaded DQN checkpoint from {checkpoint_file}")
        return agent

    if agent_type == "nfsp":
        try:
            from rlcard.agents import NFSPAgent
        except ImportError as exc:
            raise ImportError("NFSP requires PyTorch. Install torch and retry.") from exc
        agent = NFSPAgent(
            num_actions=env.num_actions,
            state_shape=env.state_shape,
            hidden_layers_sizes=[256, 256, 128],
            device=device,
            q_mlp_layers=[256, 256, 128],
            evaluate_with="average_policy",
        )
        if not checkpoint_file:
            raise ValueError("--checkpoint is required for NFSP evaluation")
        agent.load_checkpoint(checkpoint_file)
        apply_raw_action_tie_breaker(agent, raw_action_tie_breaker)
        print(f"Loaded NFSP checkpoint from {checkpoint_file}")
        return agent

    if agent_type == "ppo":
        try:
            from rlcard.agents import PtcgPPOAgent
        except ImportError as exc:
            raise ImportError("PPO requires PyTorch. Install torch and retry.") from exc
        agent = PtcgPPOAgent(
            num_actions=env.num_actions,
            state_shape=env.state_shape,
            hidden_layers_sizes=[256, 256, 128],
            device=device,
            raw_action_tie_breaker=raw_action_tie_breaker,
        )
        if not checkpoint_file:
            raise ValueError("--checkpoint is required for PPO evaluation")
        agent.load_checkpoint(checkpoint_file)
        apply_raw_action_tie_breaker(agent, raw_action_tie_breaker)
        print(f"Loaded PPO checkpoint from {checkpoint_file}")
        return agent

    raise ValueError(f"Unknown agent type: {agent_type}")


def main():
    parser = argparse.ArgumentParser(description="PTCG Evaluation")
    parser.add_argument("--deck-a", type=str, required=True,
                        help="Path to deck A (agent plays player 0)")
    parser.add_argument("--deck-b", type=str, required=True,
                        help="Path to deck B (agent plays player 1)")
    parser.add_argument("--num-games", type=int, default=200,
                        help="Number of evaluation games")
    parser.add_argument("--opponent", type=str, default="random",
                        choices=["random", "simplebot", "dqn", "nfsp", "ppo"],
                        help="Opponent type")
    parser.add_argument("--opponent-checkpoint", type=str, default=None,
                        help="Path to DQN/NFSP/PPO opponent checkpoint directory or .pt file")
    parser.add_argument("--agent", type=str, default="simplebot",
                        choices=["random", "simplebot", "dqn", "nfsp", "ppo"],
                        help="Player 0 agent type; checkpoint type is inferred when --checkpoint is set")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to DQN/NFSP/PPO checkpoint directory or .pt file")
    parser.add_argument("--raw-action-tie-breaker", type=str, default="heuristic",
                        choices=["first", "heuristic"],
                        help="Concrete raw action selector for DQN/NFSP/PPO template actions")
    args = parser.parse_args()

    device = get_device()
    seed_everything(args.seed)
    env = make_env(args.deck_a, args.deck_b, args.seed)

    # Create agents
    try:
        agent = create_ptcg_agent(
            args.agent,
            env,
            device,
            checkpoint=args.checkpoint,
            seed=args.seed,
            raw_action_tie_breaker=args.raw_action_tie_breaker,
        )
        opponent = create_ptcg_agent(
            args.opponent,
            env,
            device,
            checkpoint=args.opponent_checkpoint,
            seed=args.seed + 1,
            raw_action_tie_breaker=args.raw_action_tie_breaker,
        )
    except (ImportError, ValueError, FileNotFoundError) as exc:
        print(str(exc))
        sys.exit(1)

    print(f"Running {args.num_games} evaluation games...")
    payoff_sum = [0.0, 0.0]
    wins = 0
    losses = 0
    ties = 0
    for game_index in range(args.num_games):
        game_seed = args.seed + game_index
        seed_everything(game_seed)
        game_env = make_env(args.deck_a, args.deck_b, game_seed)
        game_env.set_agents([agent, opponent])
        _, payoffs = game_env.run(is_training=False)
        payoff_sum[0] += float(payoffs[0])
        payoff_sum[1] += float(payoffs[1])
        if payoffs[0] > 0:
            wins += 1
        elif payoffs[0] < 0:
            losses += 1
        else:
            ties += 1

    num_games = max(args.num_games, 1)
    avg_payoffs = [payoff_sum[0] / num_games, payoff_sum[1] / num_games]

    print(f"\nResults (player 0 payoff):")
    print(f"  Average payoff: {avg_payoffs[0]:.4f}")
    print(f"  Opponent payoff: {avg_payoffs[1]:.4f}")

    win_rate = wins / num_games
    print(f"  Wins/Losses/Ties: {wins}/{losses}/{ties}")
    print(f"  Approx win rate: {win_rate:.2%}")


if __name__ == "__main__":
    main()
