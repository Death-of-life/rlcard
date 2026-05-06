#!/usr/bin/env python3
"""Self-play training script for PTCG RL.

Usage:
    python examples/run_ptcg_selfplay.py \
        --algorithm dqn \
        --deck-a /path/to/deck_a.txt \
        --deck-b /path/to/deck_b.txt \
        --episodes 1000 \
        --eval-every 100 \
        --log-dir experiments/ptcg-dqn-run1
"""

import argparse
import os
import sys

import numpy as np

import rlcard
from rlcard.agents import RandomAgent
from rlcard.utils import (
    get_device,
    reorganize,
    set_seed,
    tournament,
    Logger,
)

# Import NN agents (require torch)
try:
    from rlcard.agents import DQNAgent, NFSPAgent
    HAVE_TORCH = True
except ImportError:
    HAVE_TORCH = False


def create_agent(algorithm, num_actions, state_shape, device):
    """Create an RL agent."""
    if algorithm == "dqn":
        return DQNAgent(
            num_actions=num_actions,
            state_shape=state_shape,
            mlp_layers=[256, 256, 128],
            device=device,
            replay_memory_size=20000,
            replay_memory_init_size=1000,
            update_target_estimator_every=500,
            epsilon_decay_steps=50000,
            batch_size=32,
            learning_rate=0.0001,
        )
    elif algorithm == "nfsp":
        return NFSPAgent(
            num_actions=num_actions,
            state_shape=state_shape,
            hidden_layers_sizes=[256, 256, 128],
            device=device,
            anticipatory_param=0.1,
            batch_size=128,
            rl_learning_rate=0.0001,
            sl_learning_rate=0.001,
            q_replay_memory_size=20000,
            q_replay_memory_init_size=2000,
            q_epsilon_decay_steps=50000,
            q_batch_size=64,
            q_mlp_layers=[256, 256, 128],
            evaluate_with='average_policy',
        )
    elif algorithm == "random":
        return RandomAgent(num_actions=num_actions)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")


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


def make_env(deck_a, deck_b, seed):
    """Create a PTCG environment with a given seed."""
    return rlcard.make("ptcg", config={
        "deck_a": deck_a,
        "deck_b": deck_b,
        "seed": seed,
        "max_ply": 10000,
        "record_trace": False,
    })


def main():
    parser = argparse.ArgumentParser(description="PTCG Self-Play Training")
    parser.add_argument("--algorithm", type=str, default="dqn",
                        choices=["dqn", "nfsp", "random"],
                        help="RL algorithm")
    parser.add_argument("--deck-a", type=str, required=True,
                        help="Path to deck A")
    parser.add_argument("--deck-b", type=str, required=True,
                        help="Path to deck B")
    parser.add_argument("--opponent", type=str, default="self",
                        choices=["self", "random", "simplebot"],
                        help="Opponent type")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--episodes", type=int, default=50000,
                        help="Number of training episodes")
    parser.add_argument("--eval-every", type=int, default=1000,
                        help="Evaluate every N episodes")
    parser.add_argument("--num-eval-games", type=int, default=200,
                        help="Number of evaluation games")
    parser.add_argument("--save-every", type=int, default=5000,
                        help="Save checkpoint every N episodes")
    parser.add_argument("--log-dir", type=str, default="experiments/ptcg-run",
                        help="Log directory")
    parser.add_argument("--raw-action-tie-breaker", type=str, default="heuristic",
                        choices=["first", "heuristic"],
                        help="Concrete raw action selector for DQN/NFSP template actions")
    args = parser.parse_args()

    if not HAVE_TORCH and args.algorithm in ("dqn", "nfsp"):
        raise ImportError(
            "DQN/NFSP require PyTorch. Install it with: pip install torch matplotlib"
        )

    device = get_device()
    set_seed(args.seed)

    os.makedirs(args.log_dir, exist_ok=True)

    # Create a temporary env just to get state_shape
    temp_env = make_env(args.deck_a, args.deck_b, args.seed)
    state_shape = temp_env.state_shape
    num_actions = temp_env.num_actions
    print(f"State shape: {state_shape}")
    print(f"Num actions: {num_actions}")

    # Create agents
    rl_agent = create_agent(args.algorithm, num_actions, state_shape, device)
    apply_raw_action_tie_breaker(rl_agent, args.raw_action_tie_breaker)
    if args.opponent == "self":
        opponent = rl_agent
    elif args.opponent == "random":
        opponent = RandomAgent(num_actions=num_actions)
    elif args.opponent == "simplebot":
        from rlcard.agents.ptcg_simplebot_agent import PtcgSimpleBotAgent
        opponent = PtcgSimpleBotAgent(seed=args.seed + 1)
    else:
        opponent = rl_agent

    print(f"Algorithm: {args.algorithm}")
    print(f"Opponent: {args.opponent}")
    print(f"Episodes: {args.episodes}")
    print(f"Log dir: {args.log_dir}")
    print()

    with Logger(args.log_dir) as logger:
        for episode in range(1, args.episodes + 1):
            # Use varying seed for each episode
            episode_seed = args.seed + episode
            env = make_env(args.deck_a, args.deck_b, episode_seed)
            env.set_agents([rl_agent, opponent])

            if hasattr(rl_agent, "sample_episode_policy"):
                rl_agent.sample_episode_policy()

            trajectories, payoffs = env.run(is_training=True)

            # Reorganize and feed to agent
            if args.algorithm != "random":
                transitions = reorganize(trajectories, payoffs)
                train_players = range(len(transitions)) if args.opponent == "self" else [0]
                for player_id in train_players:
                    for ts in transitions[player_id]:
                        rl_agent.feed(ts)

            # Evaluate periodically
            if episode % args.eval_every == 0:
                eval_env = make_env(args.deck_a, args.deck_b, args.seed + 100000 + episode)
                eval_env.set_agents([rl_agent, opponent])
                avg_payoffs = tournament(eval_env, args.num_eval_games)
                print(f"\nEpisode {episode}/{args.episodes}: "
                      f"avg_payoff_p0 = {avg_payoffs[0]:.4f}, "
                      f"avg_payoff_p1 = {avg_payoffs[1]:.4f}")
                logger.log_performance(episode, avg_payoffs[0])

            # Progress
            if episode % 100 == 0:
                print(f"Episode {episode}/{args.episodes}: payoff = {payoffs}")

            # Save checkpoint
            if args.save_every and episode % args.save_every == 0:
                save_dir = os.path.join(args.log_dir, f"checkpoint_{episode}")
                os.makedirs(save_dir, exist_ok=True)
                if hasattr(rl_agent, "save_checkpoint"):
                    rl_agent.save_checkpoint(save_dir)
                    print(f"Checkpoint saved to {save_dir}")

        # Final save
        save_dir = os.path.join(args.log_dir, "final")
        os.makedirs(save_dir, exist_ok=True)
        if hasattr(rl_agent, "save_checkpoint"):
            rl_agent.save_checkpoint(save_dir)
            print(f"Final model saved to {save_dir}")

    print("Training complete.")


if __name__ == "__main__":
    main()
