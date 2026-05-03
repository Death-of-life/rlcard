#!/usr/bin/env python3
"""Self-play training script for PTCG RL.

Usage:
    python examples/run_ptcg_selfplay.py \\
        --algorithm nfsp \\
        --deck-a /path/to/deck_a.txt \\
        --deck-b /path/to/deck_b.txt \\
        --episodes 50000 \\
        --eval-every 1000 \\
        --num-eval-games 200 \\
        --log-dir experiments/ptcg-nfsp-run1
"""

import argparse
import os
import sys

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


def create_agent(algorithm, num_actions, device):
    """Create an RL agent."""
    if algorithm == "dqn":
        return DQNAgent(
            num_actions=num_actions,
            state_shape=None,  # Auto-detect from env
            mlp_layers=[256, 256, 128],
            device=device,
        )
    elif algorithm == "nfsp":
        return NFSPAgent(
            num_actions=num_actions,
            state_shape=None,
            hidden_layers_sizes=[256, 256, 128],
            device=device,
        )
    elif algorithm == "random":
        return RandomAgent(num_actions=num_actions)
    elif algorithm == "simplebot":
        from rlcard.agents.ptcg_simplebot_agent import PtcgSimpleBotAgent
        return PtcgSimpleBotAgent()
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")


def main():
    parser = argparse.ArgumentParser(description="PTCG Self-Play Training")
    parser.add_argument("--algorithm", type=str, default="nfsp",
                        choices=["dqn", "nfsp", "random"],
                        help="RL algorithm")
    parser.add_argument("--deck-a", type=str, required=True,
                        help="Path to deck A")
    parser.add_argument("--deck-b", type=str, required=True,
                        help="Path to deck B")
    parser.add_argument("--opponent", type=str, default="self",
                        choices=["self", "random", "simplebot", "checkpoint"],
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
    args = parser.parse_args()

    if not HAVE_TORCH and args.algorithm in ("dqn", "nfsp"):
        raise ImportError(
            "DQN/NFSP require PyTorch. Install torch and retry, "
            "or use --algorithm random."
        )

    device = get_device()
    set_seed(args.seed)
    
    os.makedirs(args.log_dir, exist_ok=True)

    # Environment config
    env_config = {
        "deck_a": args.deck_a,
        "deck_b": args.deck_b,
        "seed": args.seed,
        "max_ply": 10000,
        "record_trace": False,
    }

    # Make environment
    env = rlcard.make("ptcg", config=env_config)
    eval_env = rlcard.make("ptcg", config=env_config)

    # Create agents
    if args.algorithm == "random":
        agents = [
            RandomAgent(env.num_actions),
            RandomAgent(env.num_actions),
        ]
    else:
        rl_agent = create_agent(args.algorithm, env.num_actions, device)
        if args.opponent == "self":
            opponent = rl_agent  # Self-play
        elif args.opponent == "random":
            opponent = RandomAgent(env.num_actions)
        elif args.opponent == "simplebot":
            from rlcard.agents.ptcg_simplebot_agent import PtcgSimpleBotAgent
            opponent = PtcgSimpleBotAgent()
        else:
            opponent = rl_agent
        agents = [rl_agent, opponent]

    env.set_agents(agents)

    print(f"Algorithm: {args.algorithm}")
    print(f"Opponent: {args.opponent}")
    print(f"Episodes: {args.episodes}")
    print(f"Log dir: {args.log_dir}")
    print()

    with Logger(args.log_dir) as logger:
        for episode in range(1, args.episodes + 1):
            trajectories, payoffs = env.run(is_training=True)

            # Reorganize and feed to agents
            if args.algorithm != "random":
                trajectories = reorganize(trajectories, payoffs)
                for ts in trajectories[0]:
                    rl_agent.feed(ts)

            # Evaluate periodically
            if episode % args.eval_every == 0:
                eval_env.set_agents(agents)
                avg_payoffs = tournament(eval_env, args.num_eval_games)
                print(f"Episode {episode}: avg payoffs = {avg_payoffs}")
                logger.log_performance(episode, avg_payoffs[0])

            # Save checkpoint
            if args.save_every and episode % args.save_every == 0:
                save_dir = os.path.join(args.log_dir, f"checkpoint_{episode}")
                os.makedirs(save_dir, exist_ok=True)
                if hasattr(rl_agent, "save_checkpoint"):
                    rl_agent.save_checkpoint(save_dir)

        # Final save
        save_dir = os.path.join(args.log_dir, "final")
        os.makedirs(save_dir, exist_ok=True)
        if hasattr(rl_agent, "save_checkpoint"):
            rl_agent.save_checkpoint(save_dir)

    print("Training complete.")


if __name__ == "__main__":
    main()
