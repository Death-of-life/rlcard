#!/usr/bin/env python3
"""Evaluation script for PTCG RL agents.

Usage:
    python examples/evaluate_ptcg.py \\
        --deck-a /path/to/deck_a.txt \\
        --deck-b /path/to/deck_b.txt \\
        --num-games 200 \\
        --opponent random|simplebot \\
        [--checkpoint /path/to/checkpoint]
"""

import argparse
import sys

import rlcard
from rlcard.agents import RandomAgent
from rlcard.utils import get_device, set_seed, tournament


def main():
    parser = argparse.ArgumentParser(description="PTCG Evaluation")
    parser.add_argument("--deck-a", type=str, required=True,
                        help="Path to deck A (agent plays player 0)")
    parser.add_argument("--deck-b", type=str, required=True,
                        help="Path to deck B (agent plays player 1)")
    parser.add_argument("--num-games", type=int, default=200,
                        help="Number of evaluation games")
    parser.add_argument("--opponent", type=str, default="random",
                        choices=["random", "simplebot"],
                        help="Opponent type")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to DQN/NFSP checkpoint directory")
    args = parser.parse_args()

    device = get_device()
    set_seed(args.seed)

    env_config = {
        "deck_a": args.deck_a,
        "deck_b": args.deck_b,
        "seed": args.seed,
        "max_ply": 10000,
    }

    env = rlcard.make("ptcg", config=env_config)

    # Create agents
    if args.checkpoint:
        try:
            from rlcard.agents import DQNAgent
            agent = DQNAgent(
                num_actions=env.num_actions,
                state_shape=None,
                mlp_layers=[256, 256, 128],
                device=device,
            )
            agent.load_checkpoint(args.checkpoint)
            print(f"Loaded checkpoint from {args.checkpoint}")
        except ImportError:
            print("DQN/NFSP requires PyTorch. Install torch and retry.")
            sys.exit(1)
    else:
        # Default: use SimpleBot for demonstration
        from rlcard.agents.ptcg_simplebot_agent import PtcgSimpleBotAgent
        agent = PtcgSimpleBotAgent()

    if args.opponent == "random":
        opponent = RandomAgent(env.num_actions)
    else:
        from rlcard.agents.ptcg_simplebot_agent import PtcgSimpleBotAgent
        opponent = PtcgSimpleBotAgent()

    env.set_agents([agent, opponent])

    print(f"Running {args.num_games} evaluation games...")
    avg_payoffs = tournament(env, args.num_games)

    print(f"\nResults (player 0 payoff):")
    print(f"  Average payoff: {avg_payoffs[0]:.4f}")
    print(f"  Opponent payoff: {avg_payoffs[1]:.4f}")

    win_rate = (avg_payoffs[0] + 1.0) / 2.0  # Approximate from payoff
    print(f"  Approx win rate: {win_rate:.2%}")


if __name__ == "__main__":
    main()
