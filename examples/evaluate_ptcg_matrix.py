#!/usr/bin/env python3
"""Run a reproducible PTCG evaluation matrix across fixed deck pairs."""

import argparse
import csv
import json
import math
import os
import random
from collections import Counter

import numpy as np
import rlcard
from evaluate_ptcg import create_ptcg_agent
from rlcard.utils import get_device
from rlcard.utils.ptcg_deck_pool import (
    deck_archetype,
    deck_label,
    deck_path,
    deck_split,
    load_deck_pool,
    manual_deck,
)


DEFAULT_DECKS = [
    manual_deck("beijingsha", "/Users/easygod/Downloads/Battle Subway 北京沙.txt"),
    manual_deck("longshenzhu", "/Users/easygod/Downloads/龙神柱 Battle Subway.txt"),
    manual_deck("mengleigu", "/Users/easygod/Downloads/猛雷鼓 (1).txt"),
]


CSV_FIELDS = [
    "agent",
    "opponent",
    "deck_p0",
    "deck_p1",
    "deck_p0_archetype",
    "deck_p1_archetype",
    "deck_p0_split",
    "deck_p1_split",
    "deck_p0_path",
    "deck_p1_path",
    "seed",
    "num_games",
    "wins",
    "losses",
    "ties",
    "avg_payoff",
    "approx_win_rate",
    "ci95_low",
    "ci95_high",
    "avg_ply",
    "fallback_count",
    "action_template_counts",
]


def parse_deck_arg(value):
    if "=" not in value:
        raise argparse.ArgumentTypeError("--deck must use NAME=/absolute/path format")
    name, path = value.split("=", 1)
    name = name.strip()
    path = path.strip()
    if not name or not path:
        raise argparse.ArgumentTypeError("--deck must use NAME=/absolute/path format")
    return manual_deck(name, path)


def make_env(deck_a, deck_b, seed, max_ply):
    return rlcard.make("ptcg", config={
        "deck_a": deck_a,
        "deck_b": deck_b,
        "seed": seed,
        "max_ply": max_ply,
        "record_trace": True,
    })


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)


def validate_decks(decks):
    for deck in decks:
        if not os.path.isfile(deck_path(deck)):
            raise FileNotFoundError(f"Deck '{deck_label(deck)}' not found: {deck_path(deck)}")


def summarize_ci(win_rate, num_games):
    if num_games <= 0:
        return 0.0, 0.0
    margin = 1.96 * math.sqrt(win_rate * (1.0 - win_rate) / num_games)
    return max(0.0, win_rate - margin), min(1.0, win_rate + margin)


def trace_for_env(env):
    ptcg_env = getattr(getattr(env, "game", None), "_ptcg_env", None)
    if ptcg_env is None:
        return []
    try:
        return ptcg_env.action_trace()
    except Exception:
        return []


def evaluate_pair(args, device, deck_p0, deck_p1, pair_index):
    deck_p0_name, deck_p0_path = deck_label(deck_p0), deck_path(deck_p0)
    deck_p1_name, deck_p1_path = deck_label(deck_p1), deck_path(deck_p1)
    pair_seed = args.seed + pair_index * 100000

    agent_env = make_env(deck_p0_path, deck_p1_path, pair_seed, args.max_ply)
    agent = create_ptcg_agent(
        args.agent,
        agent_env,
        device,
        checkpoint=args.checkpoint,
        seed=pair_seed,
        raw_action_tie_breaker=args.raw_action_tie_breaker,
    )
    opponent = create_ptcg_agent(
        args.opponent,
        agent_env,
        device,
        checkpoint=args.opponent_checkpoint,
        seed=pair_seed + 1,
        raw_action_tie_breaker=args.raw_action_tie_breaker,
    )

    wins = 0
    losses = 0
    ties = 0
    payoff_sum = 0.0
    ply_sum = 0
    fallback_count = 0
    template_counts = Counter()

    for game_index in range(args.num_games):
        game_seed = pair_seed + game_index
        seed_everything(game_seed)
        env = make_env(deck_p0_path, deck_p1_path, game_seed, args.max_ply)
        env.set_agents([agent, opponent])
        _, payoffs = env.run(is_training=False)
        payoff = float(payoffs[0])
        payoff_sum += payoff
        if payoff > 0:
            wins += 1
        elif payoff < 0:
            losses += 1
        else:
            ties += 1

        trace = trace_for_env(env)
        ply_sum += len(trace)
        for entry in trace:
            if entry.get("fallback"):
                fallback_count += 1
            template_id = entry.get("chosen_template_id")
            if template_id is not None:
                template_counts[str(template_id)] += 1

    win_rate = wins / max(args.num_games, 1)
    ci95_low, ci95_high = summarize_ci(win_rate, args.num_games)
    avg_payoff = payoff_sum / max(args.num_games, 1)
    avg_ply = ply_sum / max(args.num_games, 1)

    return {
        "agent": args.agent,
        "opponent": args.opponent,
        "deck_p0": deck_p0_name,
        "deck_p1": deck_p1_name,
        "deck_p0_archetype": deck_archetype(deck_p0),
        "deck_p1_archetype": deck_archetype(deck_p1),
        "deck_p0_split": deck_split(deck_p0),
        "deck_p1_split": deck_split(deck_p1),
        "deck_p0_path": deck_p0_path,
        "deck_p1_path": deck_p1_path,
        "seed": pair_seed,
        "num_games": args.num_games,
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "avg_payoff": avg_payoff,
        "approx_win_rate": win_rate,
        "ci95_low": ci95_low,
        "ci95_high": ci95_high,
        "avg_ply": avg_ply,
        "fallback_count": fallback_count,
        "action_template_counts": dict(sorted(template_counts.items())),
    }


def write_outputs(log_dir, rows, args, decks):
    os.makedirs(log_dir, exist_ok=True)

    csv_path = os.path.join(log_dir, "results.csv")
    with open(csv_path, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            csv_row = row.copy()
            csv_row["action_template_counts"] = json.dumps(
                row["action_template_counts"],
                sort_keys=True,
                ensure_ascii=False,
            )
            writer.writerow(csv_row)

    json_path = os.path.join(log_dir, "results.json")
    with open(json_path, "w") as jsonfile:
        json.dump({
            "agent": args.agent,
            "opponent": args.opponent,
            "checkpoint": args.checkpoint,
            "opponent_checkpoint": args.opponent_checkpoint,
            "raw_action_tie_breaker": args.raw_action_tie_breaker,
            "deck_pool": args.deck_pool,
            "split": args.split,
            "seed": args.seed,
            "num_games": args.num_games,
            "decks": [
                {
                    "name": deck_label(deck),
                    "archetype": deck_archetype(deck),
                    "split": deck_split(deck),
                    "path": deck_path(deck),
                }
                for deck in decks
            ],
            "rows": rows,
        }, jsonfile, indent=2, sort_keys=True, ensure_ascii=False)

    return csv_path, json_path


def main():
    parser = argparse.ArgumentParser(description="PTCG checkpoint/deck evaluation matrix")
    parser.add_argument("--agent", choices=["random", "simplebot", "dqn", "nfsp", "ppo"],
                        default="simplebot", help="Player 0 agent type")
    parser.add_argument("--checkpoint", default=None,
                        help="Player 0 DQN/NFSP/PPO checkpoint directory or .pt file")
    parser.add_argument("--opponent", choices=["random", "simplebot", "dqn", "nfsp", "ppo"],
                        default="random", help="Player 1 agent type")
    parser.add_argument("--opponent-checkpoint", default=None,
                        help="Player 1 DQN/NFSP/PPO checkpoint directory or .pt file")
    parser.add_argument("--deck", action="append", type=parse_deck_arg,
                        help="Deck entry in NAME=/absolute/path format; repeat to override defaults")
    parser.add_argument("--deck-pool", default=None,
                        help="JSON deck-pool file; mutually exclusive with --deck")
    parser.add_argument("--split", default="train",
                        help="Deck-pool split to evaluate, or 'all'")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed")
    parser.add_argument("--num-games", type=int, default=200, help="Games per ordered deck pair")
    parser.add_argument("--max-ply", type=int, default=10000, help="Maximum ply per game")
    parser.add_argument("--log-dir", default=None, help="Output directory")
    parser.add_argument("--raw-action-tie-breaker", choices=["first", "heuristic"],
                        default="heuristic",
                        help="Concrete raw action selector for DQN/NFSP/PPO template actions")
    args = parser.parse_args()

    if args.deck_pool and args.deck:
        parser.error("--deck-pool and --deck are mutually exclusive")
    if args.deck_pool:
        decks = load_deck_pool(args.deck_pool, split=args.split)
    else:
        decks = args.deck if args.deck else DEFAULT_DECKS
        validate_decks(decks)

    if args.log_dir is None:
        args.log_dir = f"experiments/ptcg-eval-{args.agent}-vs-{args.opponent}-seed{args.seed}"

    device = get_device()
    rows = []
    pair_index = 0
    for deck_p0 in decks:
        for deck_p1 in decks:
            print(
                f"Evaluating {args.agent}({deck_label(deck_p0)}) vs "
                f"{args.opponent}({deck_label(deck_p1)}) for {args.num_games} games...",
                flush=True,
            )
            row = evaluate_pair(args, device, deck_p0, deck_p1, pair_index)
            rows.append(row)
            print(
                f"  payoff={row['avg_payoff']:.4f}, "
                f"win_rate={row['approx_win_rate']:.2%}, "
                f"ci95=[{row['ci95_low']:.2%}, {row['ci95_high']:.2%}]",
                flush=True,
            )
            pair_index += 1

    csv_path, json_path = write_outputs(args.log_dir, rows, args, decks)
    print(f"\nWrote {len(rows)} rows to {csv_path}")
    print(f"Wrote JSON results to {json_path}")


if __name__ == "__main__":
    main()
