#!/usr/bin/env python3
"""PPO self-play training for PTCG."""

import argparse
import csv
import os
import random

import numpy as np
import rlcard
from rlcard.agents import RandomAgent
from rlcard.agents.ptcg_ppo_agent import PtcgPPOAgent, compute_gae
from rlcard.utils import get_device
from rlcard.utils.ptcg_deck_pool import (
    deck_label,
    deck_path,
    load_deck_pool,
    manual_deck,
    sample_deck_pair,
)


CSV_FIELDS = [
    "update",
    "episodes",
    "steps",
    "avg_payoff_p0",
    "eval_avg_payoff_p0",
    "policy_loss",
    "value_loss",
    "entropy",
    "approx_kl",
    "clip_fraction",
]


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)


def make_env(deck_a, deck_b, seed, max_ply):
    return rlcard.make("ptcg", config={
        "deck_a": deck_a,
        "deck_b": deck_b,
        "seed": seed,
        "max_ply": max_ply,
        "record_trace": False,
    })


def apply_raw_action_tie_breaker(agent, mode):
    if mode == "heuristic":
        from rlcard.agents.ptcg_raw_tiebreaker import PtcgRawActionTieBreaker
        agent.set_raw_action_tie_breaker(PtcgRawActionTieBreaker())
    elif mode == "first":
        agent.set_raw_action_tie_breaker(None)
    else:
        raise ValueError(f"Unknown raw action tie-breaker: {mode}")


def create_opponent(opponent, num_actions, seed):
    if opponent == "random":
        return RandomAgent(num_actions=num_actions)
    if opponent == "simplebot":
        from rlcard.agents.ptcg_simplebot_agent import PtcgSimpleBotAgent
        return PtcgSimpleBotAgent(seed=seed)
    raise ValueError(f"Unsupported non-self opponent: {opponent}")


def resolve_decks(args):
    if args.deck_pool:
        return load_deck_pool(args.deck_pool, split=args.train_split)
    if not args.deck_a or not args.deck_b:
        raise ValueError("--deck-a and --deck-b are required when --deck-pool is not set")
    return [
        manual_deck("deck_a", args.deck_a),
        manual_deck("deck_b", args.deck_b),
    ]


def fixed_pair_or_sample(decks, args, rng):
    if args.deck_pool:
        return sample_deck_pair(decks, rng=rng, mode=args.deck_sample_mode)
    return decks[0], decks[1]


def empty_rollout():
    return {
        "obs": [],
        "actions": [],
        "log_probs": [],
        "values": [],
        "returns": [],
        "advantages": [],
        "legal_masks": [],
    }


def add_player_trajectory(rollout, player_steps, payoff, gamma, gae_lambda):
    if not player_steps:
        return
    rewards = np.zeros(len(player_steps), dtype=np.float32)
    rewards[-1] = float(payoff)
    values = np.asarray([step["value"] for step in player_steps], dtype=np.float32)
    advantages, returns = compute_gae(rewards, values, gamma=gamma, gae_lambda=gae_lambda)
    for step, advantage, ret in zip(player_steps, advantages, returns):
        rollout["obs"].append(step["obs"])
        rollout["actions"].append(step["action"])
        rollout["log_probs"].append(step["log_prob"])
        rollout["values"].append(step["value"])
        rollout["advantages"].append(float(advantage))
        rollout["returns"].append(float(ret))
        rollout["legal_masks"].append(step["legal_mask"])


def collect_episode(agent, opponent, args, deck_a, deck_b, episode_seed):
    seed_everything(episode_seed)
    env = make_env(deck_path(deck_a), deck_path(deck_b), episode_seed, args.max_ply)
    player_steps = {0: [], 1: []}
    state, player_id = env.reset()

    while not env.is_over():
        trainable_turn = args.opponent == "self" or player_id == 0
        if trainable_turn:
            action, info = agent.sample_action(state, deterministic=False)
            player_steps[player_id].append({
                "obs": np.asarray(state["obs"], dtype=np.float32).ravel(),
                "action": int(info["action"]),
                "log_prob": float(info["log_prob"]),
                "value": float(info["value"]),
                "legal_mask": np.asarray(info["legal_mask"], dtype=bool),
            })
            use_raw = agent.use_raw
        else:
            action = opponent.step(state)
            use_raw = opponent.use_raw

        state, player_id = env.step(action, use_raw)

    payoffs = env.get_payoffs()
    rollout = empty_rollout()
    train_players = (0, 1) if args.opponent == "self" else (0,)
    for train_player in train_players:
        add_player_trajectory(
            rollout,
            player_steps[train_player],
            payoffs[train_player],
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
        )
    return rollout, payoffs


def merge_rollout(target, source):
    for key in target:
        target[key].extend(source[key])


def evaluate(agent, args, decks, update_index):
    if args.num_eval_games <= 0:
        return None
    rng = random.Random(args.seed + 9000000 + update_index)
    payoff_sum = 0.0
    for game_index in range(args.num_eval_games):
        deck_a, deck_b = fixed_pair_or_sample(decks, args, rng)
        game_seed = args.seed + 7000000 + update_index * 10000 + game_index
        seed_everything(game_seed)
        env = make_env(deck_path(deck_a), deck_path(deck_b), game_seed, args.max_ply)
        if args.opponent == "self":
            eval_opponent = agent
        else:
            eval_opponent = create_opponent(args.opponent, env.num_actions, game_seed + 1)
        env.set_agents([agent, eval_opponent])
        _, payoffs = env.run(is_training=False)
        payoff_sum += float(payoffs[0])
    return payoff_sum / max(args.num_eval_games, 1)


def write_training_config(args, decks, state_shape, num_actions):
    os.makedirs(args.log_dir, exist_ok=True)
    config_path = os.path.join(args.log_dir, "training_config.txt")
    with open(config_path, "w", encoding="utf-8") as file:
        for key, value in sorted(vars(args).items()):
            file.write(f"{key}={value}\n")
        file.write(f"state_shape={state_shape}\n")
        file.write(f"num_actions={num_actions}\n")
        file.write("decks:\n")
        for deck in decks:
            file.write(f"  {deck_label(deck)}={deck_path(deck)}\n")


def append_performance(log_dir, row):
    csv_path = os.path.join(log_dir, "performance.csv")
    exists = os.path.isfile(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(description="PTCG PPO Training")
    parser.add_argument("--deck-a", default=None, help="Path to fixed player 0 deck")
    parser.add_argument("--deck-b", default=None, help="Path to fixed player 1 deck")
    parser.add_argument("--deck-pool", default=None, help="JSON deck pool path")
    parser.add_argument("--train-split", default="train", help="Deck-pool split used for training")
    parser.add_argument("--deck-sample-mode", default="archetype-balanced",
                        choices=["archetype-balanced", "uniform"],
                        help="How to sample deck pairs from a deck pool")
    parser.add_argument("--opponent", default="self", choices=["self", "random", "simplebot"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-ply", type=int, default=10000)
    parser.add_argument("--rollout-episodes", type=int, default=128)
    parser.add_argument("--updates", type=int, default=500)
    parser.add_argument("--eval-every", type=int, default=25)
    parser.add_argument("--num-eval-games", type=int, default=50)
    parser.add_argument("--save-every", type=int, default=50)
    parser.add_argument("--log-dir", default="experiments/ptcg-ppo-run")
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-ratio", type=float, default=0.2)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--update-epochs", type=int, default=4)
    parser.add_argument("--minibatch-size", type=int, default=2048)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--raw-action-tie-breaker", default="heuristic",
                        choices=["first", "heuristic"])
    args = parser.parse_args()

    seed_everything(args.seed)
    device = get_device()
    decks = resolve_decks(args)
    rng = random.Random(args.seed)
    first_deck_a, first_deck_b = fixed_pair_or_sample(decks, args, rng)
    temp_env = make_env(deck_path(first_deck_a), deck_path(first_deck_b), args.seed, args.max_ply)
    state_shape = temp_env.state_shape
    num_actions = temp_env.num_actions

    agent = PtcgPPOAgent(
        num_actions=num_actions,
        state_shape=state_shape,
        hidden_layers_sizes=[256, 256, 128],
        learning_rate=args.learning_rate,
        device=device,
        raw_action_tie_breaker=args.raw_action_tie_breaker,
    )
    apply_raw_action_tie_breaker(agent, args.raw_action_tie_breaker)
    opponent = None if args.opponent == "self" else create_opponent(args.opponent, num_actions, args.seed + 1)

    os.makedirs(args.log_dir, exist_ok=True)
    write_training_config(args, decks, state_shape, num_actions)
    print(f"State shape: {state_shape}")
    print(f"Num actions: {num_actions}")
    print(f"Opponent: {args.opponent}")
    print(f"Rollout episodes/update: {args.rollout_episodes}")
    print(f"Updates: {args.updates}")
    print(f"Log dir: {args.log_dir}")
    print()

    total_episodes = 0
    total_steps = 0
    for update_index in range(1, args.updates + 1):
        rollout = empty_rollout()
        payoff_sum_p0 = 0.0
        for episode_index in range(args.rollout_episodes):
            total_episodes += 1
            episode_seed = args.seed + total_episodes
            deck_a, deck_b = fixed_pair_or_sample(decks, args, rng)
            episode_rollout, payoffs = collect_episode(agent, opponent, args, deck_a, deck_b, episode_seed)
            merge_rollout(rollout, episode_rollout)
            payoff_sum_p0 += float(payoffs[0])

        total_steps += len(rollout["obs"])
        metrics = agent.update(
            rollout,
            clip_ratio=args.clip_ratio,
            vf_coef=args.vf_coef,
            ent_coef=args.ent_coef,
            update_epochs=args.update_epochs,
            minibatch_size=args.minibatch_size,
            max_grad_norm=args.max_grad_norm,
        )

        eval_payoff = None
        if args.eval_every and update_index % args.eval_every == 0:
            eval_payoff = evaluate(agent, args, decks, update_index)

        row = {
            "update": update_index,
            "episodes": total_episodes,
            "steps": total_steps,
            "avg_payoff_p0": payoff_sum_p0 / max(args.rollout_episodes, 1),
            "eval_avg_payoff_p0": "" if eval_payoff is None else eval_payoff,
            **metrics,
        }
        append_performance(args.log_dir, row)
        print(
            f"Update {update_index}/{args.updates}: "
            f"episodes={total_episodes}, steps={total_steps}, "
            f"avg_payoff_p0={row['avg_payoff_p0']:.4f}, "
            f"policy_loss={metrics['policy_loss']:.4f}, "
            f"value_loss={metrics['value_loss']:.4f}, "
            f"entropy={metrics['entropy']:.4f}, "
            f"eval={eval_payoff if eval_payoff is not None else 'skip'}",
            flush=True,
        )

        if args.save_every and update_index % args.save_every == 0:
            save_dir = os.path.join(args.log_dir, f"checkpoint_{update_index}")
            agent.save_checkpoint(save_dir)
            print(f"Checkpoint saved to {save_dir}", flush=True)

    final_dir = os.path.join(args.log_dir, "final")
    agent.save_checkpoint(final_dir)
    print(f"Final model saved to {final_dir}")
    print("PPO training complete.")


if __name__ == "__main__":
    main()
