#!/usr/bin/env python3
"""Unified PTCG experiment CLI.

Use this as the primary entrypoint for training, evaluation, matrix runs,
background launch, and status checks.
"""

import argparse
import csv
import json
import math
import os
import random
import shutil
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import rlcard
from rlcard.agents import RandomAgent
from rlcard.agents.ptcg_ppo_agent import PtcgPPOAgent, compute_gae
from rlcard.utils import Logger, get_device, reorganize, set_seed, tournament
from rlcard.utils.ptcg_deck_pool import (
    deck_archetype,
    deck_label,
    deck_path,
    deck_split,
    load_deck_pool,
    manual_deck,
    sample_deck_pair,
)


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_DECKS = [
    manual_deck("beijingsha", "/Users/easygod/Downloads/Battle Subway 北京沙.txt"),
    manual_deck("longshenzhu", "/Users/easygod/Downloads/龙神柱 Battle Subway.txt"),
    manual_deck("mengleigu", "/Users/easygod/Downloads/猛雷鼓 (1).txt"),
]

PERFORMANCE_FIELDS = [
    "update",
    "episodes",
    "steps",
    "avg_payoff_p0",
    "policy_loss",
    "value_loss",
    "entropy",
    "approx_kl",
    "clip_fraction",
    "epochs_trained",
]

EVAL_FIELDS = [
    "update",
    "opponent",
    "deck_split",
    "num_games",
    "wins",
    "losses",
    "ties",
    "avg_payoff",
    "approx_win_rate",
    "avg_ply",
]

MATRIX_FIELDS = [
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


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)


def make_env(deck_a, deck_b, seed, max_ply=10000, record_trace=False):
    return rlcard.make("ptcg", config={
        "deck_a": deck_a,
        "deck_b": deck_b,
        "seed": seed,
        "max_ply": max_ply,
        "record_trace": record_trace,
    })


def parse_deck_arg(value):
    if "=" not in value:
        raise argparse.ArgumentTypeError("--deck must use NAME=/absolute/path format")
    name, path = value.split("=", 1)
    name = name.strip()
    path = path.strip()
    if not name or not path:
        raise argparse.ArgumentTypeError("--deck must use NAME=/absolute/path format")
    return manual_deck(name, path)


def validate_decks(decks):
    for deck in decks:
        if not os.path.isfile(deck_path(deck)):
            raise FileNotFoundError(f"Deck '{deck_label(deck)}' not found: {deck_path(deck)}")


def resolve_decks(args, split_attr="split"):
    deck_pool = getattr(args, "deck_pool", None)
    deck_entries = getattr(args, "deck", None)
    split = getattr(args, split_attr, "train")
    if deck_pool and deck_entries:
        raise ValueError("--deck-pool and --deck are mutually exclusive")
    if deck_pool:
        return load_deck_pool(deck_pool, split=split)
    if deck_entries:
        validate_decks(deck_entries)
        return deck_entries
    validate_decks(DEFAULT_DECKS)
    return DEFAULT_DECKS


def resolve_train_decks(args):
    if args.deck_pool:
        return load_deck_pool(args.deck_pool, split=args.train_split)
    if not args.deck_a or not args.deck_b:
        raise ValueError("--deck-a and --deck-b are required when --deck-pool is not set")
    decks = [manual_deck("deck_a", args.deck_a), manual_deck("deck_b", args.deck_b)]
    validate_decks(decks)
    return decks


def sample_train_pair(decks, args, rng):
    if args.deck_pool:
        return sample_deck_pair(decks, rng=rng, mode=args.deck_sample_mode)
    return decks[0], decks[1]


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


def resolve_checkpoint_file(checkpoint, agent_type=None):
    if not checkpoint:
        return None
    if os.path.isfile(checkpoint):
        return checkpoint
    if not os.path.isdir(checkpoint):
        raise FileNotFoundError(f"Checkpoint path does not exist: {checkpoint}")

    candidates = []
    if agent_type in (None, "dqn"):
        candidates.append(os.path.join(checkpoint, "checkpoint_dqn.pt"))
    if agent_type in (None, "nfsp"):
        candidates.append(os.path.join(checkpoint, "checkpoint_nfsp.pt"))
    if agent_type in (None, "ppo"):
        candidates.append(os.path.join(checkpoint, "checkpoint_ppo.pt"))
    existing = [path for path in candidates if os.path.isfile(path)]
    if len(existing) == 1:
        return existing[0]
    if len(existing) > 1:
        raise ValueError(f"Multiple checkpoint files found; specify --agent: {existing}")
    raise FileNotFoundError(f"No checkpoint file found in {checkpoint}")


def infer_checkpoint_agent_type(checkpoint_file):
    name = os.path.basename(checkpoint_file).lower()
    if "dqn" in name:
        return "dqn"
    if "nfsp" in name:
        return "nfsp"
    if "ppo" in name:
        return "ppo"

    import torch
    checkpoint = torch.load(checkpoint_file, map_location="cpu", weights_only=False)
    agent_type = checkpoint.get("agent_type", "").lower()
    if "dqn" in agent_type:
        return "dqn"
    if "nfsp" in agent_type:
        return "nfsp"
    if "ppo" in agent_type:
        return "ppo"
    raise ValueError(f"Cannot infer checkpoint type from {checkpoint_file}")


def create_ptcg_agent(agent_type, env, device, checkpoint=None, seed=None, raw_action_tie_breaker="heuristic"):
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
        from rlcard.agents import DQNAgent
        agent = DQNAgent(
            num_actions=env.num_actions,
            state_shape=env.state_shape,
            mlp_layers=[256, 256, 128],
            device=device,
        )
        if not checkpoint_file:
            raise ValueError("--checkpoint is required for DQN")
        agent.load_checkpoint(checkpoint_file)
        return apply_raw_action_tie_breaker(agent, raw_action_tie_breaker)
    if agent_type == "nfsp":
        from rlcard.agents import NFSPAgent
        agent = NFSPAgent(
            num_actions=env.num_actions,
            state_shape=env.state_shape,
            hidden_layers_sizes=[256, 256, 128],
            q_mlp_layers=[256, 256, 128],
            evaluate_with="average_policy",
            device=device,
        )
        if not checkpoint_file:
            raise ValueError("--checkpoint is required for NFSP")
        agent.load_checkpoint(checkpoint_file)
        return apply_raw_action_tie_breaker(agent, raw_action_tie_breaker)
    if agent_type == "ppo":
        agent = PtcgPPOAgent(
            num_actions=env.num_actions,
            state_shape=env.state_shape,
            hidden_layers_sizes=[256, 256, 128],
            device=device,
            raw_action_tie_breaker=raw_action_tie_breaker,
        )
        if not checkpoint_file:
            raise ValueError("--checkpoint is required for PPO")
        agent.load_checkpoint(checkpoint_file)
        return apply_raw_action_tie_breaker(agent, raw_action_tie_breaker)
    raise ValueError(f"Unknown agent type: {agent_type}")


def create_legacy_agent(algorithm, num_actions, state_shape, device):
    from rlcard.agents import DQNAgent, NFSPAgent
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
    if algorithm == "nfsp":
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
            evaluate_with="average_policy",
        )
    raise ValueError(f"Unsupported legacy algorithm: {algorithm}")


def empty_rollout():
    return {
        "obs": [],
        "actions": [],
        "log_probs": [],
        "values": [],
        "returns": [],
        "advantages": [],
        "legal_masks": [],
        "player_ids": [],
        "deck_p0": [],
        "deck_p1": [],
        "payoffs": [],
        "ply": [],
    }


def add_player_trajectory(rollout, player_steps, payoff, gamma, gae_lambda, deck_a, deck_b, ply):
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
        rollout["player_ids"].append(step["player_id"])
        rollout["deck_p0"].append(deck_label(deck_a))
        rollout["deck_p1"].append(deck_label(deck_b))
        rollout["payoffs"].append(float(payoff))
        rollout["ply"].append(int(ply))


def merge_rollout(target, source):
    for key in target:
        target[key].extend(source[key])


def opponent_for_stage(args, env, device, seed, checkpoint_cache, checkpoint_rng):
    if args.stage == "selfplay":
        return None
    if args.stage == "warmup":
        opponent_name = "simplebot" if args.opponent == "self" else args.opponent
        return create_ptcg_agent(opponent_name, env, device, seed=seed, raw_action_tie_breaker=args.raw_action_tie_breaker)
    if args.stage == "league":
        checkpoints = checkpoint_pool_files(args.checkpoint_pool)
        if not checkpoints:
            raise ValueError(f"No PPO checkpoints found in checkpoint pool: {args.checkpoint_pool}")
        checkpoint_file = checkpoint_rng.choice(checkpoints)
        if checkpoint_file not in checkpoint_cache:
            checkpoint_cache[checkpoint_file] = create_ptcg_agent(
                "ppo",
                env,
                device,
                checkpoint=checkpoint_file,
                seed=seed,
                raw_action_tie_breaker=args.raw_action_tie_breaker,
            )
        return checkpoint_cache[checkpoint_file]
    raise ValueError(f"Unknown PPO training stage: {args.stage}")


def collect_ppo_episode(agent, opponent, args, deck_a, deck_b, episode_seed):
    seed_everything(episode_seed)
    env = make_env(deck_path(deck_a), deck_path(deck_b), episode_seed, args.max_ply)
    player_steps = {0: [], 1: []}
    state, player_id = env.reset()
    ply = 0

    while not env.is_over():
        trainable_turn = args.stage == "selfplay" or player_id == 0
        if trainable_turn:
            action, info = agent.sample_action(state, deterministic=False)
            player_steps[player_id].append({
                "obs": np.asarray(state["obs"], dtype=np.float32).ravel(),
                "action": int(info["action"]),
                "log_prob": float(info["log_prob"]),
                "value": float(info["value"]),
                "legal_mask": np.asarray(info["legal_mask"], dtype=bool),
                "player_id": int(player_id),
            })
            use_raw = agent.use_raw
        else:
            action = opponent.step(state)
            use_raw = opponent.use_raw
        state, player_id = env.step(action, use_raw)
        ply += 1

    payoffs = env.get_payoffs()
    rollout = empty_rollout()
    train_players = (0, 1) if args.stage == "selfplay" else (0,)
    for train_player in train_players:
        add_player_trajectory(
            rollout,
            player_steps[train_player],
            payoffs[train_player],
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            deck_a=deck_a,
            deck_b=deck_b,
            ply=ply,
        )
    return rollout, payoffs, ply


def append_csv(path, fieldnames, row):
    exists = os.path.isfile(path)
    with open(path, "a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def write_json(path, data):
    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, sort_keys=True, ensure_ascii=False, default=str)


def summarize_games(agent, opponent, decks, args, device, update_index, opponent_name, split_label):
    rng = random.Random(args.seed + 8000000 + update_index)
    wins = losses = ties = 0
    payoff_sum = 0.0
    ply_sum = 0
    for game_index in range(args.num_eval_games):
        deck_a, deck_b = sample_train_pair(decks, args, rng)
        seed = args.seed + 9000000 + update_index * 10000 + game_index
        env = make_env(deck_path(deck_a), deck_path(deck_b), seed, args.max_ply, record_trace=True)
        if opponent_name == "self":
            eval_opponent = agent
        else:
            eval_opponent = create_ptcg_agent(opponent_name, env, device, seed=seed + 1, raw_action_tie_breaker=args.raw_action_tie_breaker)
        env.set_agents([agent, eval_opponent])
        _, payoffs = env.run(is_training=False)
        payoff = float(payoffs[0])
        payoff_sum += payoff
        if payoff > 0:
            wins += 1
        elif payoff < 0:
            losses += 1
        else:
            ties += 1
        ply_sum += len(trace_for_env(env))
    num_games = max(args.num_eval_games, 1)
    return {
        "update": update_index,
        "opponent": opponent_name,
        "deck_split": split_label,
        "num_games": args.num_eval_games,
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "avg_payoff": payoff_sum / num_games,
        "approx_win_rate": wins / num_games,
        "avg_ply": ply_sum / num_games,
    }


def checkpoint_pool_files(path):
    if not path:
        return []
    root = Path(path)
    if root.is_file() and root.name == "checkpoint_ppo.pt":
        return [str(root)]
    return sorted(str(item) for item in root.rglob("checkpoint_ppo.pt"))


def prune_checkpoints(checkpoint_root, keep_last):
    if not keep_last or keep_last <= 0:
        return
    root = Path(checkpoint_root)
    if not root.is_dir():
        return
    numbered = []
    for child in root.iterdir():
        if child.is_dir() and child.name.startswith("checkpoint_"):
            try:
                numbered.append((int(child.name.split("_", 1)[1]), child))
            except ValueError:
                continue
    for _, path in sorted(numbered)[:-keep_last]:
        shutil.rmtree(path, ignore_errors=True)


def save_ppo_checkpoint(agent, save_dir, update_index, total_episodes, total_steps, rng, args):
    agent.training_state = {
        "update": update_index,
        "episodes": total_episodes,
        "steps": total_steps,
        "python_random_state": repr(rng.getstate()),
        "seed": args.seed,
        "stage": args.stage,
    }
    agent.save_checkpoint(save_dir)


def train_ppo(args):
    seed_everything(args.seed)
    device = get_device()
    decks = resolve_train_decks(args)
    rng = random.Random(args.seed)
    first_deck_a, first_deck_b = sample_train_pair(decks, args, rng)
    temp_env = make_env(deck_path(first_deck_a), deck_path(first_deck_b), args.seed, args.max_ply)

    agent = PtcgPPOAgent(
        num_actions=temp_env.num_actions,
        state_shape=temp_env.state_shape,
        hidden_layers_sizes=[256, 256, 128],
        learning_rate=args.learning_rate,
        device=device,
        raw_action_tie_breaker=args.raw_action_tie_breaker,
    )
    if args.resume_from:
        agent.load_checkpoint(resolve_checkpoint_file(args.resume_from, "ppo"))
        apply_raw_action_tie_breaker(agent, args.raw_action_tie_breaker)

    state = agent.training_state or {}
    start_update = int(state.get("update", 0))
    total_episodes = int(state.get("episodes", 0))
    total_steps = int(state.get("steps", 0))

    os.makedirs(args.log_dir, exist_ok=True)
    write_json(os.path.join(args.log_dir, "training_config.json"), {
        "args": vars(args),
        "state_shape": temp_env.state_shape,
        "num_actions": temp_env.num_actions,
        "decks": [{key: deck.get(key) for key in ("name", "archetype", "split", "path")} for deck in decks],
        "git_status": git_status_summary(),
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    })

    checkpoint_cache = {}
    checkpoint_rng = random.Random(args.seed + 123456)
    print(f"Algorithm: ppo")
    print(f"Stage: {args.stage}")
    print(f"State shape: {temp_env.state_shape}")
    print(f"Num actions: {temp_env.num_actions}")
    print(f"Updates: {args.updates}")
    print(f"Rollout episodes/update: {args.rollout_episodes}")
    print(f"Log dir: {args.log_dir}")
    print()

    for update_index in range(start_update + 1, start_update + args.updates + 1):
        rollout = empty_rollout()
        payoff_sum_p0 = 0.0
        ply_sum = 0
        for _ in range(args.rollout_episodes):
            total_episodes += 1
            episode_seed = args.seed + total_episodes
            deck_a, deck_b = sample_train_pair(decks, args, rng)
            episode_env = make_env(deck_path(deck_a), deck_path(deck_b), episode_seed, args.max_ply)
            opponent = opponent_for_stage(args, episode_env, device, episode_seed + 1, checkpoint_cache, checkpoint_rng)
            episode_rollout, payoffs, ply = collect_ppo_episode(agent, opponent, args, deck_a, deck_b, episode_seed)
            merge_rollout(rollout, episode_rollout)
            payoff_sum_p0 += float(payoffs[0])
            ply_sum += ply
            if args.max_steps_per_update and len(rollout["obs"]) >= args.max_steps_per_update:
                break

        total_steps += len(rollout["obs"])
        metrics = agent.update(
            rollout,
            clip_ratio=args.clip_ratio,
            vf_coef=args.vf_coef,
            ent_coef=args.ent_coef,
            update_epochs=args.update_epochs,
            minibatch_size=args.minibatch_size,
            max_grad_norm=args.max_grad_norm,
            target_kl=args.target_kl,
        )
        row = {
            "update": update_index,
            "episodes": total_episodes,
            "steps": total_steps,
            "avg_payoff_p0": payoff_sum_p0 / max(args.rollout_episodes, 1),
            **metrics,
        }
        append_csv(os.path.join(args.log_dir, "performance.csv"), PERFORMANCE_FIELDS, row)

        if args.eval_every and update_index % args.eval_every == 0:
            for opponent_name in parse_csv_list(args.eval_opponents):
                eval_row = summarize_games(
                    agent, None, decks, args, device, update_index, opponent_name, args.train_split
                )
                append_csv(os.path.join(args.log_dir, "eval.csv"), EVAL_FIELDS, eval_row)

        print(
            f"Update {update_index}: episodes={total_episodes}, steps={total_steps}, "
            f"avg_payoff_p0={row['avg_payoff_p0']:.4f}, "
            f"policy_loss={metrics['policy_loss']:.4f}, value_loss={metrics['value_loss']:.4f}, "
            f"entropy={metrics['entropy']:.4f}, kl={metrics['approx_kl']:.4f}, "
            f"avg_ply={ply_sum / max(args.rollout_episodes, 1):.1f}",
            flush=True,
        )

        if args.save_every and update_index % args.save_every == 0:
            save_dir = os.path.join(args.log_dir, "checkpoints", f"checkpoint_{update_index}")
            save_ppo_checkpoint(agent, save_dir, update_index, total_episodes, total_steps, rng, args)
            prune_checkpoints(os.path.join(args.log_dir, "checkpoints"), args.checkpoint_keep_last)
            print(f"Checkpoint saved to {save_dir}", flush=True)

    final_dir = os.path.join(args.log_dir, "final")
    save_ppo_checkpoint(agent, final_dir, start_update + args.updates, total_episodes, total_steps, rng, args)
    print(f"Final model saved to {final_dir}")


def train_legacy(args):
    if not args.deck_a or not args.deck_b:
        raise ValueError("DQN/NFSP legacy training requires --deck-a and --deck-b")
    device = get_device()
    set_seed(args.seed)
    os.makedirs(args.log_dir, exist_ok=True)
    temp_env = make_env(args.deck_a, args.deck_b, args.seed, args.max_ply)
    agent = create_legacy_agent(args.algorithm, temp_env.num_actions, temp_env.state_shape, device)
    apply_raw_action_tie_breaker(agent, args.raw_action_tie_breaker)
    if args.opponent == "self":
        opponent = agent
    elif args.opponent == "random":
        opponent = RandomAgent(num_actions=temp_env.num_actions)
    else:
        from rlcard.agents.ptcg_simplebot_agent import PtcgSimpleBotAgent
        opponent = PtcgSimpleBotAgent(seed=args.seed + 1)

    with Logger(args.log_dir) as logger:
        for episode in range(1, args.episodes + 1):
            env = make_env(args.deck_a, args.deck_b, args.seed + episode, args.max_ply)
            env.set_agents([agent, opponent])
            if hasattr(agent, "sample_episode_policy"):
                agent.sample_episode_policy()
            trajectories, payoffs = env.run(is_training=True)
            transitions = reorganize(trajectories, payoffs)
            train_players = range(len(transitions)) if args.opponent == "self" else [0]
            for player_id in train_players:
                for transition in transitions[player_id]:
                    agent.feed(transition)
            if args.eval_every and episode % args.eval_every == 0:
                eval_env = make_env(args.deck_a, args.deck_b, args.seed + 100000 + episode, args.max_ply)
                eval_env.set_agents([agent, opponent])
                avg_payoffs = tournament(eval_env, args.num_eval_games)
                logger.log_performance(episode, avg_payoffs[0])
                print(f"Episode {episode}: avg_payoff_p0={avg_payoffs[0]:.4f}")
            if args.save_every and episode % args.save_every == 0:
                save_dir = os.path.join(args.log_dir, f"checkpoint_{episode}")
                os.makedirs(save_dir, exist_ok=True)
                agent.save_checkpoint(save_dir)
        final_dir = os.path.join(args.log_dir, "final")
        os.makedirs(final_dir, exist_ok=True)
        agent.save_checkpoint(final_dir)


def run_train(args):
    if args.algorithm == "ppo":
        train_ppo(args)
    else:
        train_legacy(args)


def run_eval(args):
    seed_everything(args.seed)
    device = get_device()
    env = make_env(args.deck_a, args.deck_b, args.seed, args.max_ply)
    agent = create_ptcg_agent(
        args.agent, env, device, checkpoint=args.checkpoint, seed=args.seed,
        raw_action_tie_breaker=args.raw_action_tie_breaker,
    )
    opponent = create_ptcg_agent(
        args.opponent, env, device, checkpoint=args.opponent_checkpoint, seed=args.seed + 1,
        raw_action_tie_breaker=args.raw_action_tie_breaker,
    )

    wins = losses = ties = 0
    payoff_sum = 0.0
    for game_index in range(args.num_games):
        game_seed = args.seed + game_index
        seed_everything(game_seed)
        game_env = make_env(args.deck_a, args.deck_b, game_seed, args.max_ply)
        game_env.set_agents([agent, opponent])
        _, payoffs = game_env.run(is_training=False)
        payoff = float(payoffs[0])
        payoff_sum += payoff
        if payoff > 0:
            wins += 1
        elif payoff < 0:
            losses += 1
        else:
            ties += 1
    num_games = max(args.num_games, 1)
    print("Results (player 0 payoff):")
    print(f"  Average payoff: {payoff_sum / num_games:.4f}")
    print(f"  Wins/Losses/Ties: {wins}/{losses}/{ties}")
    print(f"  Approx win rate: {wins / num_games:.2%}")


def run_trace(args):
    seed_everything(args.seed)
    device = get_device()
    os.makedirs(args.log_dir, exist_ok=True)
    env = make_env(args.deck_a, args.deck_b, args.seed, args.max_ply, record_trace=True)
    agent = create_ptcg_agent(
        args.agent, env, device, checkpoint=args.checkpoint, seed=args.seed,
        raw_action_tie_breaker=args.raw_action_tie_breaker,
    )
    opponent = create_ptcg_agent(
        args.opponent, env, device, checkpoint=args.opponent_checkpoint, seed=args.seed + 1,
        raw_action_tie_breaker=args.raw_action_tie_breaker,
    )

    env.set_agents([agent, opponent])
    _, payoffs = env.run(is_training=False)
    trace = trace_for_env(env)
    action_template_counts = Counter()
    fallback_count = 0
    for entry in trace:
        if entry.get("fallback"):
            fallback_count += 1
        template_id = entry.get("chosen_template_id")
        if template_id is not None:
            action_template_counts[str(template_id)] += 1

    result = {
        "agent": args.agent,
        "opponent": args.opponent,
        "checkpoint": args.checkpoint,
        "opponent_checkpoint": args.opponent_checkpoint,
        "deck_a": args.deck_a,
        "deck_b": args.deck_b,
        "seed": args.seed,
        "max_ply": args.max_ply,
        "raw_action_tie_breaker": args.raw_action_tie_breaker,
        "payoffs": [float(payoff) for payoff in payoffs],
        "winner": 0 if payoffs[0] > payoffs[1] else 1 if payoffs[1] > payoffs[0] else None,
        "ply": len(trace),
        "fallback_count": fallback_count,
        "action_template_counts": dict(sorted(action_template_counts.items())),
        "trace": trace,
    }
    output_path = args.output or os.path.join(
        args.log_dir,
        f"trace_{args.agent}_vs_{args.opponent}_seed{args.seed}.json",
    )
    write_json(output_path, result)
    readable_path = args.readable_output or os.path.splitext(output_path)[0] + ".txt"
    with open(readable_path, "w", encoding="utf-8") as file:
        file.write(render_trace_timeline(result))
    print(f"Payoffs: {result['payoffs']}")
    print(f"Winner: {result['winner']}")
    print(f"Ply: {result['ply']}")
    print(f"Fallback count: {result['fallback_count']}")
    print(f"Wrote trace to {output_path}")
    print(f"Wrote readable log to {readable_path}")


def render_trace_timeline(result):
    outcome = "Tie"
    if result["winner"] is not None:
        outcome = f"P{int(result['winner']) + 1} wins"
    lines = [
        (
            f"Replay rlcard-ptcg-trace | seed {result['seed']} | "
            f"{result['deck_a']} vs {result['deck_b']}"
        ),
        (
            f"Agents: P1 {result['agent']} ({result.get('checkpoint') or 'no checkpoint'}) "
            f"vs P2 {result['opponent']} ({result.get('opponent_checkpoint') or 'no checkpoint'})"
        ),
        f"Outcome: {outcome} | payoffs {result['payoffs']}",
        f"Frames: {result['ply']} | fallback_count {result['fallback_count']}",
        "",
    ]
    for entry in result["trace"]:
        snapshot = entry.get("observation_snapshot") or {}
        action_label = readable_action_label(entry, snapshot)
        lines.append(
            f"#{int(entry.get('ply', 0)):04} T{snapshot.get('turn', '?')} "
            f"P{int(entry.get('actor', 0)) + 1} {action_label}"
        )
        lines.append(
            f"      P1 {board_summary_from_snapshot(snapshot, 0)} | "
            f"P2 {board_summary_from_snapshot(snapshot, 1)}"
        )
        pending_choice = snapshot.get("pending_choice")
        if pending_choice:
            lines.append(f"      choice {pending_choice}")
        if entry.get("fallback"):
            lines.append(f"      fallback legal={entry.get('legal_action_ids', [])}")
    return "\n".join(lines) + "\n"


def readable_action_label(entry, snapshot):
    action_id = entry.get("action_id", "")
    label = action_id
    actor_hand = (((snapshot or {}).get("self") or {}).get("hand") or [])
    if ":hand:" in action_id and isinstance(actor_hand, list):
        parts = action_id.split(":")
        try:
            hand_index = parts.index("hand") + 1
            card = actor_hand[int(parts[hand_index])]
            if isinstance(card, dict):
                label = f"{action_id} [{card.get('full_name') or card.get('logic_key') or 'card'}]"
        except (ValueError, IndexError, TypeError):
            pass
    template_id = entry.get("chosen_template_id")
    if template_id is not None:
        label = f"{label} (template {template_id})"
    return label


def board_summary_from_snapshot(snapshot, player):
    active_player = snapshot.get("active_player")
    if active_player == player:
        side = snapshot.get("self") or {}
    else:
        side = snapshot.get("opponent") or {}
    active = slot_summary(side.get("active"))
    bench = side.get("bench") or []
    bench_count = sum(1 for slot in bench if slot)
    deck_count = (side.get("deck") or {}).get("count", "?")
    prizes = (side.get("prizes") or {}).get("remaining", "?")
    hand = side.get("hand")
    if isinstance(hand, list):
        hand_count = len(hand)
    elif isinstance(hand, dict):
        hand_count = hand.get("count", "?")
    else:
        hand_count = "?"
    return f"Active {active}; Bench {bench_count}; Hand {hand_count}; Deck {deck_count}; Prize {prizes}"


def slot_summary(slot):
    if not slot:
        return "-"
    name = slot.get("name") or slot.get("logic_key") or "Pokemon"
    hp = slot.get("current_hp", "?")
    max_hp = slot.get("hp", "?")
    energy = slot.get("energy_count", 0)
    damage = slot.get("damage", 0)
    return f"{name} HP {hp}/{max_hp} dmg {damage} E{energy}"


def trace_for_env(env):
    ptcg_env = getattr(getattr(env, "game", None), "_ptcg_env", None)
    if ptcg_env is None:
        return []
    try:
        return ptcg_env.action_trace()
    except Exception:
        return []


def summarize_ci(win_rate, num_games):
    if num_games <= 0:
        return 0.0, 0.0
    margin = 1.96 * math.sqrt(win_rate * (1.0 - win_rate) / num_games)
    return max(0.0, win_rate - margin), min(1.0, win_rate + margin)


def evaluate_matrix_pair(args, device, deck_p0, deck_p1, pair_index):
    seed = args.seed + pair_index * 100000
    agent_env = make_env(deck_path(deck_p0), deck_path(deck_p1), seed, args.max_ply, record_trace=True)
    agent = create_ptcg_agent(args.agent, agent_env, device, args.checkpoint, seed, args.raw_action_tie_breaker)
    opponent = create_ptcg_agent(args.opponent, agent_env, device, args.opponent_checkpoint, seed + 1, args.raw_action_tie_breaker)
    wins = losses = ties = 0
    payoff_sum = 0.0
    ply_sum = 0
    fallback_count = 0
    template_counts = Counter()
    for game_index in range(args.num_games):
        game_seed = seed + game_index
        seed_everything(game_seed)
        env = make_env(deck_path(deck_p0), deck_path(deck_p1), game_seed, args.max_ply, record_trace=True)
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
    ci_low, ci_high = summarize_ci(win_rate, args.num_games)
    return {
        "agent": args.agent,
        "opponent": args.opponent,
        "deck_p0": deck_label(deck_p0),
        "deck_p1": deck_label(deck_p1),
        "deck_p0_archetype": deck_archetype(deck_p0),
        "deck_p1_archetype": deck_archetype(deck_p1),
        "deck_p0_split": deck_split(deck_p0),
        "deck_p1_split": deck_split(deck_p1),
        "deck_p0_path": deck_path(deck_p0),
        "deck_p1_path": deck_path(deck_p1),
        "seed": seed,
        "num_games": args.num_games,
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "avg_payoff": payoff_sum / max(args.num_games, 1),
        "approx_win_rate": win_rate,
        "ci95_low": ci_low,
        "ci95_high": ci_high,
        "avg_ply": ply_sum / max(args.num_games, 1),
        "fallback_count": fallback_count,
        "action_template_counts": dict(sorted(template_counts.items())),
    }


def run_matrix(args):
    decks = resolve_decks(args)
    device = get_device()
    rows = []
    pair_index = 0
    for deck_p0 in decks:
        for deck_p1 in decks:
            print(f"Evaluating {args.agent}({deck_label(deck_p0)}) vs {args.opponent}({deck_label(deck_p1)})...")
            row = evaluate_matrix_pair(args, device, deck_p0, deck_p1, pair_index)
            rows.append(row)
            print(f"  payoff={row['avg_payoff']:.4f}, win_rate={row['approx_win_rate']:.2%}")
            pair_index += 1

    os.makedirs(args.log_dir, exist_ok=True)
    csv_path = os.path.join(args.log_dir, "results.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=MATRIX_FIELDS)
        writer.writeheader()
        for row in rows:
            csv_row = row.copy()
            csv_row["action_template_counts"] = json.dumps(row["action_template_counts"], sort_keys=True)
            writer.writerow(csv_row)
    json_path = os.path.join(args.log_dir, "results.json")
    write_json(json_path, {
        "args": vars(args),
        "rows": rows,
        "decks": [{key: deck.get(key) for key in ("name", "archetype", "split", "path")} for deck in decks],
    })
    print(f"Wrote {len(rows)} rows to {csv_path}")
    print(f"Wrote JSON results to {json_path}")


def run_launch(args):
    child_args = args.child_args
    if child_args and child_args[0] == "--":
        child_args = child_args[1:]
    if not child_args:
        raise ValueError("launch requires a subcommand after --")
    run_dir = args.run_dir or os.path.join(
        "experiments",
        f"ptcg-launch-{time.strftime('%Y%m%d-%H%M%S')}",
    )
    os.makedirs(run_dir, exist_ok=True)
    command = [sys.executable, str(Path(__file__).resolve()), *child_args]
    write_json(os.path.join(run_dir, "command.json"), {
        "cwd": str(ROOT_DIR),
        "command": command,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    })

    run_sh = os.path.join(run_dir, "run.sh")
    with open(run_sh, "w", encoding="utf-8") as file:
        file.write("#!/usr/bin/env bash\n")
        file.write("set -o pipefail\n")
        file.write(f"cd {shell_quote(str(ROOT_DIR))}\n")
        file.write(f"echo status=running > {shell_quote(os.path.join(run_dir, 'status.txt'))}\n")
        file.write(" ".join(shell_quote(part) for part in command))
        file.write("\n")
        file.write("code=$?\n")
        file.write(f"echo \"$code\" > {shell_quote(os.path.join(run_dir, 'exit_code'))}\n")
        file.write(f"echo exit_code=$code >> {shell_quote(os.path.join(run_dir, 'status.txt'))}\n")
        file.write('exit "$code"\n')
    os.chmod(run_sh, 0o755)
    stdout_path = os.path.join(run_dir, "stdout.log")
    with open(stdout_path, "ab", buffering=0) as stdout:
        process = subprocess.Popen(
            [run_sh],
            cwd=str(ROOT_DIR),
            stdout=stdout,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )
    with open(os.path.join(run_dir, "pid"), "w", encoding="utf-8") as file:
        file.write(str(process.pid))
    print("Started background PTCG experiment.")
    print(f"PID: {process.pid}")
    print(f"Run dir: {run_dir}")
    print(f"Live log: tail -f {stdout_path}")


def run_status(args):
    run_dir = args.run_dir
    pid_path = os.path.join(run_dir, "pid")
    status_path = os.path.join(run_dir, "status.txt")
    exit_path = os.path.join(run_dir, "exit_code")
    stdout_path = os.path.join(run_dir, "stdout.log")
    if os.path.isfile(pid_path):
        pid = Path(pid_path).read_text(encoding="utf-8").strip()
        alive = subprocess.run(["ps", "-p", pid], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0
        print(f"PID: {pid} ({'running' if alive else 'not running'})")
    if os.path.isfile(status_path):
        print(Path(status_path).read_text(encoding="utf-8").strip())
    if os.path.isfile(exit_path):
        print(f"exit_code={Path(exit_path).read_text(encoding='utf-8').strip()}")
    for name in ("performance.csv", "eval.csv", "results.csv", "results.json"):
        path = os.path.join(run_dir, name)
        if os.path.exists(path):
            print(f"{name}: present")
    if args.tail and os.path.isfile(stdout_path):
        print(f"--- tail {stdout_path} ---")
        with open(stdout_path, encoding="utf-8", errors="replace") as file:
            lines = file.readlines()[-args.tail:]
        print("".join(lines), end="")


def parse_csv_list(value):
    return [item.strip() for item in value.split(",") if item.strip()]


def git_status_summary():
    try:
        result = subprocess.run(
            ["git", "status", "--short"],
            cwd=str(ROOT_DIR),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        return result.stdout.strip().splitlines()
    except Exception:
        return []


def shell_quote(value):
    return "'" + str(value).replace("'", "'\"'\"'") + "'"


def add_shared_eval_args(parser):
    parser.add_argument("--agent", choices=["random", "simplebot", "dqn", "nfsp", "ppo"], default="simplebot")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--opponent", choices=["random", "simplebot", "dqn", "nfsp", "ppo"], default="random")
    parser.add_argument("--opponent-checkpoint", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-games", type=int, default=200)
    parser.add_argument("--max-ply", type=int, default=10000)
    parser.add_argument("--raw-action-tie-breaker", choices=["first", "heuristic"], default="heuristic")


def build_parser():
    parser = argparse.ArgumentParser(description="Unified PTCG experiment CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train = subparsers.add_parser("train", help="Train PPO/DQN/NFSP")
    train.add_argument("--algorithm", choices=["ppo", "dqn", "nfsp"], default="ppo")
    train.add_argument("--stage", choices=["warmup", "selfplay", "league"], default="selfplay")
    train.add_argument("--deck-a", default=None)
    train.add_argument("--deck-b", default=None)
    train.add_argument("--deck-pool", default=None)
    train.add_argument("--train-split", default="train")
    train.add_argument("--deck-sample-mode", choices=["archetype-balanced", "uniform"], default="archetype-balanced")
    train.add_argument("--opponent", choices=["self", "random", "simplebot"], default="self")
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--max-ply", type=int, default=10000)
    train.add_argument("--log-dir", default="experiments/ptcg-run")
    train.add_argument("--raw-action-tie-breaker", choices=["first", "heuristic"], default="heuristic")
    train.add_argument("--episodes", type=int, default=50000)
    train.add_argument("--rollout-episodes", type=int, default=64)
    train.add_argument("--updates", type=int, default=1000)
    train.add_argument("--eval-every", type=int, default=20)
    train.add_argument("--num-eval-games", type=int, default=50)
    train.add_argument("--eval-opponents", default="random,simplebot")
    train.add_argument("--save-every", type=int, default=50)
    train.add_argument("--checkpoint-keep-last", type=int, default=5)
    train.add_argument("--resume-from", default=None)
    train.add_argument("--checkpoint-pool", default=None)
    train.add_argument("--learning-rate", type=float, default=3e-4)
    train.add_argument("--gamma", type=float, default=0.99)
    train.add_argument("--gae-lambda", type=float, default=0.95)
    train.add_argument("--clip-ratio", type=float, default=0.2)
    train.add_argument("--vf-coef", type=float, default=0.5)
    train.add_argument("--ent-coef", type=float, default=0.01)
    train.add_argument("--update-epochs", type=int, default=4)
    train.add_argument("--minibatch-size", type=int, default=1024)
    train.add_argument("--max-grad-norm", type=float, default=0.5)
    train.add_argument("--target-kl", type=float, default=0.03)
    train.add_argument("--max-steps-per-update", type=int, default=0)
    train.set_defaults(func=run_train)

    eval_parser = subparsers.add_parser("eval", help="Evaluate one deck pair")
    add_shared_eval_args(eval_parser)
    eval_parser.add_argument("--deck-a", required=True)
    eval_parser.add_argument("--deck-b", required=True)
    eval_parser.set_defaults(func=run_eval)

    trace = subparsers.add_parser("trace", help="Run one game and export action trace JSON")
    add_shared_eval_args(trace)
    trace.set_defaults(num_games=1)
    trace.add_argument("--deck-a", required=True)
    trace.add_argument("--deck-b", required=True)
    trace.add_argument("--log-dir", default="experiments/ptcg-traces")
    trace.add_argument("--output", default=None)
    trace.add_argument("--readable-output", default=None)
    trace.set_defaults(func=run_trace)

    matrix = subparsers.add_parser("matrix", help="Evaluate an ordered deck matrix")
    add_shared_eval_args(matrix)
    matrix.add_argument("--deck", action="append", type=parse_deck_arg)
    matrix.add_argument("--deck-pool", default=None)
    matrix.add_argument("--split", default="train")
    matrix.add_argument("--log-dir", required=True)
    matrix.set_defaults(func=run_matrix)

    launch = subparsers.add_parser("launch", help="Run a subcommand in the background")
    launch.add_argument("--run-dir", default=None)
    launch.add_argument("child_args", nargs=argparse.REMAINDER)
    launch.set_defaults(func=run_launch)

    status = subparsers.add_parser("status", help="Inspect a background run")
    status.add_argument("--run-dir", required=True)
    status.add_argument("--tail", type=int, default=40)
    status.set_defaults(func=run_status)
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    try:
        args.func(args)
    except (ImportError, ValueError, FileNotFoundError) as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
