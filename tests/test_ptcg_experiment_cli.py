import importlib.util
from pathlib import Path


def load_cli_module():
    path = Path(__file__).resolve().parents[1] / "examples" / "ptcg_experiment.py"
    spec = importlib.util.spec_from_file_location("ptcg_experiment", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_train_ppo_parser_accepts_stage_and_deck_pool():
    module = load_cli_module()
    parser = module.build_parser()
    args = parser.parse_args([
        "train",
        "--algorithm", "ppo",
        "--stage", "selfplay",
        "--deck-pool", "configs/ptcg_deck_pool.json",
        "--rollout-episodes", "2",
        "--updates", "1",
    ])
    assert args.command == "train"
    assert args.algorithm == "ppo"
    assert args.stage == "selfplay"
    assert args.rollout_episodes == 2
    assert args.updates == 1


def test_matrix_parser_accepts_ppo_checkpoint_and_split():
    module = load_cli_module()
    parser = module.build_parser()
    args = parser.parse_args([
        "matrix",
        "--agent", "ppo",
        "--checkpoint", "experiments/model/final",
        "--opponent", "simplebot",
        "--deck-pool", "configs/ptcg_deck_pool.json",
        "--split", "train",
        "--num-games", "1",
        "--log-dir", "experiments/matrix",
    ])
    assert args.command == "matrix"
    assert args.agent == "ppo"
    assert args.split == "train"
    assert args.num_games == 1


def test_launch_parser_captures_child_command_after_separator():
    module = load_cli_module()
    parser = module.build_parser()
    args = parser.parse_args([
        "launch",
        "--run-dir", "experiments/bg",
        "--",
        "matrix",
        "--agent", "ppo",
        "--num-games", "1",
    ])
    assert args.command == "launch"
    assert args.run_dir == "experiments/bg"
    assert args.child_args == ["--", "matrix", "--agent", "ppo", "--num-games", "1"]
