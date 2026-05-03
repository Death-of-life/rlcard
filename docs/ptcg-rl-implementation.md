# PTCG 强化学习框架实施文档

## 概述

本项目基于 [PTCG 强化学习详细设计](https://github.com/anomalyco/opencode/issues) 文档，完成了 deckgym-core (Rust) 与 rlcard (Python) 之间的 PTCG 强化学习框架改造。

核心架构：

```
Rust (deckgym-core)
  src/ptcg                 唯一规则引擎（未修改）
  src/ptcg/rl              RL 观测、动作视图、PyO3 绑定（新增）

Python (rlcard-ptcg)
  rlcard/envs/ptcg.py      RLCard Env 适配
  rlcard/games/ptcg/       PyO3 thin facade
  rlcard/agents/*          复用 DQN/NFSP/RandomAgent
  examples/                自对弈、评估脚本
```

## 当前进度（2026-05-03）

当前仓库已经进入“PTCG RL 框架可冒烟运行”的阶段，但还没有到可以长期训练强模型的阶段。

已完成：

- Rust 侧已新增 `src/ptcg/rl` 和 PyO3 绑定，`deckgym_ptcg.PtcgEnv` 可被 Python 调用。
- Python 侧已新增 `rlcard/envs/ptcg.py`、`rlcard/games/ptcg/`、`rlcard/agents/ptcg_simplebot_agent.py`。
- 自对弈和评估脚本已存在：`examples/run_ptcg_selfplay.py`、`examples/evaluate_ptcg.py`。
- PTCG env 测试已存在：`tests/envs/test_ptcg_env.py`。
- 使用 `/Users/easygod/code/deckgym-core/.venv/bin/python` 运行 `tests/envs/test_ptcg_env.py`，当前 12 个测试通过。
- 使用同一 venv 运行 `examples/evaluate_ptcg.py --num-games 3 --opponent random` 可以完成 simplebot/random 冒烟评估。

当前实际观测：

- `rlcard.make("ptcg", ...)` 可以 reset，当前运行时 `obs` 维度观察值为 68。
- `legal_actions` 是 template id 的 `OrderedDict`，`raw_legal_actions` 是 Rust 输出的具体合法动作视图。
- 当前 `raw_legal_actions` 数量可能大于 template 数量，因为多个具体动作会映射到同一个 template。

当前阻塞和不一致：

- 系统 `python3` 环境缺少 `deckgym_ptcg`，需要使用 deckgym-core 的 `.venv`，或先执行 `maturin develop`。
- deckgym-core `.venv` 当前缺少 `torch`，DQN/NFSP 训练还不能直接启动。
- deckgym-core 普通 `cargo test` 当前会因为 PyO3 `extension-module` 链接 Python 符号失败；需要把 Python extension build 和 Rust 原生测试 build 拆开。
- `README.md`、多数上游 RLCard docs、`rlcard/models` 仍有旧 RLCard 残留，需要后续清理。
- 当前训练脚本传入 `state_shape=None`，但现有 DQN/NFSP 网络不会自动推断维度；后续必须从 `env.reset()` 推断 `state_shape=[len(state["obs"])]`。

## 后续路线图

目标先定义为：在北京沙、龙神柱、猛雷鼓三套固定牌组矩阵上，模型稳定超过 random/simplebot/heuristic baseline，并通过固定协议的人类盲测逐步接近 50% 胜率区间。不把开放格式泛化作为近期验收标准。

### Phase 0：环境与仓库卫生

- 修复 deckgym-core 的 PyO3 构建配置，使 `maturin develop` 和普通 `cargo test` 可以各自稳定运行。
- 统一 Python 环境说明：默认使用 `/Users/easygod/code/deckgym-core/.venv`，安装 `maturin`、`pytest`、`torch`、`matplotlib` 和本地 `rlcard-ptcg`。
- 清理 `README.md`、旧 RLCard docs、旧 `rlcard/models` 残留，避免误导后续 agent 或实验记录。
- 将验证命令固定到文档和 `AGENTS.md`，减少系统 `python3` 与 venv 混用。

### Phase 1：训练闭环修复

- 修改训练脚本，让 DQN/NFSP 在创建 agent 前 reset env，并显式设置 `state_shape=[obs_dim]`。
- 明确 raw action agent 与 template action agent 的兼容边界，保证 `PtcgSimpleBotAgent(use_raw=True)` 和神经网络 agent 能在同一个 env.run 流程里稳定工作。
- 补 checkpoint 加载、保存和继续训练路径，评估脚本支持 DQN/NFSP checkpoint 类型。
- 固定 seed、牌组、opponent，输出最小评估矩阵。

### Phase 2：强基线实验

- 先跑 random、simplebot、heuristic、self checkpoint 的 pairwise 胜率矩阵。
- 记录 win rate、tie rate、average ply、illegal/fallback action count、action template distribution。
- 对北京沙、龙神柱、猛雷鼓三套牌分别保存 baseline 曲线和 replay/trace 样本。
- 用小规模 DQN/NFSP 自对弈确认损失下降、checkpoint 可加载、评估结果可复现。

### Phase 3：强模型升级

- 引入 action 参数化或 learned tie-break scorer，减少粗 template 动作造成的信息损失。
- 从 card bucket 特征升级到 card id embedding 或 logic_key embedding。
- 增加并行 rollout，优先保证本机 Mac 可小规模运行，同时预留远端单卡 GPU 配置。
- 引入 PPO 或 actor-critic self-play，并维护 league/checkpoint pool，降低策略退化风险。

### Phase 4：人类水平评估

- 建立固定真人对局协议：固定三套牌、固定 seed 或固定开局集合、隐藏信息安全、记录完整 trace/replay。
- 与 simplebot、heuristic、历史 checkpoint 和真人结果统一出一张评估表。
- 只有当模型在固定牌组矩阵中稳定超过 heuristic/simplebot，并在人类盲测中接近 50% 胜率区间，才把阶段目标描述为“接近人类水平”。

## 新增/修改文件清单

### 1. Rust 侧 (deckgym-core)

#### 新增文件

| 文件 | 行数 | 描述 |
|------|------|------|
| `src/ptcg/rl/mod.rs` | ~1250 | RL 核心模块：Observation、LegalActionView、StepResult、EpisodeSummary、action_id ↔ Action 转换、template_id 映射、隐藏信息安全观测、PtcgRlEnv |
| `src/ptcg/rl/pyo3.rs` | ~120 | `#[pyclass] PtcgEnv` PyO3 绑定类 |
| `pyproject.toml` | 12 | Maturin 构建配置，模块名 `deckgym_ptcg` |

#### 修改文件

| 文件 | 改动 |
|------|------|
| `Cargo.toml` | 添加 `pyo3 = { version = "0.22", features = ["serde", "extension-module"] }` 依赖；`crate-type = ["rlib", "cdylib"]` |
| `src/lib.rs` | 添加 `#[pymodule] fn deckgym_ptcg` 注册 Python 模块 |
| `src/ptcg/mod.rs` | 添加 `pub mod rl;` |

### 2. Python 侧 (rlcard)

#### 新增文件

| 文件 | 行数 | 描述 |
|------|------|------|
| `rlcard/envs/ptcg.py` | ~130 | PtcgEnv(Env) 子类，实现 `_extract_state`、`_decode_action`、`_get_legal_actions`、`get_payoffs`、`get_perfect_information` |
| `rlcard/games/ptcg/__init__.py` | ~120 | PtcgGame thin facade，封装 `deckgym_ptcg.PtcgEnv`，提供 `init_game()`、`step()`、`get_state()`、`get_num_players()`(2)、`get_num_actions()`(15) 等接口 |
| `rlcard/agents/ptcg_simplebot_agent.py` | ~75 | 启发式 baseline agent，优先级：attack > attach_energy > evolve > play_supporter > play_item > play_stadium > retreat > use_ability > pass_turn |
| `examples/run_ptcg_selfplay.py` | ~120 | 自对弈训练脚本，支持 `--algorithm dqn\|nfsp`、`--deck-a`、`--deck-b`、`--opponent self\|random\|simplebot`、`--episodes`、`--eval-every`、`--log-dir` 等参数 |
| `examples/evaluate_ptcg.py` | ~90 | 评估脚本，支持 `--num-games`、`--opponent random\|simplebot`、`--checkpoint` 等参数 |
| `tests/envs/test_ptcg_env.py` | ~200 | 12 个 pytest 测试用例 |
| `docs/ptcg-rl-implementation.md` | 本文档 | 实施记录 |

#### 修改文件

| 文件 | 改动 |
|------|------|
| `rlcard/__init__.py` | 更新版本号 `2.0.0`，包名改为 `rlcard-ptcg` |
| `rlcard/envs/__init__.py` | 移除所有旧 env 注册，只保留 `ptcg` env |
| `rlcard/agents/__init__.py` | 移除 CFR、DMC、Human agents 引用 |
| `rlcard/agents/random_agent.py` | 修复 `eval_step` 的 `raw_legal_actions` 处理（适配 PTCG 的 dict 格式） |
| `rlcard/utils/__init__.py` | 移除 `pettingzoo_utils` 导入 |
| `rlcard/utils/utils.py` | 删除扑克相关函数（`init_standard_deck`、`init_54_deck`、`rank2int`、`elegent_form`、`print_card`）；`get_device` 改为可选依赖 torch；修复 `tournament` 函数 |
| `setup.py` | 包名 `rlcard-ptcg`、添加 `deckgym-ptcg` 依赖、升级 `requires-python >= 3.9` |

#### 删除

| 路径 | 数量 |
|------|------|
| `rlcard/envs/` 非 PTCG 文件 | 8 个（blackjack, bridge, doudizhu, gin_rummy, leducholdem, limitholdem, mahjong, nolimitholdem, uno） |
| `rlcard/games/` 非 PTCG 目录 | 9 个 |
| `rlcard/agents/` 非 RL 文件 | human_agents/, dmc_agent/, cfr_agent.py, pettingzoo_agents.py |
| `examples/` 旧示例 | evaluate.py, human/, pettingzoo/, run_cfr.py, run_dmc.py, run_random.py, run_rl.py, scripts/ |
| `tests/` 旧测试 | agents/, envs/, games/, models/, utils/ |

## 核心数据结构

### Observation（观测）

```json
{
    "player": 0,
    "turn": 3,
    "phase": "PlayerTurn",
    "active_player": 0,
    "public_features": [0.003, 1.0, 0.0, 1.0, ...],  // 所有玩家共享
    "private_features": [0.142, 0.0, 0.285, ...],      // 仅当前玩家
    "action_mask": [true, true, false, ...],            // 长度 15
    "legal_action_ids": ["pass-turn", "play-basic:hand:0:bench:0", ...],
    "raw_public_state": { ... }                         // 隐藏信息安全的 JSON
}
```

### LegalActionView（合法动作视图）

```json
{
    "id": "play-basic:hand:0:bench:0",
    "template_id": 1,
    "actor": 0,
    "label": "play 蓋諾賽克特 SV6a-040 to bench 1",
    "kind": "play_basic",
    "source_card": "蓋諾賽克特 SV6a-040",
    "target": "bench-0",
    "payload": {"card_name": "...", "card_id": "...", "hand_index": 0}
}
```

### Template ID 映射（15 种粗粒度动作）

| ID | 名称 | 描述 |
|----|------|------|
| 0 | pass_turn | 结束回合 |
| 1 | play_basic | 从手牌放置基础宝可梦到备战区 |
| 2 | evolve | 进化场上宝可梦 |
| 3 | attach_energy_active | 给战斗区宝可梦附加能量 |
| 4 | attach_energy_bench | 给备战区宝可梦附加能量 |
| 5 | attach_tool | 附加道具 |
| 6 | play_item | 使用物品卡 |
| 7 | play_supporter | 使用支援者卡 |
| 8 | play_stadium | 放置竞技场卡 |
| 9 | retreat | 撤退 |
| 10 | attack | 攻击 |
| 11 | use_ability | 使用特性 |
| 12 | use_stadium | 使用竞技场效果 |
| 13 | use_trainer_in_play | 使用场上训练家效果（VSTAR 等） |
| 14 | resolve_choice | 处理选择提示 |

### Action ID 格式

```
pass-turn
play-basic:hand:<idx>:bench:<idx>
evolve:hand:<idx>:target:<slot>           # slot = active | bench-<idx>
attach-energy:hand:<idx>:target:<slot>
attach-tool:hand:<idx>:target:<slot>
play-trainer:hand:<idx>:target:<slot>     # slot = active | bench-<idx> | none
play-stadium:hand:<idx>
retreat:bench:<idx>
attack:<idx>
ability:<slot>:<power_idx>
stadium:use
trainer-in-play:<slot>:<trainer_idx>
choice:<option>                           # option = index:<n> | bool:true|false | none
```

### RLCard State Dict

```python
{
    "obs": np.ndarray,           # 拼接 public_features + private_features + action_mask
    "legal_actions": OrderedDict({template_id: None, ...}),
    "raw_obs": dict,             # JSON observation
    "raw_legal_actions": [dict], # LegalActionView 列表
    "action_record": [],         # [(player_id, action_id), ...]
    "action_mask": np.ndarray,   # bool 数组，长度 15
}
```

## 隐藏信息安全策略

训练观测默认隐藏以下信息：

| 区域 | 己方可见 | 对手可见 |
|------|----------|----------|
| 手牌 | 完整卡牌列表 | 仅数量 |
| 牌库 | 仅数量 | 仅数量 |
| 奖赏卡 | 仅剩余数量 | 仅剩余数量 |
| 弃牌区 | 完整卡牌列表 | 完整卡牌列表 |
| 放逐区 | 完整卡牌列表 | 完整卡牌列表 |
| 战斗区/备战区 | 完整状态（HP、伤害、能量、阶段、异常状态） | 完整状态 |

## 公开特征维度

### public_features (~54 维)

- 对局特征 (4): turn 归一化、phase PlayerTurn/BetweenTurns、is_my_turn
- 己方区域计数 (6): deck/hand/discard/lost_zone/prizes/bench 归一化
- 对手区域计数 (6): 同上
- 己方棋盘特征 (15): 战斗区 HP/伤害/能量/阶段/异常状态/道具数 + 备战区平均 HP/平均伤害/数量
- 对手棋盘特征 (15): 同上

### private_features (~8 维)

- 手牌类型桶 (7): pokemon_basic/pokemon_evo/energy/trainer_item/trainer_supporter/trainer_stadium/trainer_tool 计数归一化

### action_mask (15 维)

- 15 个 template 的 bool mask

**总计 obs 维度**：由运行时 observation 决定，当前在目标牌组冒烟环境中观察为 68 维 `float32`。训练脚本必须从 `env.reset()` 的首个 state 推断 `state_shape=[len(state["obs"])]`。

## 环境配置

```python
import rlcard
from rlcard.agents import RandomAgent

env = rlcard.make("ptcg", config={
    "deck_a": "/path/to/deck_a.txt",    # 必需
    "deck_b": "/path/to/deck_b.txt",    # 必需
    "seed": 42,                          # 随机种子
    "max_ply": 10000,                    # 单局最大 ply
    "hide_private_information": True,    # 隐藏对手私密信息
    "record_trace": False,               # 记录每步 trace
    "illegal_action_policy": "error",   # "error" | "sample_legal"
})

env.set_agents([RandomAgent(15), RandomAgent(15)])
trajectories, payoffs = env.run(is_training=True)
```

## 构建与运行

```bash
# 1. 构建 Rust PyO3 包
cd /Users/easygod/code/deckgym-core
python3 -m venv .venv
.venv/bin/pip install maturin
.venv/bin/python -m maturin develop

# 2. 安装 rlcard
/Users/easygod/code/deckgym-core/.venv/bin/pip install -e /Users/easygod/code/rlcard

# 3. 运行测试
cd /Users/easygod/code/rlcard
/Users/easygod/code/deckgym-core/.venv/bin/python -m pytest tests/envs/test_ptcg_env.py -v

# 4. 评估
/Users/easygod/code/deckgym-core/.venv/bin/python examples/evaluate_ptcg.py \
    --deck-a "/Users/easygod/Downloads/Battle Subway 北京沙.txt" \
    --deck-b "/Users/easygod/Downloads/龙神柱 Battle Subway.txt" \
    --num-games 100 --opponent random

# 5. 训练（需要 PyTorch）
/Users/easygod/code/deckgym-core/.venv/bin/pip install torch matplotlib
/Users/easygod/code/deckgym-core/.venv/bin/python examples/run_ptcg_selfplay.py \
    --algorithm dqn \
    --deck-a "/path/to/deck_a.txt" \
    --deck-b "/path/to/deck_b.txt" \
    --episodes 1000 --eval-every 100 \
    --log-dir experiments/ptcg-dqn-run1
```

当前注意事项：

- 若使用系统 `python3`，可能会因为未安装 `deckgym_ptcg` 而无法 import `rlcard`。
- 在训练脚本修复前，DQN/NFSP 不能依赖 `state_shape=None` 自动推断观测维度。
- 普通 Rust 测试需要先修复 PyO3 `extension-module` 链接配置；在此之前优先用 Python env 测试验证绑定行为。

## 测试覆盖

| 测试 | 文件 | 说明 |
|------|------|------|
| test_action_id_roundtrip | Rust | action_id → Action → action_id 往返 |
| test_env_creation | Python | env 创建成功，num_players=2, num_actions=15 |
| test_reset | Python | reset 返回合法 state 和 player_id |
| test_legal_actions_not_empty | Python | 初始状态有合法动作 |
| test_step_cycle | Python | 连续 10 步不报错 |
| test_random_vs_random_one_game | Python | 单局完成 |
| test_random_vs_random_100_games | Python | 100 局无报错 |
| test_hidden_info | Python | 对手手牌/牌库/奖赏卡不含具体 card_id |
| test_illegal_action_error | Python | 默认策略下非法动作报错 |
| test_illegal_action_sample_legal | Python | sample_legal 策略下降级到合法动作 |
| test_payoffs | Python | 对局结束 payoff 正确 (+1/-1/0) |
| test_action_id_roundtrip | Python | 每个合法 action_id 可执行 |
| test_trace_recording | Python | trace 记录每步动作 |

## 后续升级方向

1. **特征工程细化**: 加入 card embedding、logic_key one-hot（目标牌组专用）
2. **动作模型升级**: 从 deterministic tie-break 升级为 action-parameter head 或 pointer network
3. **算法升级**: 从 DQN/NFSP 升级到 PPO/actor-critic 自对弈
4. **并行 rollout**: 使用 Rust rayon 并行评估，提升采样效率
5. **League/self-play pool**: 避免策略退化
6. **Replay 可视化**: 基于 deckgym replay JSON 的 UI 面板
7. **人类对局 imitation learning**: 导入人类 replay 数据

---

*最后更新: 2026-05-03*
