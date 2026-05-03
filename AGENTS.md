# Coding Guidelines

本仓库是 `rlcard-ptcg` 训练层，用于把 deckgym-core 的 PTCG 规则引擎接入 RLCard 风格的强化学习、评估、日志和实验流程。

## Project Direction

- `/Users/easygod/code/deckgym-core/src/ptcg` 是唯一 PTCG 规则引擎。
- 本仓库只负责 Python/RLCard 适配、agent、训练脚本、评估脚本、日志、trace 和实验编排。
- Battle-Subway at `/Users/easygod/code/Battle-Subway` 只作为规则迁移、simplebot 思路和 debug UI 参考，不作为训练运行时依赖。
- 首期目标是固定牌组矩阵上的强基线和可复现实验，再逐步逼近人类水平。
- 默认目标牌组：
  - `/Users/easygod/Downloads/Battle Subway 北京沙.txt`
  - `/Users/easygod/Downloads/龙神柱 Battle Subway.txt`
  - `/Users/easygod/Downloads/猛雷鼓 (1).txt`

## Boundaries

- 不在 Python、CLI、UI、训练脚本或测试中实现 PTCG 卡牌规则、合法性判断或状态推进。
- 不把 Battle-Subway TypeScript 嵌入训练循环或作为运行时决策依赖。
- 不泄漏隐藏信息：训练 observation 不能包含对手手牌、对手牌库顺序、对手奖赏卡具体内容。
- 不直接枚举完整 PTCG 全局组合动作表；Python 侧继续使用 Rust 动态合法动作和固定 template mask。
- 不把文档里的“模型接近人类水平”写成已达成结论，除非有固定评估协议和结果支持。
- 不回滚无关用户改动；这个工作树长期可能是脏的。

## RLCard Interface

- `rlcard.make("ptcg", config=...)` 是 Python 侧入口。
- `legal_actions` 保持 `OrderedDict[int, None]`，key 是粗粒度 template id。
- `raw_legal_actions` 保持 Rust `LegalActionView` 列表，包含具体 `action_id`、`kind`、`payload` 等调试信息。
- `obs` 是 `public_features + private_features + action_mask` 拼接后的 `np.ndarray`。
- 训练脚本必须从 `env.reset()` 的首个 state 推断 `state_shape=[len(state["obs"])]`，不要假设 `state_shape=None` 会自动生效。
- raw action agent（例如 `PtcgSimpleBotAgent`）必须设置 `use_raw=True`，神经网络 agent 使用 template id 并设置 `use_raw=False`。

## Default Workflow

1. 先阅读相关文档：
   - `/Users/easygod/code/deckgym-core/.codex/ptcg-rl-design.md`
   - `/Users/easygod/code/deckgym-core/AGENTS.md`
   - `docs/ptcg-rl-implementation.md`
2. 如需验证 PyO3 绑定，先在 deckgym-core 构建：

```bash
cd /Users/easygod/code/deckgym-core
.venv/bin/python -m maturin develop
```

3. 在同一个 venv 中安装/验证 rlcard：

```bash
cd /Users/easygod/code/rlcard
/Users/easygod/code/deckgym-core/.venv/bin/pip install -e .
/Users/easygod/code/deckgym-core/.venv/bin/python -m pytest tests/envs/test_ptcg_env.py -q
```

4. 做评估冒烟：

```bash
cd /Users/easygod/code/rlcard
/Users/easygod/code/deckgym-core/.venv/bin/python examples/evaluate_ptcg.py \
  --deck-a "/Users/easygod/Downloads/Battle Subway 北京沙.txt" \
  --deck-b "/Users/easygod/Downloads/龙神柱 Battle Subway.txt" \
  --num-games 3 \
  --opponent random
```

## Validation Priorities

- PTCG env smoke：`tests/envs/test_ptcg_env.py` 必须能用 deckgym-core venv 通过。
- 训练闭环：DQN/NFSP 初始化时必须显式使用运行时观测维度。
- 评估闭环：固定 seed、固定牌组、固定 opponent，输出 win rate、tie rate、average ply、illegal/fallback count、action template distribution。
- 长期强模型路线：先 simplebot/heuristic/checkpoint baseline，再 PPO 或 actor-critic、并行 rollout、league self-play、checkpoint pool。

