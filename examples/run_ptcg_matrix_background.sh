#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/Users/easygod/code/rlcard"
PYTHON_BIN="${PYTHON_BIN:-/Users/easygod/code/deckgym-core/.venv/bin/python}"

AGENT="${AGENT:-nfsp}"
OPPONENT="${OPPONENT:-simplebot}"
CHECKPOINT="${CHECKPOINT:-/Users/easygod/code/rlcard/experiments/ptcg-nfsp-tiebreaker-test50/final}"
OPPONENT_CHECKPOINT="${OPPONENT_CHECKPOINT:-}"
NUM_GAMES="${NUM_GAMES:-1000}"
SEED="${SEED:-42}"
MAX_PLY="${MAX_PLY:-10000}"
RAW_ACTION_TIE_BREAKER="${RAW_ACTION_TIE_BREAKER:-heuristic}"
RUN_NAME="${RUN_NAME:-}"
DECK_POOL="${DECK_POOL:-}"
SPLIT="${SPLIT:-train}"

DECK_BEIJINGSHA="${DECK_BEIJINGSHA:-/Users/easygod/Downloads/Battle Subway 北京沙.txt}"
DECK_LONGSHENZHU="${DECK_LONGSHENZHU:-/Users/easygod/Downloads/龙神柱 Battle Subway.txt}"
DECK_MENGLEIGU="${DECK_MENGLEIGU:-/Users/easygod/Downloads/猛雷鼓 (1).txt}"

usage() {
  cat <<'USAGE'
Run the fixed 3x3 PTCG deck evaluation matrix in the background.

Environment overrides:
  AGENT=dqn|nfsp|ppo|random|simplebot
  CHECKPOINT=/path/to/checkpoint_dir_or_pt
  OPPONENT=random|simplebot|dqn|nfsp|ppo
  OPPONENT_CHECKPOINT=/path/to/opponent_checkpoint
  DECK_POOL=configs/ptcg_deck_pool.json
  SPLIT=train|validation|holdout|all
  NUM_GAMES=1000
  SEED=42
  RAW_ACTION_TIE_BREAKER=heuristic|first
  RUN_NAME=optional-readable-name
  PYTHON_BIN=/Users/easygod/code/deckgym-core/.venv/bin/python

Examples:
  AGENT=dqn CHECKPOINT=experiments/ptcg-dqn-run1/final OPPONENT=random NUM_GAMES=1000 \
    examples/run_ptcg_matrix_background.sh

  AGENT=nfsp CHECKPOINT=experiments/ptcg-nfsp-run1/final OPPONENT=simplebot NUM_GAMES=500 \
    RUN_NAME=nfsp-vs-simplebot-500 examples/run_ptcg_matrix_background.sh

Outputs:
  experiments/ptcg-matrix-<run>/command.txt
  experiments/ptcg-matrix-<run>/pid
  experiments/ptcg-matrix-<run>/stdout.log
  experiments/ptcg-matrix-<run>/results.csv
  experiments/ptcg-matrix-<run>/results.json
USAGE
}

quote_for_runner() {
  local value="$1"
  value="${value//\\/\\\\}"
  value="${value//\"/\\\"}"
  value="${value//\$/\\\$}"
  value="${value//\`/\\\`}"
  printf '"%s"' "$value"
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python not executable: $PYTHON_BIN" >&2
  exit 1
fi

if [[ -n "$DECK_POOL" ]]; then
  if [[ ! -f "$DECK_POOL" ]]; then
    echo "Deck pool not found: $DECK_POOL" >&2
    exit 1
  fi
else
  for deck in "$DECK_BEIJINGSHA" "$DECK_LONGSHENZHU" "$DECK_MENGLEIGU"; do
    if [[ ! -f "$deck" ]]; then
      echo "Deck not found: $deck" >&2
      exit 1
    fi
  done
fi

if [[ "$AGENT" =~ ^(dqn|nfsp|ppo)$ && -z "$CHECKPOINT" ]]; then
  echo "CHECKPOINT is required when AGENT is $AGENT" >&2
  exit 1
fi

if [[ "$OPPONENT" =~ ^(dqn|nfsp|ppo)$ && -z "$OPPONENT_CHECKPOINT" ]]; then
  echo "OPPONENT_CHECKPOINT is required when OPPONENT is $OPPONENT" >&2
  exit 1
fi

timestamp="$(date +%Y%m%d-%H%M%S)"
if [[ -z "$RUN_NAME" ]]; then
  RUN_NAME="${AGENT}-vs-${OPPONENT}-${RAW_ACTION_TIE_BREAKER}-${NUM_GAMES}g-seed${SEED}-${timestamp}"
fi

LOG_DIR="${LOG_DIR:-$ROOT_DIR/experiments/ptcg-matrix-${RUN_NAME}}"
mkdir -p "$LOG_DIR"

cmd=(
  "$PYTHON_BIN" "$ROOT_DIR/examples/evaluate_ptcg_matrix.py"
  --agent "$AGENT"
  --opponent "$OPPONENT"
  --num-games "$NUM_GAMES"
  --seed "$SEED"
  --max-ply "$MAX_PLY"
  --raw-action-tie-breaker "$RAW_ACTION_TIE_BREAKER"
  --log-dir "$LOG_DIR"
)

if [[ -n "$DECK_POOL" ]]; then
  cmd+=(--deck-pool "$DECK_POOL" --split "$SPLIT")
else
  cmd+=(
    --deck "beijingsha=$DECK_BEIJINGSHA"
    --deck "longshenzhu=$DECK_LONGSHENZHU"
    --deck "mengleigu=$DECK_MENGLEIGU"
  )
fi

if [[ -n "$CHECKPOINT" ]]; then
  cmd+=(--checkpoint "$CHECKPOINT")
fi

if [[ -n "$OPPONENT_CHECKPOINT" ]]; then
  cmd+=(--opponent-checkpoint "$OPPONENT_CHECKPOINT")
fi

{
  printf 'cwd=%s\n' "$ROOT_DIR"
  printf 'started_at=%s\n' "$(date '+%Y-%m-%dT%H:%M:%S%z')"
  printf 'command_args:\n'
  for arg in "${cmd[@]}"; do
    printf '  %s\n' "$arg"
  done
} > "$LOG_DIR/command.txt"

runner="$LOG_DIR/run.sh"
{
  printf '#!/usr/bin/env bash\n'
  printf 'set -o pipefail\n'
  printf 'cd %s\n' "$(quote_for_runner "$ROOT_DIR")"
  printf 'echo "status=running" > %s\n' "$(quote_for_runner "$LOG_DIR/status.txt")"
  printf "echo \"process_started_at=\$(date '+%%Y-%%m-%%dT%%H:%%M:%%S%%z')\" >> %s\n" "$(quote_for_runner "$LOG_DIR/status.txt")"
  for arg in "${cmd[@]}"; do
    printf '%s ' "$(quote_for_runner "$arg")"
  done
  printf '\n'
  printf 'code=$?\n'
  printf 'echo "$code" > %s\n' "$(quote_for_runner "$LOG_DIR/exit_code")"
  printf "echo \"finished_at=\$(date '+%%Y-%%m-%%dT%%H:%%M:%%S%%z')\" >> %s\n" "$(quote_for_runner "$LOG_DIR/status.txt")"
  printf 'echo "exit_code=$code" >> %s\n' "$(quote_for_runner "$LOG_DIR/status.txt")"
  printf 'exit "$code"\n'
} > "$runner"
chmod +x "$runner"

(
  cd "$ROOT_DIR"
  nohup "$runner" > "$LOG_DIR/stdout.log" 2>&1 < /dev/null &
  pid=$!
  disown "$pid" 2>/dev/null || true
  echo "$pid" > "$LOG_DIR/pid"
)

pid="$(cat "$LOG_DIR/pid")"
echo "Started PTCG matrix evaluation in background."
echo "PID: $pid"
echo "Log dir: $LOG_DIR"
echo "Live log: tail -f '$LOG_DIR/stdout.log'"
echo "Results: $LOG_DIR/results.csv"
