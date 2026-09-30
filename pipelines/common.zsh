# Shared paths for the localized historical Zsh recipes. Source this file;
# it does not launch a model or a child pipeline.
export FORGE_ROOT="${${(%):-%x}:A:h:h}"
export FORGE_PIPELINE_DIR="$FORGE_ROOT/pipelines/m3u"
export FORGE_WORK_DIR="${FORGE_WORK_DIR:-$HOME/Models/anemll-forge/experiments}"
export FORGE_MODEL_DIR="${FORGE_MODEL_DIR:-${MODEL:-$HOME/Models/Qwen3.8-27B}}"
export FORGE_DFLASH_WORK_DIR="${FORGE_DFLASH_WORK_DIR:-${WORK:-$FORGE_WORK_DIR/dflash2}}"
export FORGE_DRAFTER_DIR="${FORGE_DRAFTER_DIR:-${DRAFTER:-$HOME/Models/DFlash2-27B}}"
export FORGE_PYTHON="${FORGE_PYTHON:-$FORGE_ROOT/.venv/bin/python}"
if [[ ! -x "$FORGE_PYTHON" ]]; then
  print -u2 -- "Missing Python: $FORGE_PYTHON. Set FORGE_PYTHON to your prepared environment."
  return 1
fi

export MODEL="$FORGE_MODEL_DIR" DRAFTER="$FORGE_DRAFTER_DIR"
export WIKI="${WIKI:-$FORGE_WORK_DIR/wikitext}"
export TRACE="${TRACE:-$FORGE_WORK_DIR/kl}" OUT="${OUT:-$FORGE_WORK_DIR/runs}"
export WORK="$FORGE_DFLASH_WORK_DIR"
export KL_TRACE="${KL_TRACE:-$TRACE/trace.npz}"
export HEAD_EXPORT="${HEAD_EXPORT:-$OUT/export/mix25in_mixr/lm_head.safetensors}"
export REF_CODE="${REF_CODE:-$FORGE_ROOT/vendor/dflash_reference}"
mkdir -p "$FORGE_WORK_DIR" "$TRACE" "$OUT" "$WORK" || return 1
