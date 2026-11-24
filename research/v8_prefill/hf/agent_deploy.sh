#!/bin/zsh
# Staged download / build / deploy / test of the V8-96K tree8 + sliced build, for coding agents and people
# (22 Nov 2026; AGENTS.md has the rules, DEPLOY.md the explanations). Every stage is idempotent, prints one
# "STAGE <name>: PASS | FAIL | HUMAN ..." line at the end and exits with:
#   0 pass   1 fail   2 GPU tables do not match this Mac (run tables-rebuild)   3 disk below the guard
#   4 port busy   5 smoke checksum differs   10 a human must act (SIP, the aned ACL: needs Recovery or sudo)
#   zsh agent_deploy.sh check|download|venv|bridge|compile|engines|tables|tables-rebuild|serve|smoke|pi|status|stop|all
# pi: the Pi profile + TPS footer (examples/pi/install_pi_v8_96k.py; PI_ARGS, e.g. "--set-default"; backs up what it changes)
# Variables: REPO (~/Models/anemll-forge-qwen3.8-27B-V8-96K), PY ($REPO/.venv/bin/python: its file name must be
# "python", the compile cache is keyed by it), PORT (8765), LOGDIR (~/Library/Logs/anemll-forge), GUARD_GB (25),
# tables-rebuild: EXPORT_DIR, MODEL, BPY (DEPLOY.md 5b).
# VARIANT=sa (24 Nov 2026): the super-adapter repo anemll/anemll-forge-qwen3.8-27B-V8-96K-SuperAdapter (build
# coreai/m6-96k-dec8ts-sa, kit gpu/w_dec8ts_sa, its own smoke checksum); default: the V8-96K repo.
set -u
if [ "${VARIANT:-base}" = sa ]; then
  HF_DEF=anemll/anemll-forge-qwen3.8-27B-V8-96K-SuperAdapter; BN=m6-96k-dec8ts-sa; KN=w_dec8ts_sa; SMOKE_DEF=9756d8d8e446b83f
else
  HF_DEF=anemll/anemll-forge-qwen3.8-27B-V8-96K; BN=m6-96k-dec8ts; KN=w_dec8ts; SMOKE_DEF=29aef2ec9db18a3f
fi
REPO=${REPO:-$HOME/Models/${HF_DEF:t}}; REPO=${REPO:a}
CODE=$REPO/code/anemll-forge; V=$CODE/research/v8_prefill
PY=${PY:-$REPO/.venv/bin/python}; PORT=${PORT:-8765}; LOGDIR=${LOGDIR:-$HOME/Library/Logs/anemll-forge}
GUARD_GB=${GUARD_GB:-25}; BUILD=$REPO/coreai/$BN; KIT=$REPO/gpu/$KN; TABLES=${TABLES:-$REPO/gpu/${KN}_local}
SMOKE=${SMOKE:-$SMOKE_DEF}; KIT_OS=26C5047e; HF_ID=${HF_ID:-$HF_DEF}
DRAFT=$REPO/drafter/all3/dflash2_lut4_gptq_nohead.aimodel
mkdir -p $LOGDIR
say() { print -r -- "$*"; }
done_() { say "STAGE $1: $2"; exit ${3:-0}; }
free_gb() { df -k / | awk 'NR==2 {printf "%d", $4 / 1048576}'; }
port_busy() { lsof -nP -iTCP:$PORT -sTCP:LISTEN >/dev/null 2>&1; }

check() {
  local bad=0 human=0 n
  n=$(sysctl -n machdep.cpu.brand_string 2>/dev/null)
  [[ $n == *M6* ]] && say "ok    chip: $n" || { say "FAIL  chip: $n (the build needs an M6: FP8 attention)"; bad=1; }
  n=$(( $(sysctl -n hw.memsize) / 1073741824 ))
  (( n >= 32 )) && say "ok    memory: ${n} GB" || { say "FAIL  memory: ${n} GB (32 GB needed)"; bad=1; }
  n=$(sw_vers -productVersion); local b=$(sw_vers -buildVersion)
  [[ $n == 27.3* ]] && say "ok    macOS $n ($b)" || say "WARN  macOS $n ($b): tested on 27.3 only"
  [ $b = $KIT_OS ] || say "WARN  macOS build $b != the kit's $KIT_OS: expect stage tables to exit 2 (then tables-rebuild)"
  if csrutil status 2>/dev/null | grep -q disabled; then say "ok    SIP disabled"
  else say "HUMAN SIP is enabled. A person must: restart into macOS Recovery, Terminal: csrutil disable, restart (README: Requirements, root and SIP). It lowers the security of the whole Mac."; human=1; fi
  if ls /Library/Caches/com.apple.aned >/dev/null 2>&1; then say "ok    aned cache readable"
  else say "HUMAN aned cache not readable. A person must run once (sudo): sudo chmod -R +a \"user:$(id -un) allow list,search,read,readattr,readextattr,readsecurity,file_inherit,directory_inherit\" /Library/Caches/com.apple.aned"; human=1; fi
  xcrun --find swiftc >/dev/null 2>&1 && say "ok    swiftc $(xcrun swiftc --version 2>/dev/null | head -1 | cut -c1-60)" || { say "FAIL  no swiftc: xcode-select --install"; bad=1; }
  n=$(free_gb); (( n >= GUARD_GB )) && say "ok    internal disk free ${n} GB" || { say "FAIL  internal disk free ${n} GB < ${GUARD_GB} GB (compile caches)"; bad=1; }
  if [ -x $PY ]; then
    [ ${PY:t} = python ] || say "WARN  PY file name is ${PY:t}: use .../bin/python (the compile cache and the table lookup are keyed by \"python\")"
    $PY -c "import numpy, tokenizers, huggingface_hub" 2>/dev/null && say "ok    python $PY" || say "TODO  python $PY lacks packages: stage venv"
  else say "TODO  no $PY: stages download, then venv"; fi
  [ -e $BUILD/manifest.json ] && say "ok    build downloaded" || say "TODO  build not downloaded: stage download"
  port_busy && say "WARN  port $PORT in use ($(lsof -tnP -iTCP:$PORT -sTCP:LISTEN | head -1)): stage serve needs it free" || say "ok    port $PORT free"
  (( human )) && done_ check "HUMAN (see the HUMAN lines; nothing was changed)" 10
  (( bad )) && done_ check FAIL 1
  done_ check PASS
}

download() {
  local HF=$(command -v hf || true); [ -x $REPO/.venv/bin/hf ] && HF=$REPO/.venv/bin/hf
  [ -n "$HF" ] || done_ download "FAIL: no hf CLI (python3 -m pip install --user 'huggingface_hub>=1.0', then retry)" 1
  mkdir -p $REPO
  $HF download $HF_ID --local-dir $REPO --include "coreai/$BN/*" "gpu/$KN/*" "drafter/*" "model/*" \
    "code/*" "*.md" "*.json" LICENSE NOTICE || done_ download FAIL 1
  [ -e $BUILD/manifest.json ] && [ -e $KIT/kit.json ] && [ -e $V/hf/serve_v8_96k.sh ] || done_ download "FAIL: files missing after download" 1
  done_ download "PASS ($(du -sh $REPO | cut -f1))"
}

venv() {
  local P311=$(command -v python3.11 || command -v python3)
  [ -x $PY ] || $P311 -m venv $REPO/.venv || done_ venv FAIL 1
  $PY -m pip install -q --upgrade pip && $PY -m pip install -q -r $CODE/requirements-inference.txt || done_ venv FAIL 1
  $PY -c "import numpy, tokenizers, huggingface_hub" || done_ venv FAIL 1
  done_ venv "PASS ($($PY --version))"
}

bridge() {
  (cd $CODE && bash coreai/swift_bridge/build.sh) > $LOGDIR/agent_bridge.log 2>&1 || done_ bridge "FAIL (log $LOGDIR/agent_bridge.log)" 1
  done_ bridge PASS
}

compile() {
  (( $(free_gb) >= GUARD_GB )) || done_ compile "FAIL: disk below ${GUARD_GB} GB" 3
  port_busy && done_ compile "FAIL: a server listens on $PORT (one model on the Neural Engine at a time: stop it first)" 4
  say "compiling (about 30 min; log $LOGDIR/agent_compile.log)"
  (cd $CODE && MPSGRAPH_ANE_BONDED_COMPILE_MODE=2 $PY forge.py compile --build $BUILD --draft $DRAFT) > $LOGDIR/agent_compile.log 2>&1 \
    || done_ compile "FAIL (log $LOGDIR/agent_compile.log)" 1
  grep "done in" $LOGDIR/agent_compile.log | tail -1
  done_ compile PASS
}

engines() {
  port_busy && pgrep -f "v8_server.py.*--port $PORT" >/dev/null && done_ engines "FAIL: a server on $PORT has the dylibs loaded: stop first" 4
  zsh $V/pipe/engine/build_engines.sh uw > $LOGDIR/agent_engines.log 2>&1 || done_ engines "FAIL (log $LOGDIR/agent_engines.log)" 1
  cat $LOGDIR/agent_engines.log
  done_ engines PASS
}

tables() {
  $PY $V/pipe/gpu_tables_kit.py install $KIT $BUILD $TABLES > $LOGDIR/agent_tables.log 2>&1; local rc=$?
  cat $LOGDIR/agent_tables.log
  (( rc == 2 )) && done_ tables "MISMATCH: this Mac's compiled programs differ from the kit's: run stage tables-rebuild" 2
  (( rc )) && done_ tables "FAIL (log $LOGDIR/agent_tables.log)" 1
  done_ tables "PASS ($TABLES)"
}

tables_rebuild() {
  [ -n "${EXPORT_DIR:-}" ] && [ -n "${MODEL:-}" ] && [ -n "${BPY:-}" ] || done_ tables-rebuild "FAIL: set EXPORT_DIR, MODEL, BPY (DEPLOY.md 5b)" 1
  BUILD=$BUILD OUT=$TABLES PY=$PY V8_V02=$CODE zsh $V/pipe/build_gpu_tables.sh > $LOGDIR/agent_tables_rebuild.log 2>&1 \
    || done_ tables-rebuild "FAIL (log $LOGDIR/agent_tables_rebuild.log)" 1
  $PY $V/pipe/gpu_tables_kit.py verify $TABLES $KIT | tail -3
  done_ tables-rebuild "PASS ($TABLES; differences from the kit above are expected in .tab files only)"
}

serve() {
  port_busy && done_ serve "FAIL: port $PORT in use" 4
  REPO=$REPO BUILD=$BUILD TABLES=$TABLES PORT=$PORT LOGDIR=$LOGDIR PY=$PY nohup zsh $V/hf/serve_v8_96k.sh > $LOGDIR/agent_serve.out 2>&1 &
  local L=$LOGDIR/v8_server.log i
  for i in $(seq 1 300); do
    grep -q "serving OpenAI API" $L 2>/dev/null && break
    grep -q "^Traceback\|missing " $LOGDIR/agent_serve.out $L 2>/dev/null && { tail -20 $LOGDIR/agent_serve.out $L; done_ serve FAIL 1; }
    kill -0 $! 2>/dev/null || { tail -20 $LOGDIR/agent_serve.out $L 2>/dev/null; done_ serve "FAIL: launcher exited" 1; }
    sleep 2
  done
  grep -q "serving OpenAI API" $L || done_ serve "FAIL: not serving after 600 s (log $L)" 1
  local ok=1
  for want in "libgpu_v8_f8q_uw.dylib" "libdec_gpu_tree_uw.dylib" "tree: BANNER mode=tree8 policy=branched" "slice: BANNER sliced decode entries [24, 40, 48, 56]K"; do
    grep -qF "$want" $L && say "ok    banner: $want" || { say "FAIL  banner missing: $want"; ok=0; }
  done
  (( ok )) || done_ serve "FAIL: unexpected configuration (log $L)" 1
  done_ serve "PASS (pid $(lsof -tnP -iTCP:$PORT -sTCP:LISTEN | head -1), http://127.0.0.1:$PORT/v1, log $L)"
}

smoke() {
  port_busy || done_ smoke "FAIL: nothing on $PORT (stage serve)" 1
  $PY $V/bench_tree/tree_bench.py --prompts $V/bench_tree/smoke_2k_v1.json --rungs 2k --tasks 1,8 --label agent_smoke \
    --out $LOGDIR/agent_smoke.json --log $LOGDIR/v8_server.log --idle-s 5 > $LOGDIR/agent_smoke.log 2>&1 || { tail -20 $LOGDIR/agent_smoke.log; done_ smoke FAIL 1; }
  local line=$(grep "SUMMARY" $LOGDIR/agent_smoke.log | tail -1); say "$line"
  [[ $line == *"output checksum $SMOKE"* ]] && done_ smoke "PASS (replies token for token as on the development M6)"
  done_ smoke "DIFFERS: expected checksum $SMOKE. On macOS $KIT_OS this points at a setup problem; on another build the compiler can change numerics (compare the replies in $LOGDIR/agent_smoke.json)" 5
}

pi() {
  python3 $CODE/examples/pi/install_pi_v8_96k.py --port $PORT ${=PI_ARGS:-} || done_ pi FAIL 1
  done_ pi "PASS (in Pi: /reload, /model -> ane-qwen38/qwen38-27b-ane-v8-96k)"
}

status() {
  port_busy || done_ status "not serving on $PORT" 1
  curl -s -m 5 http://127.0.0.1:$PORT/health | head -c 400; echo
  done_ status "serving (pid $(lsof -tnP -iTCP:$PORT -sTCP:LISTEN | head -1))"
}

stop() { PORT=$PORT zsh $V/pipe/srv_stop.sh; done_ stop PASS; }

case ${1:-} in
  check) check ;; download) download ;; venv) venv ;; bridge) bridge ;; compile) compile ;; engines) engines ;;
  tables) tables ;; tables-rebuild) tables_rebuild ;; serve) serve ;; smoke) smoke ;; pi) pi ;; status) status ;; stop) stop ;;
  all)
    for s in check download venv bridge compile engines tables serve smoke; do
      say "== $s"; zsh ${0:a} $s; rc=$?
      (( rc )) && { say "stopped at stage $s (exit $rc)"; exit $rc; }
    done ;;
  *) say "usage: zsh agent_deploy.sh check|download|venv|bridge|compile|engines|tables|tables-rebuild|serve|smoke|pi|status|stop|all"; exit 1 ;;
esac
