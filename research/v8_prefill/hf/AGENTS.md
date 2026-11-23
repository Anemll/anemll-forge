# AGENTS.md: download, compile, deploy and test (for coding agents)

You are setting up the research build of Qwen3.8-27B that runs on the Neural Engine and the GPU of an **M6 Mac**
together (`coreai/m6-96k-dec8ts`, tree verification, 96K context) and serves an OpenAI-compatible API on
`127.0.0.1:8765`. One script does the work in stages: `code/anemll-forge/research/v8_prefill/hf/agent_deploy.sh`.
[DEPLOY.md](DEPLOY.md) explains each step; read it when a stage fails.

## Rules

1. **Never change system security yourself.** The build needs System Integrity Protection off and a read ACL on
   `/Library/Caches/com.apple.aned`. When `check` prints `HUMAN` lines (exit 10), stop and give those lines to the
   person verbatim. Disabling SIP needs macOS Recovery and lowers the security of the whole Mac; the ACL needs `sudo`.
   Do not run `sudo`, `csrutil` or `chmod` on system paths unless the person tells you to in this session.
2. **Never delete caches.** Do not remove anything under `/Library/Caches/com.apple.aned` or
   `~/Library/Caches/coreai-cache`; the GPU tables point into those compiled programs.
3. **One model on the Neural Engine at a time.** Do not compile or serve while another Neural Engine server or job is
   running; `compile` and `serve` refuse when the port is taken. Ask before stopping a server you did not start.
4. **One compile at a time**, and keep at least 25 GB free on the internal disk (stages exit 3 below it).
5. **Use the venv's `python`** (file name exactly `python`, default `$REPO/.venv/bin/python`): the compile cache and
   the GPU-table lookup are keyed by the executable name.
6. **Do not rebuild the engine dylibs while a server has them loaded** (`engines` refuses).
7. **Stop at the first non-zero exit** and report the stage, the exit code and the last lines it printed. Do not
   work around a failed check.
8. Do not upload logs or replies anywhere; they stay on this Mac.

## Bootstrap

```
export REPO=~/Models/anemll-forge-qwen3.8-27B-V8-96K
python3 -m pip install --user 'huggingface_hub>=1.0'        # only if `hf` is missing
hf download anemll/anemll-forge-qwen3.8-27B-V8-96K --local-dir $REPO --include "code/*" "*.md"
A=$REPO/code/anemll-forge/research/v8_prefill/hf/agent_deploy.sh
zsh $A check
```

## Stages

Run them in this order (`zsh $A all` runs them in sequence and stops at the first failure):

| stage | does | time | pass means |
|---|---|---|---|
| `check` | chip, memory, macOS, SIP, aned ACL, swiftc, disk, python, port: read-only | seconds | `STAGE check: PASS` (TODO lines are fine before `download` / `venv`) |
| `download` | the build, GPU-table kit, drafter, model files, code (about 15 GB) | network | files present |
| `venv` | `$REPO/.venv` with `requirements-inference.txt` (Python 3.11) | 5 min | imports work |
| `bridge` | builds the Core AI Swift bridge | 1 min | |
| `compile` | `forge.py compile` of the 8 packages, head and drafter for this Mac's Neural Engine | about 30 min | `done in ...` |
| `engines` | builds `libgpu_v8_f8q_uw.dylib`, `libdec_gpu_tree_uw.dylib` | 1 min | API check passes |
| `tables` | installs the GPU tables: every package's compiled program must match the kit (`MATCH`) | seconds | 8 x MATCH |
| `serve` | starts the server in the background, waits for it, checks the configuration banners | seconds to 2 min | `STAGE serve: PASS (pid ...)` |
| `smoke` | 2 greedy requests; the output checksum must be `29aef2ec9db18a3f` | 1 min | `STAGE smoke: PASS` |
| `status` / `stop` | health check / stop the server on `$PORT` | seconds | |

## Exit codes

| code | meaning | what to do |
|---|---|---|
| 0 | pass | next stage |
| 1 | failure | report the stage's last lines and its log in `~/Library/Logs/anemll-forge/agent_<stage>.log` |
| 2 | `tables`: this Mac's compiled programs differ from the kit's (another macOS build) | ask the person whether to run `tables-rebuild` (25 to 40 min, needs the release export and the builder environment: DEPLOY.md 5b) |
| 3 | internal disk below 25 GB | ask the person to free space; delete nothing yourself |
| 4 | port in use / a server has the dylibs loaded | ask the person; do not kill unknown processes |
| 5 | `smoke`: checksum differs | on macOS 26C5047e: a setup problem (report it); on another build: compiler numerics, report the replies |
| 10 | a human must act (SIP, ACL) | stop; give the `HUMAN` lines to the person |

## Using it

```
curl -s http://127.0.0.1:8765/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"qwen38-27b-ane","messages":[{"role":"user","content":"Hello"}],"max_tokens":64}'
```

Streaming, tool calls and `chat_template_kwargs: {"enable_thinking": true|false}` are supported. The server log is
`~/Library/Logs/anemll-forge/v8_server.log`. The memory watchdog stops the server above 12 GB of footprint
(`FORGE_FP_MAX_MB`); restart it with `zsh $A serve`. After a macOS update, run `compile`, `tables`, `serve` and
`smoke` again.
