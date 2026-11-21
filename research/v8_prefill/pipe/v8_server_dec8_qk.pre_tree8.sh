#!/bin/zsh
# 19 Nov 2026 serving launcher: v8_server_dec8.sh (FP8 x prefill, decode keep-alive) + the staged Q.K epilogue
# (QKE=tg, attn_qk8isk_tg, bit-exact; results/19nov section 8) in libgpu_v8_f8q.dylib. Rollback: v8_server_dec8.sh
# (the plain launcher, libgpu_v8_f8.dylib, served attn_qk8isk), or QKE= with this one.
export V8_SRV_DYLIB=${V8_SRV_DYLIB:-libgpu_v8_f8q.dylib} QKE=${QKE-tg}
exec /Users/anemll/Documents/Codex/ios-prefill-20261030/v8_work/pipe/v8_server_dec8.sh "$@"
