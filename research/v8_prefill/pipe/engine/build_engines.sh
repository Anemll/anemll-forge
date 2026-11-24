#!/bin/zsh
# Build the GPU engine dylibs of the served V8 forge from source (22 Nov 2026). Xcode command line tools (swiftc) on
# macOS 27, Apple silicon. The Metal kernels are not compiled here: each dylib compiles its .metal sources at load
# (prefill: V8_KERNELS=kernels.metal + kernels_fold / kernels_f8 / kernels_qk.metal next to it; decode: DEC_METAL).
#   zsh engine/build_engines.sh [all|uw|served]   (default uw)
#     uw      libgpu_v8_f8q_uw.dylib + libdec_gpu_tree_uw.dylib: the KV-wrap leak fix (v8_unwrap / v8_kv8_clear /
#             dec_unwrap; results/22nov/memleak). Serve with V8_SRV_DYLIB=libgpu_v8_f8q_uw.dylib
#             DEC_DYLIB=libdec_gpu_tree_uw.dylib v8_server_dec8_qk.sh. Since 24 Nov 2026 the GPU engine's low-rank
#             correction takes its rank from the table files (64 as before, up to 192: the super-adapter build
#             v8_dec8ts_sa, adapter/); identical output at rank 64 (smoke 29aef2ec9db18a3f).
#     served  the launcher defaults' names, from the *.pre_unwrap.swift sources (the 19 Nov builds, leak included);
#             written as *_rebuilt.dylib so the served files are never overwritten
#     all     both
# OUT=dir writes the dylibs there instead of engine/ (never rebuild a dylib in place while a forge has it loaded).
# Prints each dylib's sha256 (the server banner shows the first 16 hex digits) and checks its exported C API.
set -eu
cd ${0:a:h}
what=${1:-uw}
O=${OUT:-.}; mkdir -p $O
build() {   # source module out
  set -- $1 $2 $O/$3
  swiftc -O -emit-library -module-name $2 -o $3 $1 -framework Metal -framework Accelerate 2>&1 | grep -v -i warning | grep -v '^ ' || true
  [ -s $3 ] || { echo "build failed: $3"; exit 1; }
  echo "$3 sha256 $(shasum -a 256 $3 | cut -c1-16) ($(stat -f %z $3) bytes, from $1 sha256 $(shasum -a 256 $1 | cut -c1-16))"
}
api() {   # dylib symbol...
  local f=$1; shift
  for s in "$@"; do nm -gU $f | grep -q " _$s\$" || { echo "$f: missing $s"; exit 1; }; done
}
if [ $what = uw ] || [ $what = all ]; then
  build gpu_v8_f8q.swift gpuv8 libgpu_v8_f8q_uw.dylib
  api $O/libgpu_v8_f8q_uw.dylib v8_init v8_wrap v8_unwrap v8_kv8_seg v8_kv8_clear
  build dec_gpu_tree.swift dec_gpu_tree libdec_gpu_tree_uw.dylib
  api $O/libdec_gpu_tree_uw.dylib dec_init dec_wrap dec_unwrap dec_seg dec_clear dec_set_tree
fi
if [ $what = served ] || [ $what = all ]; then
  build gpu_v8_f8q.pre_unwrap.swift gpuv8 libgpu_v8_f8q_rebuilt.dylib
  api $O/libgpu_v8_f8q_rebuilt.dylib v8_init v8_wrap v8_kv8_seg
  build dec_gpu_tree.pre_unwrap.swift dec_gpu_tree libdec_gpu_tree_rebuilt.dylib
  api $O/libdec_gpu_tree_rebuilt.dylib dec_init dec_wrap dec_seg dec_clear dec_set_tree
fi
