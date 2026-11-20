#!/bin/zsh
# DeltaNet alone (19 Nov 2026): the served GDN call and its parts. LUT GEMM forms (I8TV; _nodec / _nomma are
# timing diagnostics, wrong output) and recurrence forms (REC16 / REC8 / REC=pf|px), PROFILE per-kernel ms. Forge stays up
# (idle); GPU only, one-layer manifest.
O=/Volumes/SSD4TB/anemll-forge-research/contention/gdn; PY=/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll-forge/.venv/bin/python
export MANIFEST=$O/manifest_L0.json
cd /Users/anemll/Documents/Codex/ios-prefill-20261030/v8_work/pipe/unit/power
one() { local name=$1; shift; env "$@" $PY gdn_prof.py $O/s_$name 60 > $O/s_$name.log 2>&1; env "$@" PROFILE=1 $PY gdn_prof.py $O/p_$name 30 > $O/p_$name.log 2>&1
  echo "$name: $(grep '^{' $O/s_$name.log | $PY -c 'import sys,json; d=json.loads(sys.stdin.read()); print(d["ms_median"], "ms/call")') | $(grep -E 'lut_gemm|gdn_rec' $O/p_$name.log | awk '{printf "%s %.3f ms/call; ", $1, $2/34}')"; }
one served I8TV=_k256
one k256db I8TV=_k256db
one k128 I8TV=_k128
one k128nodec I8TV=_k128_nodec
one k64 I8TV=
one k64nodec I8TV=_nodec
one k64nomma I8TV=_nomma
one k512 I8TV=_k512
one m64k128 I8TV=_m64k128
one rec16 I8TV=_k256 REC16=1
one rec8 I8TV=_k256 REC8=1
one recpf I8TV=_k256 REC=pf
one recpx I8TV=_k256 REC=px
