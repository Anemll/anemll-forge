"""Applied 16 Nov 2026 (see the v8_work notes): gdn_gates_nv / gdn_conv_qkv_nv (copies of gdn_gates / gdn_conv_qkv
with a valid-row count: pad rows get beta 0 / decay 1, the conv history rolls from the last valid rows) and
v8_submit_pair_nv in gpu_v8.swift; the original kernels are unchanged so every older engine dylib keeps working.
Backups: kernels.pre_nv.metal, gpu_v8.pre_nv.swift."""
