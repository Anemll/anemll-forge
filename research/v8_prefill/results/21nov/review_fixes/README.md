# 21 Nov 2026: fixes from the outside review (`results/22nov_review/REVIEW.md`)

All CPU checks; nothing restarted. The running forge (pid 73029) still has the code from before these fixes; they
take effect at the next restart.

| review finding | verified | fix (file) | check |
|---|---|---|---|
| High: a 'system' snapshot with no visible drafter slots (system cut <= 8 tokens) makes `_ring_restore` raise (`reshape(-1, 0)`); on a 'sysk' restore after the base restore has set tokens and DeltaNet state but before the KV write-back, so a retry continues on the other prompt's KV rows | yes (numpy ValueError; reviewer's cache_checks) | `pipe/v8_server.py`: empty ring returns early; `_restore` all or nothing: any failure after the base restore -> `_restore_cold` (drop that snapshot, reset model / drafter, fed = [], rewind V8, cold scrub if set) | `results/22nov_review/cache_checks.py` (v8 gate) passes; `serve_t/restore_fault_unit.py` 4/4 (`restore_fault_unit.log`) |
| High at the 96K entry: the tree mask set before the first-use plan bind, outside the finally; setter inside `assert` | yes (fake driver: mask (5, 9) left on after a bind failure; under -O the runs see the chain mask) | `pipe/tree_verify.py` `call`: plan first, checked `dec_set_tree` calls, chain mask restored in finally (split tree off if that fails) | `serve_t/tree_mask_unit.py` 3/3 at -O0 and -O1; the old file fails 3 and 2 (`tree_mask_unit.log`) |
| Medium: split tree steps not charged to `wall_ms` / `gpu_ms` | yes | `tree_verify.py`: tree split steps add both | `tree_mask_unit.py` (gpu_ms 2.0 on the normal case) |
| Experimental: capped sampled chain samples row k with all 7 drafts in the DRY history | yes (TREE_POLICY=cap only) | `_chain`: `hist + drafts[:k]` | `tree_sampled_unit.py` still 3/3 (`tree_sampled_unit_rerun.log`) |
| Medium: non-system snapshots restore drafter rows by position only (0/4 after a ring wrap) | yes (reviewer's cache_checks) | not changed (measure acceptance after real rewinds first) | |
| High (release / v02 standalone): a 'system' snapshot reused over another prompt's target KV rows | yes (reviewer's toy runtime) | not changed: the release file has Anemll's uncommitted edits; V8 serving uses `_prune` + 'sysk' | |

**Verify widths 9-15 (review addendum):** `v8/build_gdn_wy_probe.py` `tri_solve_probe` uses `rows // 8` full blocks,
so at 9-15 rows it returns only 8 rows (`gdn_partial_solve_unit.log`). `v8/gdn_partial_solve.py` adds the 8-row-block
solve with a last partial block (bfs and blk forms, `inv_unit_lower` per diagonal block): rows 8-16 match a float64
triangular solve at 0.7-1.1e-7 (fp32) and 4.3-7.2e-4 (fp16, the same level as 8 and 16 rows). Not yet wired into a
probe build; the whole 4-layer probe module also needs a CPU equivalence check at each width before any compile.
