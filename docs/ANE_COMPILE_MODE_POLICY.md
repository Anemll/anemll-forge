# Spec: ANE bonded compile mode policy by SoC generation

**Status:** Implemented 2026-10-04 (pending reproduction on M5 Pro/Max). Selected defaults: an explicit mode override is honored and logged; pre-M5 fails on model-loading paths (`quick-test`/`serve`/`compile`); `doctor` warns only. See §5 for the decisions.

## 1. Problem

`MPSGRAPH_ANE_BONDED_COMPILE_MODE` selects which MPSGraph ANew procedure variant is
specialized: `0` keeps bonded + non-bonded, `2` keeps bonded only (the Core AI default),
and `1` is the value that made the full target/drafter load succeed on an Apple M5.

Observed on **Apple M5 (Mac17,2, macOS 26B5091g, "H17")** on 2026-10-04:

- With the runtime default `2`, the real `forge.py serve` path fails to compile the
  target/drafter packages with `_ANECompiler : ANECCompile() FAILED … "Compiler internal
  error: Couldn't do topological sort"`.
- With `MPSGRAPH_ANE_BONDED_COMPILE_MODE=1` exported, `forge.py serve` proceeds to
  compile all 18 packages and start serving. Reproduced by the maintainer.
- The earlier M5 diagnostic (`docs/SESSION_LESSONS.md`, September 29) showed mode `0` also
  fails (32 topological-sort failures) and consumes disk faster than mode `2`.

Therefore the compile-mode default that is correct for M6 is not correct for M5, and
running on pre-M5 silicon is not a supported target.

## 2. Evidence summary

| SoC | ANE behavior | `soc-generation` | Source |
|---|---|---|---|
| **M5 base (H17)** | `2` fails quick-test/serve compile (topological sort); `1` succeeds; `0` fails and costs more disk | `"H17"` | maintainer run 2026-10-04; `docs/SESSION_LESSONS.md` (Sept 29) |
| **M6 family (H18)** | `2` is the measured default; parity with `0`, lower program memory, release measurements use `2` | `"H18"` | confirmed by maintainer (`ioreg`); `docs/research/M6_ANE_COMPUTE_2026-10-02.md`, `docs/research/M6_COMPUTE_ACCELERATION_2026-10-03.md` |
| **Newer than M6 (H19+)** | Unmeasured; defaults to the M6 policy (`2`) | `>= "H19"` | maintainer directive 2026-10-04 |
| **M5 Pro / M5 Max (H17)** | `1` required (maintainer); expected to follow base M5 | expected `"H17"`, sub-variant unconfirmed in repo | maintainer directive 2026-10-04; `RESULTS_M5_MAX.md` (M5 Max, `h17c`, `Mac17,6`), single-op probes only |
| **pre-M5 (H16 and below)** | Not a supported target; no release measurements | `< "H17"` | README (M5/M6 target), `docs/VALIDATION.md` (“Not covered: … M5-family hardware”) |

Mode `1` is undocumented by Apple. Its semantics are not published; it is treated here as
an empirically required switch on H17, not as a general recommendation.

## 3. Detection

The policy is keyed on **SoC generation**, not marketing name. Detection order:

1. **Primary — Core AI architecture code** from the loaded package/device descriptor where
   available (`h17*`, `h18*`); `COREAI_ARCH` already overrides the build arch
   (`coreai/qwen38_coreai_build.py`).
2. **Fallback — platform generation**, in this order:
   - `ioreg` `soc-generation` property (`"H17"`, `"H18"`), or
   - `sysctl -n machdep.cpu.brand_string` → parse `Apple M<generation>` (observed:
     `Apple M5`), or
   - `sysctl -n hw.model` model class (`Mac17,*` ≈ M5 family, `Mac18,*` ≈ M6 family,
     per `docs/verifier_len.md` M6 = `Mac18,5` and `RESULTS_M5_MAX.md` M5 Max = `Mac17,6`).

Generation order used for decisions: `H17 < H18`, and anything `< H17` is pre-M5.

Detection MUST distinguish the M5 sub-variant, because only the base M5 has been
observed:

- **M5 base:** `Apple M5` with no `Pro`/`Max` suffix; observed `Mac17,2`.
- **M5 Pro / M5 Max:** `Apple M5 Pro` / `Apple M5 Max`; M5 Max observed as `Mac17,6`
  (`RESULTS_M5_MAX.md`) and an ANE target of `h17c`.

Detection MUST be centralized in one helper returning a normalized class
(`pre_m5 | m5 | m5_pro_max | m6 | unknown`) plus the raw source string, so the error and
logs are consistent across `forge.py`, `coreai_compile.py`, and the wrapper.

## 4. Requirements

Normative language: **MUST**, **SHOULD**, **MAY**.

- **R1 (M5 family):** On any M5-family SoC (`H17`, including base M5, M5 Pro and M5 Max),
  the runtime and compile paths **MUST** set the effective
  `MPSGRAPH_ANE_BONDED_COMPILE_MODE` to `1` unless the user explicitly overrides it
  (see R5). Base M5 confirmed on 2026-10-04; M5 Pro/Max are a maintainer directive
  (see §9).
- **R2 (pre-M5):** On a SoC older than M5 (`< H17`), the runtime and compile paths **MUST
  fail fast before loading or compiling any package**, with a clear, actionable error that
  names the detected generation and the supported set. They **MUST NOT** attempt to compile
  or serve, even when an explicit mode override is set.
- **R3 (M6):** On M6-family (`H18`), the existing measured default `2` **MUST** be
  preserved.
- **R3b (newer than M6):** On a detected generation `H19` or later, the runtime **MUST**
  default to the M6 settings (bonded mode `2`). It **MUST** log that the generation is
  unvalidated and that M6 settings are being assumed, and an explicit override **MUST** be
  honored.
- **R4 (cache correctness):** Changing the effective mode **MUST** leave the existing
  cross-mode cache handling intact: cached specializations compiled in another mode are
  purged and recompiled (`scripts/qwen38_coreai_model.py`, `_align_mode`). A policy-driven
  mode change therefore starts a cold compile exactly once.
- **R5 (explicit override):** A user-provided `MPSGRAPH_ANE_BONDED_COMPILE_MODE` in the
  environment **MUST** be honored on supported generations (`M5`, `M6`, `H19+`) and on an
  undetectable chip, because every setter uses `setdefault`. It **MUST NOT** re-enable a
  pre-M5 SoC (R2). The startup log and `/health` **MUST** report both the detected
  generation and the effective mode so an override is never silent.
- **R6 (diagnostics):** Startup **MUST** print one line identifying the generation source
  and the effective mode, e.g. `ANE compile mode policy: M5 (H17) -> bonded mode 1`
  (and `override by env` when applicable). `GET /health` **SHOULD** expose
  `soc_generation` and `ane_bonded_compile_mode`.
- **R7 (undetectable generation):** When the generation cannot be parsed at all (not a
  recognized `H<number>`/brand/model), the runtime **MUST** fail closed by default unless
  the user explicitly overrides the mode; `COREAI_ALLOW_UNKNOWN_SOC=1` **MAY** be provided
  as an explicit escape hatch. A *detected but newer* generation (`H19+`) is **not**
  unknown and follows R3b.

## 5. Decisions and remaining questions

Resolved during implementation:

1. ~~Does “M5*” include M6?~~ **Resolved:** M6 (`H18`) defaults to `2`; `H19+` also uses M6 settings.
2. ~~M5 Pro / M5 Max default.~~ **Resolved:** all `H17` (base M5, M5 Pro, M5 Max) → `1`.
3. ~~Override allowed?~~ **Resolved:** an explicit `MPSGRAPH_ANE_BONDED_COMPILE_MODE` is honored and logged.
4. ~~Fail scope.~~ **Resolved:** fail on model-loading commands; warn-only on `doctor`.

Still open:

5. **M5 Pro/Max reproduction.** A full target/drafter compile on an M5 Max has not been run
   in-tree; the directive should be confirmed by one successful load there.
6. **Wrapper behavior:** `scripts/qwen38_server.sh` currently delegates the decision to
   `forge.py`; confirm that is preferred over an explicit pre-check in the shell.

## 6. Acceptance criteria

- Unit tests with a mocked detector cover `pre_m5`, `m5`, `m5_pro_max`, `m6`, `unknown`:
  - `m5` → effective mode `1`;
  - `m5_pro_max` → effective mode `1`, override honored;
  - `m6` → effective mode `2`;
  - `h19+` (newer) → effective mode `2` + unvalidated-generation warning, override honored;
  - `pre_m5` → fails before any package load, non-zero exit, message includes generation;
  - undetectable `unknown` → fails unless `COREAI_ALLOW_UNKNOWN_SOC=1`.
- A user env override is honored on `m5`/`m6` and reported in the startup line and `/health`.
- Switching generation/mode purges stale cache entries (`_align_mode`) and recompiles once.
- Docs updated: `README.md`, `docs/SERVER.md` (env-var table), `docs/SESSION_LESSONS.md`
  (supersede the "mode is not the variable" conclusion for the mode-`1` case), and
  `docs/VALIDATION.md` (mark M5 mode-`1` evidence as a single-machine, single-run finding
  until reproduced).

## 7. Implementation touch points (for the follow-up change)

| File | Change |
|---|---|
| `scripts/qwen38_coreai_model.py` (`MODE_ENV`, `_align_mode`) | central generation helper; default `1` on H17, refuse `< H17`, default M6 settings (`2`) on H18+ |
| `forge.py` (`compile`, `serve` env assembly, `doctor`) | call helper; fail fast; report mode |
| `scripts/coreai_compile.py` | same policy before compiling |
| `scripts/qwen38_server.sh` | propagate detection/override; surface error |
| `scripts/qwen38_server.py` | `/health` fields |
| `coreai/qwen38_coreai_build.py` (`_ARCH`) | keep `h17*`/`h18*` arching consistent with detection |
| tests + docs | see §6 |

## 8. Non-goals

- Re-deriving Apple's bonded/non-bonded semantics or proving physical ANE placement.
- Claiming a throughput or quality gain from mode `1`; the requirement is only that M5
  targets compile and load successfully.
- Changing the M6 release defaults or measured results.

## 9. Validation status

This spec is based on: one successful **base M5** mode-`1` full load (2026-10-04,
maintainer), the September 29 M5 diagnostic record, and existing M6 research.

- Mode `1` has **not** been quality- or throughput-validated on M5 base.
- **M5 Pro / M5 Max:** required by a maintainer directive (2026-10-04) but **not yet
  reproduced in the repo**. The only M5 Max material in-tree is single-op vector-LUT
  placement/timing research (`RESULTS_M5_MAX.md`, `RESULTS_M5_MAX_INT8.md`), not a full
  target/drafter compile. Treat as a firm requirement with no in-repo reproduction; a
  reproduction pass on an M5 Max is still owed.
- M6 measurements remain on mode `2` and are unchanged.
- **Generations newer than M6 (`H19+`)** are unmeasured and inherit M6 settings by
  directive; this is not a validation of those chips.

Treat the policy as a raw requirement pending implementation and a reproduction pass.
