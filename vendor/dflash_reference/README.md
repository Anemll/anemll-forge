# DFlash reference source

This directory contains Z Lab's `dflash/model.py` and its original MIT [license](LICENSE), copied unchanged from the local `dflash-07ebd93` reference bundle. Its package metadata identifies [z-lab/dflash](https://github.com/z-lab/dflash) as the source repository. The bundle label has not been independently verified as a Git revision; [PROVENANCE.json](PROVENANCE.json) records exact source hashes.

Forge's `scripts/dflash2_drafter_ref.py validate` uses this local source by default to compare the ANE-shaped Torch implementation against the reference on identical inputs. Set `REF_CODE` to override it deliberately. The reference needs Torch and Transformers from the prepared environment, the target BF16 checkpoint (`MODEL`), and the DFlash2 BF16 checkpoint (`DRAFTER`). It does not download weights or provide the converted Core AI drafter; see [the speculative decoding guide](../../docs/SPECULATIVE_DECODING.md) for the serving pair.

Only the reference module needed by this validation path is included. This is not an installation of DFlash's CLI or its other backends. The root MIT license does not replace the upstream copyright notice here.
