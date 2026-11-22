# Figure sources

All performance values refer to Qwen3.8-27B and DFlash2 on the Apple M6 in the November 20–22, 2026 paper. The figures use two different verifier baselines. The tree8 experiment compares to the eight-row chain. The sliced-entry experiment compares to tree8 with full-width KV entries.

| Figure | Evidence in the M6 checkout, relative to `research/v8_prefill` |
|---|---|
| Hardware placement | `TREE_VERIFY.md` §§2.1–2.2. The split decode path shown for 96K uses GPU attention twice per eight-layer package. |
| Tree8 method | `TREE_VERIFY.md` §§3.1–3.5. The eight-node example is illustrative. The generator derives the visibility matrix from the parent of each row. See the 20 Nov D4 form comparison for the chain-mask control. The runtime gathers tap features separately after compaction. |
| Speed and width | `TREE_VERIFY.md` §§4.3 and 5.1, `results/21nov/README.md`, and the reversed sampled block and 8–16-row hardware table in `results/21nov/YOLO_REPORT.md` §§2, 4. Width data are the mean of two timing blocks for layers 0–3. The statement about greedy throughput comes from the measured-cost replay. |
| Sliced KV | `TREE_VERIFY.md` §§3.6 and 4.4, `results/22nov/README.md` (four-entry ABBA and exclusion of seed 7100), and `results/22nov/sliced/block2/compare.json`. The two markers represent two data selections. |

Figures 2–4 are available as PNG and editable SVG. Regenerate them with Python, NumPy and Matplotlib from this directory using `python3 make_figures.py .`.

## Hardware illustration

Figure 1 was made with built-in ImageGen using the following prompt.

```text
Use case: scientific-educational
Create a landscape graphical abstract to accompany a computer-systems research paper. Make a precise, restrained scientific illustration on a pure white page, wide horizontal 3:2. Subject is logical compute placement for one Apple M6 running Qwen3.8-27B. It is an elegant functional schematic, not a photograph or die floorplan. Ink navy text, muted dark teal for ANE, muted warm ochre for GPU, pale neutral gray memory. Generous margins, journal-quality high legibility, large lettering. Arrange two peer processor tiles across the upper half. The left is labeled “Neural Engine” and contains exactly two lines “DFlash2 drafter” and “Feed-forward layers”. The right is labeled “GPU” and contains exactly two lines “Prefill token mixers” and “96K verify attention”. Each tile can have a small abstract line-engraved compute icon above its heading. Below the pair, one common wide memory bank is labeled “Shared INT8 KV cache”. Draw a simple separate bidirectional connection from each processor tile to the shared cache, without implying simultaneous operation. Top title exactly “Where the work runs”. Small subtitle exactly “Qwen3.8-27B on Apple M6”. Use these seven text elements plus the subtitle exactly, without adding any legend, slogans, metrics, vendor logos, figure numbering or explanatory copy. Crisp flat, white background, no gradients, no drop shadows. Every label large enough to read when reduced for a double-column figure.
```

The [claim audit](CLAIM_AUDIT.md) connects the method diagrams to primary papers and the local hardware evidence.
