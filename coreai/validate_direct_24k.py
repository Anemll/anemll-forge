#!/usr/bin/env python3
"""Opt-in M5 Pro / 24 GB validation of an experimental direct-attention ladder.

Run with the inference venv, outside the sandbox, and no other model server.
This is a smoke/parity check, not a general model-quality evaluation.
"""
import argparse
import gc
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', required=True, type=Path)
    parser.add_argument('--build', required=True, type=Path)
    parser.add_argument('--ctx', type=int, default=24576, help='largest experimental context to validate')
    parser.add_argument('--m5pro-24gb', action='store_true', help='opt into validation on M5 Pro with 24 GB Unified Memory')
    args = parser.parse_args()
    if not args.m5pro_24gb:
        parser.error('Experimental validation requires explicit --m5pro-24gb opt-in')
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
    from qwen38_hardware_profile import M5PRO_24GB_PROFILE, validate_hardware_profile
    bundle, build = args.bundle.resolve(), args.build.resolve()
    report = build / 'ladder_validation.json'
    if report.exists():
        parser.error('Refusing to overwrite a validation report')
    manifest = json.loads((build / 'manifest.json').read_text())
    if manifest.get('hardware_profile') != M5PRO_24GB_PROFILE:
        parser.error('Validator requires a marked M5 Pro / 24 GB experimental build')
    try:
        validate_hardware_profile(manifest)
    except ValueError as exc:
        parser.error(str(exc))
    ladder = [c for c in manifest['ctxs'] if c <= args.ctx]
    if len(ladder) < 2 or ladder[0] != 8192 or ladder[-1] != args.ctx:
        parser.error('Build must provide an 8K ladder ending at --ctx')
    target, baseline = args.ctx, ladder[-2]
    cutoff = manifest.get('kv_len', {}).get(str(baseline), baseline)
    os.environ.update(MODEL=str(bundle / 'model'), COREAI_DIR=str(build),
                      EXPORT_DIR=build.name, ANE_OUT=str(bundle), COREAI_BRIDGE='1',
                      EMBED_NPY=str(bundle / 'model/embed_tokens_fp16.npy'),
                      DRAFTER=str(bundle / 'drafter'), COREAI_DRAFTER_COMPUTE='ane',
                      MPSGRAPH_ANE_BONDED_COMPILE_MODE='2')
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
    import numpy as np
    from transformers import AutoTokenizer
    from qwen38_coreai_model import CoreAIQwen
    from dflash2_coreai_drafter import CoreAIDrafter
    from dflash2_ane_drafter import load_codebooks
    from hf_release import finite_logits, speculative_greedy

    def wired():
        out = subprocess.run(['vm_stat'], check=True, capture_output=True, text=True).stdout
        page = int(out.split('page size of ')[1].split()[0])
        return next(int(line.split()[-1].rstrip('.')) for line in out.splitlines()
                    if 'wired down' in line) * page / 2**30

    tok = AutoTokenizer.from_pretrained(str(bundle / 'model'), local_files_only=True)
    vocab = json.loads((bundle / 'model/config.json').read_text())['text_config']['vocab_size']
    stops = {tok.convert_tokens_to_ids(t) for t in ('<|im_end|>', '<|endoftext|>')} | {248044}

    def encode(content):
        text = tok.apply_chat_template([{'role': 'user', 'content': content}], tokenize=False,
                                      add_generation_prompt=True, enable_thinking=False)
        return tok.encode(text, add_special_tokens=False)

    question = 'What is the capital of France? Answer in one short sentence.'
    unit = 'Calibration record: the instrument is stable.\n'
    short_ids = encode(unit * 16 + '\nIgnore the calibration records. ' + question)
    assert len(short_ids) > 64
    model = CoreAIQwen(root=build, ctx=baseline, ladder=ladder)
    result = dict(build='experimental direct-attention build (local path omitted)',
                  hardware_profile=manifest['hardware_profile'],
                  contexts=model.ladder, checks={}, wired_loaded_gib=wired())
    result['prefill_contexts'] = model.pctxs
    logits = finite_logits(model.feed(short_ids), (vocab,), 'baseline prompt')
    snap = model.snapshot()
    anchor = int(np.argmax(logits))
    a = finite_logits(model.call([anchor]), (1, vocab), 'baseline continuation')[0].astype(np.float32)
    model.accept(1)
    model.restore(snap)
    model.resize(baseline)  # restore may shrink to 8K; compare an actual baseline-to-target resize
    keep = model.hi
    cached = [{name: view[:, :keep].copy() for name, (_, view) in ch.items()} for ch in model.kv]
    position, pending = model.pos, model.pending
    model.resize(target)
    assert (model.pos, model.pending) == (position, pending)
    for before, after in zip(cached, model.kv, strict=True):
        for name, value in before.items():
            assert np.array_equal(value, after[name][1][:, :keep]), name
    b = finite_logits(model.call([anchor]), (1, vocab), 'extended continuation')[0].astype(np.float32)
    model.accept(1)
    cosine = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
    same_top1 = int(np.argmax(a)) == int(np.argmax(b))
    assert cosine > .999 and same_top1, (cosine, same_top1)
    result['checks']['resize_parity'] = dict(kv_prefix_exact=True, position_preserved=True,
                                            logits_cosine=cosine, same_top1=same_top1)
    print('PASS resize parity: ' + json.dumps(result['checks']['resize_parity']), flush=True)
    del snap, cached, a, b
    gc.collect()

    cfg = json.loads((bundle / 'drafter/config.json').read_text())
    drafter = CoreAIDrafter(bundle / 'drafter/dflash2_lut4_gptq.aimodel', cfg,
                           load_codebooks(bundle / 'drafter'), model.emb)
    model.reset(shrink=False)
    target_has_prefill = model.has_prefill()
    start = time.perf_counter()
    output, stats = speculative_greedy(model, drafter, short_ids, vocab, 24, stops)
    assert model.ctx == target
    text = tok.decode(output, skip_special_tokens=True)
    assert text.strip() and 'Paris' in text, text
    short_key = f'{target // 1024}k_speculative'
    result['checks'][short_key] = dict(text=text, seconds=time.perf_counter() - start,
                                               prompt_tokens=len(short_ids), batched_prefill=target_has_prefill, **stats)
    print(f'PASS {target // 1024}K prefill/generation: ' + json.dumps(result['checks'][short_key]), flush=True)

    model.reset()
    drafter.reset()
    assert model.ctx == 8192
    count = 1 + (cutoff + 256) // max(1, len(tok.encode(unit, add_special_tokens=False)))
    long_ids = encode(unit * count + '\nIgnore the calibration records. ' + question)
    assert cutoff < len(long_ids) < min(cutoff + 2048, model.cap(target) - 24), len(long_ids)
    original_add = drafter.add_context
    progress = 0

    def add_context(feats, positions):
        nonlocal progress
        assert np.isfinite(feats).all()
        original_add(feats, positions)
        done = int(positions[-1]) + 1
        if done - progress >= 1024:
            print(f'long prompt: {done}/{len(long_ids)} positions; ctx={model.ctx}; '
                  f'wired={wired():.2f} GiB', flush=True)
            progress = done

    drafter.add_context = add_context
    events = len(model.stats['resize'])
    start = time.perf_counter()
    output, stats = speculative_greedy(model, drafter, long_ids, vocab, 24, stops)
    assert model.ctx == target and model.pos > cutoff
    resize = model.stats['resize'][events:]
    for before, after in zip(ladder[:-1], ladder[1:]):
        assert any(e['from'] == before and e['to'] == after for e in resize), resize
    text = tok.decode(output, skip_special_tokens=True)
    assert text.strip() and 'Paris' in text, text
    result['checks']['boundary_speculative'] = dict(prompt_tokens=len(long_ids), final_position=model.pos,
        context=model.ctx, resize=resize, text=text, seconds=time.perf_counter() - start,
        wired_gib=wired(), **stats)
    print('PASS boundary generation: ' + json.dumps(result['checks']['boundary_speculative']), flush=True)
    report.write_text(json.dumps(result, indent=2) + '\n')
    print('PASS full context ladder; report=' + str(report), flush=True)


if __name__ == '__main__':
    main()
