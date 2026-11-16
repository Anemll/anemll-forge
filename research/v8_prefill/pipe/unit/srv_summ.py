import sys, json
for l in sys.stdin:
    try:
        j = json.loads(l)
        s = j.get('server', {}); p = s.get('prefill') or {}
        print(j.get('config', ''), j['request'], 'prompt', j['prompt_tokens'], 'ttft %.2fs' % j['ttft_s'],
              '| server prefill %s tok in %.2fs (%.0f tok/s, cached %s)' % (p.get('tokens'), p.get('seconds', 0), p.get('tokens_per_second', 0), p.get('cached_tokens')),
              '| decode %.1f tok/s' % j['decode_tok_s'], '|', j['text'][:110].replace(chr(10), ' '), flush=True)
    except Exception:
        print(l.strip()[:300], flush=True)
