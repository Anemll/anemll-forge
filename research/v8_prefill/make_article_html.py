"""Writes ARTICLE_ANE_GPU_M6.html next to the Markdown article: the article text embedded, rendered in the browser by
marked.js (cdnjs, pinned), slides from img/ by relative path. Rerun after editing the article.
    python3 make_article_html.py [--embed]      (--embed: slides as data URIs, one self-contained file)"""
import base64, json, os, re, sys

D = os.path.dirname(os.path.abspath(__file__))
md = open(os.path.join(D, "ARTICLE_ANE_GPU_M6.md")).read()
if "--embed" in sys.argv:
    md = re.sub(r"!\[([^\]]*)\]\((img/[^)]+\.png)\)",
                lambda m: f"![{m.group(1)}](data:image/png;base64,{base64.b64encode(open(os.path.join(D, m.group(2)), 'rb').read()).decode()})", md)
title = re.search(r"^# (.+)$", md, re.M).group(1)
html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/marked/12.0.2/marked.min.js"></script>
<style>
:root{{--fg:#1f2328;--muted:#57606a;--line:#d0d7de;--code:#f6f8fa;--bg:#fff}}
@media (prefers-color-scheme: dark){{:root{{--fg:#e6edf3;--muted:#9da7b3;--line:#30363d;--code:#161b22;--bg:#0d1117}}}}
body{{font:16px/1.6 -apple-system,system-ui,sans-serif;max-width:900px;margin:40px auto;padding:0 16px;color:var(--fg);background:var(--bg)}}
img{{max-width:100%;border:1px solid var(--line);border-radius:6px;margin:8px 0}}
table{{border-collapse:collapse;margin:12px 0;font-size:14px;display:block;overflow-x:auto}}
th,td{{border:1px solid var(--line);padding:5px 9px}} th{{background:var(--code)}}
code{{background:var(--code);padding:1px 4px;border-radius:4px;font-size:90%}} pre{{background:var(--code);padding:12px;overflow:auto}}
h1,h2{{border-bottom:1px solid var(--line);padding-bottom:.3em}} em{{color:var(--muted)}} a{{color:#0969da}}
</style></head><body><main id="c">Loading…</main>
<script>
const md = {json.dumps(md)};
const slug = s => s.toLowerCase().replace(/<[^>]+>/g, "").replace(/[^\\w\\s-]/g, "").trim().replace(/\\s+/g, "-");
marked.use({{renderer: {{heading(text, level) {{ return `<h${{level}} id="${{slug(text)}}">${{text}}</h${{level}}>`; }}}}}});
document.getElementById("c").innerHTML = marked.parse(md);
</script>
</body></html>
"""
out = os.path.join(D, "ARTICLE_ANE_GPU_M6.html")
open(out, "w").write(html)
print(out, f"{os.path.getsize(out) / 1024:.0f} KB")
