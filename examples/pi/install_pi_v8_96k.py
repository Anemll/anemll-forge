"""Install the Pi profile for the V8-96K hybrid server (22 Nov 2026): merges into an existing Pi agent directory, never
replaces it. Every file it changes is first copied to `<file>.bak-v8-96k-<time>`; nothing else in the files is
touched (other providers, models, keys and settings stay as they are).

  models.json                    provider `ane-qwen38` (created when missing, base URL 127.0.0.1:PORT/v1) and its model
                                 `qwen38-27b-ane-v8-96k` (added, or replaced when present): 98,304-token window,
                                 32,768 max output, thinking through the chat template (examples/pi/v8_96k/models.json)
  settings.json                  compaction override for that model (reserve 32,768, keep 8,192 recent tokens);
                                 with --set-default also defaultProvider / defaultModel
  live-throughput-config.json    healthUrls["ane-qwen38"] = http://127.0.0.1:PORT/health: the TPS footer reads the
                                 server's own decode and prefill counters (exact for tree / speculative decoding)
  extensions/live-throughput-status.ts   the footer extension (live tok/s, TTFT, final decode and prefill tok/s)

    python3 examples/pi/install_pi_v8_96k.py [--pi-dir DIR] [--port 8765] [--model-id ID] [--set-default]
                                             [--no-extension] [--dry-run]
The Pi directory defaults to $PI_CODING_AGENT_DIR, else ~/.pi/agent. Afterwards run /reload in Pi (restart Pi for the
compaction settings) and pick the model with /model."""
import argparse, filecmp, json, os, shutil, sys, tempfile, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
EX = HERE / "v8_96k"
EXT = HERE / "extensions" / "live-throughput-status.ts"
PROVIDER = "ane-qwen38"
STAMP = time.strftime("%Y%m%d-%H%M%S")


def load(p: Path):
    return json.loads(p.read_text()) if p.exists() else {}


class Writer:
    def __init__(self, dry):
        self.dry, self.changed = dry, []

    def json(self, p: Path, new, old):
        if new == old:
            print(f"unchanged {p}")
            return
        self.changed.append(p)
        if self.dry:
            print(f"would update {p}")
            return
        p.parent.mkdir(parents=True, exist_ok=True)
        if p.exists():
            shutil.copy2(p, p.with_name(p.name + f".bak-v8-96k-{STAMP}"))
        fd, tmp = tempfile.mkstemp(dir=p.parent, prefix=p.name + ".")
        with os.fdopen(fd, "w") as f:
            f.write(json.dumps(new, indent=2, ensure_ascii=False) + "\n")
        if p.exists():
            os.chmod(tmp, p.stat().st_mode & 0o777)
        os.replace(tmp, p)
        print(f"updated {p}" + (f" (backup {p.name}.bak-v8-96k-{STAMP})" if (p.with_name(p.name + f'.bak-v8-96k-{STAMP}')).exists() else ""))

    def copy(self, src: Path, dst: Path):
        if dst.exists() and filecmp.cmp(src, dst, shallow=False):
            print(f"unchanged {dst}")
            return
        self.changed.append(dst)
        if self.dry:
            print(f"would install {dst}")
            return
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists():
            bak = dst.with_name(dst.name + f".bak-v8-96k-{STAMP}")
            shutil.move(str(dst), str(bak))   # out of Pi's *.ts discovery
            print(f"previous {dst.name} kept as {bak.name}")
        shutil.copy2(src, dst)
        print(f"installed {dst}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pi-dir", type=Path, default=Path(os.environ.get("PI_CODING_AGENT_DIR") or Path.home() / ".pi/agent"))
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--model-id", default="qwen38-27b-ane-v8-96k")
    ap.add_argument("--set-default", action="store_true", help="also make it Pi's default provider / model")
    ap.add_argument("--no-extension", action="store_true", help="leave the TPS footer extension alone")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    d = a.pi_dir.expanduser()
    w = Writer(a.dry_run)
    base = f"http://127.0.0.1:{a.port}"

    ex_m = json.loads((EX / "models.json").read_text())["providers"][PROVIDER]
    model = dict(ex_m["models"][0], id=a.model_id)
    old = load(d / "models.json")
    new = json.loads(json.dumps(old)) if old else {}
    provs = new.setdefault("providers", {})
    if PROVIDER not in provs:
        provs[PROVIDER] = dict(ex_m, baseUrl=base + "/v1", models=[])
    p = provs[PROVIDER]
    if p.get("baseUrl") != base + "/v1":
        print(f"note: provider {PROVIDER} points at {p.get('baseUrl')}, not {base}/v1: left as it is (one provider, one "
              f"server); pass --port to match, or edit models.json")
    p["models"] = [m for m in p.get("models", []) if m.get("id") != a.model_id] + [model]
    w.json(d / "models.json", new, old)

    old = load(d / "settings.json")
    new = json.loads(json.dumps(old)) if old else {}
    comp = new.setdefault("compaction", {})
    comp.setdefault("enabled", True)
    comp.setdefault("modelOverrides", {})[f"{PROVIDER}/{a.model_id}"] = {"reserveTokens": 32768, "keepRecentTokens": 8192}
    if a.set_default:
        new["defaultProvider"], new["defaultModel"] = PROVIDER, a.model_id
    w.json(d / "settings.json", new, old)

    old = load(d / "live-throughput-config.json")
    new = json.loads(json.dumps(old)) if old else {}
    hu = new.setdefault("healthUrls", {})
    health = (p.get("baseUrl") or base + "/v1").rsplit("/v1", 1)[0] + "/health"
    hu[PROVIDER] = health
    new.setdefault("healthPollMs", 1000)
    w.json(d / "live-throughput-config.json", new, old)

    if not a.no_extension:
        w.copy(EXT, d / "extensions" / EXT.name)

    print(f"{'would change' if a.dry_run else 'changed'} {len(w.changed)} file(s) in {d}. "
          f"In Pi: /reload, then /model -> {PROVIDER}/{a.model_id} (restart Pi for the compaction settings).")


if __name__ == "__main__":
    sys.exit(main())
