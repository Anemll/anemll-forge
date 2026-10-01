"""Identify and stop only the server instance recorded by the shell wrapper."""
import argparse
import ctypes
import os
from pathlib import Path
import signal
import struct
import subprocess
import sys
import time


def darwin_argv(raw):
    """Decode KERN_PROCARGS2 argv, excluding the following environment block."""
    argc = struct.unpack_from("i", raw)[0]
    offset = raw.index(b"\0", 4) + 1  # Executable path precedes padded argv.
    while offset < len(raw) and raw[offset] == 0:
        offset += 1
    return [os.fsdecode(arg) for arg in raw[offset:].split(b"\0")[:argc]]


def process_argv(pid):
    try:
        if sys.platform == "darwin":
            libc = ctypes.CDLL(None, use_errno=True)
            mib = (ctypes.c_int * 3)(1, 49, pid)  # CTL_KERN, KERN_PROCARGS2.
            size = ctypes.c_size_t()
            if libc.sysctl(mib, 3, None, ctypes.byref(size), None, 0):
                return []
            data = ctypes.create_string_buffer(size.value)
            if libc.sysctl(mib, 3, data, ctypes.byref(size), None, 0):
                return []
            return darwin_argv(data.raw[:size.value])
        return [os.fsdecode(arg) for arg in Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0") if arg]
    except (OSError, ValueError, struct.error):
        return []


def matches(argv, root, port):
    if len(argv) < 2:
        return False
    executable = Path(argv[0])
    if not executable.name.lower().startswith("python") and not executable.resolve().name.lower().startswith("python"):
        return False
    index = 2 if argv[1] == "-u" else 1
    if index >= len(argv):
        return False
    script = argv[index]
    if script == str(root / "forge.py"):
        if argv[index + 1:index + 2] != ["serve"]:
            return False
    elif script != str(root / "scripts/qwen38_server.py"):
        return False
    ports = [argv[i + 1] for i, arg in enumerate(argv[:-1]) if arg == "--port"]
    return ports[-1:] == [str(port)]


def managed_pids(root, port, pidfile):
    try:
        recorded = [int(pid) for pid in pidfile.read_text().split()]
    except (OSError, ValueError):
        return []
    owned = {pid for pid in recorded if pid > 1 and matches(process_argv(pid), root, port)}
    if not owned:
        return []
    rows = subprocess.check_output(["ps", "-axo", "pid=,ppid="], text=True).splitlines()
    parents = {}
    for row in rows:
        pid, parent = map(int, row.split())
        parents[pid] = parent
    descendants = set(owned)
    while True:
        children = {pid for pid, parent in parents.items() if parent in descendants}
        if children <= descendants:
            break
        descendants.update(children)
    owned.update(pid for pid in descendants if matches(process_argv(pid), root, port))
    return sorted(owned)


def stop(root, port, pidfile):
    pids = managed_pids(root, port, pidfile)
    if not pids:
        pidfile.unlink(missing_ok=True)
        print("not running (no verified server in the PID file)")
        return 0
    # Retain verified child PIDs if the launcher exits before its child does.
    pidfile.write_text("\n".join(map(str, pids)) + "\n")
    for sig in (signal.SIGTERM, signal.SIGKILL):
        for pid in reversed(pids):
            if matches(process_argv(pid), root, port):
                try:
                    os.kill(pid, sig)
                except ProcessLookupError:
                    pass
        for stop_attempt in range(40 if sig == signal.SIGTERM else 10):
            pids = [pid for pid in pids if matches(process_argv(pid), root, port)]
            if not pids:
                pidfile.unlink(missing_ok=True)
                print("stopped")
                return 0
            time.sleep(0.5)
    pidfile.write_text("\n".join(map(str, pids)) + "\n")
    print("server has not exited; refusing to start another instance", file=sys.stderr)
    return 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("pids", "stop"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--pidfile", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.action == "stop":
        return stop(args.root, args.port, args.pidfile)
    pids = managed_pids(args.root, args.port, args.pidfile)
    if pids:
        print(" ".join(map(str, pids)))
    return 0 if pids else 1


if __name__ == "__main__":
    raise SystemExit(main())
