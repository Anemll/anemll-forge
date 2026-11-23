"""Paced streaming (22 Nov 2026, STREAM_PACE=1): a speculative / tree verify cycle commits several tokens at once and
the release server streams each as its own chunk, back to back, so a client sees a burst of text, then a pause of one
cycle. This wraps engine.generate's on_text callback: the engine thread only queues each delta; a writer thread sends
them spaced at the recent arrival rate (mean gap of the last PACE_N deltas, at most PACE_MAX_GAP_MS), so a cycle's
tokens spread over the next cycle. A delta never waits more than PACE_LAG_MS after it was produced (then the backlog
goes out at once); the first delta of a request goes out at once (TTFT unchanged), and when generate returns the rest
is flushed at once and the writer joined, so the handler's own final chunks keep their order. Same deltas, same
chunk boundaries, same text: only the send times change. Decode speed is unaffected (the engine never waits on the
socket; it used to, a little). A client that went away (on_text returned False) stops the engine at its next delta,
as before.
    env STREAM_PACE=1, PACE_LAG_MS (250), PACE_N (16), PACE_MAX_GAP_MS (60)"""
import collections, os, threading, time

LAG = float(os.environ.get("PACE_LAG_MS", "250")) / 1000
N = int(os.environ.get("PACE_N", "16"))
MAX_GAP = float(os.environ.get("PACE_MAX_GAP_MS", "60")) / 1000
ON_TEXT_ARG = 7   # generate(ids, max_tokens, temp, top_p, top_k, stop, seed, on_text, ...)


class Pacer:
    def __init__(self, on_text, clock=time.perf_counter, sleep=None):
        self.on_text, self.clock = on_text, clock
        self.q = collections.deque()          # (t_produced, delta)
        self.arrivals = collections.deque(maxlen=N)
        self.cv = threading.Condition()
        self.final = self.stopped = False
        self.err = None
        self.sent = 0
        self.last_send = None
        self.th = threading.Thread(target=self._run, name="stream-pace", daemon=True)
        self.th.start()

    def __call__(self, delta):               # engine thread
        if self.err is not None:
            e, self.err = self.err, None
            raise e
        if self.stopped:
            return False
        t = self.clock()
        with self.cv:
            self.q.append((t, delta))
            self.arrivals.append(t)
            self.cv.notify()
        return None

    def gap(self):
        a = self.arrivals
        if len(a) < 2:
            return 0.0
        return min(MAX_GAP, (a[-1] - a[0]) / (len(a) - 1))

    def _run(self):                          # writer thread
        while True:
            with self.cv:
                while not self.q and not self.final:
                    self.cv.wait()
                if not self.q:
                    return
                t_prod, _ = self.q[0]
                now = self.clock()
                due = now if self.last_send is None else self.last_send + self.gap()
                if not self.final and due > now and now - t_prod < LAG:
                    # wait until due (a new delta, the flush or the lag limit can cut it short)
                    self.cv.wait(min(due - now, LAG - (now - t_prod)))
                    continue
                _, delta = self.q.popleft()
            if self.stopped:
                continue
            try:
                r = self.on_text(delta)
            except Exception as e:           # noqa: BLE001  (raised in the engine thread at its next delta / the end)
                self.err, r = e, False
            self.last_send = self.clock()
            self.sent += 1
            if r is False:
                self.stopped = True

    def close(self):                         # engine thread, after generate: flush the rest at once, join
        with self.cv:
            self.final = True
            self.cv.notify()
        self.th.join()
        if self.err is not None:
            e, self.err = self.err, None
            raise e


def install(eng, log):
    gen0 = eng.generate

    def generate(*a, **k):
        if len(a) <= ON_TEXT_ARG:
            return gen0(*a, **k)
        p = Pacer(a[ON_TEXT_ARG])
        a = a[:ON_TEXT_ARG] + (p,) + a[ON_TEXT_ARG + 1:]
        try:
            return gen0(*a, **k)
        finally:
            p.close()
    eng.generate = generate
    log(f"stream pace: BANNER on (lag <= {LAG * 1000:.0f} ms, gap = mean of the last {N} arrivals, "
        f"<= {MAX_GAP * 1000:.0f} ms)")
