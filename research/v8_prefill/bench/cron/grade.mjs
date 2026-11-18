// Hidden grader for the cron-library coding test (prompt: bench/cron/PROMPT.md). Not shown to the models.
//   node grade.mjs <path/to/cron.mjs> [--verbose]
// The reference below follows the prompt's spec literally; nextRuns is checked against a day-by-day scan of the
// whole 4000-day window (obviously correct: every day is tested against the day rule, every hour x minute of a
// matching day is listed), so no clever reference logic can hide a bug.
import { pathToFileURL } from "node:url";
import { performance } from "node:perf_hooks";

const file = process.argv[2];
const verbose = process.argv.includes("--verbose");
if (!file) { console.error("usage: node grade.mjs <cron.mjs> [--verbose]"); process.exit(2); }

// ---------------------------------------------------------------- reference (spec, literally)
const MONTHS = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"];
const DAYS = ["SUN", "MON", "TUE", "WED", "THU", "FRI", "SAT"];
const F = [
  { key: "minute", min: 0, max: 59, top: 59 },
  { key: "hour", min: 0, max: 23, top: 23 },
  { key: "dayOfMonth", min: 1, max: 31, top: 31 },
  { key: "month", min: 1, max: 12, top: 12, names: MONTHS, base: 1 },
  { key: "dayOfWeek", min: 0, max: 7, top: 6, names: DAYS, base: 0 },   // `*` and a/s run 0..6
];
const MACROS = { "@yearly": "0 0 1 1 *", "@annually": "0 0 1 1 *", "@monthly": "0 0 1 * *", "@weekly": "0 0 * * 0",
  "@daily": "0 0 * * *", "@midnight": "0 0 * * *", "@hourly": "0 * * * *" };
class RefError extends Error {}
const KEYS = ["minute", "hour", "dayOfMonth", "month", "dayOfWeek", "domRestricted", "dowRestricted"];

function refValue(tok, f) {
  if (/^\d+$/.test(tok)) {
    const v = parseInt(tok, 10);
    if (v < f.min || v > f.max) throw new RefError(`out of range ${tok}`);
    return v;
  }
  if (f.names) { const i = f.names.indexOf(tok.toUpperCase()); if (i >= 0) return i + f.base; }
  throw new RefError(`bad value ${tok}`);
}
function refField(text, f) {
  const set = new Set();
  for (const item of text.split(",")) {
    if (item === "") throw new RefError("empty item");
    const sl = item.split("/");
    if (sl.length > 2) throw new RefError("two steps");
    let step = 1;
    if (sl.length === 2) {
      if (!/^\d+$/.test(sl[1])) throw new RefError("bad step");
      step = parseInt(sl[1], 10);
      if (step < 1) throw new RefError("step 0");
    }
    const body = sl[0];
    let lo, hi;
    if (body === "*") { lo = f.min; hi = f.top; }
    else if (body.includes("-")) {
      const p = body.split("-");
      if (p.length !== 2) throw new RefError("bad range");
      lo = refValue(p[0], f); hi = refValue(p[1], f);
      if (lo > hi) throw new RefError("reversed range");
    } else {
      lo = refValue(body, f);
      hi = sl.length === 2 ? f.top : lo;
      if (lo > hi) throw new RefError("a/s start past the top");
    }
    for (let v = lo; v <= hi; v += step) set.add(f.key === "dayOfWeek" && v === 7 ? 0 : v);
  }
  return [...set].sort((a, b) => a - b);
}
function refParse(expr) {
  if (typeof expr !== "string") throw new RefError("not a string");
  let s = expr.trim();
  if (s.startsWith("@")) { const m = MACROS[s.toLowerCase()]; if (!m) throw new RefError("macro"); s = m; }
  const parts = s.split(/[ \t]+/);
  if (parts.length !== 5 || parts[0] === "") throw new RefError("field count");
  const out = {};
  F.forEach((f, i) => { out[f.key] = refField(parts[i], f); });
  out.domRestricted = !parts[2].startsWith("*");
  out.dowRestricted = !parts[4].startsWith("*");
  return out;
}
const DAY = 86400000, MIN = 60000;
function refNext(expr, from, n) {
  const p = refParse(expr);
  const D = new Set(p.dayOfMonth), W = new Set(p.dayOfWeek), Mo = new Set(p.month);
  const start = Math.floor(from.getTime() / MIN) * MIN + MIN;    // strictly after from, on a minute
  const end = from.getTime() + 4000 * DAY;                         // exclusive
  const res = [];
  for (let day = Math.floor(start / DAY) * DAY; day < end && res.length < n; day += DAY) {
    const d = new Date(day);
    if (!Mo.has(d.getUTCMonth() + 1)) continue;
    const dm = D.has(d.getUTCDate()), dw = W.has(d.getUTCDay());
    if (!(p.domRestricted && p.dowRestricted ? dm || dw : dm && dw)) continue;
    outer: for (const h of p.hour) for (const m of p.minute) {
      const t = day + h * 3600000 + m * MIN;
      if (t < start || t >= end) continue;
      res.push(new Date(t).toISOString());
      if (res.length === n) break outer;
    }
  }
  return res;
}

// ---------------------------------------------------------------- harness
let mod;
try { mod = await import(pathToFileURL(file).href); }
catch (e) { console.log(JSON.stringify({ file, loadError: String(e) })); process.exit(1); }
const { parse, nextRuns, CronError } = mod;
const cats = {};
function record(cat, ok, detail) {
  const c = (cats[cat] ||= { pass: 0, fail: 0, failures: [] });
  if (ok) c.pass++; else { c.fail++; if (c.failures.length < 12) c.failures.push(detail); }
}
const show = v => (v instanceof Error ? `${v.constructor.name}: ${v.message}` : JSON.stringify(v));
function tryCall(fn) { try { return { v: fn() }; } catch (e) { return { e }; } }
function pick(o) { const r = {}; for (const k of KEYS) r[k] = o?.[k]; return r; }
function checkParse(cat, expr) {
  const want = pick(refParse(expr));
  const got = tryCall(() => parse(expr));
  const ok = !got.e && JSON.stringify(pick(got.v)) === JSON.stringify(want);
  record(cat, ok, { expr, want, got: got.e ? show(got.e) : pick(got.v) });
}
function checkNext(cat, expr, from, n) {
  const want = refNext(expr, from, n);
  const got = tryCall(() => nextRuns(expr, from, n));
  let ok = !got.e && Array.isArray(got.v) && got.v.every(d => d instanceof Date);
  const gotIso = ok ? got.v.map(d => d.toISOString()) : null;
  ok = ok && JSON.stringify(gotIso) === JSON.stringify(want);
  let firstDiff = null;
  if (!ok && gotIso) { const i = want.findIndex((w, k) => w !== gotIso[k]); firstDiff = { index: i < 0 ? want.length : i, want: want[i], got: gotIso[i < 0 ? want.length : i] }; }
  record(cat, ok, { expr, from: from.toISOString(), n, wantLen: want.length, gotLen: gotIso?.length, firstDiff, error: got.e ? show(got.e) : undefined });
}
const U = s => new Date(s);

// ---------------------------------------------------------------- 1. parse: fixed cases
for (const e of [
  "* * * * *", "0 0 1 1 *", "*/15 9-17 * * MON-FRI", "5/20 * * * *", "0-10/3 * * * *", "1,2,3,5,8 * * * *",
  "0 0 * * 7", "0 0 * * 5-7", "0 0 * * 0-7", "0 0 * * */2", "0 0 * * 1/2", "0 0 * * MON/2", "0 0 * * sun,Wed,FRI",
  "0 0 1 jan-mar *", "0 0 1 JAN,jun,Dec *", "0 0 1 */3 *", "05 08 * * *", "0 0 */2 * 1", "0 0 1-7 * 1",
  "0 12 13 * 5", "59 23 31 12 *", "0 0 31 * *", " \t 1  2\t3 4   5 \t", "@yearly", "@ANNUALLY",
  "@monthly", "@Weekly", "@daily", "@midnight", "@hourly", "0 0 1,15 * 1-5", "0 0 * * 6,7", "0 22 * * 1-5",
]) checkParse("parse_fixed", e);

// ---------------------------------------------------------------- 2. nextRuns: fixed edge cases
const fixed = [
  ["0 0 29 2 *", "2096-03-01T00:00:00Z", 3],        // 2100 is not a leap year: next is 2104, then the window ends
  ["0 0 29 2 *", "1896-01-01T00:00:00Z", 5],        // 1896, then 1904 (1900 skipped)
  ["0 0 30 2 *", "2024-01-01T00:00:00Z", 1],        // never: []
  ["0 0 31 2,4,6,9,11 *", "2024-01-01T00:00:00Z", 3],
  ["0 0 31 * *", "2024-01-15T12:00:00Z", 12],
  ["0 12 13 * 5", "2024-01-01T00:00:00Z", 20],      // 13th OR Friday
  ["0 0 */2 * 1", "2024-01-01T00:00:00Z", 15],      // */2 counts as unrestricted: odd days AND Monday
  ["0 0 1-7 * 1", "2024-01-01T00:00:00Z", 15],      // both restricted: first week OR Monday
  ["0 0 * * 7", "2024-01-01T00:00:00Z", 5],
  ["0 0 * * 1/2", "2024-01-01T00:00:00Z", 10],      // Mon, Wed, Fri (a/s runs to 6, not 7)
  ["*/15 9-17 * * MON-FRI", "2024-03-08T16:50:00Z", 10],
  ["30 4 1,15 * 5", "2024-01-01T00:00:00Z", 12],
  ["@yearly", "2024-06-01T00:00:00Z", 3], ["@weekly", "2024-06-01T00:00:00Z", 3], ["@hourly", "2024-06-01T00:30:00Z", 3],
  ["* * * * *", "2024-01-01T00:00:30Z", 3],         // from with seconds
  ["0 0 * * *", "2024-01-01T00:00:00Z", 2],         // from exactly on a match: strictly after
  ["* * * * *", "2024-01-01T00:00:00.500Z", 2],
  ["59 23 31 12 *", "2024-12-31T23:59:00Z", 2],     // year boundary, from on the match
  ["0 0 1 1 *", "2024-01-01T00:00:00Z", 1000],      // window: about 11 results
  ["* * * * *", "2024-02-28T23:00:00Z", 1000],      // 1000 consecutive minutes across Feb 29
  ["0 0 * * *", "2024-03-10T09:00:00Z", 5],         // US DST start (local-time bugs show here)
  ["30 1 * * *", "2024-11-03T00:00:00Z", 3],        // US DST end
  ["0 0 1 * *", "1999-12-31T23:59:59.999Z", 3],
  ["0 0 29 2 1", "2024-01-01T00:00:00Z", 10],       // Feb 29 OR Monday
  ["0 0 15 * *", "2024-01-01T00:00:00Z", 4],
];
for (const [e, f, n] of fixed) checkNext("next_fixed", e, U(f), n);

// ---------------------------------------------------------------- 3. random differential (seeded)
let seed = 20261117;
const rnd = () => { seed = (seed + 0x6D2B79F5) | 0; let t = seed; t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
const ri = (a, b) => a + Math.floor(rnd() * (b - a + 1));
function tokenOf(v, f) {
  if (f.names && rnd() < 0.3 && !(f.key === "dayOfWeek" && v === 7)) {
    const name = f.names[v - f.base]; return rnd() < 0.5 ? name : name.toLowerCase();
  }
  return rnd() < 0.05 ? String(v).padStart(2, "0") : String(v);
}
function randItem(f) {
  const r = rnd();
  if (r < 0.35) return tokenOf(ri(f.min, f.max), f);
  // ranges start at most at the field's top (day of week: 6): "7-7" is ambiguous in the prompt (7 "only as a value or a
  // range end"), so it is a corner case below, not a random one
  if (r < 0.6) { const a = ri(f.min, f.top), b = ri(a, f.max); return `${tokenOf(a, f)}-${tokenOf(b, f)}`; }
  if (r < 0.75) return `*/${ri(1, Math.max(2, Math.floor((f.top - f.min + 1) / 2)))}`;
  if (r < 0.88) { const a = ri(f.min, f.top), b = ri(a, f.max); return `${tokenOf(a, f)}-${tokenOf(b, f)}/${ri(1, 6)}`; }
  return `${tokenOf(ri(f.min, f.top), f)}/${ri(1, 8)}`;
}
function randField(f, starP) {
  if (rnd() < starP) return "*";
  const k = rnd() < 0.6 ? 1 : ri(2, 3);
  return Array.from({ length: k }, () => randItem(f)).join(",");
}
const randExprs = [];
for (let i = 0; i < 600; i++) {
  const e = [randField(F[0], 0.15), randField(F[1], 0.3), randField(F[2], 0.55), randField(F[3], 0.6), randField(F[4], 0.55)].join(" ");
  const from = new Date(Date.UTC(ri(1995, 2085), ri(0, 11), ri(1, 28), ri(0, 23), ri(0, 59), ri(0, 59), ri(0, 999)));
  randExprs.push(e);
  checkParse("parse_random", e);
  checkNext("next_random", e, from, ri(1, 40));
}

// ---------------------------------------------------------------- 4. errors
const bad = [
  "", "   ", "* * * *", "* * * * * *", "60 * * * *", "* 24 * * *", "* * 0 * *", "* * 32 * *", "* * * 0 *", "* * * 13 *",
  "* * * * 8", "5-2 * * * *", "* * * * SAT-SUN", "*/0 * * * *", "*/ * * * *", "1-5/ * * * *", "1-5/0 * * * *",
  "1,,2 * * * *", ",1 * * * *", "1, * * * *", "* * * FOO *", "* * * * MONDAY", "* * * JANUARY *", "@reboot", "@every",
  "@daily *", "? * * * *", "* * ? * *", "* * L * *", "* * 15W * *", "* * * * 5#3", "* * * * 5L", "-1 * * * *",
  "+1 * * * *", "1.5 * * * *", "a * * * *", "MON * * * *", "* * JAN * *", "* * * * */MON", "*/1-2 * * * *",
  "1-2-3 * * * *", "1/2/3 * * * *", "-5 * * * *", "5- * * * *", "* * * * 1-", "0x5 * * * *", "1e1 * * * *",
];
for (const e of bad) {
  for (const [name, fn] of [["parse", () => parse(e)], ["nextRuns", () => nextRuns(e, U("2024-01-01T00:00:00Z"), 1)]]) {
    const r = tryCall(fn);
    record("errors_cron", !!r.e && r.e instanceof CronError, { call: name, expr: e, got: r.e ? show(r.e) : show(r.v) });
  }
}
const badArgs = [
  ["invalid Date", () => nextRuns("* * * * *", new Date("nope"), 1)], ["string from", () => nextRuns("* * * * *", "2024-01-01", 1)],
  ["null from", () => nextRuns("* * * * *", null, 1)], ["number from", () => nextRuns("* * * * *", 1704067200000, 1)],
  ["n=0", () => nextRuns("* * * * *", U("2024-01-01T00:00:00Z"), 0)], ["n=1001", () => nextRuns("* * * * *", U("2024-01-01T00:00:00Z"), 1001)],
  ["n=1.5", () => nextRuns("* * * * *", U("2024-01-01T00:00:00Z"), 1.5)], ["n='5'", () => nextRuns("* * * * *", U("2024-01-01T00:00:00Z"), "5")],
  ["n missing", () => nextRuns("* * * * *", U("2024-01-01T00:00:00Z"))], ["n=-1", () => nextRuns("* * * * *", U("2024-01-01T00:00:00Z"), -1)],
  ["n=NaN", () => nextRuns("* * * * *", U("2024-01-01T00:00:00Z"), NaN)],
];
for (const [name, fn] of badArgs) { const r = tryCall(fn); record("errors_type", !!r.e && r.e instanceof TypeError, { case: name, got: r.e ? show(r.e) : show(r.v) }); }
const cronErrorIsError = typeof CronError === "function" && new CronError("x") instanceof Error;
record("errors_type", cronErrorIsError, { case: "CronError extends Error" });

// ---------------------------------------------------------------- 5. performance (< 100 ms per call)
nextRuns("*/5 * * * *", U("2024-01-01T00:00:00Z"), 5);   // warm-up
for (const [e, f, n] of [
  ["0 0 30 2 *", "2024-01-01T00:00:00Z", 1], ["0 0 31 2,4,6,9,11 *", "2024-01-01T00:00:00Z", 1000], ["0 0 29 2 *", "2097-01-01T00:00:00Z", 1000],
  ["59 23 31 12 *", "2024-01-01T00:00:00Z", 1000], ["* * * * *", "2024-01-01T00:00:00Z", 1000], ["0 0 13 * 5", "2024-01-01T00:00:00Z", 1000],
  ["*/7 */5 */3 */2 */2", "2024-01-01T00:00:00Z", 1000], ["0 0 */2 * 1", "2024-01-01T00:00:00Z", 1000],
]) {
  const t0 = performance.now(); const r = tryCall(() => nextRuns(e, U(f), n)); const ms = performance.now() - t0;
  record("performance", !r.e && ms < 100, { expr: e, n, ms: +ms.toFixed(1), error: r.e ? show(r.e) : undefined });
}

// ---------------------------------------------------------------- 6. spec corners (reported, not in the main score)
for (const e of ["*,5 * * * *", "5,* * * * *", " @daily ", "*/100 * * * *", "0 0 * * 0-7", "0 0 * * 7-7", "0 0 1-31/40 * *", "0 0 * * sun-sat"])
  checkParse("corners", e);
checkNext("corners", "0 0 * * 0", U("2024-01-06T23:59:59.999Z"), 2);

// ---------------------------------------------------------------- report
const main = ["parse_fixed", "next_fixed", "parse_random", "next_random", "errors_cron", "errors_type", "performance"];
let P = 0, T = 0;
const summary = {};
for (const k of [...main, "corners"]) {
  const c = cats[k] || { pass: 0, fail: 0 };
  summary[k] = `${c.pass}/${c.pass + c.fail}`;
  if (main.includes(k)) { P += c.pass; T += c.pass + c.fail; }
}
console.log(JSON.stringify({ file, score: `${P}/${T}`, pct: +(100 * P / T).toFixed(1), summary }, null, 1));
for (const k of [...main, "corners"]) {
  const c = cats[k]; if (!c || !c.fail) continue;
  console.log(`\n--- ${k}: ${c.fail} failed${verbose ? "" : " (first 4)"}`);
  for (const f of c.failures.slice(0, verbose ? 12 : 4)) console.log("  " + JSON.stringify(f));
}
