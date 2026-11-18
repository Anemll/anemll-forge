// Self-test of grade.mjs's reference: refNext against a plain minute-by-minute scan, and hand-checked cases.
import { readFileSync } from "node:fs";
const src = readFileSync(new URL("./grade.mjs", import.meta.url), "utf8");
const ref = src.slice(src.indexOf("const MONTHS"), src.indexOf("// ---------------------------------------------------------------- harness"));
const m = await import("data:text/javascript," + encodeURIComponent(ref + "\nexport { refParse, refNext, F };"));
const { refParse, refNext } = m;
function scan(expr, from, n, minutes) {
  const p = refParse(expr), D = new Set(p.dayOfMonth), W = new Set(p.dayOfWeek), Mo = new Set(p.month), H = new Set(p.hour), Mi = new Set(p.minute);
  const out = []; let t = Math.floor(from.getTime() / 60000) * 60000 + 60000;
  for (let i = 0; i < minutes && out.length < n; i++, t += 60000) {
    const d = new Date(t);
    if (!Mi.has(d.getUTCMinutes()) || !H.has(d.getUTCHours()) || !Mo.has(d.getUTCMonth() + 1)) continue;
    const dm = D.has(d.getUTCDate()), dw = W.has(d.getUTCDay());
    if (p.domRestricted && p.dowRestricted ? dm || dw : dm && dw) out.push(d.toISOString());
  }
  return out;
}
let bad = 0, n = 0;
const exprs = ["* * * * *", "0 0 29 2 *", "0 12 13 * 5", "0 0 */2 * 1", "0 0 1-7 * 1", "*/15 9-17 * * MON-FRI", "5/20 3 * * 1/2",
  "0 0 31 * *", "30 4 1,15 * 5", "0-10/3 1-3 */3 JAN-mar 0,7", "@weekly", "17 */5 10-20/4 * sun-wed", "0 0 * * 5-7"];
for (const e of exprs) for (const f of ["2024-01-01T00:00:00Z", "2023-12-31T23:59:30Z", "1999-02-27T12:00:00Z"]) {
  const from = new Date(f), MINS = 3 * 366 * 1440;
  const a = refNext(e, from, 30).filter(x => new Date(x) - from < (MINS - 2) * 60000), b = scan(e, from, 30, MINS);
  n++; if (JSON.stringify(a) !== JSON.stringify(b.slice(0, a.length)) || (a.length < 30 && b.length !== a.length)) { bad++; console.log("MISMATCH", e, f, a.slice(0, 3), b.slice(0, 3)); }
}
console.log(`refNext vs minute scan: ${n - bad}/${n} agree`);
console.log("Feb29 from 2096-03-01:", refNext("0 0 29 2 *", new Date("2096-03-01T00:00:00Z"), 3));
console.log("Feb29 from 1896:", refNext("0 0 29 2 *", new Date("1896-01-01T00:00:00Z"), 5));
console.log("13th or Fri:", refNext("0 12 13 * 5", new Date("2024-01-01T00:00:00Z"), 4));
console.log("odd day AND Mon:", refNext("0 0 */2 * 1", new Date("2024-01-01T00:00:00Z"), 3));
console.log("yearly count in window:", refNext("0 0 1 1 *", new Date("2024-01-01T00:00:00Z"), 1000).length);
console.log("dow 1/2:", refParse("0 0 * * 1/2").dayOfWeek, " MON/2:", refParse("0 0 * * MON/2").dayOfWeek, " 5-7:", refParse("0 0 * * 5-7").dayOfWeek, " */2:", refParse("0 0 * * */2").dayOfWeek);
