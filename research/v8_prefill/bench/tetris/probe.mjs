// Independent probe of both Tetris builds: loads the page script with DOM stubs and a hook, then checks
// piece sequence, O rotation, multi-row and split clears, input while paused, and a legality fuzz.
import { readFileSync } from "node:fs";
import vm from "node:vm";
const [file, flavor] = process.argv.slice(2);
let src = readFileSync(file, "utf8").match(/<script>([\s\S]*?)<\/script>/)[1];
if (flavor === "forge") {
  src = src.replace("reset();\n  requestAnimationFrame(loop);",
    `globalThis.__F = { get grid(){return grid}, set grid(g){grid=g}, get current(){return current}, set current(c){current=c},
      get next(){return next}, get score(){return score}, get lines(){return lines}, get over(){return over}, get running(){return running},
      makePiece, collides, reset, hardDrop, togglePause, rotatePiece };
    reset();\n  requestAnimationFrame(loop);`);
  if (!src.includes("__F")) throw new Error("hook not injected");
}
const listeners = {};
const ctx = new Proxy({}, { get: (t, p) => (p === "globalAlpha" ? 1 : () => {}), set: () => true });
const els = {};
const mk = id => { const cl = new Set(id === "paused" || id === "gameover" ? ["hidden"] : []);
  return { id, width: 300, height: 600, textContent: "", style: {}, get offsetWidth() { return 1; },
    classList: { add: c => cl.add(c), remove: c => cl.delete(c), contains: c => cl.has(c) },
    addEventListener: (e, f) => { (listeners[id + ":" + e] ||= []).push(f); }, getContext: () => ctx }; };
let now = 0, raf = null;
const sb = { document: { getElementById: id => (els[id] ||= mk(id)), addEventListener: (e, f) => { (listeners[e] ||= []).push(f); } },
  requestAnimationFrame: cb => { raf = cb; }, performance: { now: () => now }, console, Math };
sb.globalThis = sb; vm.createContext(sb); vm.runInContext(src, sb);
const key = k => { for (const f of listeners.keydown || []) f({ key: k, preventDefault() {} }); for (const f of listeners.keyup || []) f({ key: k, preventDefault() {} }); };
const tick = ms => { now += ms; const c = raf; raf = null; c && c(now); };
// uniform adapter
const S = flavor === "forge" ? (() => { const F = sb.__F; return {
    start: () => F.reset(), get cur() { return F.current; }, get grid() { return F.grid; }, set grid(g) { F.grid = g; },
    cells: p => p.cells, type: p => p.type, get over() { return F.over; }, get score() { return F.score; }, get lines() { return F.lines; },
    hard: () => F.hardDrop(), spawnType: t => { F.current = F.makePiece(t); }, setXY: (x, y) => { F.current.x = x; F.current.y = y; },
    rotate: () => key("ArrowUp"), pause: () => key("p"), get paused() { return !F.running && !F.over; } }; })()
  : (() => { const T = sb.__TET; return {
    start: () => T.startGame(), get cur() { return T.current; }, get grid() { return T.grid; }, set grid(g) { T.grid.splice(0, 20, ...g); },
    cells: p => T.genShape(p.type).states[p.rot % 4], type: p => p.type, get over() { return T.state === "over"; }, get score() { return T.score; },
    get lines() { return T.lines; }, hard: () => { T.hardDrop(); for (let i = 0; i < 10 && T.clearing; i++) tick(50); },
    spawnType: t => T.debugSpawn(t), setXY: (x, y) => { T.current.x = x; T.current.y = y; }, rotate: () => T.rotate(),
    pause: () => T.togglePause(), get paused() { return T.state === "paused"; } }; })();
tick(16);
const out = {};
// 1. piece sequence over 28 spawns on a cleared board
S.start(); const seq = [];
for (let i = 0; i < 28; i++) { seq.push(S.type(S.cur)); S.grid = Array.from({ length: 20 }, () => Array(10).fill(null)); S.hard(); if (S.over) S.start(); }
out.sequence = seq.join(""); out.distinctTypes = new Set(seq).size;
// 2. O rotation: absolute cells before / after 4 rotations
S.start(); S.grid = Array.from({ length: 20 }, () => Array(10).fill(null)); S.spawnType("O"); S.setXY(4, 8);
const abs = () => S.cells(S.cur).map(([a, b]) => `${S.cur.x + a},${S.cur.y + b}`).sort().join(" ");
const o0 = abs(), oPath = [];
for (let i = 0; i < 4; i++) { S.rotate(); oPath.push(abs()); }
out.O_rotation = oPath.every(p => p === o0) ? "stays" : "moves: " + [o0, ...oPath].join(" | ");
// 3. split clear: rows 17 and 19 full except col 0 (row 18 has hole at col 5 too), vertical I in col 0
const G = () => Array.from({ length: 20 }, () => Array(10).fill(null));
S.start(); let g = G();
for (const r of [16, 17, 18, 19]) for (let c = 1; c < 10; c++) g[r][c] = "#abc";
g[18][5] = null; g[16][5] = null; g[15][9] = "#mark";
S.grid = g; S.spawnType("I");
if (flavor === "forge") { S.cur.cells = [[0,0],[0,1],[0,2],[0,3]]; S.setXY(0, 0); } else { S.cur.rot = 1; S.setXY(-2, 0); }
const sc0 = S.score, l0 = S.lines; S.hard();
const rowsStr = S.grid.slice(14).map(r => r.map(c => (c === "#mark" ? "M" : c ? "#" : ".")).join("")).join("/");
out.splitClear = { lines: S.lines - l0, rows14to19: rowsStr, expect: "rows 17 and 19 cleared, 16 and 18 (holes at col 5) kept and shifted down 2, M at row 17" };
// 4. input while paused
S.start(); S.grid = G(); S.spawnType("T"); S.setXY(4, 5); S.pause();
const x0 = S.cur.x; key("ArrowLeft"); key("ArrowLeft"); const y0 = S.cur.y; key("ArrowDown"); key("ArrowUp");
out.pausedInput = { paused: S.paused, xMoved: S.cur.x - x0, yMoved: S.cur.y - y0 };
S.pause();
// 5. fuzz: random keys and ticks, check legality after each action
let bad = 0, games = 0, actions = 0, totalLines = 0; let seed = 7;
const rnd = () => ((seed = (seed * 1103515245 + 12345) % 2147483648) / 2147483648);
const keys = ["ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown", " ", "ArrowLeft", "ArrowRight", "ArrowUp"];
S.start();
for (let i = 0; i < 20000; i++) {
  if (S.over) { totalLines += S.lines; games++; S.start(); }
  const r = rnd(); if (r < 0.15) tick(50 + rnd() * 400); else key(keys[Math.floor(rnd() * keys.length)]);
  if (flavor !== "forge") for (let j = 0; j < 6 && sb.__TET.clearing; j++) tick(50);
  actions++; const p = S.cur; if (!p || S.over) continue;
  for (const [a, b] of S.cells(p)) { const x = p.x + a, y = p.y + b; if (x < 0 || x > 9 || y > 19 || (y >= 0 && S.grid[y][x])) { bad++; break; } }
}
out.fuzz = { actions, games, illegalStates: bad, linesInFinishedGames: totalLines };
console.log(JSON.stringify(out, null, 1));
