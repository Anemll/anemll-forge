// Headless test harness for tetris-v8-splash-1/index.html
import { readFileSync } from "node:fs";
import vm from "node:vm";
import { strict as assert } from "node:assert";

const html = readFileSync(new URL("./index.html", import.meta.url), "utf8");
const m = html.match(/<script>([\s\S]*?)<\/script>/);
if (!m) throw new Error("no <script> found");
const src = m[1];

// ---------- DOM stubs ----------
const listeners = {}; // event -> [handler,...]
function makeCtx() {
  return new Proxy({}, {
    get(t, prop) {
      if (prop === "globalAlpha") return 1;
      return () => {};
    },
    set() { return true; },
  });
}
function makeEl(id) {
  const classes = new Set();
  if (id === "paused" || id === "gameover") classes.add("hidden");
  return {
    id,
    width: 280,
    height: 560,
    textContent: "",
    get offsetWidth() { return 100; },
    classList: {
      add: c => classes.add(c),
      remove: c => classes.delete(c),
      contains: c => classes.has(c),
    },
    addEventListener() {},
    click() {},
    style: {},
    getContext: () => makeCtx(),
  };
}
const els = {};
const documentStub = {
  getElementById: id => (els[id] ||= makeEl(id)),
  addEventListener: (ev, fn) => { (listeners[ev] ||= []).push(fn); },
};
let rafCb = null;
const sandbox = {
  document: documentStub,
  requestAnimationFrame: cb => { rafCb = cb; },
  console,
};
sandbox.globalThis = sandbox;
vm.createContext(sandbox);
vm.runInContext(src, sandbox, { filename: "tetris.js" });

const T = sandbox.__TET;
function press(k) { for (const fn of listeners["keydown"] || []) fn({ key: k, preventDefault() {} }); }
function release(k) { for (const fn of listeners["keyup"] || []) fn({ key: k, preventDefault() {} }); }
let now = 0;
function step(ms) {
  const t = (now += ms);
  const cb = rafCb; rafCb = null;
  cb(t);
}
function pieceLegal() {
  assert.equal(T.grid.length, 20);
  for (const row of T.grid) assert.equal(row.length, 10);
  const p = T.current;
  if (!p) return;
  const cells = T.genShape(p.type).states[p.rot % 4];
  for (const [cx, cy] of cells) {
    const x = p.x + cx, y = p.y + cy;
    assert.ok(x >= 0 && x < 10, `piece cell OOB x=${x}`);
    assert.ok(y < 20, `piece cell OOB y=${y}`);
    if (y >= 0) assert.ok(!T.grid[y][x], `piece overlaps filled cell at ${x},${y}`);
  }
}
let passed = 0;
function test(name, fn) {
  fn();
  passed++;
  console.log("  ok -", name);
}

console.log("tetris headless tests");

test("boots into splash state", () => {
  assert.equal(T.state, "splash");
  assert.equal(els.splash.classList.contains("hidden"), false);
});

test("Enter starts the game", () => {
  press("Enter");
  release("Enter");
  assert.equal(T.state, "playing");
  assert.ok(T.current, "a piece should spawn");
  assert.equal(els.splash.classList.contains("hidden"), true);
});

test("7-bag queue primed (5 queued after spawn+next)", () => {
  assert.equal(T.queue.length, 5);
  assert.ok(T.next, "next slot filled");
});

test("all piece shapes are well-formed", () => {
  for (const type of ["I", "O", "T", "S", "Z", "J", "L"]) {
    const s = T.genShape(type);
    assert.ok(s.color.startsWith("#"), type + " color");
    for (let r = 0; r < 4; r++) {
      const cells = s.states[r];
      assert.equal(cells.length, 4, type + " state " + r + " cell count");
      const seen = new Set();
      for (const [x, y] of cells) {
        assert.ok(x >= 0 && x < 4 && y >= 0 && y < 4, type + " offset in range");
        seen.add(x + "," + y);
      }
      assert.equal(seen.size, 4, type + " state " + r + " unique cells");
    }
  }
});

test("move clamps at both walls", () => {
  const s = T.genShape(T.current.type).size;
  for (let i = 0; i < 20; i++) T.move(-1);
  const xMin = T.current.x;
  for (let i = 0; i < 20; i++) T.move(-1);
  assert.equal(T.current.x, xMin, "clamped at left wall");
  for (let i = 0; i < 40; i++) T.move(1);
  assert.equal(T.current.x, 10 - s, "clamped at right wall");
  for (let i = 0; i < 40; i++) T.move(-1);
  assert.equal(T.current.x, 0, "back at left wall");
  pieceLegal();
});

test("rotate cycles through 4 states and stays legal", () => {
  const rot0 = T.current.rot % 4;
  for (let i = 0; i < 4; i++) { T.rotate(); pieceLegal(); }
  assert.equal(T.current.rot % 4, rot0);
});

test("hard drop scores and spawns the next piece", () => {
  const scoreBefore = T.score;
  T.hardDrop();
  assert.ok(T.score > scoreBefore, "hard drop scores >=2");
  assert.ok(T.current, "new piece spawned");
  pieceLegal();
});

test("line clear: full tetris (4 lines)", () => {
  for (let r = 16; r < 20; r++)
    for (let c = 0; c < 10; c++) if (c !== 5) T.grid[r][c] = "#fff";
  for (let r = 16; r < 20; r++) T.grid[r][5] = null; // clear leftovers in col 5
  // vertical I into column 5: state1 cells are at x+2
  T.debugSpawn("I");
  T.current.rot = 1;
  T.current.x = 3; // 3 + 2 = col 5
  T.current.y = 16;
  pieceLegal();
  const scoreBefore = T.score;
  T.hardDrop();
  let guard = 0;
  while (T.clearing && guard++ < 10) step(100); // fast-forward the clear animation
  assert.equal(T.lines, 4, "4 lines cleared");
  assert.equal(T.score, scoreBefore + 800 * T.level, "tetris score");
  assert.ok(T.grid.every(row => row.every(c => !c)), "board empty after clearing through the stack");
});

test("hold swaps piece once per piece", () => {
  T.debugSpawn("T");
  const first = T.current.type;
  const nextBefore = T.next;
  T.doHold();
  assert.equal(T.hold, first);
  assert.equal(T.current.type, nextBefore, "next slot fills field after hold");
  const y = T.current.y;
  T.doHold(); // must be ignored (canHold false)
  assert.equal(T.hold, first);
  assert.equal(T.current.y, y);
});

test("pause freezes the board", () => {
  T.debugSpawn("O");
  T.togglePause();
  assert.equal(T.state, "paused");
  assert.equal(els.paused.classList.contains("hidden"), false);
  const y = T.current.y;
  for (let i = 0; i < 50; i++) step(50); // 2.5s paused
  assert.equal(T.current.y, y, "no gravity while paused");
  T.togglePause();
  assert.equal(T.state, "playing");
  assert.equal(els.paused.classList.contains("hidden"), true);
});

test("game over on block-out", () => {
  for (let r = 0; r < 12; r++) for (let c = 0; c < 10; c++) T.grid[r][c] = "#fff";
  T.debugSpawn("T"); // spawns inside the pile
  assert.equal(T.state, "over");
  assert.equal(els.gameover.classList.contains("hidden"), false);
  T.startGame(); // restart works
  assert.equal(T.state, "playing");
});

test("keyboard: soft drop, DAS auto-shift, stop, space drop", () => {
  T.startGame();
  T.debugSpawn("O");
  const y0 = T.current.y;
  press("ArrowDown"); step(100); release("ArrowDown");
  assert.ok(T.current.y > y0, "soft drop moves piece");

  for (let i = 0; i < 3; i++) T.move(1); // start near right wall so DAS has room
  press("ArrowLeft"); step(100);
  const x100 = T.current.x;
  for (let i = 0; i < 6; i++) step(100); // DAS=168 + ARR kicks in
  const xAfter = T.current.x;
  assert.ok(xAfter <= x100 - 5, "DAS auto-shifts left repeatedly");
  release("ArrowLeft");
  const xStop = T.current.x;
  step(200);
  assert.equal(T.current.x, xStop, "no drift after key release");

  press(" "); release(" ");
  assert.ok(T.current, "space hard-drops and spawns");
  pieceLegal();
});


test("fuzz: 3000 random actions keep invariants", () => {
  const seed = { s: 12345 };
  const rnd = () => (seed.s = (seed.s * 1103515245 + 12345) % 2147483648) / 2147483648;
  const actions = ["moveL", "moveR", "rot", "soft", "hard", "hold", "pause", "step"];
  let prevScore = 0;
  for (let i = 0; i < 3000; i++) {
    if (T.state === "over" || T.state === "splash") { T.startGame(); prevScore = T.score; continue; }
    if (T.state === "paused") T.togglePause();
    if (T.state !== "playing") continue;
    const a = actions[Math.floor(rnd() * actions.length)];
    if (a === "moveL") T.move(-1);
    else if (a === "moveR") T.move(1);
    else if (a === "rot") T.rotate();
    else if (a === "soft") T.softDrop();
    else if (a === "hard") T.hardDrop();
    else if (a === "hold") T.doHold();
    else if (a === "pause") T.togglePause();
    else step(10 + Math.floor(rnd() * 300));
    if (T.state === "playing") {
      assert.ok(T.score >= prevScore, "score monotonic");
      prevScore = T.score;
      pieceLegal(); // block-out piece overlap is allowed only in "over" state
    }
  }
  for (let i = 0; i < 20; i++) step(250); // play out any pending clear
});

console.log(`\n${passed} tests passed`);
