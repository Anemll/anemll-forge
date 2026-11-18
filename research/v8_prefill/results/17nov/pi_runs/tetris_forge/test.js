// Headless smoke test: stubs DOM, loads the game script, simulates input & ticks.
"use strict";
const fs = require("fs");
const path = require("path");
const vm = require("vm");

const html = fs.readFileSync(path.join(__dirname, "index.html"), "utf8");
const m = html.match(/<script>([\s\S]*?)<\/script>/);
if (!m) throw new Error("script block not found");
const js = m[1];

let rafCbs = [];
const ctxStub = () => ({
  clearRect() {}, fillRect() {}, strokeRect() {}, beginPath() {},
  moveTo() {}, lineTo() {}, stroke() {},
  fillStyle: "", strokeStyle: "", globalAlpha: 1,
});

function makeEl(id) {
  return {
    id,
    width: 300, height: 600,
    textContent: "",
    classList: {
      _s: new Set(),
      add(c) { this._s.add(c); },
      remove(c) { this._s.delete(c); },
      contains(c) { return this._s.has(c); },
    },
    getContext: ctxStub,
    addEventListener(type, fn) {
      this._listeners = this._listeners || {};
      this._listeners[type] = fn;
    },
  };
}

const elements = {};
["board", "next", "score", "lines", "level", "overlay", "overlayTitle", "overlaySub", "resetBtn"].forEach((id) => {
  elements[id] = makeEl(id);
  elements[id].width = id === "next" ? 120 : 300;
  elements[id].height = id === "next" ? 80 : 600;
});

let now = 0;
const sandbox = {
  document: {
    getElementById: (id) => elements[id],
    addEventListener(type, fn) {
      if (type === "keydown") sandbox._keyHandler = fn;
    },
  },
  performance: { now: () => now },
  requestAnimationFrame: (fn) => { rafCbs.push(fn); },
  console,
};
vm.createContext(sandbox);
vm.runInContext(js, sandbox);

function pumpFrames(ms) {
  now += ms;
  const cbs = rafCbs;
  rafCbs = [];
  for (const cb of cbs) cb(now);
}
function press(key) {
  sandbox._keyHandler({ key, preventDefault() {} });
}

let failures = 0;
function assert(cond, msg) {
  if (cond) console.log("PASS:", msg);
  else { console.error("FAIL:", msg); failures++; }
}

// initial state
assert(elements.score.textContent === "0", "initial score is 0");
assert(elements.level.textContent === "1", "initial level is 1");

// rotate & move don't crash
press("ArrowUp"); press("ArrowDown"); press("ArrowLeft"); press("ArrowRight"); press("x"); press("z");
assert(true, "basic keys handled without throwing");

// hard drop locks the first piece -> grid has blocks, next piece spawned
press(" ");
assert(true, "hard drop without throwing");
const scoreAfterDrop = elements.score.textContent;
assert(Number(scoreAfterDrop) > 0, "hard drop awards drop-distance points, got " + scoreAfterDrop);

// let many pieces fall and hard-drop a lot to trigger line clears eventually
for (let i = 0; i < 50; i++) {
  press(" ");
  pumpFrames(50);
}
assert(true, "50 hard drops without throwing");
const lines = Number(elements.lines.textContent);
console.log("lines cleared:", lines, "level:", elements.level.textContent, "score:", elements.score.textContent);

// pause / resume (reset first if the stacking loop caused a game over)
if (elements.overlay.classList.contains("show")) {
  elements.resetBtn._listeners.click();
}
press("p");
assert(elements.overlay.classList.contains("show"), "pause shows overlay");
assert(elements.overlayTitle.textContent === "PAUSED", "pause overlay title");
press("p");
assert(!elements.overlay.classList.contains("show"), "resume hides overlay");

// gravity over time
const beforeScore = Number(elements.score.textContent);
pumpFrames(5000);
assert(true, "5s of gravity ticks without throwing");

// reset via button
elements.resetBtn._listeners.click();
assert(elements.score.textContent === "0", "reset clears score");
assert(elements.level.textContent === "1", "reset clears level");

// play until game over (stack pieces centered to fill the board)
let guard = 0;
while (!elements.overlay.classList.contains("show") && guard < 500) {
  press(" ");
  pumpFrames(10);
  guard++;
}
assert(elements.overlay.classList.contains("show"), "game over overlay eventually appears");
assert(elements.overlayTitle.textContent === "GAME OVER", "game over title shown");
assert(/Score: \d+/.test(elements.overlaySub.textContent), "game over shows final score");

// Enter restarts
press("Enter");
assert(!elements.overlay.classList.contains("show"), "Enter restarts after game over");
pumpFrames(1000);
assert(true, "game runs after restart");
assert(elements.score.textContent === "0", "restart resets score");

if (failures) {
  console.error(failures + " test(s) failed");
  process.exit(1);
} else {
  console.log("All tests passed");
}
