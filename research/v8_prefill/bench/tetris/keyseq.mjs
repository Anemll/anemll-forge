// Splash build, keyboard only: Enter to start, then Space (hard drop) with frames between; record each spawned type.
import { readFileSync } from "node:fs"; import vm from "node:vm";
const src = readFileSync(process.argv[2], "utf8").match(/<script>([\s\S]*?)<\/script>/)[1];
const L = {}; const ctx = new Proxy({}, { get: (t, p) => (p === "globalAlpha" ? 1 : () => {}), set: () => true }); const E = {};
const mk = id => { const c = new Set(); return { id, width: 1, height: 1, textContent: "", style: {}, offsetWidth: 1,
  classList: { add: x => c.add(x), remove: x => c.delete(x), contains: x => c.has(x) }, addEventListener() {}, getContext: () => ctx }; };
let raf = null, now = 0; const sb = { document: { getElementById: id => (E[id] ||= mk(id)), addEventListener: (e, f) => (L[e] ||= []).push(f) },
  requestAnimationFrame: cb => { raf = cb; }, console, Math }; sb.globalThis = sb; vm.createContext(sb); vm.runInContext(src, sb);
const down = k => L.keydown.forEach(f => f({ key: k, preventDefault() {} })), up = k => L.keyup.forEach(f => f({ key: k, preventDefault() {} }));
const frame = () => { now += 16; const c = raf; raf = null; c(now); };
frame(); down("Enter"); up("Enter"); frame();
const seq = []; for (let i = 0; i < 12; i++) { seq.push(sb.__TET.current?.type + "(next " + sb.__TET.next + ")"); down(" "); up(" "); for (let j = 0; j < 20; j++) frame(); }
console.log(seq.join(" "), "| state", sb.__TET.state);
