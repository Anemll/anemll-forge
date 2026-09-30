/**
 * Test suite for live-throughput-status.ts
 *
 * Run:  bun run ~/.pi/agent/extensions/tests/live-throughput-status.test.ts
 *
 * Lives in extensions/tests/ (no index.ts) so pi's auto-discovery
 * (`extensions/*.ts`, `extensions/*\/index.ts`) does not load it.
 */

import assert from "node:assert/strict";
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import mod, { formatDuration, parsePrefillMetrics } from "./live-throughput-status.ts";

type Status = string | undefined;

/**
 * Every harness gets its own temp state dir by default, so calibration from one
 * test cannot leak into another. Pass `dir` to share state (persistence test).
 */
function harness(mode: "tui" | "rpc" | "print" = "tui", dirOverride?: string) {
	const dir = dirOverride ?? mkdtempSync(join(tmpdir(), "ltps-"));
	const prev = process.env.PI_LIVE_THROUGHPUT_DIR;
	process.env.PI_LIVE_THROUGHPUT_DIR = dir;

	const handlers: Record<string, Function[]> = {};
	const statuses: Status[] = [];
	const pi = { on: (n: string, f: Function) => (handlers[n] ??= []).push(f) } as any;
	const ctx = {
		mode,
		model: { provider: "test", id: "m1" },
		ui: {
			setStatus: (_k: string, t?: Status) => statuses.push(t),
			theme: { fg: (_c: string, s: string) => s },
		},
	} as any;
	mod(pi);

	// Factory already resolved paths; restore env so other code is unaffected.
	if (prev === undefined) delete process.env.PI_LIVE_THROUGHPUT_DIR;
	else process.env.PI_LIVE_THROUGHPUT_DIR = prev;

	const fire = async (n: string, e: any) => {
		for (const f of handlers[n] ?? []) await f(e, ctx);
	};
	return { fire, statuses, dir, statePath: join(dir, "live-throughput-state.json") };
}

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

function textDelta(chars: number, tokens?: number) {
	return {
		message: { role: "assistant" },
		assistantMessageEvent: {
			type: "text_delta",
			delta: "x".repeat(chars),
			...(tokens !== undefined ? { partial: { usage: { output: tokens } } } : {}),
		},
	};
}
function thinkDelta(chars: number, tokens?: number) {
	return {
		message: { role: "assistant" },
		assistantMessageEvent: {
			type: "thinking_delta",
			delta: "y".repeat(chars),
			...(tokens !== undefined ? { partial: { usage: { output: tokens } } } : {}),
		},
	};
}
function assistantStart(model = "m1") {
	return { message: { role: "assistant", provider: "test", model } };
}
function end(usage: any, stopReason = "stop") {
	return { message: { role: "assistant", stopReason, usage } };
}

async function startTurn(fire: (n: string, e: any) => Promise<void>, model = "m1") {
	await fire("session_start", {});
	await fire("before_provider_request", {});
	await fire("message_start", assistantStart(model));
}

const lastTokStatus = (statuses: Status[]) =>
	[...statuses].reverse().find((s) => s?.includes("tok/s")) ?? "n/a";

let passed = 0;
function ok(name: string, fn: () => void) {
	try {
		fn();
		passed++;
		console.log(`  ✓ ${name}`);
	} catch (err) {
		console.error(`  ✗ ${name}`);
		console.error(err instanceof Error ? err.message : err);
		process.exitCode = 1;
	}
}

async function testA() {
	const { fire, statuses } = harness();
	await startTurn(fire);
	await sleep(250);
	for (let i = 0; i < 50; i++) {
		await fire("message_update", textDelta(5));
		await sleep(20);
	}
	await fire("message_end", end({ input: 1000, output: 100, cacheRead: 0, cacheWrite: 0 }));

	ok("A first frame is 'warming up'", () => {
		assert.match(statuses[2]!, /warming up/);
		assert.doesNotMatch(statuses[2]!, /tok\/s/);
	});
	ok("A final decode rate is plausible (~90-110 tok/s)", () => {
		const live = lastTokStatus(statuses);
		const m = /([\d.]+) tok\/s TTFT: ([\d.]+)s\s+(\d+) tok/.exec(live);
		assert.ok(m, `unexpected format: ${live}`);
		const v = Number(m![1]);
		assert.ok(v > 80 && v < 120, `rate ${v}`);
	});
	ok("A shows Prompt count, not Input/TTFT", () => {
		const last = statuses[statuses.length - 1]!;
		assert.match(last, /Prompt: 1\.0k tok/);
		assert.doesNotMatch(last, /Input\/TTFT/);
	});
}

async function testB() {
	const { fire, statuses } = harness();
	await startTurn(fire);
	await sleep(300);
	await fire("message_update", textDelta(396));
	await sleep(500);
	await fire("message_update", textDelta(4));
	await fire("message_end", end({ input: 1000, output: 100, cacheRead: 0, cacheWrite: 0 }));

	ok("B batched final ~2 tok/s (not 198)", () => {
		const m = /([\d.]+) tok\/s TTFT:/.exec(lastTokStatus(statuses));
		assert.ok(m, `no rate: ${lastTokStatus(statuses)}`);
		const v = Number(m![1]);
		assert.ok(v < 5, `rate ${v}`);
	});
}

async function testC() {
	const { fire, statuses } = harness();
	await startTurn(fire);
	await fire("message_update", textDelta(4));
	await fire("message_end", end({ input: 10, output: 5, cacheRead: 0, cacheWrite: 0 }));
	ok("C single-chunk output -> rate unavailable", () => {
		assert.match(statuses[statuses.length - 1]!, /rate unavailable/);
	});
}

async function testD() {
	const { fire, statuses } = harness();
	await startTurn(fire);
	await fire("message_update", textDelta(1));
	await fire("message_end", end({ input: 10, output: 0, cacheRead: 0, cacheWrite: 0 }));
	ok("D explicit zero output -> '0 tok'", () => {
		assert.match(statuses[statuses.length - 1]!, /  0 tok/);
	});
}

async function testE() {
	const { fire, statuses } = harness();
	await startTurn(fire);
	await fire("message_update", textDelta(1));
	await fire("message_end", end({ input: 10, output: 5, cacheRead: 0, cacheWrite: 0 }, "aborted"));
	ok("E aborted -> 'aborted' status", () => {
		assert.match(statuses[statuses.length - 1]!, /aborted/);
	});
}

async function testF() {
	const { fire, statuses } = harness();
	await startTurn(fire);
	await fire("message_update", textDelta(1));
	await fire("message_end", end({ input: 10, output: 5, cacheRead: 0, cacheWrite: 0 }, "error"));
	ok("F error -> 'error' status", () => {
		assert.match(statuses[statuses.length - 1]!, /error/);
	});
}

async function testG() {
	const { fire, statuses } = harness();
	await startTurn(fire);
	await sleep(250);
	for (let i = 1; i <= 60; i++) {
		await fire("message_update", textDelta(2, i));
		await sleep(15);
	}
	await fire("message_end", end({ input: 500, output: 60, cacheRead: 0, cacheWrite: 0 }));
	ok("G exact-usage provider live has no '~'", () => {
		const live = lastTokStatus(statuses);
		assert.match(live, /  60 tok/);
		assert.doesNotMatch(live, /~/);
	});
}

async function testH() {
	const dir = mkdtempSync(join(tmpdir(), "ltps-"));
	try {
		const first = harness("tui", dir);
		await startTurn(first.fire);
		await sleep(250);
		for (let i = 0; i < 60; i++) {
			await first.fire("message_update", textDelta(2));
			await sleep(15);
		}
		await first.fire("message_end", end({ input: 500, output: 60, cacheRead: 0, cacheWrite: 0 }));
		ok("H calibrates to observed ratio 2 (2 chars/token)", () => {
			const state = JSON.parse(readFileSync(first.statePath, "utf8"));
			const bucket = state.models["test/m1"].other;
			assert.ok(Math.abs(bucket.ratio - 2) < 0.01, `ratio ${bucket.ratio}`);
			assert.ok(bucket.samples >= 1);
		});

		// A fresh instance in the same dir must load the learned ratio.
		const fresh = harness("tui", dir);
		await startTurn(fresh.fire);
		await sleep(250);
		for (let i = 0; i < 60; i++) {
			await fresh.fire("message_update", textDelta(2));
			await sleep(15);
		}
		const freshLive = lastTokStatus(fresh.statuses);
		ok("H reloaded instance uses persisted ratio (~60 tok)", () => {
			const m = /\s{2,}~?([\d.]+) tok/.exec(freshLive);
			assert.ok(m, `live: ${freshLive}`);
			const v = Number(m![1]);
			assert.ok(v > 40 && v <= 60, `tokens ${v} in ${freshLive}`);
		});
	} finally {
		rmSync(dir, { recursive: true, force: true });
	}
}

async function testI() {
	const dir = mkdtempSync(join(tmpdir(), "ltps-"));
	try {
		const { fire } = harness("tui", dir);
		await startTurn(fire);
		await sleep(250);
		for (let i = 0; i < 30; i++) {
			await fire("message_update", thinkDelta(1)); // 1 char/token
			await sleep(10);
		}
		for (let i = 0; i < 30; i++) {
			await fire("message_update", textDelta(4)); // 4 chars/token
			await sleep(10);
		}
		await fire("message_end", end({ input: 10, output: 60, reasoning: 30, cacheRead: 0, cacheWrite: 0 }));
		ok("I per-content calibration stores distinct thinking/other ratios", () => {
			const state = JSON.parse(readFileSync(join(dir, "live-throughput-state.json"), "utf8"));
			const bucket = state.models["test/m1"];
			assert.ok(Math.abs(bucket.thinking.ratio - 1) < 0.01, `thinking ${bucket.thinking.ratio}`);
			assert.ok(Math.abs(bucket.other.ratio - 4) < 0.01, `other ${bucket.other.ratio}`);
		});
	} finally {
		rmSync(dir, { recursive: true, force: true });
	}
}

async function testJ() {
	const { fire, statuses } = harness();
	await fire("message_start", assistantStart()); // no before_provider_request
	await fire("message_update", textDelta(4));
	await fire("message_end", end({ input: 10, output: 5, cacheRead: 0, cacheWrite: 0 }));
	ok("J missing request hook -> TTFT flagged approximate", () => {
		assert.match(statuses[statuses.length - 1]!, /~TTFT:/);
	});
}

async function testK() {
	const dir = mkdtempSync(join(tmpdir(), "ltps-"));
	writeFileSync(
		join(dir, "live-throughput-config.json"),
		JSON.stringify({ idleClearMs: 50, updateIntervalMs: 0, minLiveWindowMs: 0 }),
	);
	try {
		const { fire, statuses } = harness("tui", dir);
		await startTurn(fire);
		await fire("message_update", textDelta(4));
		await fire("message_end", end({ input: 10, output: 5, cacheRead: 0, cacheWrite: 0 }));
		await sleep(120);
		ok("K idleClearMs config clears status after idle", () => {
			assert.equal(statuses[statuses.length - 1], undefined);
		});
	} finally {
		rmSync(dir, { recursive: true, force: true });
	}
}

async function testL() {
	const dir = mkdtempSync(join(tmpdir(), "ltps-"));
	writeFileSync(
		join(dir, "live-throughput-config.json"),
		JSON.stringify({ statusModes: ["tui"] }),
	);
	try {
		const { fire, statuses } = harness("rpc", dir);
		await startTurn(fire);
		await fire("message_update", textDelta(4));
		ok("L statusModes config suppresses RPC", () => {
			assert.equal(statuses.length, 0);
		});
	} finally {
		rmSync(dir, { recursive: true, force: true });
	}
}

async function testM() {
	const { fire, statuses } = harness("rpc");
	await startTurn(fire);
	await fire("message_update", textDelta(4));
	ok("M RPC mode receives status by default", () => {
		assert.ok(
			statuses.length > 0 &&
				statuses.some((s) => s?.includes("tok/s") || s?.includes("waiting")),
		);
	});
}

async function testN() {
	const { fire, statuses } = harness();
	await startTurn(fire);
	await fire("message_end", end({ input: 12345, output: 5, cacheRead: 0, cacheWrite: 0 }));
	ok("N large prompt count is formatted (12.3k)", () => {
		assert.match(statuses[statuses.length - 1]!, /Prompt: 12\.3k tok/);
	});
}

async function testO() {
	ok("O formatDuration scales s/m/h/d", () => {
		const cases: Array<[number, string]> = [
			[0, "0s"],
			[30_000, "30s"],
			[59_000, "59s"],
			[60_000, "1m 0s"],
			[90_000, "1m 30s"],
			[59 * 60_000, "59m 0s"],
			[3_600_000, "1h 0m"],
			[3_661_000, "1h 1m"],
			[86_399_000, "23h 59m"],
			[86_400_000, "1d 0h"],
			[90_000_000, "1d 1h"],
		];
		for (const [ms, expected] of cases) {
			assert.equal(formatDuration(ms), expected, `${ms}ms -> ${formatDuration(ms)}`);
		}
	});
}

async function testP() {
	const { fire, statuses } = harness();
	await fire("agent_start", {});
	await startTurn(fire);
	await sleep(250);
	for (let i = 0; i < 20; i++) {
		await fire("message_update", textDelta(4));
		await sleep(15);
	}
	await fire("message_end", end({ input: 500, output: 20, cacheRead: 0, cacheWrite: 0 }));
	await fire("agent_settled", {});
	ok("P agent_settled appends 'took Xs' to the final status", () => {
		const last = statuses[statuses.length - 1]!;
		assert.match(last, /took \d+s$/);
		assert.match(last, /tok\/s/);
	});
}

async function testQ() {
	const dir = mkdtempSync(join(tmpdir(), "ltps-"));
	writeFileSync(join(dir, "live-throughput-config.json"), JSON.stringify({ showElapsed: false }));
	try {
		const { fire, statuses } = harness("tui", dir);
		await fire("agent_start", {});
		await startTurn(fire);
		await fire("message_update", textDelta(4));
		await fire("message_end", end({ input: 10, output: 5, cacheRead: 0, cacheWrite: 0 }));
		await fire("agent_settled", {});
		ok("Q showElapsed:false suppresses 'took'", () => {
			assert.doesNotMatch(statuses[statuses.length - 1]!, /took/);
		});
	} finally {
		rmSync(dir, { recursive: true, force: true });
	}
}

async function testR() {
	ok("R parsePrefillMetrics sums local_compute + ttft across engines", () => {
		const text = [
			"# HELP vllm:prompt_tokens_by_source_total ...",
			'vllm:prompt_tokens_by_source_total{engine="0",source="local_compute"} 1000',
			'vllm:prompt_tokens_by_source_total{engine="0",source="local_cache_hit"} 999999',
			'vllm:prompt_tokens_by_source_total{engine="1",source="local_compute"} 200',
			'vllm:time_to_first_token_seconds_sum{engine="0"} 10.5',
			'vllm:time_to_first_token_seconds_sum{engine="1"} 12.5',
			'vllm:time_to_first_token_seconds_count{engine="0"} 1',
			'vllm:time_to_first_token_seconds_count{engine="1"} 2',
		].join("\n");
		const snap = parsePrefillMetrics(text);
		assert.equal(snap.localCompute, 1200);
		assert.equal(snap.ttftSum, 23);
		assert.equal(snap.ttftCount, 3);
	});
}

async function testS() {
	const dir = mkdtempSync(join(tmpdir(), "ltps-"));
	writeFileSync(
		join(dir, "live-throughput-config.json"),
		JSON.stringify({ metricsUrls: { test: "http://metrics.local" }, metricsTimeoutMs: 200 }),
	);
	const origFetch = globalThis.fetch;
	const snapshots = [
		'vllm:prompt_tokens_by_source_total{source="local_compute"} 1000\nvllm:time_to_first_token_seconds_sum 10\nvllm:time_to_first_token_seconds_count 1\n',
		'vllm:prompt_tokens_by_source_total{source="local_compute"} 2600\nvllm:time_to_first_token_seconds_sum 12\nvllm:time_to_first_token_seconds_count 2\n',
	];
	let call = 0;
	(globalThis as any).fetch = async () => ({
		ok: true,
		text: async () => snapshots[Math.min(call++, snapshots.length - 1)],
	});
	try {
		const { fire, statuses } = harness("tui", dir);
		await startTurn(fire); // before_provider_request snapshots #0
		await fire("message_update", textDelta(4));
		await fire("message_end", end({ input: 100, output: 5, cacheRead: 0, cacheWrite: 0 })); // samples #1
		ok("S derives prefill TPS from server metrics (1600 tok / 2s = 800)", () => {
			assert.match(statuses[statuses.length - 1]!, /800 t\/s/);
		});
	} finally {
		globalThis.fetch = origFetch;
		rmSync(dir, { recursive: true, force: true });
	}
}

async function testT() {
	const { fire, statuses } = harness();
	await startTurn(fire);
	// Big batched first chunk: ~100 estimated tokens at seed ratio 4.
	await fire("message_update", textDelta(400));
	await sleep(300); // pass minLiveWindowMs
	// Only ~1 token actually decoded inside the window.
	await fire("message_update", textDelta(4));
	ok("T live fencepost excludes chunk 1 (no inflation)", () => {
		const live = lastTokStatus(statuses);
		const m = /([\d.]+) tok\/s/.exec(live);
		assert.ok(m, `no rate: ${live}`);
		const v = Number(m![1]);
		assert.ok(v < 20, `rate ${v} (chunk 1 must be excluded)`);
	});
}

async function testU() {
	const { fire, statuses } = harness();
	await startTurn(fire);
	await fire("message_update", textDelta(4));
	await fire("message_end", end({ input: 10, output: 5, cacheRead: 0, cacheWrite: 0 }));
	await fire("session_shutdown", {});
	ok("U session_shutdown clears the status line", () => {
		assert.equal(statuses[statuses.length - 1], undefined);
	});
}

async function testV() {
	const { fire, statuses } = harness();
	await fire("agent_start", {});
	await startTurn(fire);
	await fire("message_update", textDelta(4));
	await sleep(250);
	await fire("message_update", textDelta(4));
	await fire("message_end", end({ input: 100, output: 5, cacheRead: 0, cacheWrite: 0 }));
	await fire("turn_end", {});
	const afterTurnEnd = statuses[statuses.length - 1];
	await fire("agent_settled", {});
	const finalStatus = statuses[statuses.length - 1]!;
	ok("V turn_end clears footer; agent_settled restores final line", () => {
		assert.equal(afterTurnEnd, undefined);
		assert.match(finalStatus, /tok\/s/);
		assert.match(finalStatus, /took/);
	});
}

async function testW() {
	const dir = mkdtempSync(join(tmpdir(), "ltps-"));
	writeFileSync(
		join(dir, "live-throughput-config.json"),
		JSON.stringify({ showGenerationTps: true }),
	);
	try {
		const { fire, statuses } = harness("tui", dir);
		await startTurn(fire);
		await fire("message_update", textDelta(4));
		await fire("message_end", end({ input: 100, output: 50, cacheRead: 0, cacheWrite: 0 }));
		ok("W generation throughput = output / (request -> message_end)", () => {
			assert.match(statuses[statuses.length - 1]!, /gen: [\d.]+ tok\/s/);
		});
	} finally {
		rmSync(dir, { recursive: true, force: true });
	}
}

async function testX() {
	const { fire, statuses } = harness();
	await startTurn(fire);
	// Big first chunk (~100 est. tokens), then ~1 est. token inside the window.
	await fire("message_update", textDelta(400));
	await sleep(300);
	await fire("message_update", textDelta(4));
	// No provider output usage -> final uses the estimated fallback path.
	await fire("message_end", end({ input: 10, cacheRead: 0, cacheWrite: 0 }));
	ok("X estimated final rate also excludes chunk 1", () => {
		const final = lastTokStatus(statuses);
		const m = /([\d.]+) tok\/s/.exec(final);
		assert.ok(m, `no rate: ${final}`);
		const v = Number(m![1]);
		assert.ok(v < 20, `rate ${v} (chunk 1 must be excluded)`);
	});
}

async function main() {
	console.log("live-throughput-status tests\n");
	await testA();
	await testB();
	await testC();
	await testD();
	await testE();
	await testF();
	await testG();
	await testH();
	await testI();
	await testJ();
	await testK();
	await testL();
	await testM();
	await testN();
	await testO();
	await testP();
	await testQ();
	await testR();
	await testS();
	await testT();
	await testU();
	await testV();
	await testW();
	await testX();
	console.log(`\n${passed} passed${process.exitCode ? ", some failed" : ""}`);
}

main();
