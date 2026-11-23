import type { ExtensionAPI, ExtensionContext } from "@earendil-works/pi-coding-agent";
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { homedir } from "node:os";
import { dirname, join } from "node:path";

/**
 * live-throughput-status — model-neutral TTFT / decode-TPS footer.
 *
 * Live decode TPS is exact when a provider reports cumulative `usage.output`
 * on intermediate chunks (Gemini-style / non-standard OpenAI servers). Otherwise
 * it falls back to a chars/token estimate calibrated per model and per content
 * kind (thinking vs text/tool-call), learned across messages via EWMA.
 *
 * Provider matrix (live display):
 *   - OpenAI-completions (standard) : estimated (usage arrives in final chunk)
 *   - Anthropic-messages            : estimated
 *   - OpenAI Responses              : estimated
 *   - Gemini / cumulative-usage OAI : exact
 *
 * The final decode rate uses the provider's reported output-token count and the
 * client-observed interval from the first streamed chunk to the last, excluding
 * the tokens delivered in the first chunk (which define the window start).
 *
 * Live rate display modes (switch with /tps, persisted in
 * live-throughput-state.json under "display"):
 *   - window (default): tokens over a short sliding window (default 5 s), i.e.
 *     the near-instantaneous decode rate. Shown as `20.1 tok/s (5s)`. Until
 *     the window fills it uses the time since the first chunk.
 *   - avg: the running average since the message's first chunk (the original
 *     behaviour). Shown as `16.4 tok/s avg`.
 *   - both: `20.1 tok/s (5s) | 16.4 avg`.
 * Env overrides at launch (take precedence over the persisted choice):
 *   PI_TPS_MODE=window|avg|both   PI_TPS_WINDOW_S=5   (window clamped to 1..60 s)
 * Commands: /tps (toggle window<->avg), /tps window [secs], /tps avg,
 *   /tps both [secs], /tps status.
 *
 * Server-exact mode (22 Nov 2026, anemll-forge servers): when
 * live-throughput-config.json maps the provider in "healthUrls" (provider id ->
 * the server's GET /health URL), the live rate comes from the server's own
 * decode counter (`decode_live`: tokens and seconds on the server clock, polled
 * every healthPollMs, default 1000), and the final line from the request's
 * server stats (`decode`: tokens / decode seconds, tokens per verify call;
 * `prefill`: prompt tokens actually computed / prefill seconds). The final stats
 * are used only when the server's decode token count equals usage.output (same
 * request); otherwise the client-side numbers below are shown. Speculative /
 * tree decoding delivers tokens in bursts per verify cycle and chars/token varies
 * with content, so this is the only exact readout for those servers.
 */

const STATUS_KEY = "live-throughput";

interface RatioState {
	ratio: number;
	samples: number;
}

interface ModelState {
	thinking: RatioState;
	other: RatioState;
}

type TpsMode = "window" | "avg" | "both";

interface DisplayPrefs {
	mode: TpsMode;
	windowSeconds: number;
}

interface PersistedState {
	version: number;
	models: Record<string, ModelState>;
	display?: DisplayPrefs;
}

const DEFAULT_DISPLAY: DisplayPrefs = { mode: "window", windowSeconds: 5 };
const WINDOW_MIN_S = 1;
const WINDOW_MAX_S = 60;
// Minimum sample span before a sliding-window rate is trusted; below this the
// rate falls back to the since-first-chunk average (avoids burst spikes).
const MIN_WINDOW_SPAN_MS = 1000;

function parseMode(value: unknown): TpsMode | undefined {
	if (typeof value !== "string") return undefined;
	const v = value.trim().toLowerCase();
	if (v === "window" || v === "win" || v === "inst" || v === "instant") return "window";
	if (v === "avg" || v === "average") return "avg";
	if (v === "both") return "both";
	return undefined;
}

function parseWindowSeconds(value: unknown): number | undefined {
	const n = typeof value === "string" ? Number(value.trim().replace(/s$/i, "")) : value;
	if (!isFiniteNumber(n) || n <= 0) return undefined;
	return clampNumber(n, WINDOW_MIN_S, WINDOW_MAX_S);
}

function parseDisplay(raw: unknown): DisplayPrefs | undefined {
	if (!raw || typeof raw !== "object") return undefined;
	const candidate = raw as { mode?: unknown; windowSeconds?: unknown };
	const mode = parseMode(candidate.mode);
	if (!mode) return undefined;
	return {
		mode,
		windowSeconds: parseWindowSeconds(candidate.windowSeconds) ?? DEFAULT_DISPLAY.windowSeconds,
	};
}

/** Read only the persisted display prefs (so another Pi instance's /tps choice is kept). */
function readDiskDisplay(path: string): DisplayPrefs | undefined {
	if (!existsSync(path)) return undefined;
	try {
		return parseDisplay((JSON.parse(readFileSync(path, "utf8")) as { display?: unknown }).display);
	} catch {
		return undefined;
	}
}

export interface RateSample {
	t: number; // ms
	tokens: number; // cumulative (gross) tokens at t
}

/**
 * Append a sample and prune so that exactly one sample at or before
 * (t - windowMs) is retained as the window anchor.
 */
export function pushRateSample(samples: RateSample[], sample: RateSample, windowMs: number): void {
	samples.push(sample);
	const cutoff = sample.t - windowMs;
	let drop = 0;
	while (drop + 1 < samples.length && samples[drop + 1].t <= cutoff) drop++;
	if (drop > 0) samples.splice(0, drop);
}

/**
 * Sliding-window rate (tokens/s): (tokens_now - tokens_anchor) / (t_now - t_anchor).
 * The anchor is the newest sample at or before now - window, or the first sample
 * (the first chunk) while the window is still filling. Returns undefined when
 * the span is below MIN_WINDOW_SPAN_MS.
 */
export function windowRate(
	samples: RateSample[],
	now: number,
	tokensNow: number,
): { rate: number; spanSeconds: number } | undefined {
	if (samples.length === 0) return undefined;
	const anchor = samples[0];
	const spanMs = now - anchor.t;
	if (spanMs < MIN_WINDOW_SPAN_MS) return undefined;
	return { rate: Math.max(0, tokensNow - anchor.tokens) / (spanMs / 1000), spanSeconds: spanMs / 1000 };
}

interface Config {
	charsPerTokenSeed: number;
	ewmaAlpha: number;
	ratioMin: number;
	ratioMax: number;
	updateIntervalMs: number;
	minLiveWindowMs: number;
	idleClearMs: number;
	showElapsed: boolean;
	clearOnTurnEnd: boolean;
	showGenerationTps: boolean;
	metricsUrls: Record<string, string>;
	metricsTimeoutMs: number;
	statusModes: string[];
	healthUrls: Record<string, string>;
	healthPollMs: number;
}

const DEFAULT_CONFIG: Config = {
	charsPerTokenSeed: 4,
	ewmaAlpha: 0.3,
	ratioMin: 1,
	ratioMax: 16,
	updateIntervalMs: 200,
	minLiveWindowMs: 200,
	idleClearMs: 0, // 0 = keep the final status until the next turn
	showElapsed: true, // append 'took X' at agent_settled
	clearOnTurnEnd: true, // clear the footer while idle between turns
	showGenerationTps: false, // show output/(request->message_end): gen throughput
	// provider id -> Prometheus /metrics URL. Empty by default: prefill TPS is
	// only shown when the backend exposes server-side prefill metrics.
	metricsUrls: {},
	metricsTimeoutMs: 500,
	statusModes: ["tui", "rpc"],
	// provider id -> anemll-forge GET /health URL (server-exact tokens and timing).
	healthUrls: {},
	healthPollMs: 1000,
};

function isFiniteNumber(value: unknown): value is number {
	return typeof value === "number" && Number.isFinite(value);
}

function clampNumber(value: number, min: number, max: number): number {
	return Math.min(max, Math.max(min, value));
}

function positiveNumber(value: unknown): number | undefined {
	return isFiniteNumber(value) && value > 0 ? value : undefined;
}

function loadConfig(path: string): Config {
	const config: Config = { ...DEFAULT_CONFIG };
	if (!existsSync(path)) return config;
	try {
		const raw = JSON.parse(readFileSync(path, "utf8")) as Record<string, unknown>;
		if (isFiniteNumber(raw.charsPerTokenSeed) && raw.charsPerTokenSeed > 0) {
			config.charsPerTokenSeed = raw.charsPerTokenSeed;
		}
		if (isFiniteNumber(raw.ewmaAlpha)) {
			config.ewmaAlpha = clampNumber(raw.ewmaAlpha, 0, 1);
		}
		if (isFiniteNumber(raw.ratioMin) && raw.ratioMin > 0) config.ratioMin = raw.ratioMin;
		if (isFiniteNumber(raw.ratioMax) && raw.ratioMax > config.ratioMin) {
			config.ratioMax = raw.ratioMax;
		}
		if (isFiniteNumber(raw.updateIntervalMs) && raw.updateIntervalMs >= 0) {
			config.updateIntervalMs = raw.updateIntervalMs;
		}
		if (isFiniteNumber(raw.minLiveWindowMs) && raw.minLiveWindowMs >= 0) {
			config.minLiveWindowMs = raw.minLiveWindowMs;
		}
		if (isFiniteNumber(raw.idleClearMs) && raw.idleClearMs >= 0) {
			config.idleClearMs = raw.idleClearMs;
		}
		if (typeof raw.showElapsed === "boolean") {
			config.showElapsed = raw.showElapsed;
		}
		if (typeof raw.clearOnTurnEnd === "boolean") {
			config.clearOnTurnEnd = raw.clearOnTurnEnd;
		}
		if (typeof raw.showGenerationTps === "boolean") {
			config.showGenerationTps = raw.showGenerationTps;
		}
		if (raw.metricsUrls && typeof raw.metricsUrls === "object") {
			const urls: Record<string, string> = {};
			for (const [key, value] of Object.entries(raw.metricsUrls as Record<string, unknown>)) {
				if (typeof value === "string" && value.length > 0) urls[key] = value;
			}
			config.metricsUrls = urls;
		}
		if (isFiniteNumber(raw.metricsTimeoutMs) && raw.metricsTimeoutMs > 0) {
			config.metricsTimeoutMs = raw.metricsTimeoutMs;
		}
		if (raw.healthUrls && typeof raw.healthUrls === "object") {
			const urls: Record<string, string> = {};
			for (const [key, value] of Object.entries(raw.healthUrls as Record<string, unknown>)) {
				if (typeof value === "string" && value.length > 0) urls[key] = value;
			}
			config.healthUrls = urls;
		}
		if (isFiniteNumber(raw.healthPollMs) && raw.healthPollMs >= 100) {
			config.healthPollMs = raw.healthPollMs;
		}
		if (Array.isArray(raw.statusModes)) {
			const modes = raw.statusModes.filter((m): m is string => typeof m === "string");
			if (modes.length > 0) config.statusModes = modes;
		}
	} catch {
		// Ignore malformed config and keep defaults.
	}
	return config;
}

function loadState(path: string): PersistedState {
	const empty: PersistedState = { version: 1, models: {} };
	if (!existsSync(path)) return empty;
	try {
		const raw = JSON.parse(readFileSync(path, "utf8")) as Partial<PersistedState>;
		const models: Record<string, ModelState> = {};
		if (raw.models && typeof raw.models === "object") {
			for (const [key, value] of Object.entries(raw.models)) {
				const candidate = value as Partial<ModelState>;
				const thinking = candidate?.thinking;
				const other = candidate?.other;
				if (
					isFiniteNumber(thinking?.ratio) &&
					isFiniteNumber(thinking?.samples) &&
					isFiniteNumber(other?.ratio) &&
					isFiniteNumber(other?.samples)
				) {
					models[key] = {
						thinking: { ratio: thinking.ratio, samples: thinking.samples },
						other: { ratio: other.ratio, samples: other.samples },
					};
				}
			}
		}
		return { version: 1, models, display: parseDisplay((raw as { display?: unknown }).display) };
	} catch {
		return empty;
	}
}

function saveState(path: string, state: PersistedState): void {
	try {
		mkdirSync(dirname(path), { recursive: true });
		const out: PersistedState = { version: state.version, models: state.models };
		if (state.display) out.display = state.display;
		writeFileSync(path, JSON.stringify(out, null, 2), "utf8");
	} catch {
		// Persistence is best-effort; never break the session over it.
	}
}

function deltaInfo(event: unknown): { kind: "thinking" | "other"; chars: number } | undefined {
	if (!event || typeof event !== "object") return undefined;
	const streamEvent = event as { type?: string; delta?: unknown };
	if (typeof streamEvent.delta !== "string" || streamEvent.delta.length === 0) {
		return undefined;
	}
	if (streamEvent.type === "thinking_delta") {
		return { kind: "thinking", chars: streamEvent.delta.length };
	}
	if (streamEvent.type === "text_delta" || streamEvent.type === "toolcall_delta") {
		return { kind: "other", chars: streamEvent.delta.length };
	}
	return undefined;
}

// Some providers (Gemini-style, non-standard OpenAI servers) send a cumulative
// usage object on intermediate chunks. When present, `partial.usage.output`
// gives the exact token count so far, which beats any chars/token estimate.
function exactOutputTokens(event: unknown): number {
	if (!event || typeof event !== "object") return 0;
	const partial = (event as { partial?: { usage?: { output?: unknown } } }).partial;
	const output = partial?.usage?.output;
	return isFiniteNumber(output) && output > 0 ? output : 0;
}

function seconds(milliseconds: number): number {
	return Math.max(0, milliseconds) / 1000;
}

function rate(value: number, durationSeconds: number): string {
	return (value / Math.max(0.001, durationSeconds)).toFixed(1);
}

function formatTokens(value: number): string {
	if (value >= 1_000_000) return `${(value / 1_000_000).toFixed(1)}M`;
	if (value >= 1_000) return `${(value / 1_000).toFixed(1)}k`;
	return String(value);
}

/**
 * Scale a duration for display: seconds below a minute, then minutes, hours,
 * and days. Examples: 30s, 1m 30s, 2h 5m, 1d 3h.
 */
export function formatDuration(milliseconds: number): string {
	const totalSeconds = Math.max(0, Math.round(milliseconds / 1000));
	if (totalSeconds < 60) return `${totalSeconds}s`;
	const totalMinutes = Math.floor(totalSeconds / 60);
	if (totalMinutes < 60) return `${totalMinutes}m ${totalSeconds % 60}s`;
	const totalHours = Math.floor(totalMinutes / 60);
	if (totalHours < 24) return `${totalHours}h ${totalMinutes % 60}m`;
	const days = Math.floor(totalHours / 24);
	return `${days}d ${totalHours % 24}h`;
}

interface PrefillSnapshot {
	/** vLLM tokens actually prefilled (excludes prefix-cache hits). */
	localCompute: number;
	/** Cumulative server time-to-first-token (seconds). */
	ttftSum: number;
	ttftCount: number;
}

/**
 * Parse the handful of vLLM Prometheus counters needed to derive server-side
 * prefill throughput. Prefix caching means `usage.input` (full prompt) hugely
 * overstates real prefill work, so we use `local_compute` tokens divided by the
 * server's TTFT, sampled before/after a request.
 */
export function parsePrefillMetrics(text: string): PrefillSnapshot {
	let localCompute = 0;
	let ttftSum = 0;
	let ttftCount = 0;
	for (const rawLine of text.split("\n")) {
		const line = rawLine.trim();
		if (!line || line.startsWith("#")) continue;
		const match =
			/^([a-zA-Z_:][a-zA-Z0-9_:]*)(\{[^}]*\})?\s+([0-9eE+.\-]+)$/.exec(line);
		if (!match) continue;
		const name = match[1];
		const labels = match[2] ?? "";
		const value = Number(match[3]);
		if (!Number.isFinite(value)) continue;
		if (name === "vllm:prompt_tokens_by_source_total") {
			if (/source="local_compute"/.test(labels)) localCompute += value;
		} else if (name === "vllm:time_to_first_token_seconds_sum") {
			ttftSum += value;
		} else if (name === "vllm:time_to_first_token_seconds_count") {
			ttftCount += value;
		}
	}
	return { localCompute, ttftSum, ttftCount };
}

async function fetchPrefillMetrics(
	url: string,
	timeoutMs: number,
): Promise<PrefillSnapshot | undefined> {
	if (typeof fetch !== "function") return undefined;
	const controller = new AbortController();
	const timer = setTimeout(() => controller.abort(), timeoutMs);
	try {
		const response = await fetch(url, { signal: controller.signal });
		if (!response.ok) return undefined;
		return parsePrefillMetrics(await response.text());
	} catch {
		return undefined;
	} finally {
		clearTimeout(timer);
	}
}

export interface ForgeHealth {
	decode_live?: { active?: boolean; tokens?: number; seconds?: number } | null;
	decode?: { tokens?: number; seconds?: number; tokens_per_call?: number | null } | null;
	prefill?: { tokens?: number; seconds?: number; cached_tokens?: number } | null;
}

async function fetchHealth(url: string, timeoutMs: number): Promise<ForgeHealth | undefined> {
	if (typeof fetch !== "function") return undefined;
	const controller = new AbortController();
	const timer = setTimeout(() => controller.abort(), timeoutMs);
	try {
		const response = await fetch(url, { signal: controller.signal });
		if (!response.ok) return undefined;
		const body = (await response.json()) as unknown;
		return body && typeof body === "object" ? (body as ForgeHealth) : undefined;
	} catch {
		return undefined;
	} finally {
		clearTimeout(timer);
	}
}

/** Server-exact final stats when the server's decode count matches usage.output. */
export function serverFinal(
	h: ForgeHealth | undefined,
	outputTokens: number | undefined,
): { rate: number; perCall?: number; prefillTokens?: number; prefillRate?: number } | undefined {
	const d = h?.decode;
	if (!d || outputTokens === undefined) return undefined;
	if (!isFiniteNumber(d.tokens) || d.tokens !== outputTokens) return undefined;
	if (!isFiniteNumber(d.seconds) || d.seconds <= 0) return undefined;
	const out: { rate: number; perCall?: number; prefillTokens?: number; prefillRate?: number } = {
		rate: d.tokens / d.seconds,
	};
	if (isFiniteNumber(d.tokens_per_call) && d.tokens_per_call > 0) out.perCall = d.tokens_per_call;
	const p = h?.prefill;
	if (p && isFiniteNumber(p.tokens) && p.tokens > 0 && isFiniteNumber(p.seconds) && p.seconds > 0) {
		out.prefillTokens = p.tokens;
		out.prefillRate = p.tokens / p.seconds;
	}
	return out;
}

export default function (pi: ExtensionAPI) {
	const agentDir =
		process.env.PI_LIVE_THROUGHPUT_DIR || join(homedir(), ".pi", "agent");
	const config = loadConfig(join(agentDir, "live-throughput-config.json"));
	const state = loadState(join(agentDir, "live-throughput-state.json"));
	const statePath = join(agentDir, "live-throughput-state.json");

	// ---- live-rate display prefs -----------------------------------------
	// Precedence: env (PI_TPS_MODE / PI_TPS_WINDOW_S) > persisted /tps choice > default.
	const display: DisplayPrefs = {
		mode: parseMode(process.env.PI_TPS_MODE) ?? state.display?.mode ?? DEFAULT_DISPLAY.mode,
		windowSeconds:
			parseWindowSeconds(process.env.PI_TPS_WINDOW_S) ??
			state.display?.windowSeconds ??
			DEFAULT_DISPLAY.windowSeconds,
	};

	/** Save calibration, preserving whatever display prefs are on disk unless overridden. */
	function persistState(displayOverride?: DisplayPrefs): void {
		const onDisk = displayOverride ?? readDiskDisplay(statePath) ?? state.display;
		state.display = onDisk;
		saveState(statePath, state);
	}

	function displayLabel(): string {
		const w = `${display.windowSeconds}s window`;
		if (display.mode === "window") return `window (${w})`;
		if (display.mode === "avg") return "avg (running average since first chunk)";
		return `both (${w} + avg)`;
	}

	// ---- calibration ------------------------------------------------------
	function getModelState(key: string): ModelState {
		let model = state.models[key];
		if (!model) {
			model = {
				thinking: { ratio: config.charsPerTokenSeed, samples: 0 },
				other: { ratio: config.charsPerTokenSeed, samples: 0 },
			};
			state.models[key] = model;
		}
		return model;
	}

	function ratioFor(key: string, kind: "thinking" | "other"): number {
		const bucket = getModelState(key)[kind];
		return bucket.samples > 0 ? bucket.ratio : config.charsPerTokenSeed;
	}

	function calibrateObserved(key: string, kind: "thinking" | "other", observed: number): void {
		if (!isFiniteNumber(observed) || observed < config.ratioMin || observed > config.ratioMax) {
			return;
		}
		const bucket = getModelState(key)[kind];
		bucket.ratio =
			bucket.samples === 0
				? observed
				: bucket.ratio * (1 - config.ewmaAlpha) + observed * config.ewmaAlpha;
		bucket.samples++;
	}

	function calibrate(key: string, kind: "thinking" | "other", chars: number, tokens: number): void {
		if (tokens <= 0 || chars <= 0) return;
		calibrateObserved(key, kind, chars / tokens);
	}

	function calibrateFromUsage(
		key: string,
		outputTokens: number,
		reasoning: unknown,
	): void {
		const reasoningTokens = positiveNumber(reasoning);
		if (reasoningTokens !== undefined && streamedThinkingChars > 0) {
			calibrate(key, "thinking", streamedThinkingChars, reasoningTokens);
		}
		const otherTokens = outputTokens - (reasoningTokens ?? 0);
		if (otherTokens > 0 && streamedOtherChars > 0) {
			calibrate(key, "other", streamedOtherChars, otherTokens);
		}
		// No reasoning split reported: fall back to a blended sample so both
		// live buckets still learn from the content that actually streamed.
		if (reasoningTokens === undefined && outputTokens > 0) {
			const totalChars = streamedThinkingChars + streamedOtherChars;
			if (totalChars > 0) {
				const blended = totalChars / outputTokens;
				if (streamedThinkingChars > 0) calibrateObserved(key, "thinking", blended);
				if (streamedOtherChars > 0) calibrateObserved(key, "other", blended);
			}
		}
	}

	function estimateTokens(key: string): number {
		return (
			streamedThinkingChars / ratioFor(key, "thinking") +
			streamedOtherChars / ratioFor(key, "other")
		);
	}

	// ---- status / lifecycle ----------------------------------------------
	function setStatus(ctx: ExtensionContext, text?: string): void {
		if (!config.statusModes.includes(ctx.mode)) return;
		lastStatusText = text ?? "";
		ctx.ui.setStatus(
			STATUS_KEY,
			text === undefined ? undefined : ctx.ui.theme.fg("accent", `⚡ ${text}`),
		);
	}

	let idleTimer: ReturnType<typeof setTimeout> | undefined;
	function clearIdleTimer(): void {
		if (idleTimer !== undefined) {
			clearTimeout(idleTimer);
			idleTimer = undefined;
		}
	}
	function scheduleIdleClear(ctx: ExtensionContext): void {
		clearIdleTimer();
		if (config.idleClearMs > 0) {
			idleTimer = setTimeout(() => {
				idleTimer = undefined;
				setStatus(ctx, undefined);
			}, config.idleClearMs);
		}
	}

	// ---- per-turn state ---------------------------------------------------
	let modelKey = "default";
	let currentProvider = "";
	let lastStatusText = "";
	let agentStartedAt: number | undefined;
	let prefillBefore: Promise<PrefillSnapshot | undefined> | undefined;
	let requestStartedAt: number | undefined;
	let requestHookSeen = false;
	let ttftEstimated = false;
	let firstOutputAt: number | undefined;
	let lastOutputAt: number | undefined;
	let ttftSeconds: number | undefined;
	let streamedThinkingChars = 0;
	let streamedOtherChars = 0;
	let firstChunkTokens = 0;
	let lastExactOutput = 0;
	let sawExactStream = false;
	let lastDisplayAt = 0;
	let rateSamples: RateSample[] = [];
	// server-exact live state (healthUrls)
	let healthUrl: string | undefined;
	let healthTimer: ReturnType<typeof setInterval> | undefined;
	let healthBusy = false;
	let serverTokens = 0;
	let serverSeconds = 0;
	let serverSamples: RateSample[] = [];
	let livePrefill: string | undefined; // "4.0k @ 300 t/s" for this request, from /health prefill
	let estAtPoll = 0; // client estimate at the last poll: the count between polls = server count + streamed since
	let liveCtx: ExtensionContext | undefined;

	function stopHealth(): void {
		if (healthTimer !== undefined) {
			clearInterval(healthTimer);
			healthTimer = undefined;
		}
	}

	function renderServerLive(ctx: ExtensionContext): void {
		if (serverTokens <= 0 || serverSeconds <= 0) return;
		const avgText = (serverTokens / serverSeconds).toFixed(1);
		const win = windowRate(serverSamples, serverSeconds * 1000, serverTokens);
		const winText = win ? win.rate.toFixed(1) : avgText;
		const winLabel = `(${display.windowSeconds}s)`;
		let rateText: string;
		if (display.mode === "avg") rateText = `${avgText} tok/s avg`;
		else if (display.mode === "both") rateText = `${winText} tok/s ${winLabel} | ${avgText} avg`;
		else rateText = `${winText} tok/s ${winLabel}`;
		const ttft0 = ttftPart();
		const ttft = ttft0 && livePrefill ? `${ttft0} (${livePrefill})` : ttft0;
		// exact at each poll; between polls, plus the (estimated) tokens streamed since, so the count moves smoothly
		const between = Math.max(0, estimateTokens(modelKey) - estAtPoll);
		const tokenText = `${formatTokens(Math.round(serverTokens + between))} tok`;
		setStatus(ctx, ttft ? `${rateText} ${ttft}   ${tokenText}` : `${rateText}   ${tokenText}`);
	}

	async function pollHealth(): Promise<void> {
		if (healthBusy || !healthUrl) return;
		healthBusy = true;
		try {
			const h = await fetchHealth(healthUrl, config.metricsTimeoutMs);
			const live = h?.decode_live;
			if (!healthTimer || !live || live.active !== true) return;
			const tokens = live.tokens;
			const secs = live.seconds;
			if (!isFiniteNumber(tokens) || !isFiniteNumber(secs) || tokens <= 0 || secs <= 0) return;
			if (tokens < serverTokens || secs < serverSeconds) serverSamples = []; // a new decode began
			serverTokens = tokens;
			serverSeconds = secs;
			estAtPoll = estimateTokens(modelKey);
			if (!livePrefill) {
				// decode_live is active, so the prefill stats are this request's
				const p = h?.prefill;
				if (p && isFiniteNumber(p.tokens) && p.tokens > 0 && isFiniteNumber(p.seconds) && p.seconds > 0) {
					livePrefill = `${formatTokens(p.tokens)} @ ${formatTokens(Math.round(p.tokens / p.seconds))} t/s`;
				}
			}
			if (serverSamples.length === 0) serverSamples = [{ t: 0, tokens: 0 }];
			pushRateSample(serverSamples, { t: secs * 1000, tokens }, display.windowSeconds * 1000);
			if (liveCtx) renderServerLive(liveCtx);
		} finally {
			healthBusy = false;
		}
	}

	function startHealth(ctx: ExtensionContext, provider: string): void {
		stopHealth();
		healthUrl = config.healthUrls[provider];
		serverTokens = 0;
		serverSeconds = 0;
		serverSamples = [];
		estAtPoll = 0;
		livePrefill = undefined;
		liveCtx = ctx;
		if (!healthUrl) return;
		healthTimer = setInterval(() => void pollHealth(), config.healthPollMs);
	}

	function resetCounters(): void {
		ttftEstimated = false;
		firstOutputAt = undefined;
		lastOutputAt = undefined;
		ttftSeconds = undefined;
		streamedThinkingChars = 0;
		streamedOtherChars = 0;
		firstChunkTokens = 0;
		lastExactOutput = 0;
		sawExactStream = false;
		lastDisplayAt = 0;
		rateSamples = [];
	}

	function resetTurn(): void {
		clearIdleTimer();
		stopHealth();
		resetCounters();
		prefillBefore = undefined;
		requestStartedAt = undefined;
		requestHookSeen = false;
	}

	function ttftPart(): string | undefined {
		if (ttftSeconds === undefined) return undefined;
		return `${ttftEstimated ? "~" : ""}TTFT: ${ttftSeconds.toFixed(2)}s`;
	}

	function warmingStatus(): string {
		const ttft = ttftPart();
		return ttft ? `${ttft}   warming up…` : "TTFT: waiting";
	}

	// ---- /tps command -----------------------------------------------------
	pi.registerCommand("tps", {
		description:
			"Live TPS display: /tps (toggle window/avg), /tps window [secs], /tps avg, /tps both [secs], /tps status",
		getArgumentCompletions: (prefix: string) => {
			const items = ["window", "avg", "both", "status", "window 5", "window 10"];
			return items
				.filter((v) => v.startsWith(prefix.trim().toLowerCase()))
				.map((v) => ({ value: v, label: v }));
		},
		handler: async (args, ctx) => {
			const parts = args.trim().split(/\s+/).filter((p) => p.length > 0);
			const first = parts[0]?.toLowerCase();
			if (first === "status" || first === "show") {
				const envNote =
					process.env.PI_TPS_MODE || process.env.PI_TPS_WINDOW_S ? " (env override set)" : "";
				ctx.ui.notify(`TPS live display: ${displayLabel()}${envNote}`, "info");
				return;
			}
			let mode: TpsMode | undefined;
			let secs: number | undefined;
			if (!first) {
				mode = display.mode === "avg" ? "window" : "avg";
			} else if (first === "toggle") {
				mode = display.mode === "avg" ? "window" : "avg";
			} else {
				mode = parseMode(first);
				secs = parseWindowSeconds(first);
				if (mode) secs = parts[1] !== undefined ? parseWindowSeconds(parts[1]) : undefined;
				else if (secs !== undefined) mode = display.mode === "avg" ? "window" : display.mode;
				if (!mode || (parts[1] !== undefined && secs === undefined)) {
					ctx.ui.notify(
						"Usage: /tps | /tps window [1-60] | /tps avg | /tps both [1-60] | /tps status",
						"warning",
					);
					return;
				}
			}
			display.mode = mode;
			if (secs !== undefined) display.windowSeconds = secs;
			persistState({ ...display });
			ctx.ui.notify(`TPS live display: ${displayLabel()} (saved)`, "info");
		},
	});

	// ---- events -----------------------------------------------------------
	pi.on("session_start", async (_event, ctx) => {
		resetTurn();
		setStatus(ctx, "TTFT: waiting");
	});

	pi.on("model_select", async (_event, ctx) => {
		resetTurn();
		setStatus(ctx, undefined);
	});

	pi.on("session_shutdown", async (_event, ctx) => {
		clearIdleTimer();
		stopHealth();
		persistState();
		// Drop the status so it does not linger after the session ends.
		setStatus(ctx, undefined);
	});

	// Whole-request wall clock: one agent run per user prompt, including every
	// tool-calling turn and auto-retry, until Pi settles. Displayed once, at
	// agent_settled — never at intermediate turns.
	pi.on("agent_start", async (_event, ctx) => {
		// Keep the earliest start so auto-retries stay inside one settle window.
		if (agentStartedAt === undefined) agentStartedAt = performance.now();
		lastStatusText = "";
		setStatus(ctx, undefined);
	});

	pi.on("agent_settled", async (_event, ctx) => {
		if (agentStartedAt === undefined) return;
		const elapsed = performance.now() - agentStartedAt;
		agentStartedAt = undefined;
		if (!config.showElapsed) return;
		const base = lastStatusText;
		setStatus(
			ctx,
			base ? `${base}   took ${formatDuration(elapsed)}` : `took ${formatDuration(elapsed)}`,
		);
		scheduleIdleClear(ctx);
	});

	// Clear the footer while the agent is idle between turns/tool calls instead
	// of leaving a stale status frozen on screen. lastStatusText is intentionally
	// kept so agent_settled can rebuild the complete final line (with 'took X').
	pi.on("turn_end", async (_event, ctx) => {
		if (!config.clearOnTurnEnd) return;
		// Only clear while an agent run is still active. A late turn_end after
		// agent_settled must not wipe the final line.
		if (agentStartedAt === undefined) return;
		clearIdleTimer();
		// setStatus() rewrites lastStatusText, so preserve it across the clear.
		const preserved = lastStatusText;
		setStatus(ctx, undefined);
		lastStatusText = preserved;
	});

	// Fired right before the provider request is sent. When a metrics URL is
	// configured for this provider, snapshot prefill counters BEFORE the request
	// so we can diff them against a post-request sample.
	pi.on("before_provider_request", async (_event, ctx) => {
		const provider = ctx.model?.provider ?? currentProvider;
		const url = provider ? config.metricsUrls[provider] : undefined;
		if (url) {
			prefillBefore = fetchPrefillMetrics(url, config.metricsTimeoutMs);
			await prefillBefore;
		} else {
			prefillBefore = undefined;
		}
		requestStartedAt = performance.now();
		requestHookSeen = true;
	});

	pi.on("message_start", async (event, ctx) => {
		if (event.message.role !== "assistant") return;
		// Timestamps from the just-fired before_provider_request are retained;
		// only per-output counters are reset here.
		resetCounters();
		currentProvider = event.message.provider;
		modelKey = `${event.message.provider}/${event.message.model}`;
		if (requestStartedAt === undefined) {
			// Fallback for a provider that does not emit the request hook.
			requestStartedAt = performance.now();
			requestHookSeen = false;
		}
		startHealth(ctx, currentProvider);
		setStatus(ctx, "TTFT: waiting");
	});

	pi.on("message_update", async (event, ctx) => {
		if (event.message.role !== "assistant") return;
		const info = deltaInfo(event.assistantMessageEvent);
		if (!info) return;

		const now = performance.now();
		const exact = exactOutputTokens(event.assistantMessageEvent);

		if (firstOutputAt === undefined) {
			firstOutputAt = now;
			ttftSeconds = seconds(now - (requestStartedAt ?? now));
			ttftEstimated = !requestHookSeen || requestStartedAt === undefined;
			firstChunkTokens =
				exact > 0 ? exact : Math.max(1, info.chars / ratioFor(modelKey, info.kind));
			rateSamples = [{ t: now, tokens: firstChunkTokens }];
			lastDisplayAt = now;
			setStatus(ctx, warmingStatus());
		}

		lastOutputAt = now;
		if (info.kind === "thinking") streamedThinkingChars += info.chars;
		else streamedOtherChars += info.chars;

		if (exact > lastExactOutput) {
			const deltaTokens = exact - lastExactOutput;
			lastExactOutput = exact;
			if (deltaTokens > 0) {
				calibrate(modelKey, info.kind, info.chars, deltaTokens);
				sawExactStream = true;
			}
		}

		// Record a (time, cumulative tokens) sample on every chunk (not just on
		// display ticks) so the sliding window anchor is precise.
		const sampleTokens = lastExactOutput > 0 ? lastExactOutput : estimateTokens(modelKey);
		if (rateSamples.length > 0 && now > rateSamples[0].t) {
			pushRateSample(rateSamples, { t: now, tokens: sampleTokens }, display.windowSeconds * 1000);
		}

		if (now - lastDisplayAt < config.updateIntervalMs) return;
		lastDisplayAt = now;
		if (serverTokens > 0) {
			renderServerLive(ctx); // server-exact (refreshes TTFT); the poller updates the rate
			return;
		}

		const decodeSeconds = seconds(now - firstOutputAt);
		if (decodeSeconds * 1000 < config.minLiveWindowMs) {
			setStatus(ctx, warmingStatus());
			return;
		}

		const exactLive = lastExactOutput > 0;
		// Fencepost consistency with message_end: the live window starts at the
		// arrival of the first chunk, so the first chunk's tokens must be excluded
		// from the numerator (otherwise the initial rate is inflated, badly so when
		// a gateway batches a large first chunk).
		const grossTokens = exactLive ? lastExactOutput : estimateTokens(modelKey);
		const liveTokens = Math.max(0, grossTokens - firstChunkTokens);
		if (liveTokens <= 0) {
			setStatus(ctx, warmingStatus());
			return;
		}
		const approx = exactLive ? "" : "~";
		const ttft = ttftPart();
		const avgText = rate(liveTokens, decodeSeconds);
		const win = windowRate(rateSamples, now, grossTokens);
		const winText = win ? win.rate.toFixed(1) : avgText; // <1 s of data: same as avg
		const winLabel = `(${display.windowSeconds}s)`;
		let rateText: string;
		if (display.mode === "avg") rateText = `${avgText} tok/s avg`;
		else if (display.mode === "both") rateText = `${winText} tok/s ${winLabel} | ${avgText} avg`;
		else rateText = `${winText} tok/s ${winLabel}`;
		const tokenText = `${approx}${formatTokens(Math.round(liveTokens))} tok`;
		setStatus(ctx, ttft ? `${rateText} ${ttft}   ${tokenText}` : `${rateText}   ${tokenText}`);
	});

	pi.on("message_end", async (event, ctx) => {
		if (event.message.role !== "assistant") return;

		const stopReason = event.message.stopReason;
		if (stopReason === "aborted" || stopReason === "error") {
			setStatus(ctx, stopReason === "aborted" ? "aborted" : "error");
			resetTurn();
			return;
		}

		const finalHealthUrl = healthTimer !== undefined ? healthUrl : undefined;
		stopHealth();
		const usage = event.message.usage;
		const rawOutput = isFiniteNumber(usage?.output) ? usage.output : undefined;
		const outputTokens = positiveNumber(rawOutput);
		const rawInput = isFiniteNumber(usage?.input) ? usage.input : 0;
		const rawCacheWrite = isFiniteNumber(usage?.cacheWrite) ? usage.cacheWrite : 0;

		// Learn ratios from the completed message unless the provider already
		// gave us exact mid-stream samples (which were calibrated incrementally).
		if (!sawExactStream && outputTokens !== undefined) {
			calibrateFromUsage(modelKey, outputTokens, usage?.reasoning);
		}
		// Never attribute more first-chunk tokens than were produced.
		if (outputTokens !== undefined) {
			firstChunkTokens = Math.min(firstChunkTokens, outputTokens);
		}
		persistState();

		// Server-side prefill TPS (vLLM): diff local_compute tokens and TTFT
		// counters sampled before and after this request. Prefix caching makes
		// usage.input (the full prompt) useless for this, so the server counters
		// are the only honest per-request prefill rate available.
		let prefillTps: number | undefined;
		if (prefillBefore) {
			const url = config.metricsUrls[currentProvider];
			const before = await prefillBefore;
			prefillBefore = undefined;
			if (before && url) {
				const after = await fetchPrefillMetrics(url, config.metricsTimeoutMs);
				if (after) {
					const prefilledTokens = after.localCompute - before.localCompute;
					const prefillSeconds = after.ttftSum - before.ttftSum;
					if (prefilledTokens > 0 && prefillSeconds > 0) {
						prefillTps = prefilledTokens / prefillSeconds;
					}
				}
			}
		}
		const prefillPart =
			prefillTps !== undefined ? `   ${formatTokens(Math.round(prefillTps))} t/s` : "";

		// Generation throughput (arhen): output / (request sent -> message_end).
		// Spans queue + prefill + TTFT + generation, but excludes tool execution
		// (unlike 'took X', which is agent_start -> agent_settled).
		let generationPart = "";
		if (
			config.showGenerationTps &&
			outputTokens !== undefined &&
			requestStartedAt !== undefined
		) {
			const genSeconds = seconds(performance.now() - requestStartedAt);
			if (genSeconds > 0) {
				generationPart = `   gen: ${rate(outputTokens, genSeconds)} tok/s`;
			}
		}

		const decodeSeconds =
			firstOutputAt !== undefined && lastOutputAt !== undefined
				? seconds(lastOutputAt - firstOutputAt)
				: undefined;

		const ttft = ttftPart();
		// Prompt token count is factual; a tokens/TTFT rate is NOT server prefill
		// throughput (TTFT includes network, queue, scheduling, first-token decode),
		// so we deliberately show the raw count instead of a misleading ratio.
		const promptTokens = rawInput + rawCacheWrite;
		const promptPart = promptTokens > 0 ? `   Prompt: ${formatTokens(promptTokens)} tok` : "";

		const decodeTokens =
			outputTokens !== undefined ? Math.max(0, outputTokens - firstChunkTokens) : undefined;

		let rateText: string | undefined;
		let tokenText: string;
		let rateUnavailable = false;

		if (rawOutput === 0) {
			tokenText = "0 tok";
		} else if (
			decodeTokens !== undefined &&
			decodeTokens > 0 &&
			decodeSeconds !== undefined &&
			decodeSeconds > 0
		) {
			rateText = `${rate(decodeTokens, decodeSeconds)} tok/s`;
			tokenText = `${formatTokens(outputTokens!)} tok`;
		} else if (outputTokens !== undefined) {
			tokenText = `${formatTokens(outputTokens)} tok`;
			rateUnavailable = true;
		} else if (
			streamedThinkingChars + streamedOtherChars > 0 &&
			decodeSeconds !== undefined &&
			decodeSeconds > 0
		) {
			// Same fencepost rule as the exact path: the final rate numerator must
			// exclude the first chunk, while the token count still reports the total.
			const estimatedTotal = estimateTokens(modelKey);
			const estimatedDecode = Math.max(0, estimatedTotal - firstChunkTokens);
			tokenText = `~${formatTokens(Math.round(estimatedTotal))} tok`;
			if (estimatedDecode > 0) {
				rateText = `${rate(estimatedDecode, decodeSeconds)} tok/s`;
			}
		} else {
			tokenText = "no output tokens";
		}

		// Server-exact final stats (healthUrls): replace the client-side rate when the
		// server's decode count is this message's usage.output.
		if (finalHealthUrl && outputTokens !== undefined) {
			const sf = serverFinal(await fetchHealth(finalHealthUrl, config.metricsTimeoutMs), outputTokens);
			if (sf) {
				rateText = `${sf.rate.toFixed(1)} tok/s`;
				tokenText = `${formatTokens(outputTokens)} tok`;
				rateUnavailable = false;
				if (sf.perCall !== undefined) tokenText += ` (${sf.perCall.toFixed(2)}/verify)`;
				if (sf.prefillRate !== undefined && !prefillPart) {
					tokenText += `   prefill ${formatTokens(sf.prefillTokens!)} @ ${formatTokens(Math.round(sf.prefillRate))} t/s`;
				}
			}
		}

		let statusText: string;
		if (rateText) {
			statusText = ttft ? `${rateText} ${ttft}   ${tokenText}` : `${rateText}   ${tokenText}`;
		} else {
			statusText = `${ttft ? `${ttft}   ` : ""}${tokenText}`;
			if (rateUnavailable) statusText += "   rate unavailable";
		}
		if (prefillPart) statusText += prefillPart;
		if (generationPart) statusText += generationPart;
		statusText += promptPart;

		setStatus(ctx, statusText);
		scheduleIdleClear(ctx);
		requestStartedAt = undefined;
		requestHookSeen = false;
	});
}
