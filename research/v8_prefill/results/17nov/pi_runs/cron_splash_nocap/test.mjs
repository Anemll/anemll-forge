import assert from 'node:assert/strict';
import { parse, nextRuns, CronError } from './cron.mjs';

let passed = 0;
let failed = 0;

function t(name, fn) {
  try {
    fn();
    passed += 1;
  } catch (err) {
    failed += 1;
    console.error('FAIL:', name);
    console.error(err && err.stack ? err.stack : err);
  }
}

function eq(actual, expected, msg) {
  assert.deepEqual(actual, expected, msg);
}

function isCronError(err) {
  return err instanceof CronError && err instanceof Error;
}

function throwsCron(fn, name) {
  try {
    fn();
  } catch (err) {
    assert.ok(isCronError(err), name + ': expected CronError, got ' + (err && err.constructor && err.constructor.name));
    return;
  }
  assert.fail(name + ': expected CronError, nothing thrown');
}

function throwsType(fn, name) {
  try {
    fn();
  } catch (err) {
    assert.ok(err instanceof TypeError, name + ': expected TypeError, got ' + (err && err.constructor && err.constructor.name));
    return;
  }
  assert.fail(name + ': expected TypeError, nothing thrown');
}

const range = (a, b) => Array.from({ length: b - a + 1 }, (_, i) => a + i);
const D = Date.UTC;

// ---------- parse: basic fields ----------

t('parse: * * * * *', () => {
  const p = parse('* * * * *');
  eq(p.minute, range(0, 59));
  eq(p.hour, range(0, 23));
  eq(p.dayOfMonth, range(1, 31));
  eq(p.month, range(1, 12));
  eq(p.dayOfWeek, range(0, 6));
  eq(p.domRestricted, false);
  eq(p.dowRestricted, false);
});

t('parse: single values and leading zeros', () => {
  const p = parse('05 04 03 02 01');
  eq(p.minute, [5]);
  eq(p.hour, [4]);
  eq(p.dayOfMonth, [3]);
  eq(p.month, [2]);
  eq(p.dayOfWeek, [1]);
  eq(p.domRestricted, true);
  eq(p.dowRestricted, true);
});

t('parse: lists dedupe and sort', () => {
  const p = parse('5,2,5,1 2,1 * * *');
  eq(p.minute, [1, 2, 5]);
  eq(p.hour, [1, 2]);
});

t('parse: range', () => {
  const p = parse('10-12 3-4 5-5 6-7 2-4');
  eq(p.minute, [10, 11, 12]);
  eq(p.hour, [3, 4]);
  eq(p.dayOfMonth, [5]);
  eq(p.month, [6, 7]);
  eq(p.dayOfWeek, [2, 3, 4]);
});

t('parse: ranges with step', () => {
  const p = parse('10-20/5 0-23/6 1-31/10 1-12/4 0-5/2');
  eq(p.minute, [10, 15, 20]);
  eq(p.hour, [0, 6, 12, 18]);
  eq(p.dayOfMonth, [1, 11, 21, 31]);
  eq(p.month, [1, 5, 9]);
  eq(p.dayOfWeek, [0, 2, 4]);
});

t('parse: */step from min', () => {
  const p = parse('*/15 */10 */7 */3 */2');
  eq(p.minute, [0, 15, 30, 45]);
  eq(p.hour, [0, 10, 20]);
  eq(p.dayOfMonth, [1, 8, 15, 22, 29]);
  eq(p.month, [1, 4, 7, 10]);
  eq(p.dayOfWeek, [0, 2, 4, 6]);
});

t('parse: a/s means a through max', () => {
  const p = parse('5/15 20/3 25/4 9/2 4/2');
  eq(p.minute, [5, 20, 35, 50]);
  eq(p.hour, [20, 23]);
  eq(p.dayOfMonth, [25, 29]);
  eq(p.month, [9, 11]);
  eq(p.dayOfWeek, [4, 6]);
});

t('parse: day of week 7 is Sunday', () => {
  eq(parse('0 0 0 * 0'.replace('0 0 0', '0 0 1')).dayOfWeek, [0]);
  eq(parse('0 0 1 * 7').dayOfWeek, [0]);
  eq(parse('0 0 1 * 0,7').dayOfWeek, [0]);
  eq(parse('0 0 1 * 1-7').dayOfWeek, range(0, 6));
  eq(parse('0 0 1 * 5-7').dayOfWeek, [0, 5, 6]);
});

t('parse: names month and dow, case-insensitive', () => {
  eq(parse('0 0 1 JAN *').month, [1]);
  eq(parse('0 0 1 jan *').month, [1]);
  eq(parse('0 0 1 * SUN').dayOfWeek, [0]);
  eq(parse('0 0 1 * FRI').dayOfWeek, [5]);
  eq(parse('0 0 1 * MON-FRI').dayOfWeek, [1, 2, 3, 4, 5]);
  eq(parse('0 0 1 * mon-fri').dayOfWeek, [1, 2, 3, 4, 5]);
  eq(parse('0 0 1 JAN-DEC *').month, range(1, 12));
  eq(parse('0 0 1 * SUN-SAT').dayOfWeek, range(0, 6));
});

t('parse: restricted flags', () => {
  eq(parse('0 0 */2 * *').domRestricted, false);
  eq(parse('0 0 1-15 * *').domRestricted, true);
  eq(parse('0 0 * * */2').dowRestricted, false);
  eq(parse('0 0 * * 1,3').dowRestricted, true);
  eq(parse('0 0 * * *').domRestricted, false);
  eq(parse('0 0 * * *').dowRestricted, false);
});

t('parse: whitespace handling', () => {
  eq(parse('  5 4 3 2 1  ').minute, [5]);
  eq(parse('*\t*\t*\t*\t*').minute, range(0, 59));
  eq(parse('1\t2\t3\t4\t5').hour, [2]);
});

// ---------- parse: macros ----------

t('macros expand', () => {
  eq(parse('@daily'), parse('0 0 * * *'));
  eq(parse('@midnight'), parse('0 0 * * *'));
  eq(parse('@hourly').minute, [0]);
  eq(parse('@hourly').hour, range(0, 23));
  eq(parse('@weekly').dayOfWeek, [0]);
  eq(parse('@weekly').minute, [0]);
  eq(parse('@weekly').hour, [0]);
  eq(parse('@weekly').dayOfMonth, range(1, 31));
  eq(parse('@monthly').dayOfMonth, [1]);
  eq(parse('@monthly').month, range(1, 12));
  eq(parse('@yearly').dayOfMonth, [1]);
  eq(parse('@yearly').month, [1]);
  eq(parse('@annually'), parse('@yearly'));
  eq(parse('@Daily'), parse('@daily'));
  eq(parse('@YEARLY'), parse('@yearly'));
  eq(parse('@yearly').domRestricted, true);
  eq(parse('@yearly').dowRestricted, false);
});

t('unknown macros error', () => {
  throwsCron(() => parse('@foo'), '@foo');
  throwsCron(() => parse('@daily * * * *'), '@daily extra fields');
});

// ---------- parse: errors ----------

const badExpressions = [
  '* * * *',                    // 4 fields
  '* * * * * *',                // 6 fields
  '',
  '   ',
  '60 * * * *',
  '* 24 * * *',
  '0 0 0 * *',
  '0 0 32 * *',
  '0 0 * 0 *',
  '0 0 * 13 *',
  '0 0 * * 8',
  '5-2 * * * *',
  '11-10 * * * *',
  '* 1-24 * * *',
  '1-60 * * * *',
  '*/0 * * * *',
  '*/ * * * *',
  '1-5/ * * * *',
  '1-5/0 * * * *',
  '1-5/-1 * * * *',
  '1,,2 * * * *',
  '1, * * * *',
  '*, * * * *',
  ',1 * * * *',
  '1-2-3 * * * *',
  '1-2/3/4 * * * *',
  '5/ * * * *',
  '/2 * * * *',
  '-5 * * * *',
  '5- * * * *',
  '*-5 * * * *',
  '* * * * FOO',
  '* * * FOO *',
  '* * * JAN-FOO *',
  '* * * * ?',
  '* * * * L',
  '* * * * 1#2',
  '* * 1W *',
  '* * * * SUN/0',
  '* * * * MON/',
  '* * * * MON-2-3',
  '1e2 * * * *',
  '+1 * * * *',
  '1.0 * * * *',
  '* * JAN * *',          // name in day-of-month field
  'SUN * * * *',          // name in minute field
  'JAN * * * *',          // name in minute field
  'MON * * * *',          // dow name in minute field
  '0 0 1 * 0-7-8',        // three-part range
  '0 0 1 * 8-7',          // out of range and reversed
  '0 0 1 * 7-0',          // 7 as range start
  '0 0 1 * 7/2',          // 7 as step start
];

for (const expr of badExpressions) {
  t('parse error: ' + JSON.stringify(expr), () => throwsCron(() => parse(expr), expr));
}

t('parse error: non-string', () => {
  throwsCron(() => parse(5), 'number');
  throwsCron(() => parse(null), 'null');
  throwsCron(() => parse(undefined), 'undefined');
});

// ---------- nextRuns: matching ----------

t('nextRuns: daily midnight, strictly after', () => {
  const from = new Date(D(2024, 0, 15, 12, 34, 56, 789));
  const r = nextRuns('0 0 * * *', from, 3);
  eq(r.length, 3);
  eq(r[0].getTime(), D(2024, 0, 16, 0, 0, 0));
  eq(r[1].getTime(), D(2024, 0, 17, 0, 0, 0));
  eq(r[2].getTime(), D(2024, 0, 18, 0, 0, 0));
});

t('nextRuns: strictly after when from is exact match', () => {
  const from = new Date(D(2024, 0, 1, 0, 0, 0, 0));
  const r = nextRuns('0 0 * * *', from, 1);
  eq(r[0].getTime(), D(2024, 0, 2, 0, 0, 0));
});

t('nextRuns: minute/hour combo, from exact match', () => {
  const from = new Date(D(2024, 0, 15, 14, 30, 0, 0));
  const r = nextRuns('30 14 * * *', from, 2);
  eq(r[0].getTime(), D(2024, 0, 16, 14, 30, 0));
  eq(r[1].getTime(), D(2024, 0, 17, 14, 30, 0));
});

t('nextRuns: every minute', () => {
  const from = new Date(D(2024, 0, 1, 0, 0, 30, 123));
  const r = nextRuns('* * * * *', from, 4);
  eq(r[0].getTime(), D(2024, 0, 1, 0, 1, 0));
  eq(r[1].getTime(), D(2024, 0, 1, 0, 2, 0));
  eq(r[2].getTime(), D(2024, 0, 1, 0, 3, 0));
  eq(r[3].getTime(), D(2024, 0, 1, 0, 4, 0));
});

t('nextRuns: seconds and ms zeroed', () => {
  const r = nextRuns('0 0 * * *', new Date(D(2024, 0, 1, 0, 0)), 1);
  eq(r[0].getUTCSeconds(), 0);
  eq(r[0].getUTCMilliseconds(), 0);
});

t('nextRuns: dom AND dow (both restricted? no - both restricted = OR, use values)', () => {
  // dom=15 (restricted), dow=1 (restricted) -> both restricted -> OR: 15th OR Monday
  const from = new Date(D(2024, 0, 1, 0, 0, 0, 0)); // Mon Jan 1 2024
  const r = nextRuns('0 12 15 * 1', from, 4);
  // Jan 1 (Mon) 12:00, Jan 8 (Mon) 12:00, Jan 15 (Mon, 15th) 12:00, Jan 22 (Mon) 12:00
  eq(r[0].getTime(), D(2024, 0, 1, 12, 0));
  eq(r[1].getTime(), D(2024, 0, 8, 12, 0));
  eq(r[2].getTime(), D(2024, 0, 15, 12, 0));
  eq(r[3].getTime(), D(2024, 0, 22, 12, 0));
});

t('nextRuns: OR semantics does not double-count a day', () => {
  const from = new Date(D(2024, 0, 1, 0, 0, 0, 0));
  const r = nextRuns('0 12 1 * 1', from, 6);
  eq(r.map((d) => d.getTime()), [
    D(2024, 0, 1, 12, 0),  // Jan 1: 1st and Monday
    D(2024, 0, 8, 12, 0),  // Mon
    D(2024, 0, 15, 12, 0), // Mon
    D(2024, 0, 22, 12, 0), // Mon
    D(2024, 0, 29, 12, 0), // Mon
    D(2024, 1, 1, 12, 0),  // Feb 1: 1st
  ]);
});

t('nextRuns: one restricted field still ANDs with the other', () => {
  // dow restricted only -> day must match BOTH (dom=* matches all, dow Monday)
  const from = new Date(D(2024, 0, 1, 0, 0, 0, 0));
  const r = nextRuns('0 12 * * 1', from, 3);
  eq(r.map((d) => d.getTime()), [D(2024, 0, 1, 12, 0), D(2024, 0, 8, 12, 0), D(2024, 0, 15, 12, 0)]);
});

t('nextRuns: */2 dom still filters when dow unrestricted', () => {
  const from = new Date(D(2024, 0, 1, 0, 0, 0, 0));
  const r = nextRuns('0 0 */2 * *', from, 3);
  // dom */2 -> days 1,3,5,... ; strictly after Jan 1 00:00 -> Jan 3, 5, 7
  eq(r.map((d) => d.getTime()), [D(2024, 0, 3, 0, 0), D(2024, 0, 5, 0, 0), D(2024, 0, 7, 0, 0)]);
});

t('nextRuns: no such day (Feb 30) returns []', () => {
  eq(nextRuns('0 0 30 2 *', new Date(D(2024, 0, 1)), 1000), []);
});

t('nextRuns: leap day', () => {
  const r = nextRuns('0 0 29 2 *', new Date(D(2024, 0, 1)), 1000);
  eq(r.length, 3);
  eq(r[0].getTime(), D(2024, 1, 29, 0, 0));
  eq(r[1].getTime(), D(2028, 1, 29, 0, 0));
  eq(r[2].getTime(), D(2032, 1, 29, 0, 0));
});

t('nextRuns: yearly in 4000 day window', () => {
  const r = nextRuns('0 0 1 1 *', new Date(D(2024, 0, 1, 0, 0, 0, 0)), 1000);
  eq(r.length, 10);
  eq(r[0].getTime(), D(2025, 0, 1, 0, 0));
  eq(r[9].getTime(), D(2034, 0, 1, 0, 0));
});

t('nextRuns: month skip across year boundary', () => {
  const r = nextRuns('0 0 1 12 *', new Date(D(2024, 10, 5)), 3);
  eq(r[0].getTime(), D(2024, 11, 1, 0, 0));
  eq(r[1].getTime(), D(2025, 11, 1, 0, 0));
  eq(r[2].getTime(), D(2026, 11, 1, 0, 0));
});

t('nextRuns: monthly on the 15th', () => {
  const r = nextRuns('30 9 15 * *', new Date(D(2024, 11, 31, 23, 59, 59)), 3);
  eq(r[0].getTime(), D(2025, 0, 15, 9, 30));
  eq(r[1].getTime(), D(2025, 1, 15, 9, 30));
  eq(r[2].getTime(), D(2025, 2, 15, 9, 30));
});

t('nextRuns: window boundary respected', () => {
  const from = new Date(D(2024, 0, 1, 0, 0, 0, 0));
  const cap = from.getTime() + 4000 * 86400000;
  const r = nextRuns('0 0 * * *', from, 1000);
  eq(r.length, 1000);
  for (const d of r) {
    assert.ok(d.getTime() >= from.getTime());
    assert.ok(d.getTime() < cap);
  }
  for (let i = 1; i < r.length; i++) assert.ok(r[i] > r[i - 1]);
});

t('nextRuns: ascending and unique', () => {
  // only dow restricted -> day must match dow AND dom(*)
  const r = nextRuns('*/10 0,12 * * MON,WED', new Date(D(2024, 0, 1)), 200);
  for (let i = 1; i < r.length; i++) assert.ok(r[i].getTime() > r[i - 1].getTime());
  for (const d of r) {
    assert.ok(d instanceof Date);
    assert.ok(d.getUTCSeconds() === 0 && d.getUTCMilliseconds() === 0);
    assert.ok(d.getUTCMinutes() % 10 === 0);
    assert.ok(d.getUTCHours() === 0 || d.getUTCHours() === 12);
    const dow = d.getUTCDay();
    assert.ok(dow === 1 || dow === 3);
  }
});

// ---------- nextRuns: errors ----------

t('nextRuns: TypeError for bad from', () => {
  throwsType(() => nextRuns('* * * * *', new Date(NaN), 1), 'NaN date');
  throwsType(() => nextRuns('* * * * *', null, 1), 'null');
  throwsType(() => nextRuns('* * * * *', '2024-01-01', 1), 'string');
  throwsType(() => nextRuns('* * * * *', 1234, 1), 'number');
  throwsType(() => nextRuns('* * * * *', undefined, 1), 'undefined');
});

t('nextRuns: TypeError for bad n', () => {
  const from = new Date(D(2024, 0, 1));
  throwsType(() => nextRuns('* * * * *', from, 0), 'n=0');
  throwsType(() => nextRuns('* * * * *', from, 1.5), 'n=1.5');
  throwsType(() => nextRuns('* * * * *', from, 1001), 'n=1001');
  throwsType(() => nextRuns('* * * * *', from, -1), 'n=-1');
  throwsType(() => nextRuns('* * * * *', from, '5'), 'n=string');
  throwsType(() => nextRuns('* * * * *', from, NaN), 'n=NaN');
});

t('nextRuns: CronError for bad expressions', () => {
  const from = new Date(D(2024, 0, 1));
  throwsCron(() => nextRuns('bad', from, 1), 'bad');
  throwsCron(() => nextRuns('60 * * * *', from, 1), '60');
  throwsCron(() => nextRuns('@nope', from, 1), '@nope');
  throwsCron(() => nextRuns('*,', from, 1), 'trailing comma');
});

t('nextRuns: n bounds ok', () => {
  const r1 = nextRuns('* * * * *', new Date(D(2024, 0, 1)), 1);
  eq(r1.length, 1);
  const r1000 = nextRuns('* * * * *', new Date(D(2024, 0, 1)), 1000);
  eq(r1000.length, 1000);
});

// ---------- performance ----------

t('perf: full 4000-day scan (leap day) under 100ms', () => {
  const t0 = performance.now();
  const r = nextRuns('0 0 29 2 *', new Date(D(2024, 0, 1)), 1000);
  const dt = performance.now() - t0;
  assert.ok(dt < 100, 'took ' + dt.toFixed(1) + 'ms');
  eq(r.length, 3);
});

t('perf: sparse minutes over long window under 100ms', () => {
  const t0 = performance.now();
  const r = nextRuns('30 * * * *', new Date(D(2024, 0, 1)), 1000);
  const dt = performance.now() - t0;
  assert.ok(dt < 100, 'took ' + dt.toFixed(1) + 'ms');
  eq(r.length, 1000);
  eq(r[0].getTime(), D(2024, 0, 1, 0, 30, 0));
});

t('perf: every minute n=1000 under 100ms', () => {
  const t0 = performance.now();
  const r = nextRuns('* * * * *', new Date(D(2024, 0, 1)), 1000);
  const dt = performance.now() - t0;
  assert.ok(dt < 100, 'took ' + dt.toFixed(1) + 'ms');
  eq(r.length, 1000);
});

t('perf: daily over full window under 100ms', () => {
  const t0 = performance.now();
  const r = nextRuns('0 0 * * *', new Date(D(2024, 0, 1)), 1000);
  const dt = performance.now() - t0;
  assert.ok(dt < 100, 'took ' + dt.toFixed(1) + 'ms');
  eq(r.length, 1000);
});

console.log('passed: ' + passed + ', failed: ' + failed);
if (failed > 0) process.exit(1);
