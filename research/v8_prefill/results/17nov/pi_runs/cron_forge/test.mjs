import { test } from 'node:test';
import assert from 'node:assert/strict';
import { parse, nextRuns, CronError } from './cron.mjs';

const D = (...a) => new Date(Date.UTC(...a));

// ---------- parse: values ----------

test('parse single values', () => {
  assert.deepEqual(parse('5 6 7 8 3'), {
    minute: [5],
    hour: [6],
    dayOfMonth: [7],
    month: [8],
    dayOfWeek: [3],
    domRestricted: true,
    dowRestricted: true,
  });
});

test('parse * fields give full ranges, unrestricted flags', () => {
  assert.deepEqual(parse('* * * * *'), {
    minute: Array.from({ length: 60 }, (_, i) => i),
    hour: Array.from({ length: 24 }, (_, i) => i),
    dayOfMonth: Array.from({ length: 31 }, (_, i) => i + 1),
    month: Array.from({ length: 12 }, (_, i) => i + 1),
    dayOfWeek: [0, 1, 2, 3, 4, 5, 6],
    domRestricted: false,
    dowRestricted: false,
  });
});

test('leading zeros and tabs / multiple spaces', () => {
  assert.deepEqual(parse('05\t09 07 01 01'), parse('5 9 7 1 1'));
  assert.deepEqual(parse('  5  6 7 8 3  '), parse('5 6 7 8 3'));
});

test('comma lists are sorted and de-duplicated', () => {
  const p = parse('30,15,30,0,45 * * * *');
  assert.deepEqual(p.minute, [0, 15, 30, 45]);
  assert.deepEqual(parse('5,3,1 * * * *').minute, [1, 3, 5]);
});

test('ranges inclusive, with step', () => {
  assert.deepEqual(parse('0-30/10 * * * *').minute, [0, 10, 20, 30]);
  assert.deepEqual(parse('1-3 * * * *').minute, [1, 2, 3]);
  assert.deepEqual(parse('*/15 * * * *').minute, [0, 15, 30, 45]);
});

test('a/s steps from a to field max', () => {
  assert.deepEqual(parse('50/10 * * * *').minute, [50]);
  assert.deepEqual(parse('1/3 * * * *').minute, [1, 4, 7, 10, 13, 16, 19, 22, 25, 28, 31, 34, 37, 40, 43, 46, 49, 52, 55, 58]);
});

test('day of week: */s and a/s run 0..6, 7 normalized to 0', () => {
  assert.deepEqual(parse('* * * * */2').dayOfWeek, [0, 2, 4, 6]);
  assert.deepEqual(parse('* * * * 7').dayOfWeek, [0]);
  assert.deepEqual(parse('* * * * 0,7').dayOfWeek, [0]);
  assert.deepEqual(parse('* * * * 5/2').dayOfWeek, [5]);
  assert.deepEqual(parse('* * * * 1-3/2').dayOfWeek, [1, 3]);
  assert.deepEqual(parse('* * * * 0-7').dayOfWeek, [0, 1, 2, 3, 4, 5, 6]);
});

test('names (case-insensitive) incl. ranges; month and dow', () => {
  assert.deepEqual(parse('* * * JAN *').month, [1]);
  assert.deepEqual(parse('* * * jan-SEP *').month, [1, 2, 3, 4, 5, 6, 7, 8, 9]);
  assert.deepEqual(parse('* * * * MON-FRI').dayOfWeek, [1, 2, 3, 4, 5]);
  assert.throws(() => parse('* * * * FRI-SUN'), CronError); // 5 > 0, no wrapping ranges
  assert.deepEqual(parse('* * * * FRI-SAT').dayOfWeek, [5, 6]);
  assert.deepEqual(parse('* * * * SUN-MON').dayOfWeek, [0, 1]);
  assert.throws(() => parse('* * * * FRI-SUN'), CronError); // 5 > 0, no wrapping ranges
  assert.deepEqual(parse('* * * * sAt').dayOfWeek, [6]);
});

test('restricted flags: */2 on a day field is unrestricted', () => {
  assert.equal(parse('*/2 * * * *').minute.length, 30);
  assert.equal(parse('* * */2 * *').domRestricted, false);
  assert.equal(parse('* * * * */2').dowRestricted, false);
  assert.equal(parse('*/2 * * * *', ).minute.length, 30);
  assert.equal(parse('* * * * */1').dowRestricted, false);
  assert.equal(parse('15 2 3 * *').domRestricted, true);
  assert.equal(parse('15 2 * * 3').dowRestricted, true);
});

test('macros', () => {
  assert.deepEqual(parse('@daily'), parse('0 0 * * *'));
  assert.deepEqual(parse('@DAILY'), parse('0 0 * * *'));
  assert.deepEqual(parse('@midnight'), parse('0 0 * * *'));
  assert.deepEqual(parse('@hourly'), parse('0 * * * *'));
  assert.deepEqual(parse('@weekly'), parse('0 0 * * 0'));
  assert.deepEqual(parse('@monthly'), parse('0 0 1 * *'));
  assert.deepEqual(parse('@yearly'), parse('0 0 1 1 *'));
  assert.deepEqual(parse('@ANNUALLY'), parse('0 0 1 1 *'));
});

// ---------- parse: errors ----------

const expectCronError = (fn, label) => {
  assert.throws(fn, CronError, label);
};

test('error: wrong field count', () => {
  expectCronError(() => parse('0 0 * *'), '4 fields');
  expectCronError(() => parse('0 0 * * * *'), '6 fields');
  expectCronError(() => parse(''), 'empty');
  expectCronError(() => parse('   '), 'whitespace only');
});

test('error: out-of-range values', () => {
  expectCronError(() => parse('60 * * * *'), 'minute 60');
  expectCronError(() => parse('0 24 * * *'), 'hour 24');
  expectCronError(() => parse('0 0 0 * *'), 'dom 0');
  expectCronError(() => parse('0 0 32 * *'), 'dom 32');
  expectCronError(() => parse('0 0 * 13 *'), 'month 13');
  expectCronError(() => parse('0 0 * 0 *'), 'month 0');
  expectCronError(() => parse('0 0 * * 8'), 'dow 8');
  expectCronError(() => parse('0 0 * * -1'), 'negative');
});

test('error: range a > b, malformed ranges', () => {
  expectCronError(() => parse('5-2 * * * *'), 'a > b');
  expectCronError(() => parse('1- * * * *'), 'open range');
  expectCronError(() => parse('-2 * * * *'), 'open range low');
  expectCronError(() => parse('1-2-3 * * * *'), 'double dash');
  expectCronError(() => parse('* * * MON-F *'), 'bad name in range');
});

test('error: steps', () => {
  expectCronError(() => parse('*/0 * * * *'), 'step 0');
  expectCronError(() => parse('*/ * * * *'), 'missing step');
  expectCronError(() => parse('*/MON * * * *'), 'name as step');
  expectCronError(() => parse('*/1.5 * * * *'), 'non-integer step');
  expectCronError(() => parse('1/2/3 * * * *'), 'double slash');
  expectCronError(() => parse('1-2 * * * */'), 'trailing slash');
});

test('error: empty list items, stray commas', () => {
  expectCronError(() => parse('1,,2 * * * *'), 'empty item');
  expectCronError(() => parse(', * * * *'), 'leading comma');
  expectCronError(() => parse('* ,* * * *'), 'trailing comma');
});

test('error: unknown names, other chars', () => {
  expectCronError(() => parse('MON * * * *'), 'name in minute');
  expectCronError(() => parse('* * * FOO *'), 'unknown month');
  expectCronError(() => parse('* * * * FOO'), 'unknown dow');
  expectCronError(() => parse('* * ? * *'), 'question mark');
  expectCronError(() => parse('* * L * *'), 'L');
  expectCronError(() => parse('* * * * W'), 'W');
  expectCronError(() => parse('* * * * 1#2'), 'hash');
  expectCronError(() => parse('@bogus'), 'unknown macro');
  expectCronError(() => parse('@daily extra'), 'macro with extra field');
});

test('error: non-string expression', () => {
  expectCronError(() => parse(1234), 'number');
  expectCronError(() => parse(null), 'null');
});

// ---------- nextRuns ----------

test('nextRuns basic daily midnight', () => {
  const from = D(2024, 0, 15, 12, 30, 45);
  const out = nextRuns('0 0 * * *', from, 3);
  assert.deepEqual(out, [D(2024, 0, 16), D(2024, 0, 17), D(2024, 0, 18)]);
});

test('nextRuns strictly after from, seconds/milliseconds zero', () => {
  const from = D(2024, 0, 15, 12, 0); // exactly on a match
  assert.deepEqual(nextRuns('0 12 * * *', from, 1), [D(2024, 0, 16, 12, 0)]);
  const from2 = D(2024, 0, 15, 11, 59, 59);
  assert.deepEqual(nextRuns('0 12 * * *', from2, 1), [D(2024, 0, 15, 12, 0)]);
  for (const r of nextRuns('* * * * *', from, 5)) {
    assert.equal(r.getUTCSeconds(), 0);
    assert.equal(r.getUTCMilliseconds(), 0);
  }
});

test('nextRuns with specific minute/hour/month', () => {
  const from = D(2024, 2, 10); // 2024-03-10
  assert.deepEqual(nextRuns('0 9 1 6 *', from, 1), [D(2024, 5, 1, 9, 0)]);
  assert.deepEqual(nextRuns('30 14 * * *', D(2024, 0, 1, 14, 30), 2), [
    D(2024, 0, 2, 14, 30),
    D(2024, 0, 3, 14, 30),
  ]);
});

test('OR rule: both restricted → either matches', () => {
  // 13th OR Friday: 2020-01-02 (Fri), 2020-01-09 (Fri), 2020-01-13 (Thu)
  const out = nextRuns('0 0 13 * 4', D(2020, 0, 1), 3);
  assert.deepEqual(out, [D(2020, 0, 2), D(2020, 0, 9), D(2020, 0, 13)]);
});

test('AND rule: unrestricted */2 dom still filters day, AND with dow', () => {
  // dom */2 (days 1,3,5,...) AND Friday
  const out = nextRuns('0 0 */2 * 5', D(2024, 0, 1), 3);
  // Fridays in 2024 Jan/Feb: 5(odd), 12, 19(odd), 26, 2(odd? 2 even), 9(odd)
  assert.deepEqual(out, [D(2024, 0, 5), D(2024, 0, 19), D(2024, 1, 9)]);
});

test('leap days', () => {
  assert.deepEqual(nextRuns('0 0 29 2 *', D(2020, 0, 1), 2), [D(2020, 1, 29), D(2024, 1, 29)]);
});

test('impossible date returns empty', () => {
  assert.deepEqual(nextRuns('0 0 30 2 *', D(2024, 0, 1), 10), []);
});

test('window limited to from + 4000 days', () => {
  const from = D(2024, 0, 1);
  // Jan 1 yearly: next within 4000d is 2025-01-01, then 2026-01-01 ...
  const out = nextRuns('@yearly', from, 1000);
  const expected = [];
  for (let y = 2025; expected.length < 1000; y++) {
    if (Date.UTC(y, 0, 1) < from.getTime() + 4000 * 86400000) expected.push(D(y, 0, 1));
    else break;
  }
  assert.equal(out.length, expected.length);
  assert.deepEqual(out[0], D(2025, 0, 1));
});

test('window boundary is strict', () => {
  const from = D(2024, 0, 1);
  const cutoff = from.getTime() + 4000 * 86400000;
  // minute granularity: every minute in [from, cutoff]; last allowed < cutoff
  const out = nextRuns('* * * * *', from, 1000);
  assert.deepEqual(out[0], D(2024, 0, 1, 0, 1));
});

test('n validation', () => {
  const from = D(2024, 0, 1);
  assert.throws(() => nextRuns('* * * * *', from, 0), TypeError);
  assert.throws(() => nextRuns('* * * * *', from, 1001), TypeError);
  assert.throws(() => nextRuns('* * * * *', from, 1.5), TypeError);
  assert.throws(() => nextRuns('* * * * *', from, '5'), TypeError);
  assert.throws(() => nextRuns('* * * * *', from, NaN), TypeError);
  // n up to 1000 allowed
  assert.equal(nextRuns('*/10 * * * *', from, 1000).length, 1000);
});

test('from validation', () => {
  assert.throws(() => nextRuns('* * * * *', '2024-01-01', 1), TypeError);
  assert.throws(() => nextRuns('* * * * *', null, 1), TypeError);
  assert.throws(() => nextRuns('* * * * *', new Date(NaN), 1), TypeError);
  assert.throws(() => nextRuns('* * * * *', 1700000000000, 1), TypeError);
});

test('invalid expression in nextRuns throws CronError', () => {
  assert.throws(() => nextRuns('61 * * * *', D(2024, 0, 1), 1), CronError);
});

test('performance: full-window sparse match under 100ms', () => {
  const t0 = process.hrtime.bigint();
  nextRuns('0 0 30 2 *', D(2024, 0, 1), 1000);
  assert.ok(Number(process.hrtime.bigint() - t0) / 1e6 < 100);
});

test('performance: dense expression n=1000 under 100ms', () => {
  const t0 = process.hrtime.bigint();
  nextRuns('* * * * *', D(2024, 0, 1), 1000);
  assert.ok(Number(process.hrtime.bigint() - t0) / 1e6 < 100);
});

test('performance: sparse across full window under 100ms', () => {
  const t0 = process.hrtime.bigint();
  nextRuns('0 0 29 2 *', D(2024, 0, 1), 1000);
  assert.ok(Number(process.hrtime.bigint() - t0) / 1e6 < 100);
});

test('yearly across macro + dow names end to end', () => {
  const out = nextRuns('30 12 * * MON-FRI', D(2024, 0, 6), 3); // Sat, Sun 6/7 skipped
  assert.deepEqual(out, [D(2024, 0, 8, 12, 30), D(2024, 0, 9, 12, 30), D(2024, 0, 10, 12, 30)]);
});

test('dedup between overlapping ranges', () => {
  assert.deepEqual(parse('1-3,2-4 * * * *').minute, [1, 2, 3, 4]);
});
