import { test } from 'node:test';
import assert from 'node:assert/strict';
import { CronError, parse, nextRuns } from './cron.mjs';

const range = (lo, hi) => Array.from({ length: hi - lo + 1 }, (_, i) => lo + i);

// ---------- parse: values ----------

test('parse simple expression', () => {
  const p = parse('0 0 1 1 *');
  assert.deepEqual(p.minute, [0]);
  assert.deepEqual(p.hour, [0]);
  assert.deepEqual(p.dayOfMonth, [1]);
  assert.deepEqual(p.month, [1]);
  assert.deepEqual(p.dayOfWeek, [0, 1, 2, 3, 4, 5, 6]);
  assert.equal(p.domRestricted, true);
  assert.equal(p.dowRestricted, false);
});

test('parse all stars', () => {
  const p = parse('* * * * *');
  assert.deepEqual(p.minute, range(0, 59));
  assert.deepEqual(p.hour, range(0, 23));
  assert.deepEqual(p.dayOfMonth, range(1, 31));
  assert.deepEqual(p.month, range(1, 12));
  assert.deepEqual(p.dayOfWeek, range(0, 6));
  assert.equal(p.domRestricted, false);
  assert.equal(p.dowRestricted, false);
});

test('parse lists are sorted and deduplicated', () => {
  const p = parse('30,15,15,45,10 9-10,10 1,1 2 *');
  assert.deepEqual(p.minute, [10, 15, 30, 45]);
  assert.deepEqual(p.hour, [9, 10]);
  assert.deepEqual(p.dayOfMonth, [1]);
});

test('parse steps', () => {
  assert.deepEqual(parse('*/15 * * * *').minute, [0, 15, 30, 45]);
  assert.deepEqual(parse('5-20/5 * * * *').minute, [5, 10, 15, 20]);
  assert.deepEqual(parse('30/15 * * * *').minute, [30, 45]);
  assert.deepEqual(parse('0 3/6 * * *').hour, [3, 9, 15, 21]);
});

test('parse ranges inclusive', () => {
  assert.deepEqual(parse('10-12 * * * *').minute, [10, 11, 12]);
  assert.deepEqual(parse('* 0-0 * * *').hour, [0]);
});

test('parse leading zeros', () => {
  assert.deepEqual(parse('05 09 * * *').minute, [5]);
  assert.deepEqual(parse('05 09 * * *').hour, [9]);
});

test('parse month names case-insensitive, incl. in ranges', () => {
  assert.deepEqual(parse('0 0 1 jan *').month, [1]);
  assert.deepEqual(parse('0 0 1 Jan *').month, [1]);
  assert.deepEqual(parse('0 0 * feb-may *').month, [2, 3, 4, 5]);
  assert.deepEqual(parse('0 0 * DEC *').month, [12]);
});

test('parse day of week names', () => {
  assert.deepEqual(parse('0 0 * * SUN').dayOfWeek, [0]);
  assert.deepEqual(parse('0 0 * * mon-FRI').dayOfWeek, [1, 2, 3, 4, 5]);
  assert.deepEqual(parse('0 0 * * Sat').dayOfWeek, [6]);
});

test('parse day of week 7 becomes 0', () => {
  assert.deepEqual(parse('0 0 * * 7').dayOfWeek, [0]);
  assert.deepEqual(parse('0 0 * * 0,7').dayOfWeek, [0]);
  assert.deepEqual(parse('0 0 * * 0-7').dayOfWeek, range(0, 6));
  assert.deepEqual(parse('0 0 * * 5-7/2').dayOfWeek, [0, 5]);
});

test('parse day of week stars and steps run 0..6', () => {
  assert.deepEqual(parse('0 0 * * */2').dayOfWeek, [0, 2, 4, 6]);
  assert.deepEqual(parse('0 0 * * 1/3').dayOfWeek, [1, 4]);
  assert.deepEqual(parse('0 0 * * */7').dayOfWeek, [0]);
});

test('restricted flags', () => {
  assert.equal(parse('* * * * *').domRestricted, false);
  assert.equal(parse('* * * * *').dowRestricted, false);
  assert.equal(parse('* * * */2 *').domRestricted, false);
  assert.equal(parse('* * * * */2').dowRestricted, false);
  assert.equal(parse('5 * * * *').domRestricted, false);
  assert.equal(parse('5 1 1 * *').domRestricted, true);
  assert.equal(parse('* * * * 2').dowRestricted, true);
  assert.equal(parse('5 * 1 * 2').domRestricted, true);
  assert.equal(parse('5 * 1 * 2').dowRestricted, true);
});

test('parse macros case-insensitive', () => {
  const daily = parse('@daily');
  assert.deepEqual(daily.minute, [0]);
  assert.deepEqual(daily.hour, [0]);
  assert.deepEqual(daily.dayOfMonth, range(1, 31));
  assert.deepEqual(daily.month, range(1, 12));
  assert.deepEqual(daily.dayOfWeek, range(0, 6));
  assert.equal(daily.domRestricted, false);
  assert.equal(daily.dowRestricted, false);

  const hourly = parse('@HOURLY');
  assert.deepEqual(hourly.minute, [0]);
  assert.deepEqual(hourly.hour, range(0, 23));

  const weekly = parse('@weekly');
  assert.deepEqual(weekly.dayOfWeek, [0]);
  assert.equal(weekly.dowRestricted, true);

  const monthly = parse('@monthly');
  assert.deepEqual(monthly.minute, [0]);
  assert.deepEqual(monthly.hour, [0]);
  assert.deepEqual(monthly.dayOfMonth, [1]);
  assert.deepEqual(monthly.month, range(1, 12));

  const yearly = parse('@Annually');
  assert.deepEqual(yearly.minute, [0]);
  assert.deepEqual(yearly.hour, [0]);
  assert.deepEqual(yearly.dayOfMonth, [1]);
  assert.deepEqual(yearly.month, [1]);

  const midnight = parse('@midnight');
  assert.deepEqual(midnight, {
    minute: [0], hour: [0],
    dayOfMonth: range(1, 31), month: range(1, 12), dayOfWeek: range(0, 6),
    domRestricted: false, dowRestricted: false,
  });
});

test('parse accepts tabs and extra spaces', () => {
  const p = parse('  0 \t 0 \t 1 \t 1 \t *  ');
  assert.deepEqual(p.minute, [0]);
  assert.deepEqual(p.dayOfMonth, [1]);
});

// ---------- parse: errors ----------

const badExpressions = [
  '',
  '   ',
  '* * * *',                 // 4 fields
  '* * * * * *',             // 6 fields
  '60 * * * *',              // minute out of range
  '* 24 * * *',              // hour out of range
  '0 0 0 * *',               // dom min
  '0 0 32 * *',              // dom max
  '0 0 0 0 *',               // month out of range
  '0 0 1 13 *',              // month out of range
  '0 0 * * 8',               // dow out of range
  '0 0 * * -1',
  '5-2 * * * *',             // range a > b
  '10-9 * * * *',
  '*/0 * * * *',             // step 0
  '1-5/0 * * * *',
  '*/ * * * *',              // missing step
  '1/ * * * *',
  '1/2/3 * * * *',           // double slash
  '1,,2 * * * *',            // empty list item
  ',1 * * * *',
  '1, * * * *',
  '*-1 * * * *',
  '-1 * * * *',
  '1- * * * *',
  '1-2-3 * * * *',
  'FOO * * * *',             // unknown name in minute
  '0 FOO * * *',
  '0 0 1 foo *',             // unknown name in month
  '0 0 * * FOO',  '0 0 * ? *',               // ?
  '0 0 * L *',               // L
  '0 0 * 1W *',              // W
  '0 0 * * 1#1',             // #
  '5 * * * * extra',
  '0 0 * * 7/2',             // 7 not allowed in step start (dow)
  '0 0 * * 7-7',             // 7 not allowed as range start (dow)
  '0 0 * * MON/2/2',         // nonsense
  '@everyday',               // unknown macro
  '@daily extra',            // not a macro
  '5 * * *',                 // 4 fields
];

for (const expr of badExpressions) {
  test(`parse throws CronError for ${JSON.stringify(expr)}`, () => {
    assert.throws(() => parse(expr), CronError);
  });
}

test('parse throws CronError for non-string', () => {
  assert.throws(() => parse(null), CronError);
  assert.throws(() => parse(42), CronError);
  assert.throws(() => parse(['0', '0', '1', '1', '*']), CronError);
});

test('parse result is a CronError extending Error', () => {
  try {
    parse('bad');
  } catch (e) {
    assert.ok(e instanceof CronError);
    assert.ok(e instanceof Error);
  }
});

// ---------- nextRuns: values ----------

const UTC = (...a) => Date.UTC(...a);
const D = (...a) => new Date(UTC(...a));

test('nextRuns daily midnight, strictly after from', () => {
  const out = nextRuns('@daily', D(2024, 0, 15, 10, 30, 15, 250), 2);
  assert.equal(out.length, 2);
  assert.equal(out[0].getTime(), UTC(2024, 0, 16, 0, 0, 0));
  assert.equal(out[1].getTime(), UTC(2024, 0, 17, 0, 0, 0));
});

test('nextRuns strictly after: from at exact run time', () => {
  const out = nextRuns('@daily', D(2024, 0, 15, 0, 0, 0), 1);
  assert.equal(out[0].getTime(), UTC(2024, 0, 16));
});

test('nextRuns returns times with zero seconds and ms', () => {
  const out = nextRuns('30 14 * * *', D(2024, 0, 15, 10, 30, 999, 999), 3);
  for (const d of out) {
    assert.equal(d.getUTCSeconds(), 0);
    assert.equal(d.getUTCMilliseconds(), 0);
    assert.equal(d.getUTCMinutes(), 30);
    assert.equal(d.getUTCHours(), 14);
  }
  assert.equal(out[0].getTime(), UTC(2024, 0, 15, 14, 30));
});

test('nextRuns honors n and ascending order', () => {
  const out = nextRuns('*/15 * * * *', D(2024, 0, 1, 0, 0), 4);
  assert.equal(out.length, 4);
  for (let i = 1; i < out.length; i++) {
    assert.ok(out[i] > out[i - 1]);
  }
  assert.equal(out[0].getTime(), UTC(2024, 0, 1, 0, 15));
});

test('nextRuns every-minute from within a minute', () => {
  const out = nextRuns('* * * * *', D(2024, 0, 1, 0, 0, 30, 0), 4);
  assert.deepEqual(out.map((d) => d.getTime()), [
    UTC(2024, 0, 1, 0, 1),
    UTC(2024, 0, 1, 0, 2),
    UTC(2024, 0, 1, 0, 3),
    UTC(2024, 0, 1, 0, 4),
  ]);
});

test('nextRuns dom only (AND rule with unrestricted dow)', () => {
  const out = nextRuns('0 0 15 * *', D(2024, 0, 1, 0, 0), 2);
  assert.deepEqual(out.map((d) => d.getTime()), [UTC(2024, 0, 15), UTC(2024, 1, 15)]);
});

test('nextRuns dow only (AND rule with unrestricted dom)', () => {
  // Jan 1 2024 is a Monday; Jan 3 is Wednesday.
  const out = nextRuns('0 0 * * 3', D(2024, 0, 1, 0, 0), 3);
  assert.deepEqual(out.map((d) => d.getTime()), [
    UTC(2024, 0, 3),
    UTC(2024, 0, 10),
    UTC(2024, 0, 17),
  ]);
});

test('nextRuns both restricted uses OR', () => {
  // dom=15 OR Mondays. Jan 1 2024 is a Monday (excluded, not strictly after).
  const out = nextRuns('0 0 15 * 1', D(2024, 0, 1, 0, 0), 5);
  assert.deepEqual(out.map((d) => d.getTime()), [
    UTC(2024, 0, 8),
    UTC(2024, 0, 15), // both
    UTC(2024, 0, 22),
    UTC(2024, 0, 29),
    UTC(2024, 1, 5),
  ]);
});

test('nextRuns */2 dom still filters days when dow unrestricted', () => {
  // dom "*/2" = 1,3,...,31 ; hour 12 every month/day-of-week matching dow even.
  // Feb 1 2024 is a Thursday (dow 4, in */2 set).
  const out = nextRuns('0 12 * * */2', D(2024, 1, 1, 0, 0), 1);
  assert.equal(out[0].getTime(), UTC(2024, 1, 1, 12, 0));
});

test('nextRuns leap day only', () => {
  const out = nextRuns('0 0 29 2 *', D(2023, 0, 1), 10);
  // 2024-02-29, 2028-02-29 and 2032-02-29 all fit inside 4000 days.
  assert.deepEqual(out.map((d) => d.getTime()), [
    UTC(2024, 1, 29),
    UTC(2028, 1, 29),
    UTC(2032, 1, 29),
  ]);
});

test('nextRuns impossible date returns empty', () => {
  assert.deepEqual(nextRuns('0 0 30 2 *', D(2024, 0, 1), 100), []);
  assert.deepEqual(nextRuns('0 0 31 2 *', D(2024, 0, 1), 100), []);
  assert.deepEqual(nextRuns('0 0 31 4 *', D(2024, 0, 1), 100), []); // April
});

test('nextRuns macros', () => {
  const hourly = nextRuns('@hourly', D(2024, 0, 1, 0, 30), 3);
  assert.deepEqual(hourly.map((d) => d.getTime()), [
    UTC(2024, 0, 1, 1, 0),
    UTC(2024, 0, 1, 2, 0),
    UTC(2024, 0, 1, 3, 0),
  ]);
  const weekly = nextRuns('@weekly', D(2024, 0, 1, 0, 0), 1); // Jan 7 2024 Sunday
  assert.equal(weekly[0].getTime(), UTC(2024, 0, 7));
  const yearly = nextRuns('@yearly', D(2024, 0, 1, 0, 0), 1);
  assert.equal(yearly[0].getTime(), UTC(2025, 0, 1));
  const monthly = nextRuns('@monthly', D(2024, 0, 15, 12, 0), 1);
  assert.equal(monthly[0].getTime(), UTC(2024, 1, 1, 0, 0));
});

test('nextRuns n cap is exact', () => {
  const out = nextRuns('* * * * *', D(2024, 0, 1), 1000);
  assert.equal(out.length, 1000);
});

test('nextRuns works far in the past and future', () => {
  const out = nextRuns('0 0 1 1 *', D(2020, 11, 31, 23, 59), 1);
  assert.equal(out[0].getTime(), UTC(2021, 0, 1));
  const out2 = nextRuns('0 0 31 1 *', D(2024, 0, 1), 1);
  assert.equal(out2[0].getTime(), UTC(2024, 0, 31));
});

test('nextRuns window is 4000 days (impossible month-day returns none early)', () => {
  // Feb 30 never exists: full 4000-day scan must still be fast.
  assert.deepEqual(nextRuns('0 0 30 2 *', D(2024, 0, 1), 1000), []);
});

test('nextRuns performance: full 4000-day scan under 100ms', () => {
  const t0 = process.hrtime.bigint();
  nextRuns('0 0 30 2 *', D(2024, 0, 1), 1000);
  const ms = Number(process.hrtime.bigint() - t0) / 1e6;
  assert.ok(ms < 100, `took ${ms}ms`);

  const t1 = process.hrtime.bigint();
  nextRuns('0 0 31 2 *', D(2024, 0, 1), 1000); // never matches, worst scan
  const ms2 = Number(process.hrtime.bigint() - t1) / 1e6;
  assert.ok(ms2 < 100, `took ${ms2}ms`);
});

// ---------- nextRuns: errors ----------

test('nextRuns throws TypeError for invalid from', () => {
  assert.throws(() => nextRuns('@daily', new Date('nope'), 1), TypeError);
  assert.throws(() => nextRuns('@daily', null, 1), TypeError);
  assert.throws(() => nextRuns('@daily', 123, 1), TypeError);
  assert.throws(() => nextRuns('@daily', undefined, 1), TypeError);
  assert.throws(() => nextRuns('@daily', { toISOString: () => '' }, 1), TypeError);
});

test('nextRuns throws TypeError for invalid n', () => {
  const from = new Date();
  assert.throws(() => nextRuns('@daily', from, 0), TypeError);
  assert.throws(() => nextRuns('@daily', from, 1001), TypeError);
  assert.throws(() => nextRuns('@daily', from, 1.5), TypeError);
  assert.throws(() => nextRuns('@daily', from, '5'), TypeError);
  assert.throws(() => nextRuns('@daily', from, null), TypeError);
  assert.throws(() => nextRuns('@daily', from, NaN), TypeError);
  // boundaries are fine
  nextRuns('@daily', from, 1);
  nextRuns('@daily', from, 1000);
});

test('nextRuns throws CronError for invalid expressions', () => {
  const from = new Date();
  assert.throws(() => nextRuns('60 * * * *', from, 1), CronError);
  assert.throws(() => nextRuns('@bogus', from, 1), CronError);
  assert.throws(() => nextRuns('not an expression', from, 1), CronError);
});

test('nextRuns parses macros and reuses parse', () => {
  const out = nextRuns('@midnight', D(2024, 5, 15, 23, 59, 59), 1);
  assert.equal(out[0].getTime(), UTC(2024, 5, 16));
});
