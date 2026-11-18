// cron.mjs — a cron expression parser / next-run calculator.
// Node built-ins only, ES module, all times in UTC.

export class CronError extends Error {}

const MACROS = {
  '@yearly': '0 0 1 1 *',
  '@annually': '0 0 1 1 *',
  '@monthly': '0 0 1 * *',
  '@weekly': '0 0 * * 0',
  '@daily': '0 0 * * *',
  '@midnight': '0 0 * * *',
  '@hourly': '0 * * * *',
};

const MONTH_NAMES = {
  JAN: 1, FEB: 2, MAR: 3, APR: 4, MAY: 5, JUN: 6,
  JUL: 7, AUG: 8, SEP: 9, OCT: 10, NOV: 11, DEC: 12,
};

const DOW_NAMES = {
  SUN: 0, MON: 1, TUE: 2, WED: 3, THU: 4, FRI: 5, SAT: 6,
};

const DAY_MS = 86_400_000;

function fail(message) {
  throw new CronError(message);
}

/** Parse a single token (value or name) in a field. */
function parseValue(token, names, label) {
  if (/^\d+$/.test(token)) return parseInt(token, 10);
  if (names) {
    const v = names[token.toUpperCase()];
    if (v !== undefined) return v;
  }
  fail(`unknown value "${token}" in ${label} field`);
}

/**
 * Parse one cron field into a sorted, de-duplicated array of integers.
 * opts: { min, max, stepMax, names?, label }
 *   min/max   : allowed value bounds (dow max is 7; 7 is normalized later)
 *   stepMax   : upper bound used when a step item has no explicit end
 */
function parseField(text, opts) {
  const { min, max, stepMax, names, label } = opts;
  const values = new Set();

  for (const rawItem of text.split(',')) {
    if (rawItem === '') fail(`empty list item in ${label} field`);

    let base = rawItem;
    let step = null;
    const slash = rawItem.indexOf('/');
    if (slash !== -1) {
      base = rawItem.slice(0, slash);
      const stepText = rawItem.slice(slash + 1);
      if (!/^\d+$/.test(stepText)) {
        fail(`invalid step in "${rawItem}" of ${label} field`);
      }
      step = parseInt(stepText, 10);
      if (step < 1) fail(`step must be >= 1 in "${rawItem}" of ${label} field`);
    }

    let lo;
    let hi;
    if (base === '*') {
      lo = min;
      hi = step === null ? max : stepMax;
    } else {
      const dash = base.indexOf('-');
      if (dash !== -1) {
        if (dash === 0 || base.indexOf('-', dash + 1) !== -1) {
          fail(`invalid range "${base}" in ${label} field`);
        }
        lo = parseValue(base.slice(0, dash), names, label);
        hi = parseValue(base.slice(dash + 1), names, label);
        if (lo > hi) fail(`range start greater than end in "${base}" of ${label} field`);
      } else {
        lo = parseValue(base, names, label);
        hi = step === null ? lo : stepMax;
      }
    }

    if (lo < min || hi > max) {
      fail(`value out of range ${min}-${max} in ${label} field: "${rawItem}"`);
    }

    if (step === null) {
      for (let v = lo; v <= hi; v++) values.add(v);
    } else {
      for (let v = lo; v <= hi; v += step) values.add(v);
    }
  }

  return [...values].sort((a, b) => a - b);
}

export function parse(expr) {
  if (typeof expr !== 'string') {
    fail('expression must be a string');
  }

  let text = expr.trim();
  const macro = MACROS[text.toLowerCase()];
  if (text === '' && !macro) fail('empty expression');
  if (macro) text = macro;

  const fields = text.split(/[\t ]+/);
  if (fields.length !== 5) {
    fail(`expected 5 fields, got ${fields.length}`);
  }

  const [minuteText, hourText, domText, monthText, dowText] = fields;

  const minute = parseField(minuteText, { min: 0, max: 59, stepMax: 59, label: 'minute' });
  const hour = parseField(hourText, { min: 0, max: 23, stepMax: 23, label: 'hour' });
  const dayOfMonth = parseField(domText, { min: 1, max: 31, stepMax: 31, label: 'day of month' });
  const month = parseField(monthText, {
    min: 1, max: 12, stepMax: 12, names: MONTH_NAMES, label: 'month',
  });

  let dayOfWeek = parseField(dowText, {
    min: 0, max: 7, stepMax: 6, names: DOW_NAMES, label: 'day of week',
  });
  dayOfWeek = [...new Set(dayOfWeek.map((v) => (v === 7 ? 0 : v)))].sort((a, b) => a - b);

  return {
    minute,
    hour,
    dayOfMonth,
    month,
    dayOfWeek,
    domRestricted: domText[0] !== '*',
    dowRestricted: dowText[0] !== '*',
  };
}

function dayMatches(spec, year, month1, dayOfMonth, dayOfWeek) {
  if (!spec.month.includes(month1)) return false;

  const domOk = spec.dayOfMonth.includes(dayOfMonth);
  const dowOk = spec.dayOfWeek.includes(dayOfWeek);

  if (spec.domRestricted && spec.dowRestricted) return domOk || dowOk;
  return domOk && dowOk;
}

export function nextRuns(expr, from, n) {
  if (!(from instanceof Date) || Number.isNaN(from.getTime())) {
    throw new TypeError('from must be a valid Date');
  }
  if (!Number.isInteger(n) || n < 1 || n > 1000) {
    throw new TypeError('n must be an integer from 1 to 1000');
  }

  const spec = parse(expr);
  const fromMs = from.getTime();
  const cutoffMs = fromMs + 4000 * DAY_MS;

  const start = new Date(fromMs);
  let y = start.getUTCFullYear();
  let m = start.getUTCMonth();
  let d = start.getUTCDate();

  const monthsSet = new Set(spec.month);
  const hours = spec.hour;
  const minutes = spec.minute;
  const results = [];

  let guard = 0;
  while (results.length < n && guard++ < 4005) {
    const dayMs = Date.UTC(y, m, d);
    if (dayMs >= cutoffMs) break;

    const dow = new Date(dayMs).getUTCDay();
    if (
      monthsSet.has(m + 1) &&
      dayMatches(spec, y, m + 1, d, dow)
    ) {
      for (const h of hours) {
        for (const min of minutes) {
          const candidateMs = Date.UTC(y, m, d, h, min);
          if (candidateMs <= fromMs) continue; // must be strictly after `from`
          if (candidateMs >= cutoffMs) break; // inner minute loop is ascending
          results.push(new Date(candidateMs));
          if (results.length === n) break;
        }
        if (results.length === n) break;
      }
    }

    // Advance one calendar day (Date.UTC normalizes overflow).
    const nextDay = new Date(Date.UTC(y, m, d + 1));
    y = nextDay.getUTCFullYear();
    m = nextDay.getUTCMonth();
    d = nextDay.getUTCDate();
  }

  return results;
}
