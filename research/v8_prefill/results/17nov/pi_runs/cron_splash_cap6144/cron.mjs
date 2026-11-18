export class CronError extends Error {
  constructor(message) {
    super(message);
    this.name = 'CronError';
  }
}

const MONTH_NAMES = {
  JAN: 1, FEB: 2, MAR: 3, APR: 4, MAY: 5, JUN: 6,
  JUL: 7, AUG: 8, SEP: 9, OCT: 10, NOV: 11, DEC: 12,
};

const DOW_NAMES = {
  SUN: 0, MON: 1, TUE: 2, WED: 3, THU: 4, FRI: 5, SAT: 6,
};

const FIELDS = [
  { key: 'minute', min: 0, max: 59, names: {} },
  { key: 'hour', min: 0, max: 23, names: {} },
  { key: 'dayOfMonth', min: 1, max: 31, names: {} },
  { key: 'month', min: 1, max: 12, names: MONTH_NAMES },
  { key: 'dayOfWeek', min: 0, max: 7, names: DOW_NAMES },
];

const MACROS = {
  '@yearly': '0 0 1 1 *',
  '@annually': '0 0 1 1 *',
  '@monthly': '0 0 1 * *',
  '@weekly': '0 0 * * 0',
  '@daily': '0 0 * * *',
  '@midnight': '0 0 * * *',
  '@hourly': '0 * * * *',
};

function parseToken(tok, spec) {
  if (/^\d+$/.test(tok)) {
    const v = parseInt(tok, 10);
    if (v < spec.min || v > spec.max) {
      throw new CronError(`value ${v} out of range for ${spec.key}`);
    }
    return v;
  }
  const name = spec.names[tok.toUpperCase()];
  if (name === undefined) {
    throw new CronError(`unknown value "${tok}" for ${spec.key}`);
  }
  return name;
}

function parseField(text, spec) {
  // Returns sorted unique array of values.
  // dow step-end: step items run 0..6 (7 is only allowed as a value or range end).
  const values = new Set();
  const isDow = spec.key === 'dayOfWeek';
  const stepEndMax = isDow ? 6 : spec.max;

  for (const item of text.split(',')) {
    if (item === '') throw new CronError(`empty list item in ${spec.key} field`);
    let base = item;
    let step = null;
    const slash = item.indexOf('/');
    if (slash !== -1) {
      if (item.indexOf('/', slash + 1) !== -1) {
        throw new CronError(`invalid step item "${item}" in ${spec.key} field`);
      }
      base = item.slice(0, slash);
      const stepPart = item.slice(slash + 1);
      if (!/^\d+$/.test(stepPart)) {
        throw new CronError(`invalid step "${stepPart}" in ${spec.key} field`);
      }
      step = parseInt(stepPart, 10);
      if (step === 0) {
        throw new CronError(`step of 0 in ${spec.key} field`);
      }
    }

    let start;
    let end;
    if (base === '*') {
      if (step === null) {
        start = spec.min;
        end = spec.max;
      } else {
        // */s runs from min to step-end (dow: 0..6)
        start = spec.min;
        end = stepEndMax;
      }
    } else if (base.includes('-')) {
      const parts = base.split('-');
      if (parts.length !== 2 || parts[0] === '' || parts[1] === '') {
        throw new CronError(`invalid range "${base}" in ${spec.key} field`);
      }
      start = parseToken(parts[0], spec);
      end = parseToken(parts[1], spec);
      if (isDow && start === 7) {
        throw new CronError(`7 is not allowed as a range start in day of week`);
      }
      if (start > end) {
        throw new CronError(`range start greater than end in "${base}"`);
      }
    } else {
      start = parseToken(base, spec);
      if (step === null) {
        end = start;
      } else {
        // a/s : a through the field's maximum (dow: 6; 7/s is invalid)
        end = stepEndMax;
        if (start > end) {
          throw new CronError(`value ${start} out of range for step item in ${spec.key} field`);
        }
      }
    }

    if (step === null) {
      for (let v = start; v <= end; v++) values.add(v);
    } else {
      for (let v = start; v <= end; v += step) values.add(v);
    }
  }

  const out = [...values].map((v) => (isDow && v === 7 ? 0 : v));
  const final = [...new Set(out)].sort((a, b) => a - b);
  return final;
}

function parseParts(parts) {
  if (parts.length !== 5) {
    throw new CronError(`expected 5 fields, got ${parts.length}`);
  }
  const fields = {};
  for (let i = 0; i < 5; i++) {
    fields[FIELDS[i].key] = parseField(parts[i], FIELDS[i]);
  }
  fields.domRestricted = !parts[2].startsWith('*');
  fields.dowRestricted = !parts[4].startsWith('*');
  return fields;
}

export function parse(expr) {
  if (typeof expr !== 'string') {
    throw new CronError('cron expression must be a string');
  }
  const trimmed = expr.trim();
  if (trimmed.startsWith('@')) {
    const macro = MACROS[trimmed.toLowerCase()];
    if (!macro) throw new CronError(`unknown macro "${trimmed}"`);
    return parseParts(macro.split(/\s+/));
  }
  const parts = trimmed === '' ? [trimmed] : trimmed.split(/\s+/);
  return parseParts(parts);
}

const MS_DAY = 86400000;

export function nextRuns(expr, from, n) {
  const spec = parse(expr);
  if (!(from instanceof Date) || Number.isNaN(from.getTime())) {
    throw new TypeError('from must be a valid Date');
  }
  if (!Number.isInteger(n) || n < 1 || n > 1000) {
    throw new TypeError('n must be an integer from 1 to 1000');
  }

  const fromMs = from.getTime();
  const endMs = fromMs + 4000 * MS_DAY;

  const minutes = spec.minute;
  const hours = spec.hour;
  const monthSet = new Set(spec.month);
  const domSet = new Set(spec.dayOfMonth);
  const dowSet = new Set(spec.dayOfWeek);
  const bothRestricted = spec.domRestricted && spec.dowRestricted;

  const out = [];
  const startDayMs = Math.floor(fromMs / MS_DAY) * MS_DAY;
  outer: for (let day = startDayMs; day < endMs; day += MS_DAY) {
    const d = new Date(day);
    const month = d.getUTCMonth() + 1;
    if (!monthSet.has(month)) continue;
    const domOk = domSet.has(d.getUTCDate());
    const dowOk = dowSet.has(d.getUTCDay());
    const dayOk = bothRestricted ? (domOk || dowOk) : (domOk && dowOk);
    if (!dayOk) continue;
    for (const h of hours) {
      for (const m of minutes) {
        const ts = day + h * 3600000 + m * 60000;
        if (ts > fromMs && ts < endMs) {
          out.push(new Date(ts));
          if (out.length === n) break outer;
        }
      }
    }
  }
  return out;
}
