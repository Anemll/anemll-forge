// Small UTC cron expression library. No dependencies, no browser APIs.

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
const DOW_NAMES = { SUN: 0, MON: 1, TUE: 2, WED: 3, THU: 4, FRI: 5, SAT: 6 };

const FIELDS = [
  { key: 'minute',     min: 0, max: 59, names: null,        seven: false },
  { key: 'hour',       min: 0, max: 23, names: null,        seven: false },
  { key: 'dayOfMonth', min: 1, max: 31, names: null,        seven: false },
  { key: 'month',      min: 1, max: 12, names: MONTH_NAMES, seven: false },
  { key: 'dayOfWeek',  min: 0, max: 6,  names: DOW_NAMES,   seven: true  },
];

const MACROS = {
  '@yearly':   '0 0 1 1 *',
  '@annually': '0 0 1 1 *',
  '@monthly':  '0 0 1 * *',
  '@weekly':   '0 0 * * 0',
  '@daily':    '0 0 * * *',
  '@midnight': '0 0 * * *',
  '@hourly':   '0 * * * *',
};

const MONTH_LENGTHS = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];

function daysInMonth(year, month0) {
  if (month0 === 1) {
    const leap = year % 4 === 0 && (year % 100 !== 0 || year % 400 === 0);
    return leap ? 29 : 28;
  }
  return MONTH_LENGTHS[month0];
}

function fail(msg) {
  throw new CronError(msg);
}

function tokenValue(tok, names) {
  if (/^\d+$/.test(tok)) return Number(tok);
  if (names && Object.prototype.hasOwnProperty.call(names, tok.toUpperCase())) {
    return names[tok.toUpperCase()];
  }
  fail('Invalid value: ' + tok);
}

function parseItemInto(item, f, set) {
  const segs = item.split('/');
  if (segs.length > 2) fail('Invalid item: ' + item);
  const body = segs[0];
  let step = 1;
  if (segs.length === 2) {
    if (!/^\d+$/.test(segs[1])) fail('Invalid step: ' + item);
    step = Number(segs[1]);
    if (step < 1) fail('Step must be >= 1: ' + item);
  }
  let lo, hi;
  if (body === '*') {
    lo = f.min;
    hi = f.max;
  } else if (body.includes('-')) {
    const parts = body.split('-');
    if (parts.length !== 2) fail('Invalid range: ' + item);
    if (parts[0] === '' || parts[1] === '') fail('Invalid range: ' + item);
    const a = tokenValue(parts[0], f.names);
    const b = tokenValue(parts[1], f.names);
    const hiMax = f.seven ? 7 : f.max;
    if (a < f.min || b < f.min) fail('Value out of range: ' + item);
    if (a > hiMax || b > hiMax) fail('Value out of range: ' + item);
    if (f.seven && a === 7) fail('7 may only be a value or a range end: ' + item);
    if (a > b) fail('Range start greater than end: ' + item);
    lo = a;
    hi = b;
  } else {
    if (body === '') fail('Empty item');
    const v = tokenValue(body, f.names);
    const hiMax = f.seven ? 7 : f.max;
    if (v < f.min || v > hiMax) fail('Value out of range: ' + item);
    if (segs.length === 2) {
      if (f.seven && v === 7) fail('7 may only be a value or a range end: ' + item);
      lo = v;
      hi = f.max;
    } else {
      lo = hi = v;
    }
  }
  for (let v = lo; v <= hi; v += step) {
    set.add(f.seven && v === 7 ? 0 : v);
  }
}

function parseField(text, f) {
  const set = new Set();
  for (const item of text.split(',')) {
    parseItemInto(item, f, set);
  }
  return [...set].sort((a, b) => a - b);
}

export function parse(expr) {
  if (typeof expr !== 'string') fail('Expression must be a string');
  const trimmed = expr.trim();
  if (trimmed === '') fail('Expected 5 fields, got 0');
  let text = trimmed;
  const lower = trimmed.toLowerCase();
  if (lower.startsWith('@')) {
    if (!Object.prototype.hasOwnProperty.call(MACROS, lower)) {
      fail('Unknown macro: ' + lower);
    }
    text = MACROS[lower];
  }
  const parts = text.split(/[ \t]+/);
  if (parts.length !== 5) fail('Expected 5 fields, got ' + parts.length);
  const result = {
    minute: [],
    hour: [],
    dayOfMonth: [],
    month: [],
    dayOfWeek: [],
    domRestricted: false,
    dowRestricted: false,
  };
  for (let i = 0; i < FIELDS.length; i++) {
    result[FIELDS[i].key] = parseField(parts[i], FIELDS[i]);
  }
  result.domRestricted = !parts[2].startsWith('*');
  result.dowRestricted = !parts[4].startsWith('*');
  return result;
}

function nextAfter(sortedArr, v) {
  for (let i = 0; i < sortedArr.length; i++) {
    if (sortedArr[i] > v) return sortedArr[i];
  }
  return null;
}

function advanceDay(state) {
  state.da += 1;
  if (state.da > daysInMonth(state.y, state.mo)) {
    state.da = 1;
    state.mo += 1;
    if (state.mo === 12) {
      state.mo = 0;
      state.y += 1;
    }
  }
}

export function nextRuns(expr, from, n) {
  if (!(from instanceof Date) || Number.isNaN(from.getTime())) {
    throw new TypeError('from must be a valid Date');
  }
  if (!Number.isInteger(n) || n < 1 || n > 1000) {
    throw new TypeError('n must be an integer from 1 to 1000');
  }
  const p = parse(expr);
  const minutes = p.minute;
  const hours = p.hour;
  const months = p.month;
  const monthSet = new Set(months);
  const domSet = new Set(p.dayOfMonth);
  const dowSet = new Set(p.dayOfWeek);
  const bothRestricted = p.domRestricted && p.dowRestricted;

  const cap = from.getTime() + 4000 * 86400000;
  const start = Math.floor(from.getTime() / 60000) * 60000 + 60000;
  const sd = new Date(start);
  const state = {
    y: sd.getUTCFullYear(),
    mo: sd.getUTCMonth(),
    da: sd.getUTCDate(),
    h: sd.getUTCHours(),
    mi: sd.getUTCMinutes(),
  };

  const out = [];
  let dayKey = null;
  let dow = 0;

  while (out.length < n) {
    const t = Date.UTC(state.y, state.mo, state.da, state.h, state.mi);
    if (!Number.isFinite(t) || t >= cap) break;

    const monthNum = state.mo + 1;
    if (!monthSet.has(monthNum)) {
      const nm = nextAfter(months, monthNum);
      if (nm === null) {
        state.y += 1;
        state.mo = months[0] - 1;
      } else {
        state.mo = nm - 1;
      }
      state.da = 1;
      state.h = 0;
      state.mi = 0;
      dayKey = null;
      continue;
    }

    const key = (state.y + 100000) * 400 + state.mo * 31 + state.da;
    if (key !== dayKey) {
      dayKey = key;
      dow = new Date(Date.UTC(state.y, state.mo, state.da)).getUTCDay();
    }
    const domOk = domSet.has(state.da);
    const dowOk = dowSet.has(dow);
    const dayOk = bothRestricted ? (domOk || dowOk) : (domOk && dowOk);
    if (!dayOk) {
      advanceDay(state);
      state.h = 0;
      state.mi = 0;
      dayKey = null;
      continue;
    }

    if (!hours.includes(state.h)) {
      const nh = nextAfter(hours, state.h);
      if (nh === null) {
        advanceDay(state);
        dayKey = null;
      }
      state.h = nh === null ? 0 : nh;
      state.mi = 0;
      continue;
    }

    if (!minutes.includes(state.mi)) {
      const nm = nextAfter(minutes, state.mi);
      if (nm === null) {
        state.h += 1;
        if (state.h === 24) {
          state.h = 0;
          advanceDay(state);
          dayKey = null;
        }
      }
      state.mi = nm === null ? 0 : nm;
      continue;
    }

    out.push(new Date(t));
    state.mi += 1;
    if (state.mi === 60) {
      state.mi = 0;
      state.h += 1;
      if (state.h === 24) {
        state.h = 0;
        advanceDay(state);
        dayKey = null;
      }
    }
  }

  return out;
}
