# Cron library test prompt (as sent to Pi on 17 Nov 2026; change only the folder name per run)

Build a cron expression library in JavaScript. Save it in ./cron-splash-1/ (cron.mjs plus your tests), then test and fix until your tests pass. Use Node built-ins only: no npm packages, no browser. Do not edit files outside that folder.

## API (ES module cron.mjs)
- export class CronError extends Error {}
- export function parse(expr) returns
    { minute, hour, dayOfMonth, month, dayOfWeek,   // sorted arrays of unique integers
      domRestricted, dowRestricted }                 // booleans
- export function nextRuns(expr, from, n) returns an array of Date

## Expression format
Five fields separated by one or more spaces or tabs; leading and trailing whitespace is ignored.

  minute        0-59
  hour          0-23
  day of month  1-31
  month         1-12, or JAN-DEC
  day of week   0-7 (0 and 7 are both Sunday), or SUN-SAT

Each field is a comma-separated list of items. An item is one of:
- `*`: every value
- a value: `5`
- a range: `a-b` with a <= b, inclusive
- a step: `*/s`, `a-b/s`, or `a/s` (a through the field's maximum), where s is a number >= 1

Numbers are plain decimal digits (leading zeros allowed). Names are case-insensitive and can be used anywhere a number can except as a step, including in ranges (MON-FRI). For day of week, `*` and `a/s` run from 0 to 6; 7 may appear only as a value or a range end. In parse() output, day of week 7 becomes 0 and names become numbers.

Macros (the whole expression, case-insensitive): @yearly and @annually = "0 0 1 1 *", @monthly = "0 0 1 * *", @weekly = "0 0 * * 0", @daily and @midnight = "0 0 * * *", @hourly = "0 * * * *".

## Matching
All times are UTC; never use local time.
A minute matches when its minute, hour and month match and the day rule holds:
- A day field is unrestricted if its text starts with `*` (so `*` and `*/2`). domRestricted and dowRestricted are true when the field is restricted.
- If both day fields are restricted, the day matches when EITHER field matches.
- Otherwise the day must match BOTH fields (`*` matches every day; `*/2` still filters by its values).

## nextRuns(expr, from, n)
- from is a Date; n is an integer from 1 to 1000.
- Returns up to n matching times strictly after from, in ascending order, each at second 0 and millisecond 0.
- Only times earlier than from + 4000 days count. If fewer than n exist in that window, return the ones found (possibly an empty array). Example: "0 0 30 2 *" returns [].
- Each call must finish in under 100 ms for any valid expression.

## Errors
- parse() and nextRuns() throw CronError for any invalid expression: wrong number of fields, values out of range, a > b in a range, a step of 0 or a missing step, empty list items, unknown names or macros, and anything else not described above (for example ?, L, W, #).
- nextRuns() throws TypeError if from is not a valid Date or n is not an integer from 1 to 1000.

When done, reply with a short summary of what you built and how you tested it.
