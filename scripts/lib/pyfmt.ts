/**
 * Python-compatible text output for the evidence scripts.
 *
 * The audit files in `samples/validation/` were written by Python (`json.dumps(indent=2)` and
 * `csv.DictWriter`). These helpers reproduce that formatting byte for byte so a TS run can be
 * diffed against the Python baseline with `git diff` or `cmp`, not only through the fingerprints.
 */
import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname } from "node:path";

/** Python `repr(float)`: shortest round-trip digits, `.0` on integral values, `e-05` / `e+16` exponents. */
export function pyFloat(x: number): string {
  if (Number.isNaN(x)) return "NaN";
  if (x === Infinity) return "Infinity";
  if (x === -Infinity) return "-Infinity";
  if (x === 0) return Object.is(x, -0) ? "-0.0" : "0.0";
  const a = Math.abs(x);
  if (a >= 1e16 || a < 1e-4) {
    const [mant, exp] = x.toExponential().split("e") as [string, string];
    const sign = exp.startsWith("-") ? "-" : "+";
    const digits = exp.replace(/^[-+]/, "").padStart(2, "0");
    return `${mant}e${sign}${digits}`;
  }
  const s = String(x); // JS switches to exponent only outside [1e-7, 1e21): never here
  return /[.e]/.test(s) ? s : `${s}.0`;
}

/** Python `str()` of a scalar as written by `csv.writer` (`True`, `1.0`, `None` -> empty). */
export function pyStr(v: unknown, isInt = false): string {
  if (v === null || v === undefined) return "";
  if (typeof v === "boolean") return v ? "True" : "False";
  if (typeof v === "number") return isInt && Number.isInteger(v) ? String(v) : pyFloat(v);
  return String(v);
}

function escapeString(s: string): string {
  let out = '"';
  for (const ch of s) {
    const c = ch.codePointAt(0)!;
    if (ch === '"') out += '\\"';
    else if (ch === "\\") out += "\\\\";
    else if (ch === "\n") out += "\\n";
    else if (ch === "\r") out += "\\r";
    else if (ch === "\t") out += "\\t";
    else if (ch === "\b") out += "\\b";
    else if (ch === "\f") out += "\\f";
    else if (c < 0x20 || c > 0x7e) {
      // ensure_ascii=True: \uXXXX, astral code points as surrogate pairs
      for (let i = 0; i < ch.length; i++) out += "\\u" + ch.charCodeAt(i).toString(16).padStart(4, "0");
    } else out += ch;
  }
  return out + '"';
}

/**
 * `json.dumps(value, indent=indent)` with Python's number formatting.
 *
 * JS has no int/float distinction, so `isInt(path)` decides which numbers print without a
 * decimal point; `path` is the key chain, with `[]` for list items (e.g. `label_shape[]`).
 * Every other number prints as a Python float (`8.0`). NaN and infinities are rejected, as with
 * `allow_nan=False`.
 */
export function pyJson(value: unknown, indent = 2, isInt: (path: string) => boolean = () => false): string {
  const walk = (v: unknown, path: string, depth: number): string => {
    if (v === null || v === undefined) return "null";
    if (typeof v === "boolean") return v ? "true" : "false";
    if (typeof v === "number") {
      if (!Number.isFinite(v)) throw new Error(`Out of range float values are not JSON compliant: ${v} at ${path}`);
      return isInt(path) && Number.isInteger(v) ? String(v) : pyFloat(v);
    }
    if (typeof v === "string") return escapeString(v);
    const pad = "\n" + " ".repeat(indent * (depth + 1));
    const end = "\n" + " ".repeat(indent * depth);
    if (Array.isArray(v)) {
      if (!v.length) return "[]";
      return "[" + v.map((x) => pad + walk(x, `${path}[]`, depth + 1)).join(",") + end + "]";
    }
    const entries = Object.entries(v as Record<string, unknown>);
    if (!entries.length) return "{}";
    return (
      "{" +
      entries.map(([k, x]) => pad + escapeString(k) + ": " + walk(x, path ? `${path}.${k}` : k, depth + 1)).join(",") +
      end +
      "}"
    );
  };
  return walk(value, "", 0);
}

/** One CSV field with `csv.QUOTE_MINIMAL` quoting. */
function csvField(s: string): string {
  return /[",\r\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
}

/** `csv.DictWriter(columns, lineterminator=eol)` + `writeheader()` + `writerows(rows)`; values already strings. */
export function csvText(columns: readonly string[], rows: Record<string, string>[], eol = "\r\n"): string {
  const lines = [columns.map(csvField).join(",")];
  for (const r of rows) lines.push(columns.map((c) => csvField(r[c] ?? "")).join(","));
  return lines.join(eol) + eol;
}

/** Write a CSV, creating the parent folder. */
export function writeCsv(path: string, columns: readonly string[], rows: Record<string, string>[], eol = "\r\n"): void {
  mkdirSync(dirname(path), { recursive: true });
  writeFileSync(path, csvText(columns, rows, eol), "utf8");
}

/** RFC 4180 reader returning one record per row keyed by the header (`csv.DictReader`, utf-8-sig). */
export function readCsv(path: string): Record<string, string>[] {
  let text = readFileSync(path, "utf8");
  if (text.charCodeAt(0) === 0xfeff) text = text.slice(1);
  const rows: string[][] = [];
  let row: string[] = [];
  let field = "";
  let quoted = false;
  for (let i = 0; i < text.length; i++) {
    const ch = text[i]!;
    if (quoted) {
      if (ch === '"') {
        if (text[i + 1] === '"') {
          field += '"';
          i++;
        } else quoted = false;
      } else field += ch;
    } else if (ch === '"') quoted = true;
    else if (ch === ",") {
      row.push(field);
      field = "";
    } else if (ch === "\n" || ch === "\r") {
      if (ch === "\r" && text[i + 1] === "\n") i++;
      row.push(field);
      rows.push(row);
      row = [];
      field = "";
    } else field += ch;
  }
  if (field !== "" || row.length) {
    row.push(field);
    rows.push(row);
  }
  const [header, ...body] = rows.filter((r) => !(r.length === 1 && r[0] === ""));
  if (!header) return [];
  return body.map((r) => Object.fromEntries(header.map((h, k) => [h, r[k] ?? ""])));
}

/** Python `round(x, nd)` (correctly rounded, half to even). */
export { pyRoundN as pyRound } from "../../src/steps/np.js";
