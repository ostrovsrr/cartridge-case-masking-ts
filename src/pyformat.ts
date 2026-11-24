/**
 * Python-compatible text output for the CLI: `repr(float)`, `json.dumps(obj, indent=2)` and
 * `csv.DictWriter` rows, so `masks.json` and `summary.csv` read exactly like the Python files.
 *
 * Every number in a `Result.summary()` is a Python float, so numbers are written as floats
 * (`8.0`, not `8`); pass `ints` to keep chosen keys integral.
 */

/** Python `repr(float)` (shortest round-trip digits; exponent below 1e-4 and from 1e16). */
export function pyFloatRepr(v: number): string {
  if (Number.isNaN(v)) return "nan";
  if (!Number.isFinite(v)) return v > 0 ? "inf" : "-inf";
  if (v === 0) return Object.is(v, -0) ? "-0.0" : "0.0";
  const [mant, expS] = v.toExponential().split("e");
  const e = Number(expS);
  const neg = mant.startsWith("-");
  const digits = mant.replace("-", "").replace(".", "");
  const sign = neg ? "-" : "";
  if (e < -4 || e >= 16) {
    const m = digits.length > 1 ? `${digits[0]}.${digits.slice(1)}` : digits;
    return `${sign}${m}e${e < 0 ? "-" : "+"}${String(Math.abs(e)).padStart(2, "0")}`;
  }
  if (e < 0) return `${sign}0.${"0".repeat(-e - 1)}${digits}`;
  if (digits.length <= e + 1) return `${sign}${digits}${"0".repeat(e + 1 - digits.length)}.0`;
  return `${sign}${digits.slice(0, e + 1)}.${digits.slice(e + 1)}`;
}

/** JSON number as Python's `json` module writes a float (NaN / Infinity literals). */
function jsonFloat(v: number): string {
  if (Number.isNaN(v)) return "NaN";
  if (!Number.isFinite(v)) return v > 0 ? "Infinity" : "-Infinity";
  return pyFloatRepr(v);
}

/** `json.dumps(s)` with `ensure_ascii=True`. */
function jsonString(s: string): string {
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
      if (c > 0xffff) {
        const v = c - 0x10000;
        out += `\\u${(0xd800 + (v >> 10)).toString(16).padStart(4, "0")}\\u${(0xdc00 + (v & 0x3ff)).toString(16).padStart(4, "0")}`;
      } else out += `\\u${c.toString(16).padStart(4, "0")}`;
    } else out += ch;
  }
  return out + '"';
}

/**
 * `json.dumps(obj, indent=indent)` as Python writes it. Numbers are floats unless their key is in
 * `ints` (or they sit in an array under such a key).
 */
export function pyJsonDumps(obj: unknown, indent = 2, ints: ReadonlySet<string> = new Set()): string {
  const walk = (o: unknown, level: number, isInt: boolean): string => {
    if (o === null || o === undefined) return "null";
    if (typeof o === "boolean") return o ? "true" : "false";
    if (typeof o === "number") return isInt && Number.isInteger(o) ? String(o) : jsonFloat(o);
    if (typeof o === "string") return jsonString(o);
    const pad = "\n" + " ".repeat(indent * (level + 1));
    const end = "\n" + " ".repeat(indent * level);
    if (Array.isArray(o)) {
      if (o.length === 0) return "[]";
      return "[" + o.map((v) => pad + walk(v, level + 1, isInt)).join(",") + end + "]";
    }
    const entries = Object.entries(o as Record<string, unknown>);
    if (entries.length === 0) return "{}";
    return "{" + entries.map(([k, v]) => pad + jsonString(k) + ": " + walk(v, level + 1, ints.has(k))).join(",") + end + "}";
  };
  return walk(obj, 0, false);
}

/** `str(value)` as `csv.DictWriter` writes it (None -> empty, floats via repr, bools True/False). */
export function csvValue(v: unknown): string {
  if (v === null || v === undefined) return "";
  if (typeof v === "boolean") return v ? "True" : "False";
  if (typeof v === "number") return pyFloatRepr(v);
  return String(v);
}

/** One `csv.writer` row (QUOTE_MINIMAL, `\r\n` line terminator). */
export function csvRow(values: readonly string[]): string {
  return values.map((s) => (/[",\r\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s)).join(",") + "\r\n";
}
