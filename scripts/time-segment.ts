/**
 * Time segment() on a few scans (read time excluded), several repeats after a warm-up run.
 *   npx tsx scripts/time-segment.ts <file.x3p> [...] [--repeat N]
 */
import { readX3P } from "../src/x3p.js";
import { segment } from "../src/segment.js";

const args = process.argv.slice(2);
const ri = args.indexOf("--repeat");
const repeat = ri >= 0 ? Number(args[ri + 1]) : 3;
const files = args.filter((a, i) => !a.startsWith("--") && (ri < 0 || i !== ri + 1));
for (const f of files) {
  const s = readX3P(f);
  segment(s); // warm-up (JIT)
  const ts: number[] = [];
  for (let k = 0; k < repeat; k++) {
    const t0 = performance.now();
    segment(s);
    ts.push((performance.now() - t0) / 1000);
  }
  console.log(`${s.name}  [${s.z.rows}x${s.z.cols}]  best ${Math.min(...ts).toFixed(2)} s  mean ${(ts.reduce((a, b) => a + b) / ts.length).toFixed(2)} s`);
}
