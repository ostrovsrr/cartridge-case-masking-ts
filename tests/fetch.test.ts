import { createHash } from "node:crypto";
import { existsSync, mkdtempSync, readdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { zipSync } from "fflate";
import { afterAll, describe, expect, it, vi } from "vitest";
import { extract, fetchScan, get, loadManifest, main, MANIFEST, ROOT, verify, type Scan } from "../scripts/fetch-nbtrd.js";

const tmp = mkdtempSync(join(tmpdir(), "fetch-test-"));
afterAll(() => rmSync(tmp, { recursive: true, force: true }));

const sha = (b: Uint8Array) => createHash("sha256").update(b).digest("hex");
const payload = new Uint8Array(4096).map((_, i) => (i * 31 + 7) & 255);
const scan: Scan = {
  path: "glock/test.x3p",
  url: "https://tsapps.nist.gov/NRBTD/Studies/CartridgeMeasurement/DownloadMeasurement/00000000-0000-0000-0000-000000000000",
  size: payload.length,
  sha256: sha(payload),
};

function capture() {
  const out: string[] = [];
  const err: string[] = [];
  return { out, err, deps: { stdout: (l: string) => out.push(l), stderr: (l: string) => err.push(l) } };
}

function response(body: Uint8Array, status = 200): Response {
  return new Response(body, { status, statusText: status === 200 ? "OK" : "Service Unavailable" });
}

describe("manifests", () => {
  it("load and validate both pinned manifests", () => {
    const base = loadManifest(MANIFEST);
    expect(base).toHaveLength(34);
    const sets = new Set(base.map((s) => s.path.split("/")[0]));
    expect([...sets].sort()).toEqual(["glock", "matching", "raw"]);
    expect(loadManifest(join(ROOT, "docs/expanded-scan-manifest.csv"))).toHaveLength(118);
  });

  it("rejects malformed, duplicate or off-site rows", () => {
    const head = "file,study_url,download_url,bytes,sha256\n";
    const good = `data/raw/a.x3p,x,${scan.url},10,${"a".repeat(64)}\n`;
    const bad = [
      `data/other/a.x3p,x,${scan.url},10,${"a".repeat(64)}\n`,
      `data/raw/b.x3p,x,https://example.com/x,10,${"a".repeat(64)}\n`,
      `data/raw/b.x3p,x,${scan.url},0,${"a".repeat(64)}\n`,
      `data/raw/b.x3p,x,${scan.url},10,${"A".repeat(64)}\n`,
      `data/raw/a b.x3p,x,${scan.url},10,${"a".repeat(64)}\n`,
      good,
    ];
    for (const [i, row] of bad.entries()) {
      const p = join(tmp, `m${i}.csv`);
      writeFileSync(p, head + good + (i === bad.length - 1 ? good : row));
      expect(() => loadManifest(p)).toThrow(/invalid or duplicate manifest entry/);
    }
    const empty = join(tmp, "empty.csv");
    writeFileSync(empty, "﻿" + head);
    expect(() => loadManifest(empty)).toThrow(/manifest is empty/);
    const ok = join(tmp, "ok.csv");
    writeFileSync(ok, "﻿" + head.replace(/\n/g, "\r\n") + good.replace(/\n/g, "\r\n"));
    expect(loadManifest(ok)).toEqual([{ path: "raw/a.x3p", url: scan.url, size: 10, sha256: "a".repeat(64) }]);
  });
});

describe("download (mocked, no network)", () => {
  it("retries twice with 2 s / 4 s back-off and sends the User-Agent", async () => {
    const fetchMock = vi
      .fn<typeof fetch>()
      .mockRejectedValueOnce(new Error("ECONNRESET"))
      .mockResolvedValueOnce(response(new Uint8Array(), 503))
      .mockResolvedValueOnce(response(payload));
    const sleep = vi.fn(async (_ms: number) => {});
    const { err, deps } = capture();
    const data = await get(scan.url, { ...deps, fetch: fetchMock, sleep });
    expect(Buffer.compare(Buffer.from(data), Buffer.from(payload))).toBe(0);
    expect(fetchMock).toHaveBeenCalledTimes(3);
    expect(sleep.mock.calls.map((c) => c[0])).toEqual([2000, 4000]);
    const init = fetchMock.mock.calls[0]![1]!;
    expect((init.headers as Record<string, string>)["User-Agent"]).toBe("cartmask-forensic-research/2.0");
    expect(err).toHaveLength(2);
    expect(err[0]).toMatch(/^retry https:\/\/tsapps/);
  });

  it("gives up after three failures", async () => {
    const fetchMock = vi.fn<typeof fetch>().mockRejectedValue(new Error("down"));
    const { deps } = capture();
    await expect(get(scan.url, { ...deps, fetch: fetchMock, sleep: async () => {} })).rejects.toThrow("down");
    expect(fetchMock).toHaveBeenCalledTimes(3);
  });

  it("extracts the pinned X3P from a ZIP bundle and rejects mismatches", () => {
    const bundle = zipSync({ "readme.txt": new Uint8Array([1, 2]), "x/OTHER.X3P": new Uint8Array(4096), "x/scan.X3P": payload });
    expect(Buffer.from(extract(bundle, scan)).equals(Buffer.from(payload))).toBe(true);
    expect(extract(payload, scan)).toBe(payload);
    expect(() => extract(zipSync({ "a.x3p": new Uint8Array(4096) }), scan)).toThrow(/no X3P matching/);
  });

  it("fetchScan writes atomically and verify accepts the result", async () => {
    const target = join(tmp, "dl", "glock", "test.x3p");
    const bundle = zipSync({ "scan.x3p": payload });
    await fetchScan(scan, target, { fetch: async () => response(bundle), sleep: async () => {} });
    expect(Buffer.from(readFileSync(target)).equals(Buffer.from(payload))).toBe(true);
    expect(readdirSync(join(tmp, "dl", "glock"))).toEqual(["test.x3p"]);
    await expect(verify(scan, target)).resolves.toBeUndefined();
    await expect(verify({ ...scan, sha256: "0".repeat(64) }, target)).rejects.toThrow(/SHA-256 does not match/);
    await expect(verify({ ...scan, size: 1 }, target)).rejects.toThrow(/byte count/);
  });

  it("main counts a bad download as a failure and leaves no staging files", async () => {
    const out = join(tmp, "bad");
    const fetchMock = vi.fn<typeof fetch>().mockImplementation(async () => response(new Uint8Array([1, 2, 3])));
    const { out: lines, err, deps } = capture();
    const code = await main(["--set", "glock", "--out", out], { ...deps, fetch: fetchMock, sleep: async () => {} });
    expect(code).toBe(1);
    expect(fetchMock).toHaveBeenCalledTimes(3); // one request per scan, sequential
    expect(lines.at(-1)).toBe("0/3 scans verified");
    expect(err).toHaveLength(3);
    expect(existsSync(out) ? readdirSync(out) : []).toEqual([]);
  });
});

describe("CLI", () => {
  it("--dry-run lists pinned URLs without requests", async () => {
    const fetchMock = vi.fn<typeof fetch>();
    const { out, deps } = capture();
    expect(await main(["--set", "glock", "--dry-run", "--data-root", tmp], { ...deps, fetch: fetchMock })).toBe(0);
    expect(out).toHaveLength(3);
    expect(out[0]).toMatch(/glock.*\.x3p <- https:\/\/tsapps\.nist\.gov\//);
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it("rejects bad flag combinations like argparse (exit 2)", async () => {
    const { err, deps } = capture();
    expect(await main(["--set", "all", "--out", tmp], deps)).toBe(2);
    expect(await main(["--dry-run", "--verify-only"], deps)).toBe(2);
    expect(await main(["--set", "nope"], deps)).toBe(2);
    expect(err.join("\n")).toMatch(/--out requires a single set/);
  });

  it("--verify-only reports missing files without making requests", async () => {
    const fetchMock = vi.fn<typeof fetch>();
    const { out, err, deps } = capture();
    const code = await main(["--set", "glock", "--verify-only", "--data-root", join(tmp, "none")], { ...deps, fetch: fetchMock });
    expect(code).toBe(1);
    expect(out.at(-1)).toBe("0/3 scans verified");
    expect(err[0]).toMatch(/No such file or directory/);
    expect(fetchMock).not.toHaveBeenCalled();
  });
});

const DATA = process.env.CARTMASK_DATA;
describe.skipIf(!DATA)("--verify-only against real data (CARTMASK_DATA)", () => {
  it("all 34 baseline scans", async () => {
    const fetchMock = vi.fn<typeof fetch>();
    const { out, err, deps } = capture();
    const code = await main(["--set", "all", "--verify-only", "--data-root", DATA!], { ...deps, fetch: fetchMock });
    expect(err).toEqual([]);
    expect(code).toBe(0);
    expect(out.at(-1)).toBe("34/34 scans verified");
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it.skipIf(!existsSync(join(DATA ?? "", "expanded")))("the 118-scan expanded cohort", async () => {
    const { out, err, deps } = capture();
    const code = await main(["--set", "expanded", "--verify-only", "--data-root", DATA!], { ...deps, fetch: vi.fn() });
    expect(err).toEqual([]);
    expect(code).toBe(0);
    expect(out.at(-1)).toBe("118/118 scans verified");
  });

  it("--out points one set at a folder", async () => {
    const { out, deps } = capture();
    expect(await main(["--set", "glock", "--verify-only", "--out", join(DATA!, "glock")], deps)).toBe(0);
    expect(out.at(-1)).toBe("3/3 scans verified");
  });
});
