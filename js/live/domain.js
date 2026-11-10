// Live-solver domain: loads the packed 10 m static fields (data/live/static10.*) and builds the coarse solver
// geometry at dx = 20/30/40 m following lbm3d.build_domain of the prototype.
//
// Conventions: every 2-D array returned here is north-up, [row*nx + col], row 0 = north edge (scene z = -2995),
// col 0 = west edge (scene x = -2995). The solver flips rows (j = 0 at the south edge) when it uploads them.

export const SCENE_X0 = -2995, SCENE_Z0 = -2995, AOI_SIZE = 6000, BASE_ELEV = 83.53166;
export const KAPPA = 0.4;
export const Z0_BY_CLASS = { 10: 1.0, 20: 0.3, 30: 0.03, 40: 0.05, 50: 0.5, 60: 0.01, 80: 0.0002, 90: 0.05 };   // = lbm3d
export const Z0_DEFAULT = 0.1;

// ------------------------------------------------------------------ static10 loader
async function gunzipIfNeeded(buf) {
  const u = new Uint8Array(buf, 0, Math.min(2, buf.byteLength));
  if (!(u[0] === 0x1f && u[1] === 0x8b)) return buf;       // the server (or browser) already decoded it
  if (typeof DecompressionStream === 'undefined') throw new Error('This browser cannot decompress data (DecompressionStream missing).');
  const ds = new Response(new Blob([buf]).stream().pipeThrough(new DecompressionStream('gzip')));
  return ds.arrayBuffer();
}

async function fetchOk(url, kind) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`live solver: cannot load ${url} (HTTP ${r.status})`);
  return kind === 'json' ? r.json() : r.arrayBuffer();
}

const cache = new Map();

/** load + decode static10 (cached per base URL). Returns integer-exact planes:
 *  { nrows, ncols, demCm: Int32Array (dem in cm), lc: Uint8Array, frac25: Uint8Array (frac*25), hmeanCm: Uint16Array,
 *    dca4: Uint8Array (dcarea/4 m^2), meta } */
export function loadStatic10(base = 'data/live/') {
  if (!base.endsWith('/')) base += '/';
  if (!cache.has(base)) {
    const p = (async () => {
      const meta = await fetchOk(base + 'static10.json', 'json');
      const buf = await gunzipIfNeeded(await fetchOk(base + (meta.file || 'static10.bin.gz'), 'bin'));
      if (buf.byteLength !== meta.raw_bytes) throw new Error(`live solver: static10 has ${buf.byteLength} bytes, expected ${meta.raw_bytes}`);
      const ny = meta.grid.nrows, nx = meta.grid.ncols, n = nx * ny, P = meta.planes;
      const bytes = new Uint8Array(buf);
      // dem: row-wise first differences of cm values, int16 split in a low-byte and a high-byte plane
      const lo = bytes.subarray(P.dem.offset, P.dem.offset + n), hi = bytes.subarray(P.dem.offset + n, P.dem.offset + 2 * n);
      const demCm = new Int32Array(n);
      for (let r = 0; r < ny; r++) {
        let acc = 0;
        for (let c = 0; c < nx; c++) {
          const k = r * nx + c;
          let d = lo[k] | (hi[k] << 8);
          if (d >= 32768) d -= 65536;
          acc += d; demCm[k] = acc;
        }
      }
      const st = {
        nrows: ny, ncols: nx, meta, demCm,
        lc: bytes.slice(P.lc.offset, P.lc.offset + n),
        frac25: bytes.slice(P.frac25.offset, P.frac25.offset + n),
        hmeanCm: new Uint16Array(buf.slice(P.hmean.offset, P.hmean.offset + 2 * n)),
        dca4: bytes.slice(P.dca4.offset, P.dca4.offset + n),
      };
      const sum = a => { let s = 0; for (let i = 0; i < a.length; i++) s += a[i]; return s; };
      const ck = meta.checksums, got = { dem_sum_cm: sum(demCm), lc_sum: sum(st.lc), frac25_sum: sum(st.frac25), hmean_sum_cm: sum(st.hmeanCm), dca4_sum: sum(st.dca4) };
      for (const k of Object.keys(got)) if (got[k] !== ck[k]) throw new Error(`live solver: static10 checksum mismatch (${k})`);
      return st;
    })();
    cache.set(base, p);
    p.catch(() => cache.delete(base));
  }
  return cache.get(base);
}

// ------------------------------------------------------------------ helpers
/** numpy-compatible rounding (half to even) */
export function roundHalfEven(x) {
  const r = Math.round(x);
  return Math.abs(x - Math.trunc(x)) === 0.5 ? 2 * Math.round(x / 2) : r;
}

/** north-up [row][col] -> solver orientation [j][i] with j = 0 at the south edge (and back: the flip is its own inverse) */
export function flipRows(a, nx, ny) {
  const o = new a.constructor(a.length);
  for (let r = 0; r < ny; r++) o.set(a.subarray(r * nx, (r + 1) * nx), (ny - 1 - r) * nx);
  return o;
}

// ------------------------------------------------------------------ block aggregation (10 m -> dx)
/** dem: block mean; frac: block mean; hmean: area-weighted mean over covered 2 m sub-pixels; dcarea: block sum;
 *  roughness: drag coefficient cd = (kappa / ln(0.5 dx / z0))^2 of every 10 m cell (WorldCover z0 table, evaluated at
 *  the coarse first-cell centre 0.5 dx) averaged over the block (= the stress-conserving average). */
export function aggregate(st, dx) {
  const r = Math.round(dx / 10);
  if (Math.abs(r * 10 - dx) > 1e-9 || st.ncols % r || st.nrows % r) throw new Error(`live solver: dx must be a multiple of 10 m dividing 6000 m (got ${dx})`);
  const nx = st.ncols / r, ny = st.nrows / r, n = nx * ny, NC = st.ncols;
  const cdTab = new Float64Array(256);
  for (let c = 0; c < 256; c++) {
    const z0 = Z0_BY_CLASS[c] ?? Z0_DEFAULT;
    cdTab[c] = (KAPPA / Math.log((0.5 * dx) / z0)) ** 2;
  }
  const dem = new Float64Array(n), cnt = new Int32Array(n), frac = new Float64Array(n), hmean = new Float64Array(n),
    dcarea = new Float64Array(n), cd = new Float64Array(n);
  for (let R = 0; R < ny; R++) for (let C = 0; C < nx; C++) {
    let sd = 0, sc = 0, sh = 0, sa = 0, scd = 0;
    for (let a = 0; a < r; a++) {
      const base = (R * r + a) * NC + C * r;
      for (let b = 0; b < r; b++) {
        const k = base + b, v = st.frac25[k];
        sd += st.demCm[k]; sc += v; sh += v * st.hmeanCm[k]; sa += st.dca4[k]; scd += cdTab[st.lc[k]];
      }
    }
    const o = R * nx + C;
    dem[o] = sd / (100 * r * r);
    cnt[o] = sc;
    frac[o] = sc / (25 * r * r);
    hmean[o] = sc > 0 ? sh / sc / 100 : 0;
    dcarea[o] = 4 * sa;
    cd[o] = scd / (r * r);
  }
  return { nx, ny, dx, r, dem, cnt, frac, hmean, dcarea, cd };
}

// ------------------------------------------------------------------ new campuses
/** halls: [x0, z0, x1, z1] axis-aligned rectangles in scene metres. Exact area overlap with the coarse cells.
 *  Returns north-up per-cell roof area (m^2, clipped to the cell area) and heat (W): each campus spreads mw over its
 *  total hall area (W/m^2 = mw*1e6 / sum of hall areas); hall parts outside the 6 x 6 km box are dropped. */
export function rasterizeCampuses(list, dx, nx, ny) {
  const n = nx * ny, newArea = new Float64Array(n), heatW = new Float64Array(n), info = [];
  const X1 = SCENE_X0 + nx * dx, Z1 = SCENE_Z0 + ny * dx, cellA = dx * dx;
  for (const cp of list || []) {
    const per = new Map();
    let total = 0;
    for (const h of cp.halls || []) {
      const xa = Math.min(h[0], h[2]), xb = Math.max(h[0], h[2]), za = Math.min(h[1], h[3]), zb = Math.max(h[1], h[3]);
      if (!(xb > xa && zb > za)) continue;
      total += (xb - xa) * (zb - za);
      const x0 = Math.max(xa, SCENE_X0), x1 = Math.min(xb, X1), z0 = Math.max(za, SCENE_Z0), z1 = Math.min(zb, Z1);
      if (x1 <= x0 || z1 <= z0) continue;
      const c0 = Math.max(0, Math.floor((x0 - SCENE_X0) / dx)), c1 = Math.min(nx - 1, Math.floor((x1 - SCENE_X0) / dx));
      const r0 = Math.max(0, Math.floor((z0 - SCENE_Z0) / dx)), r1 = Math.min(ny - 1, Math.floor((z1 - SCENE_Z0) / dx));
      for (let r = r0; r <= r1; r++) {
        const oz = Math.min(z1, SCENE_Z0 + (r + 1) * dx) - Math.max(z0, SCENE_Z0 + r * dx);
        if (oz <= 0) continue;
        for (let c = c0; c <= c1; c++) {
          const ox = Math.min(x1, SCENE_X0 + (c + 1) * dx) - Math.max(x0, SCENE_X0 + c * dx);
          if (ox <= 0) continue;
          const k = r * nx + c;
          per.set(k, (per.get(k) || 0) + ox * oz);
        }
      }
    }
    const mw = Math.max(0, +cp.mw || 0);
    const flux = total > 0 ? mw * 1e6 / total : 0;
    let inDom = 0;
    for (const [k, a0] of per) {
      const a = Math.min(a0, cellA);
      newArea[k] += a; heatW[k] += a * flux; inDom += a;
    }
    info.push({ id: cp.id, mw, roofArea_m2: total, inDomain_m2: inDom, flux_wm2: flux, heatMW: inDom * flux / 1e6, cells: per.size });
  }
  for (let k = 0; k < n; k++) if (newArea[k] > cellA) newArea[k] = cellA;
  return { newArea, heatW, info };
}

// ------------------------------------------------------------------ build_domain
/** lbm3d.build_domain on the aggregated grid. opts: { heightM = 720, newArea = null (north-up m^2 of new roof per
 *  cell), extraHeight = 14, nzMin = 0 }. Returns north-up Int32Array kt (terrain-solid cells per column), nb (building
 *  cells), ktop = kt + nb (first fluid level), Float32Array cd, plus nz, zbase (m NAVD88 of level 0 bottom). */
export function buildDomain(agg, opts = {}) {
  const { heightM = 720, newArea = null, extraHeight = 14.0, nzMin = 0 } = opts;
  const { nx, ny, dx } = agg, n = nx * ny;
  let dmin = Infinity;
  for (let k = 0; k < n; k++) if (agg.dem[k] < dmin) dmin = agg.dem[k];
  const zbase = Math.floor(dmin) - dx;
  const kt = new Int32Array(n), nb = new Int32Array(n), ktop = new Int32Array(n), cd = new Float32Array(n);
  let ktopMax = 0, nbCells = 0;
  for (let k = 0; k < n; k++) {
    let frac = agg.frac[k], hmean = agg.hmean[k];
    if (newArea && newArea[k] > 0) {
      frac = Math.max(frac, newArea[k] / (dx * dx));
      hmean = Math.max(hmean, extraHeight);
    }
    kt[k] = Math.max(1, Math.ceil((agg.dem[k] - zbase) / dx - 0.5));
    nb[k] = frac >= 0.5 ? roundHalfEven(hmean / dx) : 0;
    ktop[k] = kt[k] + nb[k];
    if (ktop[k] > ktopMax) ktopMax = ktop[k];
    nbCells += nb[k];
    cd[k] = agg.cd[k];
  }
  let nz = Math.max(Math.round(heightM / dx), nzMin);
  if (ktopMax >= nz - 8) nz = ktopMax + 9;              // build_domain asserts ktop.max() < nz - 8: adapt instead
  return { nx, ny, nz, dx, zbase, kt, nb, ktop, cd, ktopMax, nbCells, dcarea: agg.dcarea, dem: agg.dem };
}
