// Small shared helpers: data loading (gzip'd typed arrays), colour ramps, easing, formatting.
import * as THREE from 'three';

export const clamp = (x, a = 0, b = 1) => Math.min(b, Math.max(a, x));
export const lerp = (a, b, t) => a + (b - a) * t;
export const smooth = (a, b, x) => { const t = clamp((x - a) / (b - a)); return t * t * (3 - 2 * t); };
export const easeInOut = t => t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;
export const easeOut = t => 1 - Math.pow(1 - t, 3);

const DT = { uint8: Uint8Array, int8: Int8Array, uint16: Uint16Array, int16: Int16Array, uint32: Uint32Array, int32: Int32Array, float32: Float32Array };

/** fetch a file as ArrayBuffer; transparently gunzips (gzip magic 1f 8b) whether or not the server already decoded it */
export async function fetchBuffer(url, onBytes) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${url}: HTTP ${res.status}`);
  let buf = await res.arrayBuffer();
  if (onBytes) onBytes(buf.byteLength);
  const u = new Uint8Array(buf, 0, Math.min(2, buf.byteLength));
  if (u[0] === 0x1f && u[1] === 0x8b) {
    if (typeof DecompressionStream === 'undefined') throw new Error('This browser cannot decompress data (DecompressionStream missing).');
    const ds = new Response(new Blob([buf]).stream().pipeThrough(new DecompressionStream('gzip')));
    buf = await ds.arrayBuffer();
  }
  return buf;
}

/** load a manifest-described array: returns { data: TypedArray, meta } */
export async function loadArray(manifest, name, base = 'data/') {
  const meta = manifest.files[name];
  if (!meta) throw new Error('not in manifest: ' + name);
  const buf = await fetchBuffer(base + name);
  const T = DT[meta.dtype] || Uint8Array;
  return { data: new T(buf), meta };
}

export const fetchJSON = async url => { const r = await fetch(url); if (!r.ok) throw new Error(`${url}: HTTP ${r.status}`); return r.json(); };

export function loadImage(url) {
  return new Promise((res, rej) => { const im = new Image(); im.decoding = 'async'; im.onload = () => res(im); im.onerror = () => rej(new Error('image ' + url)); im.src = url; });
}

// ------------------------------------------------------------------ colour
export const hex = h => new THREE.Color(h);
const srgbToLin = c => c <= 0.04045 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4);

/** colour stops [[t, '#rrggbb', alpha], ...] -> 256 x 1 RGBA DataTexture (linear colour, straight alpha) */
export function rampTexture(stops, n = 256) {
  const data = new Uint8Array(n * 4);
  const cols = stops.map(([t, h, a]) => { const c = new THREE.Color(h); return [t, c.r, c.g, c.b, a ?? 1]; });   // THREE.Color hex -> linear when colour management is on
  for (let i = 0; i < n; i++) {
    const t = i / (n - 1);
    let k = 0; while (k < cols.length - 2 && t > cols[k + 1][0]) k++;
    const a = cols[k], b = cols[k + 1]; const u = clamp((t - a[0]) / Math.max(1e-6, b[0] - a[0]));
    const s = u * u * (3 - 2 * u) * 0.35 + u * 0.65;
    for (let j = 0; j < 4; j++) data[i * 4 + j] = Math.round(clamp(lerp(a[j + 1], b[j + 1], s)) * 255);
  }
  const tex = new THREE.DataTexture(data, n, 1, THREE.RGBAFormat);
  tex.minFilter = tex.magFilter = THREE.LinearFilter; tex.wrapS = tex.wrapT = THREE.ClampToEdgeWrapping; tex.needsUpdate = true;
  tex.userData.stops = stops;
  return tex;
}

// Atlas palette ramps (warming: paper -> pale gold -> gold -> terracotta -> deep red)
export const RAMP_WARM = [[0.0, '#FBF3DC', 0.0], [0.03, '#FBEFC8', 0.0], [0.065, '#FAE6AE', 0.32], [0.104, '#F7D687', 0.55], [0.208, '#F1B12C', 0.78], [0.417, '#DE7E2C', 0.9], [0.62, '#B5432C', 0.95], [1.0, '#6E1E14', 1.0]];
// surface temperature (cool steel -> paper -> gold -> terracotta -> deep red)
export const RAMP_LST = [[0.0, '#3E5F86', 1], [0.22, '#7F9DBE', 1], [0.42, '#E9E4D6', 1], [0.6, '#F2C25A', 1], [0.78, '#D9752B', 1], [0.9, '#B5432C', 1], [1.0, '#6E1E14', 1]];

/** CSS gradient for a legend bar; `paper` blends translucent stops over the paper colour so the bar matches the map */
export function rampCSS(stops, paper = '#F3F1EA') {
  const P = new THREE.Color(paper);
  return 'linear-gradient(90deg,' + stops.map(([t, h, a]) => {
    const c = new THREE.Color(h).lerp(P, 1 - (a ?? 1));
    return `#${c.getHexString()} ${(t * 100).toFixed(1)}%`;
  }).join(',') + ')';
}

// ------------------------------------------------------------------ formatting
export const fmt = (x, d = 0) => (x === null || x === undefined || !isFinite(x)) ? '–' : Number(x).toLocaleString('en-US', { minimumFractionDigits: d, maximumFractionDigits: d });
export const compass = deg => ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE', 'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW'][Math.round(((deg % 360) + 360) % 360 / 22.5) % 16];
export const signed = (x, d = 2) => (x >= 0 ? '+' : '−') + Math.abs(x).toFixed(d);

/** tiny deterministic PRNG */
export function rng(seed = 1) { let s = seed >>> 0; return () => { s = (s + 0x6D2B79F5) >>> 0; let t = s; t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; }; }

/** simple tween manager driven by the render loop */
export class Tweens {
  constructor() { this.list = []; }
  add(dur, fn, done, ease = easeInOut) { const tw = { t0: performance.now(), dur, fn, done, ease, alive: true }; this.list.push(tw); fn(0); return tw; }
  cancelAll() { this.list.forEach(t => t.alive = false); this.list = []; }
  tick(now) {
    if (!this.list.length) return false;
    for (const tw of this.list) {
      if (!tw.alive) continue;
      const u = clamp((now - tw.t0) / tw.dur);
      tw.fn(tw.ease(u));
      if (u >= 1) { tw.alive = false; tw.done && tw.done(); }
    }
    this.list = this.list.filter(t => t.alive);
    return true;
  }
}
