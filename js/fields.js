// Archived solver output (10 m GPU runs) and observations: loading, GPU textures, streamlines, wind particles,
// switch-on animation, Landsat comparison and per-campus attribution. Also serves probes (point samples, profiles).
import * as THREE from 'three';
import { loadArray, clamp, lerp } from './util.js';
import { halfTex, setHalf, X0, X1, Z0, Z1, SIZE } from './scene.js';

const ZB = 50.0;   // m NAVD88, bottom of solver level 0

export class Fields {
  constructor(maq, manifest) {
    this.Q = maq; this.M = manifest; this.runs = {}; this.frames = null; this.landsat = {}; this.tracers = null;
    this.baseElev = manifest.frame.base_elev;
    this.state = { mode: 'none' };
    this.particles = new WindParticles(maq);
    this.volBox = { x0: X0, x1: X1, z0: Z0, z1: Z1, y0: ZB - this.baseElev, y1: ZB + 400 - this.baseElev };
  }

  // ------------------------------------------------------------------ archived runs
  async loadRun(name) {
    if (this.runs[name]) return this.runs[name];
    const M = this.M, base = `runs/${name}/`;
    const [t2m, vol, wind, lines, dt] = await Promise.all([
      loadArray(M, base + 't2m.bin.gz'), loadArray(M, base + 'vol.bin.gz'), loadArray(M, base + 'wind.bin.gz'),
      M.files[base + 'lines.bin.gz'] ? loadArray(M, base + 'lines.bin.gz') : null,
      M.files[base + 'dt2m.bin.gz'] ? loadArray(M, base + 'dt2m.bin.gz') : null]);
    const R = { name };
    R.t2m = new Float32Array(t2m.data.length); for (let i = 0; i < R.t2m.length; i++) R.t2m[i] = t2m.data[i] * 1e-4;
    R.t2mTex = halfTex(R.t2m, 600, 600);
    if (dt) { R.dt = new Float32Array(dt.data.length); for (let i = 0; i < R.dt.length; i++) R.dt[i] = dt.data[i] * 1e-4; R.dtTex = halfTex(R.dt, 600, 600); }
    R.vol = vol.data; R.volScale = vol.meta.scale;
    R.volTex = vol3d(vol.data, 300, 300, 40);
    const w = wind.data; R.u = new Float32Array(150 * 150); R.v = new Float32Array(150 * 150);
    for (let i = 0; i < R.u.length; i++) { R.u[i] = w[i] * 1e-3; R.v[i] = w[R.u.length + i] * 1e-3; }
    if (lines) R.lines = makeStreamlines(lines.data, lines.meta, this.Q);
    this.runs[name] = R;
    return R;
  }

  /** show an archived run: ground field (total or delta), 3-D volume, streamlines, wind */
  showRun(R, { ground = 'total', vol = true, lines = false } = {}) {
    const U = this.Q.U;
    this.state = { mode: 'run', run: R, ground };
    this.current = { kind: 'run', R };
    U.uFieldMode.value = 0; U.uFieldScale.value = 1; U.uSwipeOn.value = 0;
    if (ground === 'none') U.uFieldOn.value = 0;
    else { U.uField.value = ground === 'delta' && R.dtTex ? R.dtTex : R.t2mTex; U.uFieldOn.value = 1; }
    U.uFieldMax.value = ground === 'delta' ? 3.0 : 2.4;
    this.Q.CU.uRampMax.value = 2.4;
    if (vol) this.Q.setVolume(R.volTex, null, 0, R.volScale * 255, this.volBox); else this.Q.setVolume(null);
    this.showLines(lines ? R : null);
    this.particles.setField(R.u, R.v, 150, 150);
  }
  showLines(R) {
    for (const r of Object.values(this.runs)) if (r.lines) r.lines.visible = false;
    if (R && R.lines) { R.lines.visible = true; if (!R.lines.parent) this.Q.world.add(R.lines); }
  }

  // ------------------------------------------------------------------ switch-on animation (baseline)
  async loadFrames() {
    if (this.frames) return this.frames;
    const M = this.M;
    const [f2, f3] = await Promise.all([loadArray(M, 'runs/baseline/frames2d.bin.gz'), loadArray(M, 'runs/baseline/frames3d.bin.gz')]);
    const F = { f2: f2.data, n2: f2.meta.n, dt2: f2.meta.dt_s, s2: f2.meta.scale, f3: f3.data, n3: f3.meta.n, dt3: f3.meta.dt_s, s3: f3.meta.scale };
    F.buf = new Float32Array(300 * 300); F.tex = halfTex(F.buf, 300, 300);
    F.vols = []; const per = 40 * 150 * 150;
    for (let i = 0; i < F.n3; i++) F.vols.push(vol3d(F.f3.subarray(i * per, (i + 1) * per), 150, 150, 40));
    F.tmax = Math.min((F.n2 - 1) * F.dt2, (F.n3 - 1) * F.dt3);
    this.frames = F; return F;
  }
  showFrames(t) {
    const F = this.frames; if (!F) return;
    const U = this.Q.U;
    const a = clamp(t / F.dt2, 0, F.n2 - 1), i0 = Math.floor(a), i1 = Math.min(F.n2 - 1, i0 + 1), u = a - i0, N = 90000, s = F.s2;
    const A = F.f2.subarray(i0 * N, i0 * N + N), B = F.f2.subarray(i1 * N, i1 * N + N), out = F.buf;
    for (let k = 0; k < N; k++) out[k] = (A[k] + (B[k] - A[k]) * u) * s;
    setHalf(F.tex, out);
    U.uField.value = F.tex; U.uFieldOn.value = 1; U.uFieldMode.value = 0; U.uFieldScale.value = 1; U.uFieldMax.value = 2.4; U.uSwipeOn.value = 0;
    const b = clamp(t / F.dt3, 0, F.n3 - 1), j0 = Math.floor(b), j1 = Math.min(F.n3 - 1, j0 + 1);
    this.Q.setVolume(F.vols[j0], F.vols[j1], b - j0, F.s3 * 255, this.volBox);
    this.state = { mode: 'frames', t };
    this.current = { kind: 'frames', t };
  }

  // ------------------------------------------------------------------ Landsat
  async loadLandsat(id) {
    if (this.landsat[id]) return this.landsat[id];
    const M = this.M; const ent = M.landsat.find(e => e.id === id);
    const obs = await loadArray(M, `landsat/lst_${id}.bin.gz`);
    const L = { ent, obs: toFloatNodata(obs.data, 0.01, -32768) };
    L.obsTex = halfTex(L.obs, 200, 200, { nearest: true });
    if (M.files[`landsat/sim_${id}.bin.gz`]) { const sim = await loadArray(M, `landsat/sim_${id}.bin.gz`); L.sim = toFloatNodata(sim.data, 0.01, -32768); L.simTex = halfTex(L.sim, 200, 200, { nearest: true }); }
    this.landsat[id] = L; return L;
  }
  showLandsat(L, { swipe = true, x = 0 } = {}) {
    const U = this.Q.U;
    U.uField.value = L.obsTex; U.uField2.value = L.simTex || L.obsTex; U.uFieldOn.value = 1; U.uFieldMode.value = 1; U.uFieldScale.value = 1; U.uFieldScale2.value = 1;
    U.uSwipeOn.value = swipe && L.simTex ? 1 : 0; U.uSwipeX.value = x;
    const lo = Math.floor(L.ent.p5 - 1.5), hi = Math.ceil(L.ent.p95 + 1.5); U.uLstRange.value.set(lo, hi);
    this.Q.setVolume(null); this.showLines(null);
    this.state = { mode: 'lst', L }; this.current = { kind: 'lst', L };
  }

  // ------------------------------------------------------------------ per-campus attribution
  async loadTracers() {
    if (this.tracers) return this.tracers;
    const t = await loadArray(this.M, 'campus/tracers.bin.gz');
    const order = t.meta.order, vmax = t.meta.vmax, T = { data: t.data, index: {} };
    order.forEach((k, i) => T.index[parseInt(k.slice(1))] = { i, vmax: vmax[i] });
    T.tex = new THREE.DataTexture(new Uint8Array(90000), 300, 300, THREE.RedFormat, THREE.UnsignedByteType);
    T.tex.minFilter = T.tex.magFilter = THREE.LinearFilter; T.tex.flipY = false; T.tex.unpackAlignment = 1; T.tex.needsUpdate = true;
    this.tracers = T; return T;
  }
  showTracer(id) {
    const T = this.tracers; const e = T && T.index[id]; if (!e) return false;
    T.tex.image.data.set(T.data.subarray(e.i * 90000, (e.i + 1) * 90000)); T.tex.needsUpdate = true;
    const U = this.Q.U; U.uField.value = T.tex; U.uFieldScale.value = e.vmax; U.uFieldOn.value = 1; U.uFieldMode.value = 0; U.uFieldMax.value = 2.0; U.uSwipeOn.value = 0;
    this.current = { kind: 'tracer', id, vmax: e.vmax };
    T.cur = id; T.buf = T.data.subarray(e.i * 90000, (e.i + 1) * 90000);
    return true;
  }

  // ------------------------------------------------------------------ probes
  /** near-surface value of whatever is on the ground now */
  sample(x, z) {
    const c = this.current; if (!c) return null;
    const fx = (x - X0) / SIZE, fz = (z - Z0) / SIZE;
    if (c.kind === 'run') return { theta: bil(c.R.t2m, 600, 600, fx, fz), dt: c.R.dt ? bil(c.R.dt, 600, 600, fx, fz) : null, wind: this.windAt(c.R, fx, fz) };
    if (c.kind === 'frames') return { theta: bil(this.frames.buf, 300, 300, fx, fz) };
    if (c.kind === 'lst') { const L = c.L; return { lst: nearest(L.obs, 200, fx, fz), lstSim: L.sim ? nearest(L.sim, 200, fx, fz) : null }; }
    if (c.kind === 'tracer') { const T = this.tracers; return { tracer: bil(T.buf, 300, 300, fx, fz, 1 / 255 * c.vmax) }; }
    if (c.kind === 'live') return c.sample(x, z);
    return null;
  }
  windAt(R, fx, fz) { const u = bil(R.u, 150, 150, fx, fz), v = bil(R.v, 150, 150, fx, fz); return { u, v, speed: Math.hypot(u, v), from: (Math.atan2(-u, -v) * 180 / Math.PI + 360) % 360 }; }
  /** vertical profile of the 3-D field above (x, z): [{h (m above ground), th}] */
  profile(x, z) {
    const c = this.current; if (!c || c.kind !== 'run') return null;
    const R = c.R, col = clamp(Math.floor((x - X0) / 20), 0, 299), row = clamp(Math.floor((z - Z0) / 20), 0, 299);
    const g = this.Q.heightAt(x, z) + this.baseElev, out = [];
    for (let k = 0; k < 40; k++) { const zc = ZB + (k + 0.5) * 10; if (zc < g) continue; out.push({ h: zc - g, th: R.vol[(k * 300 + row) * 300 + col] * R.volScale }); }
    return out;
  }
}

// ---------------------------------------------------------------------- helpers
function vol3d(data, w, h, d) {
  const t = new THREE.Data3DTexture(data, w, h, d); t.format = THREE.RedFormat; t.type = THREE.UnsignedByteType;
  t.minFilter = t.magFilter = THREE.LinearFilter; t.wrapS = t.wrapT = t.wrapR = THREE.ClampToEdgeWrapping; t.unpackAlignment = 1; t.needsUpdate = true;
  return t;
}
export { vol3d };
function toFloatNodata(a, scale, nodata) { const o = new Float32Array(a.length); for (let i = 0; i < a.length; i++) o[i] = a[i] === nodata ? -999 : a[i] * scale; return o; }
function bil(arr, w, h, fx, fz, s = 1) {
  const x = clamp(fx * w - 0.5, 0, w - 1.001), y = clamp(fz * h - 0.5, 0, h - 1.001), j = Math.floor(x), i = Math.floor(y), ax = x - j, ay = y - i;
  return s * (arr[i * w + j] * (1 - ax) * (1 - ay) + arr[i * w + j + 1] * ax * (1 - ay) + arr[(i + 1) * w + j] * (1 - ax) * ay + arr[(i + 1) * w + j + 1] * ax * ay);
}
export { bil };
function nearest(arr, n, fx, fz) { const j = clamp(Math.floor(fx * n), 0, n - 1), i = clamp(Math.floor(fz * n), 0, n - 1); const v = arr[i * n + j]; return v < -100 ? null : v; }

function makeStreamlines(bytes, meta, Q) {
  const nl = meta.n_lines, np = meta.n_pts;
  const buf = bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
  const lens = new Uint16Array(buf, 0, nl), q = new Int16Array(buf, nl * 2, np * 4), tau = new Uint16Array(buf, nl * 2 + np * 8, np);
  const pos = [], warm = [], tt = [], seed = [];
  let o = 0;
  for (let l = 0; l < nl; l++) {
    const m = lens[l], sd = (l * 0.6180339) % 1;
    for (let k = 0; k < m - 1; k++) {
      for (const kk of [k, k + 1]) {
        const p = o + kk; pos.push(q[p * 4] / 4, q[p * 4 + 1] / 20, q[p * 4 + 2] / 4); warm.push(q[p * 4 + 3] / 1e4); tt.push(tau[p] / 10); seed.push(sd);
      }
    }
    o += m;
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3)); g.setAttribute('aWarm', new THREE.Float32BufferAttribute(warm, 1));
  g.setAttribute('aTau', new THREE.Float32BufferAttribute(tt, 1)); g.setAttribute('aSeed', new THREE.Float32BufferAttribute(seed, 1));
  const mat = new THREE.ShaderMaterial({
    transparent: true, depthWrite: false,
    uniforms: { uTime: Q.U.uTime, uRamp: Q.U.uRampW, uInk: Q.U.uInk, uA: { value: 1 } },
    vertexShader: `attribute float aWarm; attribute float aTau; attribute float aSeed; varying float vW; varying float vT; varying float vS;
      void main(){ vW = aWarm; vT = aTau; vS = aSeed; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }`,
    fragmentShader: `uniform float uTime; uniform sampler2D uRamp; uniform vec3 uInk; uniform float uA; varying float vW; varying float vT; varying float vS;
      void main(){
        float ph = fract(vT / 420.0 - uTime * 0.11 + vS);
        float pulse = smoothstep(0.82, 0.99, ph) * (1.0 - smoothstep(0.99, 1.0, ph));
        float w = smoothstep(0.04, 0.9, vW);
        vec3 warmC = texture2D(uRamp, vec2(clamp(0.18 + vW / 2.0, 0.0, 1.0), 0.5)).rgb;
        vec3 c = mix(uInk, warmC * 0.9, w);
        float a = (0.16 + 0.22 * w) + pulse * (0.35 + 0.3 * w);
        gl_FragColor = vec4(pow(c, vec3(1.0)), a * uA);
      }`
  });
  const mesh = new THREE.LineSegments(g, mat); mesh.frustumCulled = false; mesh.visible = false; mesh.renderOrder = 3;
  return mesh;
}

// ---------------------------------------------------------------------- near-surface wind particles (thin ink trails)
export class WindParticles {
  constructor(Q, n = 2600, trail = 16) {
    this.Q = Q; this.n = n; this.trail = trail; this.on = true; this.timeScale = 650;
    this.p = new Float32Array(n * 2); this.age = new Float32Array(n); this.life = new Float32Array(n);
    this.hist = new Float32Array(n * trail * 3);
    const segs = n * (trail - 1);
    this.pos = new Float32Array(segs * 2 * 3); this.alpha = new Float32Array(segs * 2);
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(this.pos, 3).setUsage(THREE.DynamicDrawUsage));
    g.setAttribute('aA', new THREE.BufferAttribute(this.alpha, 1).setUsage(THREE.DynamicDrawUsage));
    const mat = new THREE.ShaderMaterial({ transparent: true, depthWrite: false, uniforms: { uInk: Q.U.uInk, uA: { value: 0.62 } },
      vertexShader: 'attribute float aA; varying float vA; void main(){ vA = aA; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }',
      fragmentShader: 'uniform vec3 uInk; uniform float uA; varying float vA; void main(){ gl_FragColor = vec4(uInk, vA * uA); }' });
    this.mesh = new THREE.LineSegments(g, mat); this.mesh.frustumCulled = false; this.mesh.renderOrder = 4; this.mesh.visible = false;
    Q.world.add(this.mesh);
    for (let i = 0; i < n; i++) this.respawn(i, true);
  }
  setField(u, v, w, h) { this.u = u; this.v = v; this.w = w; this.h = h; this.mesh.visible = this.on && !!u; }
  setOn(on) { this.on = on; this.mesh.visible = on && !!this.u; }
  respawn(i, initial) {
    const x = X0 + Math.random() * SIZE, z = Z0 + Math.random() * SIZE;
    this.p[i * 2] = x; this.p[i * 2 + 1] = z; this.age[i] = initial ? Math.random() * 5 : 0; this.life[i] = 3.5 + Math.random() * 4.5;
    const y = this.Q.heightAt(x, z) + 24;
    for (let k = 0; k < this.trail; k++) this.hist.set([x, y, z], (i * this.trail + k) * 3);
  }
  step(dt) {
    if (!this.mesh.visible || !this.u) return;
    dt = Math.min(dt, 0.05);
    const { u, v, w, h, n, trail } = this, s = dt * this.timeScale, H = this.hist;
    for (let i = 0; i < n; i++) {
      let x = this.p[i * 2], z = this.p[i * 2 + 1];
      const fx = (x - X0) / SIZE, fz = (z - Z0) / SIZE;
      const uu = bil(u, w, h, fx, fz), vv = bil(v, w, h, fx, fz);
      x += uu * s; z -= vv * s;   // v is northward, scene z points south
      this.age[i] += dt;
      if (x < X0 || x > X1 || z < Z0 || z > Z1 || this.age[i] > this.life[i] || (Math.abs(uu) + Math.abs(vv)) < 0.05) { this.respawn(i, false); continue; }
      this.p[i * 2] = x; this.p[i * 2 + 1] = z;
      const base = i * trail * 3; H.copyWithin(base + 3, base, base + (trail - 1) * 3);
      H[base] = x; H[base + 1] = this.Q.heightAt(x, z) + 24; H[base + 2] = z;
    }
    const P = this.pos, A = this.alpha; let o = 0;
    for (let i = 0; i < n; i++) {
      const fade = Math.min(1, this.age[i] / 0.8) * Math.min(1, (this.life[i] - this.age[i]) / 1.2);
      const base = i * trail * 3;
      for (let k = 0; k < trail - 1; k++) {
        P[o * 3] = H[base + k * 3]; P[o * 3 + 1] = H[base + k * 3 + 1]; P[o * 3 + 2] = H[base + k * 3 + 2]; A[o] = fade * (1 - k / (trail - 1)); o++;
        P[o * 3] = H[base + (k + 1) * 3]; P[o * 3 + 1] = H[base + (k + 1) * 3 + 1]; P[o * 3 + 2] = H[base + (k + 1) * 3 + 2]; A[o] = fade * (1 - (k + 1) / (trail - 1)); o++;
      }
    }
    this.mesh.geometry.attributes.position.needsUpdate = true; this.mesh.geometry.attributes.aA.needsUpdate = true;
  }
}
