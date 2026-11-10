// Live mode: the WebGPU lattice-Boltzmann solver running in the browser, with weather, heat and new-campus controls.
import * as THREE from 'three';
import { fmt, compass, clamp } from './util.js';
import { halfTex, setHalf, X0, X1, Z0, Z1, SIZE } from './scene.js';
import { bil } from './fields.js';

const $ = s => document.querySelector(s);
const HALL_W = 380, HALL_D = 130, HALL_GAP = 30;   // the archived scenario campus: 3 halls, 14.8 ha of roof

export class Live {
  constructor(app) {
    this.app = app; this.solver = null; this.running = true; this.placing = false; this.campuses = []; this.nextId = 1;
    this.dx = 30; this.flux = 400; this.mw = 120; this.preset = '20240825'; this.mixing = 'conv';
    const m = app.M.landsat.find(e => e.id === this.preset).met;
    this.met = { u10: m.U10, dirFrom: m.dir, wstar: m.wstar, blh: m.blh };
    this.k = 8; this.inflight = null; this.reading = null; this.lastRead = 0; this.phase = 'idle'; this.stats = null; this.speed = 60; this.simBudget = 0;
  }

  // ---------------------------------------------------------------- enter / leave
  async enter() {
    const { Q, F } = this.app;
    this.app.Timeline.hide(); Q.showSite(null); Q.U.uDcTint.value = 0.35; Q.U.uDcGlow.value = 0; Q.U.uSel.value = -9; Q.setPhoto(false);
    Q.setView(Q.views.live, 1600);
    this.render();
    this.labels();
    this.showCampusMeshes(true);
    if (this.solver) { this.attach(); return; }
    F.showRun(this.app.base, { ground: 'none', vol: false });
    F.particles.setOn(true);
    this.app.setLegend({ title: 'Warming of the air near the ground', sub: 'live solver, lowest grid cell (0–' + this.dx + ' m)' });
    await this.start();
  }
  leave() { this.cancelPlacing(); this.showCampusMeshes(false); this.app.Q.U.uClear.value.forEach(v => v.set(1e9, 1e9, -1e9, -1e9)); this.app.Q.renderer.shadowMap.needsUpdate = true; }

  async start() {
    this.setStatus('Starting the WebGPU solver…');
    let mod;
    try { mod = await import('./live/solver.js'); }
    catch (e) { this.fail('The live solver could not be loaded (' + e.message + ').'); return; }
    if (!navigator.gpu) { this.fail('This browser has no WebGPU, so the live solver cannot run here. Recent Chrome or Edge on a desktop (or Safari 26) will work. The story and explore modes use the archived 10 m runs and work everywhere.'); return; }
    try {
      this.phase = 'init';
      const solver = await mod.createLiveSolver({ base: 'data/live/', dx: this.dx, u10Max: 8, onProgress: (t, f) => this.setStatus(`${t} ${f !== undefined ? Math.round(f * 100) + '%' : ''}`) });
      if (this.solver) this.solver.destroy();
      this.solver = solver;
    } catch (e) { console.error(e); this.fail(e.message || String(e)); return; }
    const I = this.solver.info;
    this.surfTex = halfTex(new Float32Array(I.nx * I.ny), I.nx, I.ny);
    const vol = new THREE.Data3DTexture(new Uint8Array(I.nzv * I.ny * I.nx), I.nx, I.ny, I.nzv);
    vol.format = THREE.RedFormat; vol.type = THREE.UnsignedByteType; vol.minFilter = vol.magFilter = THREE.LinearFilter; vol.unpackAlignment = 1; vol.needsUpdate = true;
    this.volTex = vol;
    this.box = { x0: X0, x1: X1, z0: Z0, z1: Z1, y0: I.zbase - I.baseElev, y1: I.zbase + I.nzv * I.dx - I.baseElev };
    this.applyMet(); this.solver.setExistingFlux(0); this.solver.setNewCampuses([]);
    this.phase = 'spinup'; this.spinS = Math.max(700, 2600 * I.dx / 30 * 0.35 + 600); this.k = 4;
    try { this.mlups = await this.solver.benchmark(120); } catch (e) { this.mlups = null; }
    this.attach(); this.render();
  }
  attach() {
    const { Q, F } = this.app;
    const U = Q.U; U.uField.value = this.surfTex; U.uFieldOn.value = 1; U.uFieldMode.value = 0; U.uFieldScale.value = 1; U.uFieldMax.value = 2.4; U.uSwipeOn.value = 0;
    Q.setVolume(this.volTex, null, 0, 3.0, this.box);
    F.showLines(null);
    F.current = { kind: 'live', sample: (x, z) => this.sample(x, z) };
    this.app.setLegend({ title: 'Warming of the air near the ground', sub: `live solver · lowest ${this.dx} m of air · °C above no data-center heat` });
  }
  fail(msg) { this.phase = 'failed'; this.failMsg = msg; this.render(); }

  // ---------------------------------------------------------------- simulation loop
  tick(dt, now) {
    if (this.app.mode !== 'live' || !this.solver || this.phase === 'failed') return;
    const S = this.solver;
    if (this.running && this.phase === 'run' && this.speed) this.simBudget = Math.min(this.simBudget + dt * this.speed, this.speed * 0.25);
    if (this.running && !this.inflight) {
      let k = this.k; const t0 = performance.now();
      if (this.phase === 'run' && this.speed) { const dts = S.info.dt; k = Math.min(k, Math.floor(this.simBudget / dts)); if (k < 1) return; this.simBudget -= k * dts; }
      this.inflight = S.step(k).then(() => {
        const ms = performance.now() - t0; this.inflight = null;
        const per = ms / k; if (k >= this.k * 0.5) this.k = clamp(Math.round(14 / Math.max(per, 0.02)), 1, 400);
        if (this.phase === 'spinup' && S.simTime >= this.spinS) this.switchOn();
      }).catch(e => { this.inflight = null; this.fail('The GPU stopped responding: ' + (e.message || e)); });
    }
    if (!this.reading && now - this.lastRead > 110) {
      this.lastRead = now;
      this.reading = S.read({ vol: true, wind: true, volMax: 3.0 }).then(r => { this.reading = null; this.onRead(r); }).catch(e => { this.reading = null; console.warn(e); });
    }
  }
  switchOn() {
    this.phase = 'run'; this.solver.setExistingFlux(this.flux); this.pushCampuses(); this.solver.resetHeat(); this.lastSimT = undefined;
    this.app.toast('Heat switched on: rooftops now release ' + this.flux + ' W/m²');
    this.render();
  }
  onRead(r) {
    if (!r || this.app.mode !== 'live') return;
    const I = this.solver.info, S = this.solver;
    // light exponential smoothing in simulated time (tau 40 s): the solver returns instantaneous, LES-like fields
    const dts = this.lastSimT === undefined ? -1 : r.simTime - this.lastSimT; this.lastSimT = r.simTime;
    const fresh = !this.sSurf || this.sSurf.length !== r.surf.length || dts < 0 || (r.vol && (!this.sVol || this.sVol.length !== r.vol.length));
    const a = fresh ? 1 : 1 - Math.exp(-dts / 40);
    if (fresh) { this.sSurf = Float32Array.from(r.surf); this.sVol = r.vol ? Float32Array.from(r.vol) : null; }
    else { const s = this.sSurf, x = r.surf; for (let i = 0; i < s.length; i++) s[i] += a * (x[i] - s[i]); if (r.vol) { const v = this.sVol, y = r.vol; for (let i = 0; i < v.length; i++) v[i] += a * (y[i] - v[i]); } }
    r = { ...r, surf: this.sSurf };
    this.last = r;
    setHalf(this.surfTex, this.sSurf);
    if (this.sVol) { const d = this.volTex.image.data, v = this.sVol; for (let i = 0; i < v.length; i++) d[i] = v[i] + 0.5; this.volTex.needsUpdate = true; }
    if ((S.flowReinits || 0) > (this._reinits ?? 0)) { if (this.phase === 'run' && this._reinits !== undefined) this.app.toast('Sharp change of wind: the flow restarts from the new profile and re-develops over a few minutes of model time', 4200); }
    this._reinits = S.flowReinits || 0;
    if (r.u) this.app.F.particles.setField(r.u, r.v, I.nx, I.ny);
    // readouts
    let mx = 0, a05 = 0; for (const v of r.surf) { if (v > mx) mx = v; if (v > 0.5) a05++; }
    let homes = 0; const H = this.app.Q.homes;
    for (const [x, z] of H) { if (bil(r.surf, I.nx, I.ny, (x - X0) / SIZE, (z - Z0) / SIZE) > 0.25) homes++; }
    this.stats = { peak: mx, a05: a05 * I.dx * I.dx / 1e4, homes };
    this.updateReadouts();
  }
  sample(x, z) {
    if (!this.last) return { theta: 0 };
    const I = this.solver.info, fx = (x - X0) / SIZE, fz = (z - Z0) / SIZE, r = this.last;
    const out = { theta: bil(r.surf, I.nx, I.ny, fx, fz) };
    if (r.u) { const u = bil(r.u, I.nx, I.ny, fx, fz), v = bil(r.v, I.nx, I.ny, fx, fz); out.wind = { u, v, speed: Math.hypot(u, v), from: (Math.atan2(-u, -v) * 180 / Math.PI + 360) % 360 }; }
    if (r.vol) {
      const col = clamp(Math.floor(fx * I.nx), 0, I.nx - 1), row = clamp(Math.floor(fz * I.ny), 0, I.ny - 1);
      const g = this.app.Q.heightAt(x, z) + I.baseElev, prof = [];
      for (let k = 0; k < I.nzv; k++) { const zc = I.zbase + (k + 0.5) * I.dx; if (zc < g) continue; prof.push({ h: zc - g, th: r.vol[(k * I.ny + row) * I.nx + col] / 255 * r.volMax }); }
      out.profile = prof;
    }
    return out;
  }

  // ---------------------------------------------------------------- controls
  applyMet() {
    const m = { ...this.met }; if (this.mixing === 'neutral') { m.wstar = 0; m.blh = 800; }
    this.solver && this.solver.setMet(m);
  }
  setPreset(id) {
    this.preset = id; const m = this.app.M.landsat.find(e => e.id === id).met;
    this.met = { u10: m.U10, dirFrom: m.dir, wstar: m.wstar, blh: m.blh }; this.applyMet(); this.render();
  }
  pushCampuses() {
    if (!this.solver) return;
    const list = this.campuses.map(c => ({ id: c.id, halls: c.halls, mw: c.mw }));
    if (this.phase === 'run') this.solver.setNewCampuses(list); else this.solver.setNewCampuses([]);
  }
  startPlacing() {
    if (this.campuses.length >= 3) { this.app.toast('Up to three new campuses at a time'); return; }
    this.placing = true; document.body.classList.add('placing'); this.app.toast('Click on the model to place the campus', 3200); this.render();
  }
  cancelPlacing() { if (!this.placing) return; this.placing = false; document.body.classList.remove('placing'); this.render(); }
  placeAt(cx, cy) {
    const Q = this.app.Q, p = Q.pickGround(cx, cy); if (!p) return;
    const x = clamp(p.x, X0 + HALL_W / 2 + 40, X1 - HALL_W / 2 - 40), z = clamp(p.z, Z0 + 260, Z1 - 260);
    const tot = 3 * HALL_D + 2 * HALL_GAP, halls = [];
    for (let k = 0; k < 3; k++) { const z0 = z - tot / 2 + k * (HALL_D + HALL_GAP); halls.push([x - HALL_W / 2, z0, x + HALL_W / 2, z0 + HALL_D]); }
    const c = { id: this.nextId++, x, z, halls, mw: this.mw };
    this.campuses.push(c); this.buildCampusMesh(c);
    this.cancelPlacing(); this.pushCampuses(); this.labels(); this.render();
  }
  removeCampus(id) {
    const i = this.campuses.findIndex(c => c.id === id); if (i < 0) return;
    const c = this.campuses[i]; if (c.mesh) { this.app.Q.world.remove(c.mesh); c.mesh.geometry.dispose(); }
    this.campuses.splice(i, 1); this.updateClear(); this.pushCampuses(); this.labels(); this.render();
  }
  buildCampusMesh(c) {
    const Q = this.app.Q, G = Q._bucket();
    for (const [a, b, cc, d] of c.halls) {
      let y0 = 1e9; for (const [xx, zz] of [[a, b], [cc, b], [a, d], [cc, d], [(a + cc) / 2, (b + d) / 2]]) y0 = Math.min(y0, Q.heightAt(xx, zz));
      Q._extrude(G, [[cc, d, cc, b, a, b, a, d]], y0 - 1.5, y0 + 14, 3, -7, 0.5);
    }
    const m = Q._bucketMesh(G, Q.bldMat); m.castShadow = m.receiveShadow = true; Q.world.add(m); c.mesh = m;
    this.updateClear();
  }
  updateClear() {
    const cl = this.app.Q.U.uClear.value; cl.forEach(v => v.set(1e9, 1e9, -1e9, -1e9));
    this.campuses.slice(0, 3).forEach((c, i) => { const h = c.halls; cl[i].set(h[0][0] - 12, h[0][1] - 12, h[2][2] + 12, h[2][3] + 12); });
    this.app.Q.renderer.shadowMap.needsUpdate = true;
  }
  showCampusMeshes(on) { for (const c of this.campuses) if (c.mesh) c.mesh.visible = on; if (on) this.updateClear(); }
  labels() {
    const M = this.app.M, list = [];
    for (const l of M.labels) if (l.prio <= 1) list.push({ id: 'p_' + l.text, text: l.text, kind: l.kind, x: l.x, y: l.kind === 'direction' ? 0 : l.y, z: l.kind === 'direction' ? Z1 - 40 : l.z, prio: l.prio + 2 });
    for (const c of this.campuses) list.push({ id: 'lc' + c.id, kind: 'site', cls: 'active', html: `<em>${c.id}</em>New campus<small>${c.mw} MW</small>`, x: c.x, y: this.app.Q.heightAt(c.x, c.z) + 16, z: c.z, prio: -1, stem: 44 });
    this.app.L.set(list);
  }
  async setGrid(dx) {
    if (dx === this.dx && this.solver) return;
    this.dx = dx; const old = this.solver; this.solver = null; this.phase = 'init'; this.last = null;
    if (old) { await (this.inflight || Promise.resolve()).catch(() => {}); old.destroy(); }
    this.render(); await this.start();
  }
  resetHeat() { if (!this.solver) return; if (this.phase === 'run') { this.solver.resetHeat(); this.lastSimT = undefined; } this.render(); }

  // ---------------------------------------------------------------- panel
  setStatus(t) { this.statusText = t; const el = $('#liveStatus'); if (el) el.innerHTML = this.statusHTML(); }
  statusHTML() {
    if (this.phase === 'failed') return `<span class="warn">${this.failMsg}</span>`;
    if (!this.solver) return this.statusText || 'Starting…';
    const I = this.solver.info;
    const cells = `${I.nx} × ${I.ny} × ${I.nz}`;
    const sp = this.phase === 'spinup' ? `<br><b>Spinning up the wind</b> before switching the heat on… ${Math.min(99, Math.round(this.solver.simTime / this.spinS * 100))}%` : '';
    return `<b>WebGPU</b> · ${I.adapter || 'GPU'} · ${I.dx} m grid, ${cells} cells${this.mlups ? ` · ${fmt(this.mlups, 0)} M updates/s` : ''}${sp}`;
  }
  render() {
    if (this.app.mode !== 'live') return;
    const rail = $('#rail'); rail.classList.remove('fade'); void rail.offsetWidth; rail.classList.add('fade');
    const M = this.app.M, presets = M.landsat.filter(e => e.met);
    const m = this.met;
    rail.innerHTML = `
      <div class="kicker"><span class="chap">Live solver · same physics, coarser grid</span></div>
      <div class="status" id="liveStatus">${this.statusHTML()}</div>
      <div class="section-h">Weather<small>observed at Dulles at the Landsat overpass</small></div>
      <div class="seg full" id="presets">${presets.map(e => `<button data-p="${e.id}" aria-pressed="${e.id === this.preset}">${this.app.dateShort(e.date)}</button>`).join('')}</div>
      <div class="windrow" style="margin-top:12px">
        <div class="dial" id="dial">${dialSVG(m.dirFrom)}</div>
        <div>
          <div class="ctrl-label" style="margin-top:0">Wind from <b id="dirRead">${compass(m.dirFrom)} · ${Math.round(m.dirFrom)}°</b></div>
          <div class="ctrl-label">Speed at 10 m <b id="spdRead">${fmt(m.u10, 1)} m/s</b></div>
          <input type="range" id="spd" min="1" max="8" step="0.1" value="${m.u10}">
          <div class="seg full" style="margin-top:10px">${[['conv', 'Sunny'], ['neutral', 'Overcast']].map(([k, l]) => `<button data-mix="${k}" aria-pressed="${this.mixing === k}" title="${k === 'conv' ? 'convective afternoon mixing' : 'no convection'}">${l}</button>`).join('')}</div>
        </div>
      </div>
      <div class="section-h">Heat</div>
      <div class="ctrl-label" style="margin-top:4px">Existing data-center roofs <b id="fluxRead">${this.flux} W/m²</b></div>
      <div class="slider"><input type="range" id="flux" min="0" max="800" step="10" value="${this.flux}"><span class="mark" style="left:${383 / 800 * 100}%">Landsat bound</span></div>
      <div class="row" style="margin-top:26px;justify-content:space-between">
        <button class="btn ${this.placing ? 'primary' : ''}" id="place">${this.placing ? 'Click on the model…' : '+ Place a new campus'}</button>
        <span class="small">three halls, 14.8 ha of roof</span>
      </div>
      <div class="ctrl-label">Heat of a new campus <b id="mwRead">${this.mw} MW</b></div>
      <input type="range" id="mw" min="40" max="300" step="10" value="${this.mw}" aria-label="New campus heat (MW)">
      ${this.campuses.length ? `<ul class="campuslist">${this.campuses.map(c => `<li><span>New campus ${c.id} · ${c.mw} MW</span><button data-rm="${c.id}" aria-label="Remove">×</button></li>`).join('')}</ul>` : ''}
      <div class="section-h">Run</div>
      <div class="row" style="justify-content:space-between">
        <div class="row" style="gap:6px"><button class="btn" id="run">${this.running ? '❚❚ Pause' : '▶ Run'}</button><button class="btn quiet" id="reset">Reset heat</button></div>
        <span class="clock" id="clock">${this.clockText()}</span>
      </div>
      <div class="row" style="margin-top:10px;justify-content:space-between"><span class="small">Speed</span><div class="seg">${[[60, '60×'], [240, '240×'], [0, 'As fast as possible']].map(([v, l]) => `<button data-speed="${v}" aria-pressed="${this.speed === v}">${l}</button>`).join('')}</div></div>
      <div class="row" style="margin-top:8px;justify-content:space-between"><span class="small">Grid spacing</span><div class="seg">${[40, 30, 20].map(d => `<button data-dx="${d}" aria-pressed="${this.dx === d}">${d} m</button>`).join('')}</div></div>
      <div class="readouts" id="readouts">${this.readoutsHTML()}</div>
      <p class="small" style="margin:12px 0 0">Live numbers come from a coarse grid and are for exploring. The calibrated 10 m results are in Story and Explore.</p>`;
    const q = s => rail.querySelector(s);
    rail.querySelectorAll('[data-p]').forEach(b => b.onclick = () => this.setPreset(b.dataset.p));
    rail.querySelectorAll('[data-mix]').forEach(b => b.onclick = () => { this.mixing = b.dataset.mix; this.applyMet(); this.render(); });
    rail.querySelectorAll('[data-dx]').forEach(b => b.onclick = () => this.setGrid(+b.dataset.dx));
    rail.querySelectorAll('[data-speed]').forEach(b => b.onclick = () => { this.speed = +b.dataset.speed; this.simBudget = 0; rail.querySelectorAll('[data-speed]').forEach(x => x.setAttribute('aria-pressed', String(x === b))); });
    rail.querySelectorAll('[data-rm]').forEach(b => b.onclick = () => this.removeCampus(+b.dataset.rm));
    const spd = q('#spd'); const pr = el => el.style.setProperty('--p', ((el.value - el.min) / (el.max - el.min) * 100) + '%');
    pr(spd); spd.oninput = () => { this.met.u10 = +spd.value; q('#spdRead').textContent = fmt(this.met.u10, 1) + ' m/s'; pr(spd); this.applyMet(); };
    const fl = q('#flux'); pr(fl); fl.oninput = () => { this.flux = +fl.value; q('#fluxRead').textContent = this.flux + ' W/m²'; pr(fl); if (this.solver && this.phase === 'run') this.solver.setExistingFlux(this.flux); };
    const mw = q('#mw'); pr(mw); mw.oninput = () => { this.mw = +mw.value; pr(mw); q('#mwRead').textContent = this.mw + ' MW'; };
    q('#place').onclick = () => this.placing ? this.cancelPlacing() : this.startPlacing();
    q('#run').onclick = () => { this.running = !this.running; this.render(); };
    q('#reset').onclick = () => this.resetHeat();
    wireDial(q('#dial'), deg => { this.met.dirFrom = deg; q('#dirRead').textContent = `${compass(deg)} · ${Math.round(deg)}°`; this.applyMet(); });
    clearInterval(this._clk); this._clk = setInterval(() => { if (this.app.mode !== 'live') return; const c = $('#clock'); if (c) c.textContent = this.clockText(); const s = $('#liveStatus'); if (s && this.phase === 'spinup') s.innerHTML = this.statusHTML(); }, 250);
  }
  clockText() {
    if (!this.solver) return '';
    if (this.phase === 'spinup') return 'no heat yet';
    const t = this.solver.simTime, m = Math.floor(t / 60), s = Math.floor(t % 60);
    return `${m} min ${String(s).padStart(2, '0')} s after switch-on`;
  }
  readoutsHTML() {
    const s = this.stats || { peak: 0, a05: 0, homes: 0 };
    return `<div><b>+${fmt(s.peak, 2)}</b><span>°C peak, near the ground</span></div><div><b>${fmt(s.a05, 0)}</b><span>ha warmed by more than ½ °C</span></div><div><b>${fmt(s.homes, 0)}</b><span>homes in air warmed by more than ¼ °C</span></div>`;
  }
  updateReadouts() { const el = $('#readouts'); if (el) el.innerHTML = this.readoutsHTML(); }
}

// ---------------------------------------------------------------------- compass dial
function dialSVG(from) {
  let ticks = '';
  for (let a = 0; a < 360; a += 15) { const r0 = a % 90 === 0 ? 41 : (a % 45 === 0 ? 44 : 46), r1 = 49; const t = a * Math.PI / 180; ticks += `<line x1="${56 + r0 * Math.sin(t)}" y1="${56 - r0 * Math.cos(t)}" x2="${56 + r1 * Math.sin(t)}" y2="${56 - r1 * Math.cos(t)}" stroke="rgba(27,31,39,${a % 90 ? .25 : .55})" stroke-width="1"/>`; }
  const lab = [['N', 0], ['E', 90], ['S', 180], ['W', 270]].map(([s, a]) => { const t = a * Math.PI / 180; return `<text x="${56 + 32 * Math.sin(t)}" y="${56 - 32 * Math.cos(t) + 3}" text-anchor="middle">${s}</text>`; }).join('');
  return `<svg viewBox="0 0 112 112"><circle cx="56" cy="56" r="49" fill="#fff" stroke="rgba(27,31,39,.18)"/>${ticks}${lab}
    <g class="arrow" style="transform-origin:56px 56px;transform:rotate(${from}deg)">
      <line x1="56" y1="12" x2="56" y2="86" stroke="#012169" stroke-width="1.6"/>
      <path d="M50 80 L56 94 L62 80 Z" fill="#012169"/>
      <circle cx="56" cy="10" r="6.5" fill="#fff" stroke="#012169" stroke-width="1.6"/>
    </g></svg>`;
}
function wireDial(el, onChange) {
  const set = e => {
    const r = el.getBoundingClientRect(); const x = e.clientX - (r.left + r.width / 2), y = e.clientY - (r.top + r.height / 2);
    let deg = (Math.atan2(x, -y) * 180 / Math.PI + 360) % 360; deg = Math.round(deg / 5) * 5 % 360;
    el.querySelector('.arrow').style.transform = `rotate(${deg}deg)`; onChange(deg);
  };
  el.addEventListener('pointerdown', e => { el.classList.add('drag'); el.setPointerCapture(e.pointerId); set(e); });
  el.addEventListener('pointermove', e => { if (el.classList.contains('drag')) set(e); });
  el.addEventListener('pointerup', () => el.classList.remove('drag'));
}
