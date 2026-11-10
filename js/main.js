// Ashburn Heat Twin — application shell: boot, render loop, modes (story / explore / live), legend, map key, probe.
import * as THREE from 'three';
import { Maquette, X0, X1, Z0, Z1 } from './scene.js';
import { Fields } from './fields.js';
import { Labels, escapeHtml } from './labels.js';
import { fetchJSON, fmt, compass, clamp, rampCSS, RAMP_WARM, RAMP_LST, signed } from './util.js';
import { Story } from './story.js';
import { Live } from './live.js';
import { aboutHTML } from './about.js';

const $ = s => document.querySelector(s);
const app = window.app = { mode: null };

async function boot() {
  const msg = $('.lmsg'), bar = $('.lbar i');
  const progress = (t, f) => { msg.textContent = { terrain: 'Loading terrain and buildings…', model: 'Building the model…', imagery: 'Draping the imagery…', runs: 'Loading the solver runs…' }[t] || t; bar.style.width = Math.round(f * 100) + '%'; };
  try {
    const gl = document.createElement('canvas').getContext('webgl2');
    if (!gl) throw new Error('This page needs WebGL 2, which this browser does not provide.');
    progress('terrain', 0.03);
    const M = app.M = await fetchJSON('data/manifest.json');
    const Q = app.Q = new Maquette($('#gl'), M);
    Q.panelShift = window.innerWidth > 900 ? 205 : 0;
    await Q.init(progress);
    progress('runs', 0.86);
    const F = app.F = new Fields(Q, M);
    app.L = new Labels($('#labels'), Q);
    app.base = await F.loadRun('baseline');
    bar.style.width = '100%';
    app.story = new Story(app); app.live = new Live(app);
    wire();
    const want = (location.hash || '').replace('#', '');
    setMode(['explore', 'live'].includes(want) ? want : 'story');
    requestAnimationFrame(loop);
    setTimeout(() => $('#loader').classList.add('done'), 250);
  } catch (e) {
    console.error(e); msg.textContent = e.message || String(e); msg.style.color = '#B5432C'; bar.style.background = '#B5432C';
  }
}

// ------------------------------------------------------------------ render loop
let last = performance.now(), fpsAcc = 0, fpsN = 0;
function loop(now) {
  const dt = (now - last) / 1000; last = now;
  if (!app.paused) frame(now, dt);
  requestAnimationFrame(loop);
}
/** one update + render; also called directly (with a scripted clock) by the film recorder */
function frame(now, dt) {
  const Q = app.Q;
  Q.tweens.tick(now);
  if (Q.controls.enabled) Q.controls.update();
  app.F.particles.step(dt);
  app.story && app.story.tick(dt, now);
  app.live && app.live.tick(dt, now);
  Q.render(now);
  app.L.update();
  updateMapKey();
  positionProbe();
  // adaptive quality: keep the frame rate up on modest GPUs
  if (!app.paused) { fpsAcc += dt; fpsN++; if (fpsAcc > 2.5) { const fps = fpsN / fpsAcc; fpsAcc = 0; fpsN = 0; adaptQuality(fps); } }
}
app.frame = frame;
function adaptQuality(fps) {
  const q = app.Q.quality;
  if (fps < 32 && q.scale > 0.6) { q.scale = Math.max(0.6, q.scale - 0.15); app.Q.CU.uSteps.value = Math.max(48, app.Q.CU.uSteps.value - 16); app.Q.resize(); }
}

// ------------------------------------------------------------------ modes
export function setMode(mode) {
  if (mode === 'about') { openAbout(); return; }
  if (app.mode === mode) return;
  const prev = app.mode; app.mode = mode;
  document.body.classList.remove('mode-story', 'mode-explore', 'mode-live'); document.body.classList.add('mode-' + mode);
  document.querySelectorAll('.modes button[data-mode]').forEach(b => b.getAttribute('data-mode') !== 'about' && b.setAttribute('aria-pressed', String(b.dataset.mode === mode)));
  closeProbe(); hideTip();
  if (prev === 'live') app.live.leave();
  if (prev === 'story') app.story.leave();
  if (prev === 'explore') Explore.leave();
  history.replaceState(null, '', mode === 'story' ? location.pathname : '#' + mode);
  if (mode === 'story') app.story.enter();
  if (mode === 'explore') Explore.enter();
  if (mode === 'live') app.live.enter();
}
app.setMode = setMode;

function wire() {
  document.querySelectorAll('.modes button[data-mode]').forEach(b => b.addEventListener('click', () => setMode(b.dataset.mode)));
  window.addEventListener('resize', () => { app.Q.panelShift = window.innerWidth > 900 ? 205 : 0; app.Q.resize(); });
  const cv = $('#gl');
  let down = null;
  cv.addEventListener('pointerdown', e => { down = { x: e.clientX, y: e.clientY, t: performance.now() }; });
  cv.addEventListener('pointerup', e => {
    if (!down) return; const moved = Math.hypot(e.clientX - down.x, e.clientY - down.y); const quick = performance.now() - down.t < 600; down = null;
    if (moved < 5 && quick && e.button === 0) onClick(e);
  });
  cv.addEventListener('pointermove', e => { if (e.buttons) { hideTip(); return; } onHover(e); });
  cv.addEventListener('pointerleave', hideTip);
  window.addEventListener('keydown', e => {
    if (e.target.closest('input, select, textarea')) return;
    if (e.key === 'Escape') { closeAbout(); closeProbe(); app.live && app.live.cancelPlacing(); const f = $('#film'); if (!f.hidden) { f.querySelector('video').pause(); f.hidden = true; app.paused = false; } }
    if (app.mode === 'story') app.story.key(e);
  });
}

// ------------------------------------------------------------------ click / hover
function onClick(e) {
  if (app.mode === 'live' && app.live.placing) { app.live.placeAt(e.clientX, e.clientY); return; }
  if (app.mode === 'story' && app.story.onClick && app.story.onClick(e)) return;
  if (app.mode === 'explore') {
    const b = app.Q.pickBuilding(e.clientX, e.clientY);
    if (b && b.kind > 1.5 && b.kind < 2.5 && b.campus >= 0) { Explore.selectCampus(b.campus); return; }
  }
  const p = app.Q.pickGround(e.clientX, e.clientY);
  if (p) openProbe(p); else closeProbe();
}
let hoverT = 0;
function onHover(e) {
  if (app.mode !== 'explore') return;
  const now = performance.now(); if (now - hoverT < 50) return; hoverT = now;
  const b = app.Q.pickBuilding(e.clientX, e.clientY);
  const U = app.Q.U;
  if (b && b.kind > 1.5 && b.kind < 2.5 && b.campus >= 0) {
    const c = app.M.campuses.find(c => c.id === b.campus);
    U.uHover.value = b.campus;
    if (c) showTip(e.clientX, e.clientY, `<b>Data-center campus ${c.id}</b><br>${fmt(c.roof_m2 / 1e4, 1)} ha of roof · <em>${fmt(c.mw, 0)} MW</em> at 400 W/m²<br><span style="opacity:.75">Click to see the warming it causes alone</span>`);
    $('#gl').style.cursor = 'pointer';
  } else { U.uHover.value = -9; hideTip(); $('#gl').style.cursor = ''; }
}
function showTip(x, y, html) { const t = $('#tip'); t.innerHTML = html; t.hidden = false; t.style.left = x + 'px'; t.style.top = y + 'px'; }
function hideTip() { $('#tip').hidden = true; if (app.Q) app.Q.U.uHover.value = -9; }
app.toast = (text, ms = 2600) => { const t = $('#toast'); t.textContent = text; t.hidden = false; clearTimeout(t._h); t._h = setTimeout(() => t.hidden = true, ms); };

// ------------------------------------------------------------------ probe
let probeAt = null;
const LC = { 10: 'tree cover', 20: 'shrubland', 30: 'grassland', 40: 'cropland', 50: 'built-up', 60: 'bare ground', 80: 'open water', 90: 'wetland' };
function openProbe(p) {
  probeAt = p;
  const F = app.F, s = F.sample(p.x, p.z) || {};
  const el = $('#probe'); let html = '';
  const lc = LC[app.Q.landcoverAt(p.x, p.z)] || '—';
  const near = nearestCampus(p.x, p.z);
  if (s.lst !== undefined) {
    html = `<h5>Surface temperature <button aria-label="Close">×</button></h5>
      <div class="big">${s.lst === null ? '—' : fmt(s.lst, 1)}<small>°C · Landsat</small></div>
      <dl><dt>Model (calibrated)</dt><dd>${s.lstSim === null || s.lstSim === undefined ? '—' : fmt(s.lstSim, 1) + ' °C'}</dd><dt>Land cover</dt><dd>${lc}</dd><dt>Nearest data center</dt><dd>${fmt(near.d, 0)} m</dd></dl>`;
  } else if (s.tracer !== undefined) {
    html = `<h5>From campus ${app.F.current.id} alone <button aria-label="Close">×</button></h5>
      <div class="big">${signed(s.tracer, 2)}<small>°C</small></div><dl><dt>Land cover</dt><dd>${lc}</dd><dt>Distance to that campus</dt><dd>${fmt(distCampus(p, app.F.current.id), 0)} m</dd></dl>`;
  } else {
    const th = s.theta ?? 0;
    html = `<h5>Air near the ground<button aria-label="Close">×</button></h5><div class="small" style="margin:-4px 0 6px">warming from data-center heat, 0–10 m</div>
      <div class="big">${signed(th, 2)}<small>°C warmer</small></div>
      <dl>${s.dt !== null && s.dt !== undefined ? `<dt>From the new campus</dt><dd>${signed(s.dt, 2)} °C</dd>` : ''}
      ${s.wind ? `<dt>Wind (near ground)</dt><dd>${fmt(s.wind.speed, 1)} m/s from ${compass(s.wind.from)}</dd>` : ''}
      <dt>Land cover</dt><dd>${lc}</dd><dt>Nearest data center</dt><dd>${fmt(near.d, 0)} m</dd><dt>Ground height</dt><dd>${fmt(p.y + app.M.frame.base_elev, 0)} m</dd></dl>`;
    const prof = F.profile(p.x, p.z) || (s.profile || null);
    if (prof && prof.length) html += profileSVG(prof);
  }
  el.innerHTML = html; el.hidden = false;
  el.querySelector('button').onclick = closeProbe;
  app.L.set(app.L.items.filter(i => i.id !== 'probe').map(stripEl).concat([{ id: 'probe', kind: 'probe', text: '', x: p.x, y: p.y, z: p.z, prio: -1, avoid: false, stem: 0 }]));
}
const stripEl = it => { const { el, w, h, _text, alpha, ...rest } = it; return rest; };
function closeProbe() { probeAt = null; $('#probe').hidden = true; if (app.L) app.L.set(app.L.items.filter(i => i.id !== 'probe').map(stripEl)); }
app.closeProbe = closeProbe;
app.refreshProbe = () => { if (probeAt) openProbe(probeAt); };
function positionProbe() {
  if (!probeAt) return; const el = $('#probe'); const s = app.Q.project(probeAt.x, probeAt.y, probeAt.z);
  const W = app.Q.width, H = app.Q.height; let x = s.x + 18, y = s.y - 40;
  const w = el.offsetWidth, h = el.offsetHeight;
  if (x + w > W - 300) x = s.x - w - 18; if (y + h > H - 20) y = H - 20 - h; if (y < 90) y = 90;
  el.style.left = x + 'px'; el.style.top = y + 'px';
}
function profileSVG(prof) {
  const W = 224, H = 92, maxH = 300, maxT = Math.max(0.3, ...prof.filter(p => p.h < maxH).map(p => p.th)) * 1.1;
  const pts = prof.filter(p => p.h <= maxH).map(p => [(p.th / maxT) * (W - 30) + 26, H - 12 - (p.h / maxH) * (H - 20)]);
  const d = pts.map((p, i) => (i ? 'L' : 'M') + p[0].toFixed(1) + ' ' + p[1].toFixed(1)).join(' ');
  return `<svg viewBox="0 0 ${W} ${H}"><line x1="26" y1="${H - 12}" x2="${W}" y2="${H - 12}" stroke="rgba(27,31,39,.25)"/><line x1="26" y1="4" x2="26" y2="${H - 12}" stroke="rgba(27,31,39,.25)"/>
    <text x="22" y="${H - 9}" text-anchor="end">0</text><text x="22" y="10" text-anchor="end">${maxH}</text><text x="2" y="${H / 2}" >m</text>
    <text x="${W}" y="${H}" text-anchor="end">${maxT.toFixed(1)} °C</text><path d="${d}" fill="none" stroke="#B5432C" stroke-width="1.6"/>
    <text x="30" y="12">warming with height</text></svg>`;
}
function nearestCampus(x, z) {
  let best = { d: 1e9, c: null };
  for (const c of app.M.campuses) { const bb = c.bbox; const dx = Math.max(bb[0] - x, 0, x - bb[2]), dz = Math.max(bb[1] - z, 0, z - bb[3]); const d = Math.hypot(dx, dz); if (d < best.d) best = { d, c }; }
  return best;
}
function distCampus(p, id) { const c = app.M.campuses.find(c => c.id === id); if (!c) return NaN; const bb = c.bbox; return Math.hypot(Math.max(bb[0] - p.x, 0, p.x - bb[2]), Math.max(bb[1] - p.z, 0, p.z - bb[3])); }

// ------------------------------------------------------------------ legend + map key
export function setLegend(spec) {
  const el = $('#legend');
  if (!spec) { el.hidden = true; document.body.classList.remove('has-legend'); return; }
  let html = `<h4>${spec.title}${spec.sub ? `<small>${spec.sub}</small>` : ''}</h4>`;
  if (spec.kind === 'lst') {
    const [lo, hi] = spec.range; html += `<div class="ramp" style="background:${rampCSS(RAMP_LST)}"></div><div class="ticks">`;
    const step = (hi - lo) > 14 ? 5 : 2;
    for (let v = Math.ceil(lo / step) * step; v <= hi; v += step) html += `<span style="left:${((v - lo) / (hi - lo) * 100).toFixed(1)}%">${v}°</span>`;
    html += '</div>';
  } else {
    const max = spec.max || 2.4; html += `<div class="ramp" style="background:${rampCSS(RAMP_WARM)}"></div><div class="ticks">`;
    for (const v of (spec.ticks || [0.25, 0.5, 1, 2])) if (v <= max * 0.93) html += `<span style="left:${(v / max * 100).toFixed(1)}%">${v === 0.25 ? '¼' : v === 0.5 ? '½' : v}</span>`;
    html += `<span style="left:100%">${max >= 2.2 ? '' : max} °C</span></div>`;
    if (spec.iso !== false) html += `<div class="iso"><i></i>contours at ¼, ½, 1 and 2 °C</div>`;
  }
  if (spec.src) html += `<div class="src">${spec.src}</div>`;
  el.innerHTML = html; el.hidden = false; document.body.classList.add('has-legend');
  requestAnimationFrame(() => document.body.style.setProperty('--legend-h', el.offsetHeight + 'px'));
}
app.setLegend = setLegend;
const _v = new THREE.Vector3();
function updateMapKey() {
  const Q = app.Q, t = Q.controls.target, cam = Q.camera;
  const a = new THREE.Vector3(t.x, t.y, t.z).project(cam), n = new THREE.Vector3(t.x, t.y, t.z - 500).project(cam);
  const ang = Math.atan2((n.x - a.x) * Q.width, (n.y - a.y) * Q.height);
  $('#mapkey .rot').setAttribute('transform', `rotate(${(ang * 180 / Math.PI).toFixed(1)} 20 20)`);
  // metres per pixel at the target, measured across the screen
  const right = new THREE.Vector3().setFromMatrixColumn(cam.matrixWorld, 0).setY(0).normalize();
  const b = t.clone().addScaledVector(right, 1000).project(cam);
  const ppm = Math.hypot((b.x - a.x) * Q.width / 2, (b.y - a.y) * Q.height / 2) / 1000;
  const opts = [100, 200, 250, 500, 1000, 2000]; let L = 1000; for (const o of opts) if (o * ppm >= 55 && o * ppm <= 130) { L = o; break; }
  const bar = $('#mapkey .bar'); bar.querySelector('i').style.width = Math.round(L * ppm) + 'px'; bar.querySelector('span').textContent = L >= 1000 ? (L / 1000) + ' km' : L + ' m';
}

// ------------------------------------------------------------------ timeline (shared by story + explore)
export const Timeline = {
  el: $('#timeline'), playing: false, t: 0, tmax: 1, onChange: null, speed: 1,
  show(tmax, onChange, { label = 'min after the heat is switched on' } = {}) {
    this.tmax = tmax; this.onChange = onChange; this.el.hidden = false; this.el.querySelector('.readout span').textContent = label;
    const r = this.el.querySelector('input'); r.oninput = () => { this.set(r.value / 1000 * this.tmax, true); this.pause(); };
    this.el.querySelector('.play').onclick = () => this.playing ? this.pause() : this.play();
    this.set(0, true);
  },
  hide() { this.pause(); this.el.hidden = true; this.onChange = null; },
  set(t, fromUser) {
    this.t = clamp(t, 0, this.tmax); const r = this.el.querySelector('input'); const p = this.t / this.tmax;
    if (!fromUser || document.activeElement !== r) r.value = Math.round(p * 1000); r.style.setProperty('--p', (p * 100).toFixed(1) + '%');
    this.el.querySelector('.readout b').textContent = Math.round(this.t / 60);
    this.onChange && this.onChange(this.t);
  },
  play() { if (this.t >= this.tmax - 1) this.set(0); this.playing = true; this.el.classList.add('playing'); },
  pause() { this.playing = false; this.el.classList.remove('playing'); },
  tick(dt) { if (!this.playing) return; const nt = this.t + dt * this.tmax / 16 * this.speed; this.set(nt); if (nt >= this.tmax) this.pause(); }
};
app.Timeline = Timeline;

// ------------------------------------------------------------------ explore mode
const Explore = app.Explore = {
  st: { run: 'baseline', ground: true, vol: true, lines: false, wind: true, dc: true, lst: false, lstDate: '20240825', photo: false, ve: 2.5, campus: null },
  async enter() {
    const Q = app.Q; Timeline.hide();
    Q.setView(Q.views.home, 1400);
    this.render(); await this.apply();
  },
  render() {
    const s = this.st, M = app.M;
    const rail = $('#rail'); rail.classList.remove('fade'); void rail.offsetWidth; rail.classList.add('fade');
    const runBtn = (k, label) => `<button data-run="${k}" aria-pressed="${s.run === k}">${label}</button>`;
    const veBtn = v => `<button data-ve="${v}" aria-pressed="${s.ve === v}">${v}×</button>`;
    rail.innerHTML = `
      <div class="section-h">Archived runs<small>10 m GPU solver</small></div>
      <div class="seg full">${runBtn('baseline', 'Existing')}${runBtn('scenario_A', '+ Site A')}${runBtn('scenario_B', '+ Site B')}${runBtn('scenario_C', '+ Site C')}</div>
      <p class="small" style="margin:8px 0 0">${s.run === 'baseline' ? 'All 54 existing campuses at 400 W/m² of roof. Wind from 210° at 3.1 m/s (Dulles, 25 Aug 2024, 11:52 EDT).' : `Existing campuses plus a hypothetical 120 MW campus at site ${s.run.slice(-1)}: ${escapeHtml(M.sites[s.run.slice(-1)].label)}.`}</p>
      <div class="section-h">Show</div>
      <ul class="layers">
        <li><label><input type="checkbox" data-k="ground" ${s.ground ? 'checked' : ''}>Warming of the air near the ground</label><span class="sw" style="background:${rampCSS(RAMP_WARM)}"></span></li>
        <li><label><input type="checkbox" data-k="vol" ${s.vol ? 'checked' : ''}>Warm air in three dimensions</label></li>
        <li><label><input type="checkbox" data-k="lines" ${s.lines ? 'checked' : ''}>Streamlines of the mean wind</label></li>
        <li><label><input type="checkbox" data-k="wind" ${s.wind ? 'checked' : ''}>Wind near the ground</label></li>
        <li><label><input type="checkbox" data-k="dc" ${s.dc ? 'checked' : ''}>Highlight data centers</label></li>
        <li><label><input type="checkbox" data-k="lst" ${s.lst ? 'checked' : ''}>Landsat surface temperature</label>
          <select data-k="lstDate">${M.landsat.filter(e => e.sim).map(e => `<option value="${e.id}" ${s.lstDate === e.id ? 'selected' : ''}>${dateShort(e.date)}</option>`).join('')}</select></li>
      </ul>
      <div class="section-h">Model</div>
      <div class="row" style="justify-content:space-between"><div class="seg">${['map', 'photo'].map(v => `<button data-base="${v}" aria-pressed="${(v === 'photo') === s.photo}">${v === 'map' ? 'Map' : 'Aerial photo'}</button>`).join('')}</div>
        <div class="row" style="gap:6px"><span class="small">Height</span><div class="seg">${[1, 2.5, 5].map(veBtn).join('')}</div></div></div>
      <div class="hint"><svg viewBox="0 0 16 16"><circle cx="8" cy="8" r="6.5" fill="none" stroke="currentColor"/><path d="M8 4.5v4M8 10.6v.9" stroke="currentColor" stroke-width="1.4" stroke-linecap="round"/></svg>
        <span>Click the ground to read the air there. Hover a data center, then click it to see the warming that campus causes on its own.</span></div>
      <div id="campusCard"></div>`;
    rail.querySelectorAll('[data-run]').forEach(b => b.onclick = () => { s.run = b.dataset.run; s.campus = null; this.render(); this.apply(); });
    rail.querySelectorAll('[data-ve]').forEach(b => b.onclick = () => { s.ve = +b.dataset.ve; app.Q.setVE(s.ve); this.render(); });
    rail.querySelectorAll('[data-base]').forEach(b => b.onclick = () => { s.photo = b.dataset.base === 'photo'; app.Q.setPhoto(s.photo); this.render(); });
    rail.querySelectorAll('input[data-k]').forEach(i => i.onchange = () => { s[i.dataset.k] = i.checked; if (i.dataset.k === 'lst' && i.checked) s.campus = null; this.apply(); });
    rail.querySelector('select[data-k]').onchange = e => { s.lstDate = e.target.value; s.lst = true; this.render(); this.apply(); };
    if (s.campus !== null) this.campusCard(s.campus);
  },
  async apply() {
    const s = this.st, Q = app.Q, F = app.F;
    Q.U.uDcTint.value = s.dc ? 1 : 0; Q.U.uDcGlow.value = 0; Q.U.uSel.value = s.campus ?? -9;
    Q.showSite(s.run === 'baseline' ? null : s.run.slice(-1));
    F.particles.setOn(s.wind);
    const R = await F.loadRun(s.run);
    if (s.lst) {
      const L = await F.loadLandsat(s.lstDate); F.showLandsat(L, { swipe: true, x: 0 });
      F.particles.setField(R.u, R.v, 150, 150);
      setLegend({ kind: 'lst', title: 'Land surface temperature', sub: `${dateLong(L.ent.date)} · Landsat left of the line, model right`, range: [Q.U.uLstRange.value.x, Q.U.uLstRange.value.y], src: 'Landsat 8/9 Collection 2 surface temperature, 30 m pixels' });
      this.swipe(true);
    } else {
      this.swipe(false);
      F.showRun(R, { ground: s.ground ? 'total' : 'none', vol: s.vol && s.campus === null, lines: s.lines });
      if (s.campus !== null) { await F.loadTracers(); F.showTracer(s.campus); }
      if (s.campus !== null) setLegend({ title: `Warming caused by campus ${s.campus} alone`, sub: 'source-tagged tracer in the mean flow, near the ground', max: 2.0 });
      else if (s.ground) setLegend({ title: 'Warming of the air near the ground', sub: '°C above the same day without data-center heat (0–10 m layer)' });
      else setLegend(null);
    }
    this.labels();
    app.refreshProbe();
  },
  labels() {
    const s = this.st, M = app.M, list = [];
    for (const l of M.labels) if (l.prio <= 2) list.push({ id: 'p_' + l.text, text: l.text, kind: l.kind, x: l.x, y: l.kind === 'direction' ? 0 : l.y, z: l.kind === 'direction' ? Z1 - 40 : l.z, prio: l.prio + 2 });
    if (s.campus !== null) { const c = M.campuses.find(c => c.id === s.campus); if (c) list.push({ id: 'campus', kind: 'campus', html: `<em>${fmt(c.mw, 0)} MW</em>&nbsp; campus ${c.id}`, x: c.x, y: c.y + 16, z: c.z, prio: -1, stem: 40 }); }
    if (s.run !== 'baseline') { const k = s.run.slice(-1), st = M.sites[k]; list.push({ id: 'site', kind: 'site', html: `<em>${k}</em>Site ${k}<small>120 MW, hypothetical</small>`, x: st.center[0], y: st.ground + 16, z: st.center[1], prio: 0, cls: 'active', stem: 44 }); }
    app.L.set(list);
  },
  async selectCampus(id) {
    const s = this.st; if (s.lst) { s.lst = false; }
    s.campus = s.campus === id ? null : id; this.render(); await this.apply();
  },
  campusCard(id) {
    const c = app.M.campuses.find(c => c.id === id), el = $('#campusCard'); if (!c || !el) return;
    el.innerHTML = `<hr><div class="section-h">Campus ${c.id}<small><button class="link" id="ccClose">show all campuses</button></small></div>
      <div class="figs" style="margin-top:0"><div class="fig"><b>${fmt(c.mw, 0)}<small>MW</small></b><span>heat at 400 W/m²</span></div><div class="fig"><b>${fmt(c.a025, 0)}<small>ha</small></b><span>air warmed by more than ¼ °C</span></div><div class="fig"><b>${fmt(c.homes025, 0)}</b><span>homes inside that contour</span></div></div>
      <p class="small">${fmt(c.roof_m2 / 1e4, 1)} ha of roof in ${c.n} buildings. Peak warming beyond its fence ${fmt(c.peak, 1)} °C; nearest home ${fmt(c.nearest_home, 0)} m away. Campus numbers are arbitrary; operators are not identified.</p>`;
    el.querySelector('#ccClose').onclick = () => this.selectCampus(id);
  },
  swipe(on) {
    let h = $('#swipeHandle');
    if (!on) { if (h) h.remove(); return; }
    if (!h) {
      h = document.createElement('div'); h.id = 'swipeHandle'; h.className = 'swipe';
      h.innerHTML = '<span class="l">Landsat</span><i></i><span class="r">Model</span>'; document.body.appendChild(h);
      let drag = false;
      h.addEventListener('pointerdown', e => { drag = true; h.setPointerCapture(e.pointerId); app.Q.controls.enabled = false; });
      h.addEventListener('pointermove', e => { if (!drag) return; const Q = app.Q; const y = Q.project(0, 0, Z0).y + 40; const p = Q.pickGround(e.clientX, y) || Q.pickGround(e.clientX, e.clientY); if (p) Q.U.uSwipeX.value = clamp(p.x, X0 + 50, X1 - 50); });
      h.addEventListener('pointerup', () => { drag = false; app.Q.controls.enabled = true; });
      const follow = () => { if (!h.isConnected) return; const Q = app.Q, x = Q.U.uSwipeX.value; const s = Q.project(x, Q.heightAt(x, Z0), Z0); h.style.transform = `translate(${s.x}px, ${s.y - 46}px)`; requestAnimationFrame(follow); };
      follow();
    }
  },
  leave() { this.swipe(false); }
};
export const dateShort = d => { const [y, m, dd] = d.split('-'); return `${+dd} ${['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'][+m - 1]} ${y}`; };
export const dateLong = dateShort;
app.dateShort = dateShort;

// ------------------------------------------------------------------ about
function openAbout() {
  const s = $('#about'); s.innerHTML = aboutHTML(app.M); s.hidden = false;
  const scrim = document.createElement('div'); scrim.className = 'scrim'; scrim.id = 'scrim'; scrim.onclick = closeAbout; document.body.appendChild(scrim);
  s.querySelector('.close').onclick = closeAbout;
  s.querySelectorAll('[data-film]').forEach(b => b.onclick = () => app.openFilm());
}
function closeAbout() { $('#about').hidden = true; const s = $('#scrim'); s && s.remove(); }
app.openFilm = () => {
  closeAbout(); const f = $('#film'), v = f.querySelector('video'); f.hidden = false; app.paused = true;
  const close = () => { v.pause(); f.hidden = true; app.paused = false; };
  f.querySelector('.close').onclick = close; f.onclick = e => { if (e.target === f) close(); };
  v.play().catch(() => {});
};

boot();
