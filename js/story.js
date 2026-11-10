// Guided story: a cover and six short chapters, each with one idea, one camera move and at most one control.
import { fmt, rampCSS, RAMP_WARM, clamp } from './util.js';
import { X0, X1, Z0, Z1 } from './scene.js';
import { vol3d } from './fields.js';

const $ = s => document.querySelector(s);

export class Story {
  constructor(app) { this.app = app; this.i = 0; this.site = 'B'; this.lstDate = '20240825'; this.deltaVols = {}; }

  get steps() {
    const M = this.app.M, B = M.baseline, cal = M.calibration;
    return [
      { id: 'cover', chap: 'A digital twin of Data Center Alley', cover: true,
        title: 'Where does the heat of Data Center Alley go?',
        body: `Northern Virginia is the world’s largest data-center market, and much of it sits here, in Ashburn. Every watt the servers use ends up as heat, and most of it leaves through the roofs. This model follows that heat through the air with a physics engine checked against satellites.`,
        enter: () => this.sceneCover() },
      { id: 'place', chap: 'The place', title: 'Six by six kilometres of Ashburn, rebuilt from public data',
        body: `Terrain from USGS 3DEP, aerial imagery from NAIP 2023 and ${fmt(M.facts.buildings)} building footprints. <b>${M.facts.n_dc_buildings}</b> of the buildings are data centers, grouped here into ${M.facts.n_campuses} campuses. Their roofs alone cover about ${M.facts.dc_roof_ha} hectares.`,
        figs: [[M.facts.n_dc_buildings, '', 'data-center buildings'], [M.facts.n_campuses, '', 'campuses'], [M.facts.dc_roof_ha, 'ha', 'of data-center roof']],
        enter: () => this.scenePlace() },
      { id: 'heat', chap: 'The heat', title: 'Every campus sheds its heat through the roof',
        body: `Chillers and dry coolers on the roofs carry the servers’ heat into the air. Here each operational roof gives off <b>400 W per square metre</b>, the upper bound that Landsat allows (chapter 4). Summed over the box, that is about ${fmt(B.heat_MW, 0)} MW.`,
        figs: [[400, 'W/m²', 'from each data-center roof'], [fmt(B.heat_MW, 0), 'MW', 'of heat in the 6 × 6 km box']], figsTwo: true,
        enter: () => this.sceneHeat() },
      { id: 'physics', chap: 'The physics', title: 'A lattice-Boltzmann solver follows the warm air',
        body: `Wind observed at Dulles on 25 August 2024, from the south-southwest at 3.1 m/s, flows over the real terrain and buildings. The warm air rises, drifts downwind and mixes into the afternoon boundary layer. <b>Press play</b> to switch the heat on.`,
        figs: [['23', 'M', 'grid cells, 10 m apart'], ['+' + fmt(B.max, 1), '°C', 'peak warming near the ground'], [fmt(B.a05, 0), 'ha', 'warmed by more than ½ °C']],
        ctrl: () => `<ul class="layers" style="margin-top:2px"><li><label><input type="checkbox" id="stSec" checked>Vertical section along the wind</label></li><li><label><input type="checkbox" id="stLines">Streamlines of the mean wind</label></li></ul>`,
        enter: () => this.scenePhysics() },
      { id: 'check', chap: 'The check', title: 'Landsat sees the roofs, but not the plume',
        body: `We fitted the model to Landsat surface temperature on two days with opposite winds and tested it on a third. It reproduces the pattern of hot roofs and cool stream valleys. The warm air drifting downwind is too faint for the satellite to see, which caps the heat at roughly 400 W/m² of roof.`,
        figs: [[fmt(cal.r2_val, 2), '', 'R² on the held-out day'], [fmt(cal.rmse_val, 1), '°C', 'typical error (RMSE)'], [fmt(cal.F_90[1], 0), 'W/m²', 'most rooftop heat the satellite allows']],
        ctrl: () => this.ctrlCheck(),
        enter: () => this.sceneCheck() },
      { id: 'choice', chap: 'The choice', title: 'One 120 MW campus, three sites, three different neighbours',
        body: `We added the same hypothetical campus at three sites and reran the solver. The amount of warm air hardly changes; who receives it does.`,
        ctrl: () => this.ctrlChoice(),
        enter: () => this.sceneChoice() },
      { id: 'live', chap: 'Your turn', title: 'Run the same physics in your browser',
        body: `The live solver runs the same lattice-Boltzmann equations on your graphics card through WebGPU, on a coarser 30 m grid. Change the wind, turn the rooftop heat up or down, and drop a campus anywhere.`,
        ctrl: () => `<div class="row" style="margin-top:4px"><button class="btn primary" id="goLive">Open the live solver <svg viewBox="0 0 14 14"><path d="M2 7h9M8 3.5 11.5 7 8 10.5" fill="none" stroke="currentColor" stroke-width="1.5"/></svg></button><button class="btn quiet" id="goExplore">Explore the archived runs</button></div>
          <p class="small" style="margin-top:12px">Needs a browser with WebGPU: recent Chrome or Edge on desktop, or Safari 26.</p>`,
        enter: () => this.sceneLive() }
    ];
  }

  enter() { this.render(); this.go(this.i, true); }
  leave() { this.app.Timeline.hide(); this.app.F.showLines(null); this.app.Q.section.visible = false; }
  key(e) {
    if (e.key === 'ArrowRight' || e.key === 'PageDown') { e.preventDefault(); this.next(); }
    if (e.key === 'ArrowLeft' || e.key === 'PageUp') { e.preventDefault(); this.prev(); }
    if (e.key === ' ' && this.steps[this.i].id === 'physics') { e.preventDefault(); const T = this.app.Timeline; T.playing ? T.pause() : T.play(); }
  }
  next() { if (this.i < this.steps.length - 1) this.go(this.i + 1); }
  prev() { if (this.i > 0) this.go(this.i - 1); }
  go(i, force) {
    if (i === this.i && !force) return;
    this.i = i; this.app.closeProbe(); this.app.Timeline.hide(); this.app.F.showLines(null); this.app.Q.section.visible = false;
    this.render(); this.steps[i].enter();
  }
  tick(dt) { this.app.Timeline.tick(dt); }

  render() {
    const st = this.steps[this.i], n = this.steps.length - 1, rail = $('#rail');
    rail.classList.remove('fade'); void rail.offsetWidth; rail.classList.add('fade');
    const figs = st.figs ? `<div class="figs ${st.figsTwo ? 'two' : ''}">${st.figs.map(([v, u, l]) => `<div class="fig"><b>${v}${u ? `<small>${u}</small>` : ''}</b><span>${l}</span></div>`).join('')}</div>` : '';
    const kick = st.cover ? `<div class="kicker"><span class="chap">${st.chap}</span></div>` : `<div class="kicker"><span class="n">${this.i} / ${n}</span><span class="chap">${st.chap}</span></div>`;
    const nav = st.cover
      ? `<div class="nav" style="border-top:0;padding-top:4px"><button class="btn primary" id="sNext">Begin <svg viewBox="0 0 14 14"><path d="M2 7h9M8 3.5 11.5 7 8 10.5" fill="none" stroke="currentColor" stroke-width="1.5"/></svg></button>
           <div class="row" style="gap:2px"><button class="btn quiet" data-goto="explore">Explore</button><button class="btn quiet" data-goto="live">Live solver</button></div></div>
         <p class="small" style="margin:14px 0 0">6 × 6 km · 10 m grid · 23 million cells · checked against Landsat</p>
         <p style="margin:14px 0 0"><button class="filmlink" data-film><i></i><span>Watch the two-minute film</span></button></p>`
      : `<div class="nav"><button class="btn quiet" id="sPrev" aria-label="Previous">← Back</button>
           <div class="steps">${this.steps.slice(1).map((s, k) => `<button data-step="${k + 1}" class="${k + 1 === this.i ? 'on' : (k + 1 < this.i ? 'done' : '')}" aria-label="Chapter ${k + 1}: ${s.chap}"></button>`).join('')}</div>
           ${this.i < n ? `<button class="btn" id="sNext">Next →</button>` : `<span style="width:64px"></span>`}</div>`;
    rail.innerHTML = `${kick}<h2 class="${st.cover ? 'cover' : ''}">${st.title}</h2><p>${st.body}</p>${figs}${st.ctrl ? `<div class="ctrl">${st.ctrl()}</div>` : ''}${nav}`;
    const q = s => rail.querySelector(s);
    q('#sNext') && (q('#sNext').onclick = () => this.next());
    q('#sPrev') && (q('#sPrev').onclick = () => this.prev());
    rail.querySelectorAll('[data-step]').forEach(b => b.onclick = () => this.go(+b.dataset.step));
    rail.querySelectorAll('[data-goto]').forEach(b => b.onclick = () => this.app.setMode(b.dataset.goto));
    rail.querySelectorAll('[data-film]').forEach(b => b.onclick = () => this.app.openFilm());
    q('#goLive') && (q('#goLive').onclick = () => this.app.setMode('live'));
    q('#goExplore') && (q('#goExplore').onclick = () => this.app.setMode('explore'));
    this.wireCtrl();
  }

  // ---------------------------------------------------------------- scenes
  base({ view, ms = 2000, dcTint = 0, dcGlow = 0, site = null, wind = false, labels = 'places' } = {}) {
    const { Q, F } = this.app;
    if (view) Q.setView(view, ms);
    Q.U.uDcTint.value = dcTint; Q.U.uDcGlow.value = dcGlow; Q.U.uSel.value = -9; Q.U.uHover.value = -9;
    Q.showSite(site); Q.setPhoto(false);
    F.particles.setOn(wind);
    if (labels === 'places') this.placeLabels();
  }
  placeLabels(extra = [], maxPrio = 2) {
    const M = this.app.M, list = [];
    for (const l of M.labels) if (l.prio <= maxPrio) list.push({ id: 'p_' + l.text, text: l.text, kind: l.kind, x: l.x, y: l.kind === 'direction' ? 0 : l.y, z: l.kind === 'direction' ? Z1 - 40 : l.z, prio: l.prio + 2 });
    this.app.L.set(extra.concat(list));
  }
  sceneCover() {
    const { Q, F } = this.app;
    this.base({ view: Q.views.home, ms: 2400, dcTint: 0.0 });
    F.showRun(this.app.base, { ground: 'total', vol: true });
    this.app.setLegend(null); this.placeLabels([], 1);
  }
  scenePlace() {
    const { Q, F } = this.app;
    this.base({ view: Q.views.home, dcTint: 1 });
    F.showRun(this.app.base, { ground: 'none', vol: false });
    this.app.setLegend({ title: 'Data-center buildings', sub: 'tagged in OpenStreetMap; footprints from Microsoft and OSM', iso: false, kind: 'none', ticks: [] });
    $('#legend').innerHTML = `<h4>Data-center buildings<small>tagged in OpenStreetMap; footprints from Microsoft and OSM</small></h4><div class="iso"><i style="border-top:9px solid #8E9DC2;width:16px;opacity:1"></i>data center <i style="border-top:9px solid #E6E2D9;width:16px;opacity:1;margin-left:10px"></i>other building</div>`;
  }
  sceneHeat() {
    const { Q, F, M } = this.app;
    this.base({ view: Q.views.cluster, dcTint: 0.4, dcGlow: 1, labels: 'none' });
    F.showRun(this.app.base, { ground: 'none', vol: false });
    const big = M.campuses.filter(c => c.mw).sort((a, b) => b.mw - a.mw).slice(0, 6);
    this.placeLabels(big.map((c, k) => ({ id: 'c' + c.id, kind: 'campus', html: `<em>${fmt(c.mw, 0)} MW</em>&nbsp; campus ${c.id}`, x: c.x, y: c.y + 16, z: c.z, prio: 0, stem: 30 + (k % 2) * 16 })), 1);
    $('#legend').hidden = false;
    $('#legend').innerHTML = `<h4>Rooftop heat rejection<small>400 W per m² of operational data-center roof</small></h4><div class="iso"><i style="border-top:9px solid #F2B524;width:16px;opacity:1"></i>heat source (roof)</div>`;
    document.body.classList.add('has-legend');
  }
  async scenePhysics() {
    const { Q, F } = this.app;
    this.base({ view: Q.views.plume, ms: 2400, wind: true, labels: 'places' });
    const fr = await F.loadFrames();
    if (this.steps[this.i].id !== 'physics') return;
    F.particles.setField(this.app.base.u, this.app.base.v, 150, 150);
    const T = this.app.Timeline;
    Q.setSection(-200, -600, 30, 4600, true);
    T.show(fr.tmax, t => F.showFrames(t));
    T.set(fr.tmax * 0.0);
    setTimeout(() => { if (this.steps[this.i].id === 'physics' && !T.playing && T.t === 0) T.play(); }, 2300);
    this.app.setLegend({ title: 'Warming of the air near the ground', sub: '°C above the same day without data-center heat · the 3-D cloud shows air warmed by more than 0.1 °C' });
  }
  ctrlCheck() {
    const M = this.app.M;
    const dates = M.landsat.filter(e => e.sim);
    return `<div class="seg full" id="lstDates">${dates.map(e => `<button data-d="${e.id}" aria-pressed="${e.id === this.lstDate}">${this.app.dateShort(e.date)}</button>`).join('')}</div>
      <p class="small" style="margin:6px 0 0">30 Jul 2023 and 16 Jul 2024 were used to fit the model; 25 Aug 2024 was held out to test it.</p>
      <div class="swipehint"><span class="chipL">Landsat</span><svg width="34" height="10" viewBox="0 0 34 10"><path d="M1 5h32M5 1 1 5l4 4M29 1l4 4-4 4" fill="none" stroke="currentColor" stroke-width="1.2"/></svg><span class="chipR">Model</span><span class="small">drag the line on the model</span></div>
      <div class="chart"><canvas id="scatter" width="664" height="360"></canvas><div class="cap" id="scatterCap"></div></div>`;
  }
  async sceneCheck() {
    const { Q, F } = this.app;
    this.base({ view: Q.views.lst, ms: 2200, labels: 'places' });
    await this.showLst();
  }
  async showLst() {
    const { Q, F } = this.app;
    const L = await F.loadLandsat(this.lstDate);
    if (this.steps[this.i].id !== 'check') return;
    F.showLandsat(L, { swipe: true, x: Q.U.uSwipeX.value && Math.abs(Q.U.uSwipeX.value) < 2800 ? Q.U.uSwipeX.value : 200 });
    this.app.Explore.swipe(true);
    const e = L.ent;
    this.app.setLegend({ kind: 'lst', title: 'Land surface temperature', sub: `${this.app.dateShort(e.date)}, about 11:46 local time`, range: [Q.U.uLstRange.value.x, Q.U.uLstRange.value.y], src: `left of the line: ${e.platform ? e.platform.replace('landsat-', 'Landsat ') : 'Landsat'} · right: calibrated model` });
    this.drawScatter(e);
  }
  drawScatter(e) {
    const cv = $('#scatter'); if (!cv || !e.hist) return;
    const ctx = cv.getContext('2d'), W = cv.width, H = cv.height, h = e.hist, n = h.n;
    ctx.clearRect(0, 0, W, H);
    const size = H - 64, ox = 74, oy = 12;
    const max = Math.max(...h.counts);
    ctx.fillStyle = '#FFFFFF'; ctx.fillRect(ox, oy, size, size);
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
      const c = h.counts[i * n + j]; if (!c) continue;
      const a = Math.pow(c / max, 0.42);
      ctx.fillStyle = `rgba(1,33,105,${(0.07 + 0.88 * a).toFixed(3)})`;
      ctx.fillRect(ox + (i / n) * size, oy + size - ((j + 1) / n) * size, size / n + 0.6, size / n + 0.6);
    }
    ctx.strokeStyle = 'rgba(181,67,44,.95)'; ctx.lineWidth = 2; ctx.setLineDash([7, 6]); ctx.beginPath(); ctx.moveTo(ox, oy + size); ctx.lineTo(ox + size, oy); ctx.stroke(); ctx.setLineDash([]);
    ctx.strokeStyle = 'rgba(27,31,39,.3)'; ctx.lineWidth = 1.5; ctx.strokeRect(ox + 0.5, oy + 0.5, size, size);
    ctx.fillStyle = '#7D828C'; ctx.font = '22px Barlow, Bahnschrift, sans-serif';
    const lo = h.lo, hi = h.hi;
    for (let v = Math.ceil(lo / 5) * 5; v <= hi; v += 5) { const t = (v - lo) / (hi - lo); ctx.textAlign = 'center'; ctx.fillText(v + '°', ox + t * size, oy + size + 28); ctx.textAlign = 'right'; ctx.fillText(v + '°', ox - 10, oy + size - t * size + 8); }
    ctx.save(); ctx.translate(22, oy + size / 2); ctx.rotate(-Math.PI / 2); ctx.textAlign = 'center'; ctx.fillStyle = '#474D59'; ctx.font = '22px "Source Sans 3", sans-serif'; ctx.fillText('model', 0, 0); ctx.restore();
    ctx.textAlign = 'center'; ctx.fillStyle = '#474D59'; ctx.font = '22px "Source Sans 3", sans-serif'; ctx.fillText('Landsat', ox + size / 2, H - 4);
    const tx = ox + size + 30; ctx.textAlign = 'left';
    ctx.fillStyle = '#1B1F27'; ctx.font = '25px Newsreader, Georgia, serif'; ctx.fillText(e.role.startsWith('held') ? 'Held-out day' : 'Calibration day', tx, oy + 28);
    ctx.fillStyle = '#474D59'; ctx.font = '22px "Source Sans 3", sans-serif';
    const lines = ['Each square bins 30 m', 'pixels; darker = more.', 'On the dashed line the', 'model matches Landsat.', '', `R² ${e.r2.toFixed(2)} · RMSE ${e.rmse.toFixed(1)} °C`];
    lines.forEach((l, k) => ctx.fillText(l, tx, oy + 66 + k * 29));
    $('#scatterCap').textContent = `${fmt(h.counts.reduce((a, b) => a + b, 0))} cloud-free pixels inside the box.`;
  }
  ctrlChoice() {
    const M = this.app.M, S = M.sites, k = this.site;
    const card = (key, title, d) => `<button class="site" data-site="${key}" aria-pressed="${key === k}"><span class="l"><i>${key}</i>${title}</span><span class="d">${d}</span></button>`;
    const row = (label, f, d = 0) => `<tr><th>${label}</th>${['A', 'B', 'C'].map(x => `<td class="${x === k ? 'on' : ''}">${fmt(f(S[x]), d)}</td>`).join('')}</tr>`;
    return `<div class="sites">${card('A', 'Cluster', `${S.A.intake} data halls breathe warmer air`)}${card('B', 'Homes', `${S.B.homes025} homes in warmer air`)}${card('C', 'Stream', `${fmt(S.C.stream05, 2)} km of stream warmed`)}</div>
      <table class="metrics"><thead><tr><th></th>${['A', 'B', 'C'].map(x => `<td>${x}</td>`).join('')}</tr></thead><tbody>
      ${row('Area warmed &gt; ½ °C (ha)', s => s.a05, 0)}
      ${row('Homes in air warmed &gt; ¼ °C', s => s.homes025)}
      ${row('Stream warmed &gt; ½ °C (km)', s => s.stream05, 2)}
      ${row('Forest warmed &gt; ½ °C (ha)', s => s.forest05, 0)}
      ${row('Data halls breathing warmer air', s => s.intake)}
      </tbody></table>
      <p class="small" style="margin:6px 0 0">${this.siteNote(k)}</p>`;
  }
  siteNote(k) {
    return {
      A: 'Inside the cluster no homes are reached, but two neighbouring data halls breathe in air more than ½ °C warmer, which makes their own cooling work harder.',
      B: 'Beside the neighbourhoods seven homes end up in air warmed by more than ¼ °C, and nine existing data halls take in warmer air.',
      C: 'In the stream woodland no homes are reached, but 0.75 km of stream and 16 ha of forest end up in air warmed by more than ½ °C.'
    }[k];
  }
  async sceneChoice() {
    const { Q, F, M } = this.app;
    const st = M.sites[this.site], t = Math.PI / 6;   // plume heads toward 030 degrees
    this.base({ view: { target: [st.center[0] + Math.sin(t) * 650, 40, st.center[1] - Math.cos(t) * 650], dist: 5600, az: 122, el: 30 }, ms: 2000, site: this.site, labels: 'none', dcTint: 0.5 });
    const R = await F.loadRun('scenario_' + this.site);
    if (this.steps[this.i].id !== 'choice') return;
    F.showRun(R, { ground: 'delta', vol: false });
    const dv = await this.deltaVolume(this.site);
    Q.setVolume(dv, null, 0, R.volScale * 255, F.volBox);
    this.siteLabels();
    this.app.setLegend({ title: 'Extra warming from the new campus', sub: '°C near the ground; same wind as chapter 3', max: 3.0 });
  }
  siteLabels() {
    const M = this.app.M;
    const extra = ['A', 'B', 'C'].map(k => { const s = M.sites[k]; return { id: 'site' + k, kind: 'site', cls: k === this.site ? 'active' : 'dim', html: `<em>${k}</em>${{ A: 'Inside the cluster', B: 'Next to homes', C: 'Stream woodland' }[k]}`, x: s.center[0], y: s.ground + 16, z: s.center[1], prio: k === this.site ? -1 : 0, stem: 46 }; });
    this.placeLabels(extra, 1);
  }
  async deltaVolume(k) {
    if (this.deltaVols[k]) return this.deltaVols[k];
    const F = this.app.F; const R = await F.loadRun('scenario_' + k), B = this.app.base;
    const d = new Uint8Array(R.vol.length); for (let i = 0; i < d.length; i++) d[i] = Math.max(0, R.vol[i] - B.vol[i]);
    return (this.deltaVols[k] = vol3d(d, 300, 300, 40));
  }
  sceneLive() {
    const { Q, F } = this.app;
    this.base({ view: Q.views.live, ms: 2000, wind: true });
    F.showRun(this.app.base, { ground: 'total', vol: true });
    this.app.setLegend({ title: 'Warming of the air near the ground', sub: 'archived 10 m run; the live solver recomputes it' });
  }

  wireCtrl() {
    const rail = $('#rail'), id = this.steps[this.i].id;
    if (id === 'physics') { const cb = rail.querySelector('#stLines'); cb.onchange = () => this.app.F.showLines(cb.checked ? this.app.base : null); const sc = rail.querySelector('#stSec'); sc.onchange = () => { this.app.Q.section.visible = sc.checked; }; }
    if (id === 'check') rail.querySelectorAll('[data-d]').forEach(b => b.onclick = () => { this.lstDate = b.dataset.d; rail.querySelectorAll('[data-d]').forEach(x => x.setAttribute('aria-pressed', String(x === b))); this.showLst(); });
    if (id === 'choice') rail.querySelectorAll('[data-site]').forEach(b => b.onclick = () => { this.site = b.dataset.site; this.render(); this.sceneChoice(); });
    if (id !== 'check') this.app.Explore.swipe(false);
  }
}
