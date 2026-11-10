// Live data-center-heat solver: WebGPU port of the prototype's numba.cuda lattice Boltzmann model
// (D3Q19 flow + D3Q7 potential-temperature anomaly, Boussinesq buoyancy), run in the browser at 20/30/40 m.
// ES module, no dependencies. See README.md in this folder for physics, conventions and validation.
import { loadStatic10, aggregate, buildDomain, rasterizeCampuses, flipRows, SCENE_X0, SCENE_Z0, BASE_ELEV, KAPPA } from './domain.js';
import { genStepWGSL, genSurfWGSL, genVolWGSL, genInitWGSL, genReinitWGSL, genStatsWGSL, paramsBytes, BINDINGS, BINDING_INFO } from './kernel.js';

// physical constants and model parameters (lbm3d.py)
const RHO_CP = 1.16 * 1005.0;   // J m-3 K-1
const T0 = 303.0;               // K, Boussinesq reference temperature
const GRAV = 9.81;
const CS = 0.16, PRT = 0.7, NU0 = 0.2, KFLOOR = 0.5, MOMFAC = 1.0, ULAT = 0.1, ALPHA = 0.14, ZCAP = 150.0, SPONGE_TAU_S = 60.0;
const U10_MIN = 0.5;            // m/s; the UI range is 1-8
const DIR_TOL = 30, CALM = 0.3, ADJ_TAU_S = 300;   // flow re-initialisation on large direction changes (see README)
const DEFAULT_MET = { u10: 3.09, dirFrom: 210, wstar: 1.67, blh: 903 };   // KIAD 25 Aug 2024 15:52 UTC

/** lbm3d.wind_profile: power law, constant above zcap; returns lattice-free m/s arrays at z = (k + 0.5) dx */
export function windProfile(U10, dirFrom, alpha, nz, dx, zcap = ZCAP) {
  const th = dirFrom * Math.PI / 180, sx = -Math.sin(th), sy = -Math.cos(th);
  const u = new Float64Array(nz), v = new Float64Array(nz), sp = new Float64Array(nz);
  for (let k = 0; k < nz; k++) {
    const z = (k + 0.5) * dx;
    sp[k] = U10 * Math.pow(Math.min(z, zcap) / 10.0, alpha);
    u[k] = sp[k] * sx; v[k] = sp[k] * sy;
  }
  return { u, v, sp };
}

/** lbm3d.kprofile: Holtslag & Boville (1993) K = kappa w_m z (1 - z/h)^2, w_m = (u*^3 + 0.6 w*^3 min(z, 0.1h)/h)^(1/3) */
export function kprofile(z, ustar, wstar, h) {
  const zz = Math.min(z, 0.1 * h);
  const wm = Math.cbrt(ustar ** 3 + 0.6 * wstar ** 3 * zz / h);
  const s = Math.min(1, Math.max(0, 1 - z / h));
  return KAPPA * wm * z * s * s;
}

const defaultUstar = u10 => KAPPA * u10 / Math.log(10 / 0.2);

function dispatchDims(threads, wg) {
  const groups = Math.ceil(threads / wg);
  const x = Math.min(groups, 65535), y = Math.ceil(groups / x);
  return { x, y, strideY: x * wg, groups };
}

export async function createLiveSolver(opts = {}) {
  const o = Object.assign({ base: 'data/live/', dx: 30, u10Max: 8, heightM: 720, onProgress: () => {}, device: null }, opts);
  const prog = (t, f) => { try { o.onProgress(t, f); } catch (e) { /* ignore UI errors */ } };
  if (!o.device && (typeof navigator === 'undefined' || !navigator.gpu))
    throw new Error('WebGPU is not available in this browser (navigator.gpu is missing). The live model needs a WebGPU browser ' +
      '(Chrome/Edge 113+, or Safari/Firefox with WebGPU enabled) and a page served over https or localhost.');
  const dx = +o.dx;
  if (![20, 30, 40].includes(dx) && !o.allowAnyDx) throw new Error(`live solver: dx must be 20, 30 or 40 m (got ${o.dx})`);
  o.u10Max = +o.u10Max; o.heightM = +o.heightM;
  if (!(o.u10Max >= 1 && o.u10Max <= 20)) throw new Error(`live solver: u10Max must be within 1-20 m/s (got ${opts.u10Max})`);
  if (!(o.heightM >= 300 && o.heightM <= 2000)) throw new Error(`live solver: heightM must be within 300-2000 m (got ${opts.heightM})`);

  prog('Loading terrain and buildings', 0.05);
  const adapterP = o.device ? Promise.resolve(null) : navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
  const st = await loadStatic10(o.base);
  prog(`Building the ${dx} m grid`, 0.2);
  const agg = aggregate(st, dx);
  const dom0 = buildDomain(agg, { heightM: o.heightM });
  const { nx, ny, nz } = dom0, N = nx * ny * nz;
  const fBytes = 19 * N * 4;

  // ---- device
  let device = o.device, adapterName = 'unknown', ownDevice = false;
  if (!device) {
    const adapter = await adapterP;
    if (!adapter) throw new Error('WebGPU: no GPU adapter is available (the GPU may be blocklisted or WebGPU disabled).');
    const lim = adapter.limits;
    if (lim.maxStorageBufferBindingSize < fBytes || lim.maxBufferSize < fBytes)
      throw new Error(`WebGPU: this GPU allows ${Math.floor(Math.min(lim.maxStorageBufferBindingSize, lim.maxBufferSize) / 2 ** 20)} MB per buffer, ` +
        `the ${dx} m grid needs ${Math.ceil(fBytes / 2 ** 20)} MB. Try a coarser grid (dx = 40 m).`);
    const info = adapter.info || {};
    adapterName = [info.vendor, info.architecture].filter(Boolean).join(' ') || info.description || 'unknown';
    try {
      device = await adapter.requestDevice({ requiredLimits: { maxStorageBufferBindingSize: lim.maxStorageBufferBindingSize, maxBufferSize: lim.maxBufferSize } });
    } catch (e) {
      throw new Error('WebGPU: could not create a device: ' + (e && e.message || e));
    }
    if (!device) throw new Error('WebGPU: could not create a device.');
    ownDevice = true;
  } else {
    const info = device.adapterInfo || {};
    adapterName = [info.vendor, info.architecture].filter(Boolean).join(' ') || 'unknown';
    if (device.limits.maxStorageBufferBindingSize < fBytes || device.limits.maxBufferSize < fBytes)
      throw new Error(`WebGPU: the supplied device allows ${Math.floor(Math.min(device.limits.maxStorageBufferBindingSize, device.limits.maxBufferSize) / 2 ** 20)} MB ` +
        `per buffer; the ${dx} m grid needs ${Math.ceil(fBytes / 2 ** 20)} MB (request higher maxStorageBufferBindingSize/maxBufferSize, or use dx = 40 m).`);
  }
  if (device.limits.maxStorageBuffersPerShaderStage < 7) throw new Error('WebGPU: needs at least 7 storage buffers per shader stage.');

  const s = new LiveSolver(device, ownDevice, adapterName, o, st, agg, dom0, prog);
  try {
    await s._init();
  } catch (e) {
    s.destroy();
    throw (e instanceof Error ? e : new Error(String(e)));
  }
  prog('Ready', 1);
  return s;
}

class LiveSolver {
  constructor(device, ownDevice, adapterName, o, st, agg, dom, prog) {
    this.device = device; this._ownDevice = ownDevice; this._o = o; this._agg = agg; this._dom = dom; this._prog = prog;
    const { nx, ny, nz, dx } = dom;
    // dt fixed per instance from the profile maximum at u10Max (lbm3d.Sim: dt = ulat dx / Umax)
    const Umax = windProfile(o.u10Max, 0, ALPHA, nz, dx).sp.reduce((a, b) => Math.max(a, b), 0);
    const dt = ULAT * dx / Umax;
    this._dt = dt; this._Umax = Umax; this._conv = dx / dt;
    this._nsp = o.nsp ?? Math.max(1, Math.round(120 / dx));
    this._nsw = o.nsw ?? Math.max(1, Math.round(160 / dx));
    this._wg = o.wg || 64;
    this._cacheNbr = o.cacheNbr ?? true;
    this._oclip = o.outflowClip === false ? 0 : 1;     // backflow-clipped outlets (live extension, see README)
    this.nzv = Math.min(nz, Math.round(420 / dx));
    const nFluid = dom.ktop.reduce((acc, kk) => acc + (nz - kk), 0);
    let dcRoof = 0; for (let k = 0; k < dom.dcarea.length; k++) dcRoof += dom.dcarea[k];
    this._flux = o.existingFlux ?? 400;
    this.info = {
      nx, ny, nz, nzv: this.nzv, dx, dt, zbase: dom.zbase, baseElev: BASE_ELEV, x0: SCENE_X0, z0: SCENE_Z0, adapter: adapterName,
      heatExistingMW: dcRoof * this._flux / 1e6, dcRoofArea_m2: dcRoof,
      // extras (not part of the app interface)
      u10Max: o.u10Max, Umax, convU: this._conv, nsp: this._nsp, nsw: this._nsw, nCells: nx * ny * nz, nFluid, top: dom.zbase + nz * dx,
    };
    this.grid = null;          // north-up geometry (ktop, kt, nb, terrain), set in _setGeometry
    this.campusInfo = [];
    this._met = null; this._inflow = null;
    this._campuses = []; this._ras = rasterizeCampuses([], dx, nx, ny);
    this._parity = 0; this._n = 0; this._steps = 0;
    this._pool = []; this._destroyed = false; this._lost = null;
    this._params = new ArrayBuffer(paramsBytes(nz));
  }

  get simTime() { return this._n * this._dt; }

  async _init() {
    const d = this.device, { nx, ny, nz } = this._dom, N = nx * ny * nz, NXY = nx * ny, wg = this._wg;
    d.lost.then(info => { this._lost = info; });
    this._prog('Allocating GPU memory', 0.3);
    const U = GPUBufferUsage;
    d.pushErrorScope('out-of-memory'); d.pushErrorScope('validation');
    const mk = (size, usage, label) => d.createBuffer({ size: Math.max(16, Math.ceil(size / 4) * 4), usage, label });
    this._bP = mk(paramsBytes(nz), U.UNIFORM | U.COPY_DST, 'params');
    this._bR = mk(16, U.UNIFORM | U.COPY_DST, 'rparams');
    this._bColv = mk(NXY * 16, U.STORAGE | U.COPY_DST | U.COPY_SRC, 'colv');
    this._bKtop = mk(NXY * 4, U.STORAGE | U.COPY_DST | U.COPY_SRC, 'ktop');
    this._bOldk = mk(NXY * 4, U.STORAGE | U.COPY_DST, 'oldktop');
    this._bF = [0, 1].map(p => mk(19 * N * 4, U.STORAGE | U.COPY_SRC | U.COPY_DST, 'f' + p));
    this._bG = [0, 1].map(p => mk(7 * N * 4, U.STORAGE | U.COPY_SRC | U.COPY_DST, 'g' + p));
    this._bSurf = mk(3 * NXY * 4, U.STORAGE | U.COPY_SRC, 'surf');
    this._nvol = this.nzv * NXY; this._nvolw = Math.ceil(this._nvol / 4);
    this._bVol = mk(this._nvolw * 4, U.STORAGE | U.COPY_SRC, 'vol');
    this._bStat = mk(16, U.STORAGE | U.COPY_SRC | U.COPY_DST, 'stat');
    const e1 = await d.popErrorScope(), e2 = await d.popErrorScope();
    if (e1 || e2) throw new Error('WebGPU: buffer allocation failed (' + ((e2 || e1).message || 'out of memory') + '). Try a coarser grid.');
    this._stOff = { surf: 0, vol: 3 * NXY * 4, stat: 3 * NXY * 4 + this._nvolw * 4 };
    this._stOff.total = this._stOff.stat + 16;

    // ---- kernels
    this._prog('Compiling the solver', 0.45);
    const dS = dispatchDims(N, wg), dC = dispatchDims(NXY, wg), dV = dispatchDims(this._nvolw, wg);
    this._dS = dS; this._dC = dC; this._dV = dV;
    const cfg = { nx, ny, nz, wg, strideY: dS.strideY, strideYVol: dV.strideY, nzv: this.nzv, cacheNbr: this._cacheNbr };
    this._cfg = cfg;
    this._layouts = {};
    for (const [k, names] of Object.entries(BINDINGS)) {
      this._layouts[k] = d.createBindGroupLayout({
        label: k, entries: names.map(nm => ({ binding: BINDING_INFO[nm][0], visibility: GPUShaderStage.COMPUTE, buffer: { type: BINDING_INFO[nm][1] } })),
      });
    }
    const [pStep, pSurf, pVol, pInit, pReinit] = await Promise.all([
      this._pipeline('step', genStepWGSL({ ...cfg, out: 'none' }), 'step'),
      this._pipeline('surf', genSurfWGSL(cfg), 'surf'),
      this._pipeline('vol', genVolWGSL(cfg), 'vol'),
      this._pipeline('init', genInitWGSL(cfg), 'init'),
      this._pipeline('reinit', genReinitWGSL(cfg), 'reinit'),
    ]);
    Object.assign(this, { _pStep: pStep, _pSurf: pSurf, _pVol: pVol, _pInit: pInit, _pReinit: pReinit });

    // ---- bind groups (parity p: state in F[p], G[p])
    const B = { P: this._bP, R: this._bR, colv: this._bColv, ktopc: this._bKtop, oldk: this._bOldk, sout: this._bSurf, vout: this._bVol, stat: this._bStat };
    this._B = B;
    this._bgStep = [0, 1].map(p => this._bindGroup('step', { ...B, fA: this._bF[p], fB: this._bF[1 - p], gA: this._bG[p], gB: this._bG[1 - p] }));
    this._bgSurf = [0, 1].map(p => this._bindGroup('surf', { ...B, fA: this._bF[p], gA: this._bG[p] }));
    this._bgVol = [0, 1].map(p => this._bindGroup('vol', { ...B, gA: this._bG[p] }));
    this._bgInit = this._bindGroup('init', { ...B, f0w: this._bF[0], f1w: this._bF[1] });
    this._bgReinit = [0, 1].map(p => this._bindGroup('reinit', { ...B, f0w: this._bF[p], gw: this._bG[p] }));

    this._prog('Initialising the flow', 0.85);
    this._setGeometry(true);
    this.setMet(this._o.met || DEFAULT_MET);
    this.resetFlow();
    await d.queue.onSubmittedWorkDone();
    if (this._lost) throw new Error('WebGPU device lost during initialisation: ' + (this._lost.message || this._lost.reason));
  }

  async _pipeline(label, code, layoutKey) {
    // (no error scopes here: several pipelines compile concurrently and scopes are a per-device stack)
    const d = this.device;
    const module = d.createShaderModule({ label, code });
    const ci = module.getCompilationInfo ? await module.getCompilationInfo() : { messages: [] };
    const errs = ci.messages.filter(m => m.type === 'error');
    if (errs.length) throw new Error(`WGSL compile error in ${label}: ` + errs.slice(0, 3).map(m => `line ${m.lineNum}: ${m.message}`).join('; '));
    try {
      return await d.createComputePipelineAsync({ label, layout: d.createPipelineLayout({ bindGroupLayouts: [this._layouts[layoutKey]] }), compute: { module, entryPoint: 'main' } });
    } catch (e) {
      throw new Error(`WebGPU pipeline ${label}: ${e.message || e}`);
    }
  }

  _bindGroup(layoutKey, bufs) {
    return this.device.createBindGroup({
      layout: this._layouts[layoutKey],
      entries: BINDINGS[layoutKey].map(nm => ({ binding: BINDING_INFO[nm][0], resource: { buffer: bufs[nm] } })),
    });
  }

  _check() {
    if (this._destroyed) throw new Error('live solver: destroyed');
    if (this._lost) throw new Error('live solver: WebGPU device lost (' + (this._lost.message || this._lost.reason || '') + ')');
  }

  // ------------------------------------------------------------------ parameters
  _writeParams() {
    const dx = this._dom.dx, dt = this._dt, nz = this._dom.nz;
    const f = new Float32Array(this._params), i = new Int32Array(this._params);
    f[0] = 0.5 + 3 * NU0 * dt / (dx * dx);              // tau0
    f[1] = 18 * Math.SQRT2 * CS * CS;                    // cs2smag
    f[2] = GRAV / T0 * dt * dt / dx;                     // gbeta
    f[3] = PRT;
    f[4] = KFLOOR * dt / (dx * dx);                      // kfloor
    f[5] = MOMFAC;
    f[6] = 1.0;                                          // srcscale (q is in K per step)
    f[7] = Math.min(0.2, dt / SPONGE_TAU_S);             // sigmax
    const inf = this._inflow;
    i[8] = inf.w; i[9] = inf.e; i[10] = inf.s; i[11] = inf.n;
    i[12] = this._nsp; i[13] = this._nsw; i[14] = this._oclip; i[15] = 0;
    const pr = this._prof;
    for (let k = 0; k < nz; k++) { f[16 + 4 * k] = pr.ul[k]; f[17 + 4 * k] = pr.vl[k]; f[18 + 4 * k] = pr.kl[k]; f[19 + 4 * k] = 0; }
    this.device.queue.writeBuffer(this._bP, 0, this._params);
  }

  /** { u10, dirFrom, wstar, blh, ustar (optional, default kappa u10 / ln(10/0.2)) }; applies from the next step */
  setMet(m = {}) {
    this._check();
    const prev = this._met || DEFAULT_MET;
    const u10 = Math.min(Math.max(+(m.u10 ?? prev.u10), U10_MIN), this._o.u10Max);
    const dirFrom = (((+(m.dirFrom ?? prev.dirFrom)) % 360) + 360) % 360;
    const wstar = Math.max(0, +(m.wstar ?? prev.wstar));
    const blh = Math.min(Math.max(+(m.blh ?? prev.blh), 50), 5000);
    const ustar = (m.ustar !== undefined && m.ustar !== null) ? Math.max(0, +m.ustar) : defaultUstar(u10);
    this._met = { u10, dirFrom, wstar, blh, ustar };
    const { nz, dx } = this._dom, dt = this._dt;
    const wp = windProfile(u10, dirFrom, ALPHA, nz, dx);
    const ul = new Float32Array(nz), vl = new Float32Array(nz), kl = new Float32Array(nz), Kphys = new Float64Array(nz);
    const kmaxLat = 1 / 6;                        // background tau <= 1 (never reached in the UI range)
    for (let k = 0; k < nz; k++) {
      ul[k] = wp.u[k] / this._conv; vl[k] = wp.v[k] / this._conv;
      Kphys[k] = kprofile((k + 0.5) * dx, ustar, wstar, blh);
      kl[k] = Math.min(Kphys[k] * dt / (dx * dx), kmaxLat);
    }
    this._prof = { ul, vl, kl, Kphys };
    const th = dirFrom * Math.PI / 180, ux = -Math.sin(th), uy = -Math.cos(th), eps = 1e-3;
    // inflow faces: the lbm3d eps rule (option sideInflowDeg > 0, off by default: a face whose normal wind component is
    // within sin(sideInflowDeg) of zero is also an inflow face; not needed with the backflow-clipped outlets)
    const side = Math.sin(Math.max(0, this._o.sideInflowDeg ?? 0) * Math.PI / 180);
    this._inflow = { w: +(ux > eps || Math.abs(ux) < side), e: +(ux < -eps || Math.abs(ux) < side),
      s: +(uy > eps || Math.abs(uy) < side), n: +(uy < -eps || Math.abs(uy) < side) };
    this._writeParams();
    // A turn of the wind by more than DIR_TOL relative to the direction the interior flow has adjusted to (lagged by
    // ADJ_TAU_S) puts the old flow against the new inflow faces and gives strong transient jets (the prototype never
    // changes the wind during a run). Such a change re-initialises the flow exactly like the initial state (equilibrium
    // with the new profile at the local height above terrain); theta is kept and simTime keeps running.
    const ref = this._flowRef, applied = this._applied;
    let reinit = false;
    if (ref && applied && this._o.metReinit !== false) {
      const ang = (a, b) => Math.abs(((a - b + 540) % 360) - 180);
      const turn = Math.max(ang(dirFrom, ref.dir), ang(dirFrom, applied.dir));
      if (u10 > CALM && Math.max(ref.u, applied.u) > CALM && turn > (this._o.dirTol ?? DIR_TOL)) reinit = true;
    }
    this._applied = { dir: dirFrom, u: u10 };
    if (!ref || reinit) this._flowRef = { dir: dirFrom, u: u10 };
    if (reinit) { this._reinitFlow(); this.flowReinits = (this.flowReinits || 0) + 1; }
  }

  // f -> equilibrium with the current inflow profile (as at start-up) in both population buffers; theta kept
  _reinitFlow() {
    const enc = this.device.createCommandEncoder(), pass = enc.beginComputePass();
    pass.setPipeline(this._pInit); pass.setBindGroup(0, this._bgInit);
    pass.dispatchWorkgroups(this._dS.x, this._dS.y); pass.end();
    this.device.queue.submit([enc.finish()]);
  }

  // lagged inflow the interior flow has adjusted to (exponential, time constant ADJ_TAU_S of simulated time)
  _advanceFlowRef(n) {
    const ref = this._flowRef, ap = this._applied;
    if (!ref || !ap) return;
    const a = 1 - Math.exp(-n * this._dt / ADJ_TAU_S);
    ref.dir = (((ref.dir + (((ap.dir - ref.dir + 540) % 360) - 180) * a) % 360) + 360) % 360;
    ref.u += (ap.u - ref.u) * a;
  }

  /** W per m^2 of operational data-center roof */
  setExistingFlux(wm2) {
    this._check();
    this._flux = Math.max(0, +wm2 || 0);
    this.info.heatExistingMW = this.info.dcRoofArea_m2 * this._flux / 1e6;
    this._writeColumns();
  }

  /** [{ id, halls: [[x0, z0, x1, z1], ...] (scene metres), mw }] — replaces the previous list */
  setNewCampuses(list) {
    this._check();
    this._campuses = (list || []).map(c => ({ id: c.id, mw: +c.mw || 0, halls: (c.halls || []).map(h => h.slice(0, 4).map(Number)) }));
    const { nx, ny, dx } = this._dom;
    this._ras = rasterizeCampuses(this._campuses, dx, nx, ny);
    this.campusInfo = this._ras.info;
    this._setGeometry(false);
  }

  // geometry (kt/ktop with the campus rule of build_domain) + per-column source; uploads what changed
  _setGeometry(first) {
    const { nx, ny } = this._dom;
    const dom = buildDomain(this._agg, { heightM: this._o.heightM, newArea: this._ras.newArea, nzMin: this._dom.nz });
    if (dom.nz !== this._dom.nz) {
      // nz is fixed per instance; a campus can raise ktop by at most a level, so this cannot trigger in practice
      dom.nz = this._dom.nz;
    }
    const oldKtopS = this._ktopS;
    const ktS = flipRows(dom.kt, nx, ny), ktopS = flipRows(dom.ktop, nx, ny), cdS = flipRows(dom.cd, nx, ny);
    this._ktS = ktS; this._ktopS = ktopS; this._cdS = cdS;
    this.grid = { ktop: dom.ktop, kt: dom.kt, nb: dom.nb, demMean: Float32Array.from(dom.dem), zbase: dom.zbase, dx: dom.dx, nx, ny, nz: this._dom.nz };
    const q = this.device.queue;
    let changed = first;
    if (!first) for (let k = 0; k < ktopS.length; k++) if (ktopS[k] !== oldKtopS[k]) { changed = true; break; }
    if (changed) {
      q.writeBuffer(this._bKtop, 0, Uint32Array.from(ktopS));
      if (!first) {
        // cells that were solid and are fluid now: rest equilibrium, theta = 0 (in the current state buffers)
        let grow = false;
        for (let k = 0; k < ktopS.length; k++) if (ktopS[k] < oldKtopS[k]) { grow = true; break; }
        if (grow) {
          q.writeBuffer(this._bOldk, 0, Uint32Array.from(oldKtopS));
          const enc = this.device.createCommandEncoder(), pass = enc.beginComputePass();
          pass.setPipeline(this._pReinit); pass.setBindGroup(0, this._bgReinit[this._parity]);
          pass.dispatchWorkgroups(this._dC.x, this._dC.y); pass.end();
          q.submit([enc.finish()]);
        }
      }
      let nFluid = 0; const nz = this._dom.nz;
      for (let k = 0; k < ktopS.length; k++) nFluid += nz - ktopS[k];
      this.info.nFluid = nFluid;
    }
    this._writeColumns();
  }

  // per-column vec4<u32>: kt, ktop, cd (f32 bits), source q in K per step (f32 bits)
  _writeColumns() {
    const { nx, ny, dx } = this._dom, NXY = nx * ny, dt = this._dt;
    const buf = new ArrayBuffer(NXY * 16), u = new Uint32Array(buf), f = new Float32Array(buf);
    const dca = this._agg.dcarea, heat = this._ras.heatW, kq = dt / (RHO_CP * dx * dx * dx);
    let totalW = 0;
    for (let jj = 0; jj < ny; jj++) {
      const row = ny - 1 - jj;                   // solver row j -> north-up row
      for (let ii = 0; ii < nx; ii++) {
        const c = jj * nx + ii, cn = row * nx + ii;
        const W = dca[cn] * this._flux + heat[cn];
        totalW += W;
        u[4 * c] = this._ktS[c]; u[4 * c + 1] = this._ktopS[c]; f[4 * c + 2] = this._cdS[c]; f[4 * c + 3] = W * kq;
      }
    }
    this._colBuf = buf;
    this._heatMW = totalW / 1e6;
    this.device.queue.writeBuffer(this._bColv, 0, buf);
  }

  // ------------------------------------------------------------------ state
  /** theta -> 0 everywhere, flow kept, simTime -> 0 */
  resetHeat() {
    this._check();
    const enc = this.device.createCommandEncoder();
    enc.clearBuffer(this._bG[0]); enc.clearBuffer(this._bG[1]);
    this.device.queue.submit([enc.finish()]);
    this._n = 0; this._steps = 0;
  }

  /** full re-init: equilibrium with the inflow profile at the local height above terrain, theta -> 0, simTime -> 0 */
  resetFlow() {
    this._check();
    const enc = this.device.createCommandEncoder(), pass = enc.beginComputePass();
    pass.setPipeline(this._pInit); pass.setBindGroup(0, this._bgInit);
    pass.dispatchWorkgroups(this._dS.x, this._dS.y); pass.end();
    enc.clearBuffer(this._bG[0]); enc.clearBuffer(this._bG[1]);
    this.device.queue.submit([enc.finish()]);
    this._parity = 0; this._n = 0; this._steps = 0;
    if (this._applied) this._flowRef = { ...this._applied };
  }

  /** encode n steps in one command buffer and submit; resolves when the GPU has finished them */
  step(n = 1) {
    try { this._check(); } catch (e) { return Promise.reject(e); }
    n = Math.max(0, Math.floor(n));
    if (n > 0) this._encodeSteps(n, this._pStep, this._bgStep);
    return this.device.queue.onSubmittedWorkDone();
  }

  _encodeSteps(n, pipe, bgs) {
    // one submit; very long requests are split into command buffers of <= 1000 steps (keeps each one short)
    const { x, y } = this._dS, cbs = [];
    for (let done = 0; done < n;) {
      const m = Math.min(1000, n - done);
      const enc = this.device.createCommandEncoder(), pass = enc.beginComputePass();
      pass.setPipeline(pipe);
      for (let s = 0; s < m; s++) {
        pass.setBindGroup(0, bgs[this._parity]);
        pass.dispatchWorkgroups(x, y);
        this._parity ^= 1;
      }
      pass.end();
      cbs.push(enc.finish());
      done += m;
    }
    this.device.queue.submit(cbs);
    this._n += n; this._steps += n;
    this._advanceFlowRef(n);
  }

  // ------------------------------------------------------------------ read-back
  _takeStaging() {
    return this._pool.pop() || this.device.createBuffer({ size: this._stOff.total, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST, label: 'staging' });
  }

  _giveStaging(b) {
    if (!this._destroyed && this._pool.length < 4) this._pool.push(b); else b.destroy();
  }

  /** async snapshot of the current state; never blocks stepping (mapAsync on a staging buffer) */
  async read({ vol = true, wind = true, volMax = 3.0 } = {}) {
    this._check();
    const d = this.device, { nx, ny } = this._dom, NXY = nx * ny;
    const vm = volMax > 0 ? +volMax : 3.0;
    d.queue.writeBuffer(this._bR, 0, new Float32Array([this._conv, 255 / vm, 0, 0]));
    const stg = this._takeStaging();
    const enc = d.createCommandEncoder();
    enc.clearBuffer(this._bStat);
    const pass = enc.beginComputePass();
    pass.setPipeline(this._pSurf); pass.setBindGroup(0, this._bgSurf[this._parity]); pass.dispatchWorkgroups(this._dC.x, this._dC.y);
    pass.setPipeline(this._pVol); pass.setBindGroup(0, this._bgVol[this._parity]); pass.dispatchWorkgroups(this._dV.x, this._dV.y);
    pass.end();
    enc.copyBufferToBuffer(this._bSurf, 0, stg, 0, (wind ? 3 : 1) * NXY * 4);
    if (vol) enc.copyBufferToBuffer(this._bVol, 0, stg, this._stOff.vol, this._nvolw * 4);
    enc.copyBufferToBuffer(this._bStat, 0, stg, this._stOff.stat, 16);
    d.queue.submit([enc.finish()]);
    const simTime = this.simTime, steps = this._steps, heatMW = this._heatMW;
    try {
      await stg.mapAsync(GPUMapMode.READ);
    } catch (e) {
      try { stg.destroy(); } catch (_) { /* */ }
      throw new Error('live solver: read failed (' + (this._destroyed ? 'solver destroyed' : (e.message || e)) + ')');
    }
    const ab = stg.getMappedRange();
    const surf = new Float32Array(ab.slice(0, NXY * 4));
    const u = wind ? new Float32Array(ab.slice(NXY * 4, 2 * NXY * 4)) : null;
    const v = wind ? new Float32Array(ab.slice(2 * NXY * 4, 3 * NXY * 4)) : null;
    const volArr = vol ? new Uint8Array(ab.slice(this._stOff.vol, this._stOff.vol + this._nvol)) : null;
    const tmax = new Float32Array(ab.slice(this._stOff.stat, this._stOff.stat + 4))[0];
    stg.unmap();
    this._giveStaging(stg);
    return { simTime, surf, u, v, vol: volArr, volMax: vm, nzv: this.nzv, thetaMax: tmax, heatMW, steps };
  }

  /** steps the model (the state advances) and returns MLUPS (all nx*ny*nz cells, like the prototype) */
  async benchmark(nSteps = 300) {
    this._check();
    await this.step(Math.min(20, nSteps));
    const t0 = performance.now();
    let left = nSteps;
    while (left > 0) { const m = Math.min(left, 500); this._encodeSteps(m, this._pStep, this._bgStep); left -= m; }
    await this.device.queue.onSubmittedWorkDone();
    const ms = performance.now() - t0;
    return this.info.nCells * nSteps / (ms * 1e3);
  }

  destroy() {
    if (this._destroyed) return;
    this._destroyed = true;
    const bufs = [this._bP, this._bR, this._bColv, this._bKtop, this._bOldk, this._bSurf, this._bVol, this._bStat, this._bDbg, this._bPart,
      ...(this._bF || []), ...(this._bG || []), ...this._pool];
    for (const b of bufs) if (b) try { b.destroy(); } catch (e) { /* */ }
    this._pool = [];
    if (this._ownDevice) try { this.device.destroy(); } catch (e) { /* */ }
  }

  // ------------------------------------------------------------------ validation hooks (not part of the app interface)
  get _dbg() {
    if (!this.__dbg) this.__dbg = this._makeDbg();
    return this.__dbg;
  }

  _makeDbg() {
    const self = this, d = this.device;
    const nPart = this._dS.x * this._dS.y;
    const readBuf = async (buf, size) => {
      const stg = d.createBuffer({ size, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
      const enc = d.createCommandEncoder(); enc.copyBufferToBuffer(buf, 0, stg, 0, size); d.queue.submit([enc.finish()]);
      await stg.mapAsync(GPUMapMode.READ);
      const out = stg.getMappedRange().slice(0); stg.unmap(); stg.destroy();
      return out;
    };
    const ensureDbg = async () => {
      if (self._pStepOut) return;
      const { nx, ny, nz } = self._dom, N = nx * ny * nz;
      self._bDbg = d.createBuffer({ size: N * 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST, label: 'dbg' });
      self._bPart = d.createBuffer({ size: nPart * 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC, label: 'part' });
      [self._pStepOut, self._pStepAcc, self._pStats] = await Promise.all([
        self._pipeline('stepOut', genStepWGSL({ ...self._cfg, out: 'out' }), 'stepDbg'),
        self._pipeline('stepAcc', genStepWGSL({ ...self._cfg, out: 'acc' }), 'stepDbg'),
        self._pipeline('stats', genStatsWGSL(self._cfg), 'stats'),
      ]);
      const B = self._B;
      self._bgStepDbg = [0, 1].map(p => self._bindGroup('stepDbg', { ...B, dbg: self._bDbg, fA: self._bF[p], fB: self._bF[1 - p], gA: self._bG[p], gB: self._bG[1 - p] }));
      self._bgStats = [0, 1].map(p => self._bindGroup('stats', { ...B, fA: self._bF[p], gA: self._bG[p], part: self._bPart }));
      const enc = d.createCommandEncoder(); enc.clearBuffer(self._bDbg); d.queue.submit([enc.finish()]);
    };
    return {
      wgsl() { return { step: genStepWGSL({ ...self._cfg, out: 'none' }), surf: genSurfWGSL(self._cfg), vol: genVolWGSL(self._cfg), init: genInitWGSL(self._cfg) }; },
      domain() {
        const pf = new Float32Array(self._params.slice(0)), pi = new Int32Array(self._params.slice(0));
        const colU = new Uint32Array(self._colBuf), colF = new Float32Array(self._colBuf), NXY = self._dom.nx * self._dom.ny;
        const kt = new Int16Array(NXY), ktop = new Int16Array(NXY), cd = new Float32Array(NXY), q = new Float32Array(NXY);
        for (let c = 0; c < NXY; c++) { kt[c] = colU[4 * c]; ktop[c] = colU[4 * c + 1]; cd[c] = colF[4 * c + 2]; q[c] = colF[4 * c + 3]; }
        return {
          nx: self._dom.nx, ny: self._dom.ny, nz: self._dom.nz, dx: self._dom.dx, zbase: self._dom.zbase, dt: self._dt, conv: self._conv,
          params: { tau0: pf[0], cs2smag: pf[1], gbeta: pf[2], prt: pf[3], kfloor: pf[4], momfac: pf[5], srcscale: pf[6], sigmax: pf[7], nsp: pi[12], nsw: pi[13], oclip: pi[14] },
          inflow: { ...self._inflow }, met: { ...self._met }, uin: self._prof.ul, vin: self._prof.vl, kbg: self._prof.kl, Kphys: Array.from(self._prof.Kphys),
          kt, ktop, cd, q, heatMW: self._heatMW, parity: self._parity, steps: self._steps, campusInfo: self.campusInfo,
        };
      },
      async readState() {
        const N = self.info.nCells, p = self._parity;
        return { f: new Float32Array(await readBuf(self._bF[p], 19 * N * 4)), g: new Float32Array(await readBuf(self._bG[p], 7 * N * 4)) };
      },
      writeState(f, g) {
        if (f) d.queue.writeBuffer(self._bF[self._parity], 0, f);
        if (g) d.queue.writeBuffer(self._bG[self._parity], 0, g);
      },
      async readDbg() { await ensureDbg(); return new Float32Array(await readBuf(self._bDbg, self.info.nCells * 16)); },
      async clearDbg() { await ensureDbg(); const enc = d.createCommandEncoder(); enc.clearBuffer(self._bDbg); d.queue.submit([enc.finish()]); },
      /** n steps; the last one writes (theta, ux, uy, uz) of every fluid cell to the debug buffer (lbm_step do_out) */
      async stepOut(n) {
        await ensureDbg();
        if (n > 1) self._encodeSteps(n - 1, self._pStep, self._bgStep);
        self._encodeSteps(1, self._pStepOut, self._bgStepDbg);
        return d.queue.onSubmittedWorkDone();
      },
      /** n steps accumulating (theta, ux, uy, uz) into the debug buffer (lbm_step do_acc) */
      async stepAcc(n) { await ensureDbg(); self._encodeSteps(n, self._pStepAcc, self._bgStepDbg); return d.queue.onSubmittedWorkDone(); },
      async stats() {
        await ensureDbg();
        const enc = d.createCommandEncoder();
        enc.clearBuffer(self._bStat);
        const pass = enc.beginComputePass();
        pass.setPipeline(self._pStats); pass.setBindGroup(0, self._bgStats[self._parity]); pass.dispatchWorkgroups(self._dS.x, self._dS.y); pass.end();
        d.queue.submit([enc.finish()]);
        const part = new Float32Array(await readBuf(self._bPart, nPart * 16));
        const stat = new Uint32Array(await readBuf(self._bStat, 16));
        let th = 0, ke = 0, dr = 0, nf = 0;
        for (let k = 0; k < nPart; k++) { th += part[4 * k]; ke += part[4 * k + 1]; dr += part[4 * k + 2]; nf += part[4 * k + 3]; }
        const dx = self._dom.dx, c = self._conv;
        return {
          simTime: self.simTime, steps: self._steps, nFluid: nf, nonFinite: stat[1], thetaMax: new Float32Array(stat.buffer.slice(0, 4))[0],
          heatContentMJ: th * RHO_CP * dx * dx * dx / 1e6, keMean: ke / Math.max(nf, 1) * c * c, rhoDevMean: dr / Math.max(nf, 1),
          umax: new Float32Array(stat.buffer.slice(8, 12))[0] * c,
        };
      },
    };
  }
}
