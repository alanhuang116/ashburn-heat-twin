# Live solver (WebGPU)

In-browser port of the prototype's data-center heat model for the 6 × 6 km Ashburn box. It is a 3-D lattice Boltzmann
solver: D3Q19 flow plus D3Q7 potential-temperature anomaly θ with Boussinesq buoyancy, run live at 20, 30 or 40 m. The
WGSL step kernel is generated section by section from the prototype's `lbm_kernel_gen.py`. Lattice tables, unrolling,
boundary conditions and order of operations are the same. On identical inputs it reproduces the numba reference to
float precision. **Everything it computes is simulated, not observed.**

| file | what |
|---|---|
| `solver.js` | host module: `createLiveSolver()` (interface below), buffers, pipelines, asynchronous read-back |
| `kernel.js` | WGSL generator: the fused step (mirrors `gen_source()`), near-surface and volume extraction, init, diagnostics |
| `domain.js` | loads `data/live/static10.*`, block-aggregates the 10 m inputs, applies `build_domain`'s rules, rasterizes campuses |
| `data/live/static10.json` + `static10.bin.gz` | 0.53 MB, 10 m north-up: DEM (1 cm), WorldCover class, building cover fraction, mean building height, operational data-center roof area (`tools/live/pack_live.py`) |
| `tests/live/validate.html`, `tools/live/run_headless.js`, `tools/live/validate_live.py` | validation page, headless-Chrome runner (static server + dump endpoint), numba reference driver; collected results in `tests/live/validation_summary.json` |

## Quick start

```js
import { createLiveSolver } from './js/live/solver.js';
const solver = await createLiveSolver({ dx: 30, onProgress: (text, f) => console.log(text, f) });
solver.setMet({ u10: 3.09, dirFrom: 210, wstar: 1.67, blh: 903 });     // KIAD 25 Aug 2024
solver.setExistingFlux(400);
solver.setNewCampuses([{ id: 'A', mw: 120, halls: [[-205, 475, 175, 605], [-205, 635, 175, 765], [-205, 795, 175, 925]] }]);
// per frame: ~40 steps at 30 m is ~10 ms of GPU time and ~10 s of simulated time
await solver.step(40);
const r = await solver.read();                // ~3-4 ms; r.surf, r.u, r.v (north-up), r.vol ([k][row][col] u8)
```

## Interface (as specified)

```js
createLiveSolver({ base = 'data/live/', dx = 30, u10Max = 8, heightM = 720, onProgress, device = null })  // -> Promise<solver>
solver.info     // { nx, ny, nz, nzv, dx, dt, zbase, baseElev: 83.53166, x0: -2995, z0: -2995, adapter, heatExistingMW, dcRoofArea_m2, ...extras }
solver.setMet({ u10, dirFrom, wstar, blh, ustar })   // sync, applies from the next step
solver.setExistingFlux(wm2)                          // W per m2 of operational data-center roof
solver.setNewCampuses([{ id, halls: [[x0, z0, x1, z1], ...], mw }])
solver.resetHeat(); solver.resetFlow()
solver.step(n)                                       // Promise, resolves on queue.onSubmittedWorkDone()
solver.simTime
await solver.read({ vol = true, wind = true, volMax = 3.0 })
  // { simTime, surf, u, v, vol, volMax, nzv, thetaMax, heatMW, steps }
await solver.benchmark(nSteps = 300)                 // MLUPS
solver.destroy()
```

What the values mean:
- **Rejections.** `createLiveSolver` rejects with a readable `Error` in these cases: no `navigator.gpu`; no adapter or
  device; a buffer limit that is too small (the 20 m grid needs one 246 MB buffer, and the message suggests 40 m); a
  WGSL or pipeline error; out-of-memory during allocation.
- **`read().surf`, `.u`, `.v`.** These are taken in the first fluid cell above terrain or roof (`ktop`). They are the
  pre-collision moments of the current state: θ in K, and the force-corrected velocity `(j + F/2)/ρ` in m/s. This is
  exactly what the reference `lbm_step` writes to `theta_out`/`u_out` on its next step. `u` is east and `v` is north.
  Arrays are north-up `[row][col]`: row 0 is the north edge (z = −2995), col 0 the west edge (x = −2995).
- **`read().vol`.** The same θ, as `[k][row][col]`, with `round(θ·255/volMax)` clamped to 0–255. Solid cells are 0.
  Level k spans `zbase + k·dx … zbase + (k+1)·dx`.
- **`thetaMax`.** The unclamped maximum θ over the fluid cells of those `nzv` levels.
- **`heatMW`.** The heat release currently applied: existing flux × roof area, plus the new campuses' in-domain MW.
- **Read-back mechanics.** `read()` encodes two small kernels, copies into a pooled staging buffer and awaits
  `mapAsync`. It never blocks stepping. Measured latency is 3–4 ms median and ≤ 10 ms max, and with reads at 10 Hz the
  step throughput is unchanged (table below).
- **`simTime`.** Steps since the last `resetHeat`/`resetFlow`, times `dt`.
- **`benchmark()`.** It really steps, so it advances the simulation.

Per-grid facts with u10Max = 8 (dt is fixed per instance: `dt = 0.1·dx/Umax`, Umax = 8·15^0.14 = 11.69 m/s):

| dx | grid | levels / top | zbase (m NAVD88) | dt | nsp (top sponge) | nsw (outflow sponge) | nzv | GPU memory (f + g, ×2) |
|---|---|---|---|---|---|---|---|---|
| 20 m | 300 × 300 × 36 | 720 m | 41 | 0.1711 s | 6 | 8 | 21 (420 m) | 674 MB |
| 30 m | 200 × 200 × 24 | 720 m | 31 | 0.2567 s | 4 | 5 | 14 (420 m) | 200 MB |
| 40 m | 150 × 150 × 18 | 720 m | 21 | 0.3422 s | 3 | 4 | 11 (440 m) | 84 MB |

## Physics

The physics is the prototype's (`lbm3d.py`, `lbm_kernel_gen.py`):
- **Flow and closure.** Regularized BGK with Smagorinsky Cs = 0.16. A Holtslag–Boville K-profile background viscosity
  comes from u\*, w\* and blh. If u\* is omitted it defaults to κ·u10/ln(10/0.2), which gives 0.32 m/s at 3.09 m/s. K
  is clipped to τ ≤ 1 in lattice units; this is a safety net only and is never reached in the UI range. When blh is
  above the 720 m lid, the profile is simply cut at the lid.
- **θ.** D3Q7 with eddy diffusivity νsgs/Prt + K + kfloor, where Prt = 0.7, kfloor = 0.5 m²/s and ν0 = 0.2 m²/s.
- **Forcing.** Guo forcing with Boussinesq buoyancy, T0 = 303 K.
- **Surface drag.** A limited log-law drag in the first fluid cell above terrain, in columns without buildings.
- **Sponges.**
  - Top sponge: `nsp = round(120/dx)` cells; it relaxes velocity toward the inflow profile with σ = min(0.2, dt/60 s) and damps θ.
  - Outflow viscous sponge: τ += 0.15·rr² over `round(160/dx)` cells (the prototype's 16-cell rule).
- **Boundaries.**
  - Terrain and buildings: half-way bounce-back.
  - Inflow faces: equilibrium with the power-law profile (α = 0.14, constant above 150 m). Faces are chosen by the
    prototype's eps rule from the direction only.
  - Other faces: pressure outlet (ρ = 1, non-equilibrium extrapolation).
  - Lid: free slip.
  - θ: 0 at inflow faces, zero-gradient at outflow faces and the lid.
- **Heat source.** Rooftop heat is injected in the first fluid cell above the surface, `Q·dt/(ρcp·dx³)` per step with
  ρcp = 1.16 × 1005.
- **Initial state.** Equilibrium with the inflow profile at the local height above terrain, θ = 0.

**Coarse domain** (`domain.js`, following `build_domain`):
- **Aggregation of the 10 m fields.**
  - dem: block mean.
  - frac: block mean (exact integer counts).
  - hmean: area-weighted mean over covered sub-pixels.
  - dcarea: block sum.
  - roughness: mean of the per-10 m drag coefficients `(κ/ln(0.5·dx/z0))²`, each evaluated at the coarse first-cell
    centre. This is the stress-conserving average. z0 comes from the WorldCover table.
- **Vertical layout.**
  - `zbase = floor(min dem) − dx`.
  - `kt = max(1, ceil((dem − zbase)/dx − 0.5))`.
  - `nb = round_half_even(hmean/dx)` where frac ≥ 0.5, otherwise 0.
  - `ktop = kt + nb`.
  - `nz = round(720/dx)`, raised automatically if `ktop.max() ≥ nz − 8`. This never triggers: ktop.max() is 4, 3 and 2.
- **What is resolved at each dx.** At 30 and 40 m almost no buildings are solid (42 and 3 columns), and the 14 m data
  halls are not solid (nb = 0). Their heat goes into the 0–30 m or 0–40 m first cell. At 20 m, 5,390 columns carry a
  building level.
- **New campuses.**
  - Rasterization: exact rectangle–cell overlap, in scene metres.
  - Heat: W/m² = mw·10⁶ / Σ hall area. Hall parts outside the box are dropped.
  - Geometry: `frac = max(frac, area/dx²)` and `hmean = max(hmean, 14 m)`, as in `build_domain`. So at 20 m every cell
    with ≥ 50 % hall cover becomes one solid level, and the heat goes into the cell above the roof. Cells uncovered by
    a removed campus restart at rest.

## Deviations from the prototype (deliberate, each switchable)

1. **Backflow-clipped outlets** (`outflowClip`, default on).
   - *Why.* The prototype's pressure outlet lets air back in, which produces 25–35 m/s jets and NaNs within 600–4,400
     s in three situations: at 1 m/s with strong heating, when the wind is exactly along a grid axis (180° at 4 m/s,
     40 m), and after abrupt changes of the wind. The unmodified numba kernel blows up identically: same step, same
     33.1 m/s. The prototype never ran these regimes (its cases are 2.6–5.7 m/s at oblique directions).
   - *Change.* The prototype's outlet `f_q + w_q(1−ρn)E(u)` is a non-equilibrium extrapolation toward ρ = 1 with the
     cell's own velocity. The live version uses that velocity with any inward normal component clipped to zero:
     `f_q + w_q[E(u_clip) − ρn·E(u)]`.
   - *Effect.* It is mathematically identical wherever there is no backflow. For the 25 Aug validation case it changes
     the 600 s mean surface θ by a relative L2 of 2.7e-4 (correlation 0.99999995). `outflowClip: false` runs the
     prototype's outlet bit-for-bit; that path is a uniform branch.
2. **Flow re-initialisation on large turns of the wind** (`metReinit`, default on).
   - *When.* `setMet` with a direction more than 30° from the direction the interior flow has adjusted to (an
     exponential lag with a 300 s time constant).
   - *What.* The flow is re-initialised exactly like the initial state, but with the new profile. θ is kept and
     `simTime` continues. It is counted in `solver.flowReinits`.
   - *Why.* Without it, an abrupt 120–180° turn puts the old flow against the new inflow faces and gives transient jets
     of up to ~20 m/s. Speed changes and smaller turns apply directly.
3. **Fixed dt per instance from `u10Max`** (as specified). At 3.09 m/s this runs the lattice at Mach 0.04, where the
   prototype ran at 0.1. Repeating the 25 Aug case with u10Max = 3.09 changes the 600 s mean surface θ only slightly:
   correlation 0.9985 with the u10Max = 8 run, peak 2.17 vs 2.27 K, mean −1.6 %.
4. **Smaller changes.**
   - u10 is clamped to [0.5, u10Max], blh to [50, 5000] m and w\* to ≥ 0.
   - Per-column source in K per step (srcscale = 1).
   - Arithmetic is f32. numba promotes most of the prototype kernel to f64 because its literals and scalar arguments
     are f64; storage is f32 in both.
   - An optional `sideInflowDeg` rule, which makes faces within N° of parallel to the wind inflow faces, exists but is
     off. It is not needed once the outlets are clipped.

## Validation

**Method.**
1. The page builds the domain in JS and dumps it: kt, ktop, cd, per-column source, profiles and parameters.
2. It also dumps the initial or developed state, runs, and dumps the outputs.
3. `validate_live.py` runs the reference kernel on the same domain and state. The kernel is generated by the
   prototype's `gen_source()` and driven like `lbm3d.Sim.run`: python-float scalars, int16 kt/ktop, uint8 flag.
4. The reference is modified only by explicit, asserted text patches: the sponge width when it is not 16, and the
   outlet clip when the WebGPU run uses it.
5. The page's `resetFlow` state is bit-identical to `lbm3d.Sim`'s numpy initialisation (max 0 ulp).
6. The JS domain equals an independent numpy mirror of `build_domain` (kt, ktop, cd, zbase all exact), and that mirror
   at 10 m reproduces `lbm3d.build_domain` exactly. At 20 m with campuses A+B, exactly the 828 cells with ≥ 50 % hall
   cover gain a level.

**Short horizon** (relative L2 against numba; "field" is all fluid cells of the step output; "surf" is `read()`):

| case | dx | inflow faces | outlet | pre + steps | field θ | field ux | field uy | surf θ | surf u | surf v | state f | u8 volume equal / max diff |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cold start, heat on | 30 | w s | prototype, nsw 16 (unpatched kernel) | 0 + 300 | 2.8e-6 | 5.2e-6 | 3.1e-6 | 2.6e-6 | 2.1e-5 | 9.8e-6 | 2.9e-7 | 100 % / 0 |
| developed, campus A | 30 | w s | clip | 2000 + 300 | 3.2e-6 | 1.1e-5 | 7.2e-6 | 3.3e-6 | 2.7e-5 | 2.4e-5 | 4.1e-7 | 99.999 % / 1 |
| developed, campuses A+B solid | 20 | w s | clip | 1500 + 300 | 2.9e-6 | 9.2e-6 | 6.0e-6 | 3.1e-6 | 3.5e-5 | 2.6e-5 | 3.5e-7 | 99.9996 % / 1 |
| 6 m/s from 75°, campus C | 40 | e n | clip | 1000 + 500 | 3.5e-6 | 3.1e-6 | 5.9e-6 | 3.1e-6 | 1.2e-5 | 2.0e-5 | 3.3e-7 | 99.9996 % / 1 |
| 1 m/s, w\* 0, 800 W/m², A+B | 40 | w n | clip | 1500 + 500 | 1.6e-5 | 1.5e-5 | 2.4e-5 | 9.6e-6 | 5.8e-5 | 8.5e-5 | 7.1e-7 | 99.992 % / 1 |

All cases pass the < 1e-4 criterion. uz has the same absolute error (rms error ≤ 1.6e-5 of the rms speed); its
relative L2 is larger only because |uz| is small. Against the unpatched (no-clip) reference, the clip shows up only
where backflow occurs: surface u/v differ by 4–6e-4 for the developed 30 m case and by 0.7–2 % for the 1 m/s convective
case.

**Long horizon: 25 Aug 2024 at 30 m.** U10 3.09 m/s from 210°, u\* 0.32, w\* 1.67, blh 903 m, 400 W/m². 2200 s
spin-up plus a 600 s mean (8,572 + 2,338 steps; WebGPU 4.5 s wall, numba 18.7 s). The time mean is accumulated every
step on both sides (`do_acc`).

| quantity (600 s mean) | WebGPU | numba (same outlet) | numba (prototype outlet) |
|---|---|---|---|
| near-surface θ: pattern correlation vs WebGPU | — | 0.9999999987 (rel. L2 4.3e-5) | 0.99999995 (rel. L2 2.7e-4) |
| near-surface θ max / mean | 2.2745 / 0.15256 K | 2.2745 / 0.15256 K | 2.2747 / 0.15256 K |
| domain-integrated heat | 759,721 MJ | 759,723 MJ | 759,726 MJ |
| heat injected / advected out (lateral faces, mean fields) | 742.6 / 732.0 MW (98.6 %) | 742.6 / 732.0 MW | 742.6 / 732.0 MW |
| plume (heat centroid vs source centroid) | 1070 m toward 22.1°, mean height 111 m above ground | 1070 m, 22.1°, 111 m | same |

Sanity checks:
- **Downwind.** The plume goes downwind: the wind blows toward 30°, and the heat centroid is 1070 m toward 22°. The
  archived 10 m baseline has 1064 m toward 20.9°, at 108 m.
- **Heat budget.** 98.6 % of the injected heat leaves through the lateral faces in the mean. The rest is the top
  sponge and remaining storage. The mass imbalance is 0.03 %.
- **Early budget.** In a cold-started run with heat on, the heat content after 51 s is 44,251 MJ against 44,280 MJ
  injected.

**Against the archived 10 m baseline** (same weather and flux), with the 10 m field block-averaged to 30 m:
- **First fluid cell** (0–10 m at 10 m vs 0–30 m at 30 m): pattern correlation 0.965, peak ratio 1.03, mean ratio 0.85.
- **Same 0–30 m layer** (mean of the first three 10 m cells): correlation 0.976, peak ratio 1.34, mean ratio 0.98.

Read: the 30 m field reproduces the baseline's pattern; it is smoother and its near-surface mean is 15 % weaker. The
unresolved data halls put their heat straight into the 30 m surface layer, so peaks over the big roofs are not diluted.
(The u10Max = 3.09 run gives 0.968, 0.99 and 0.84.)

**Stability** (all with the defaults; no NaN or Inf anywhere, domain-wide max speed bounded):
- **UI matrix, 10,000 steps per run, every combination at 20, 30 and 40 m** (1,711, 2,567 and 3,422 s): 18 of 18 runs
  stable. The combinations:
  - u10 1 m/s from 300° with w\* 2.5, blh 2000, 800 W/m², campuses A+B;
  - 1 m/s with w\* 0, blh 500, 800 W/m², A+B;
  - 8 m/s from 45°, neutral, 800 W/m², C;
  - 4 m/s from 180° (axis-aligned), flux 0, A;
  - 2 m/s from 270° (axis-aligned), BC;
  - a switching run (3 → 8 m/s with the wind turned from 210° to 90° and A+B added; then 1 m/s from 315° at 800 W/m²
    with w\* 2.5 and blh 2000; then 5 m/s from 0° neutral with the campuses removed).
- **Weak wind for 6000 s:** stable at every dx. Domain-wide maximum speed 2.1–5.8 m/s, i.e. buoyant updrafts. With
  the prototype's outlet these blow up after 600–4,400 s.
- **36 met transitions** (12 per dx): abrupt 25°, 30°, 120° and 180° turns, 8 → 1 and 1 → 8 m/s, and ramped turns and
  reversals. All stable, with transient maxima ≤ 1.7 × the inflow peak.

## Performance

Headless Chrome 157 with WebGPU (D3D12) on an RTX 4090 ("nvidia lovelace"). MLUPS counts all nx·ny·nz cells, like the
prototype. The GPU was shared with the desktop (other processes used 36–47 %), and repeated sessions vary by about
±20 %. The table shows the final session.

| dx | MLUPS (workgroup 64) | ms per step | simulated / wall time | `read()` latency median / max | steps per s with `step(20)` loop: alone / with 10 Hz reads |
|---|---|---|---|---|---|
| 20 m | 4,280 (3,530 earlier session) | 0.76 | 226 × | 4.2 / 9.9 ms | 1,201 / 1,199 |
| 30 m | 4,100 (3,450) | 0.23 | 1,100 × | 3.2 / 4.7 ms | 3,329 / 3,324 |
| 40 m | 6,600 (5,860) | 0.06 | 5,600 × | 3.0 / 5.3 ms | 6,216 / 6,233 |

Notes:
- **Bandwidth.** The kernel is bandwidth-bound, at about 800 GB/s of population traffic at 20–30 m. At 40 m the 84 MB
  state largely stays in the 4090's 72 MB L2. The 30 m target was ≥ 300 MLUPS.
- **Reference speed.** The numba reference runs the same 30 m grid at 560–585 MLUPS: its arithmetic is f64.
- **Workgroup size** 64 vs 128 differs by less than 1 %. f16 storage was not needed.
- **Call overhead.** A loop of awaited `step(20)` calls is limited by per-call latency, especially at 40 m. Use larger
  `n` per frame, or keep one call in flight.

## Spin-up

Cold start (`resetFlow`, heat on from t = 0), 25 Aug weather. "Settled" means the 60 s running mean stays within 2 %
(near-surface mean wind) or 5 % (heat content, mean near-surface θ) of its mean over the last quarter of the run.

| dx | near-surface wind settled | heat content settled | mean near-surface θ settled | wall time to heat-settled |
|---|---|---|---|---|
| 20 m | 605 s (3,538 steps) | 1,499 s (8,758 steps) | 1,399 s (8,178) | ~6.6 s |
| 30 m | 771 s (3,003 steps) | 1,532 s (5,967 steps) | 1,401 s (5,460) | ~1.4 s |
| 40 m | 863 s (2,523 steps) | 1,667 s (4,872 steps) | 1,558 s (4,553) | ~0.3 s |
| 30 m, 8 m/s | 410 s (1,599) | 751 s (2,925) | 711 s (2,769) | ~0.7 s |
| 30 m, 1 m/s | 2,543 s (9,906) | 2,803 s (10,920) | 2,142 s (8,346) | ~2.6 s |

The domain-mean kinetic energy is dominated by the flow aloft, which starts at the inflow profile, so it settles within
30 s. A change of flux or campuses re-equilibrates on roughly the heat time scale. The prototype used 2200 s of
spin-up plus a 600 s average at 3.1 m/s.

## Known limitations

- **Coarse vertical resolution.** The near-surface value is the mean of a 20–40 m deep first cell (the 10 m model used
  0–10 m). At 30 and 40 m the buildings are essentially unresolved, and drag acts only through z0. Treat the 20 m grid
  as the most faithful.
- **Validation scope.** Validation against the 10 m model covers only the 25 Aug 2024 weather. The weak-wind and
  axis-aligned regimes are stable but unvalidated, and the outlet clip is an extension the prototype does not have.
- **Visible artefacts.**
  - A re-initialisation after a large wind turn resets the near-surface flow pattern; it re-develops in about 600–900 s
    of simulated time.
  - The 120–160 m sponge zones along the outflow faces and the lid distort the flow locally.
  - θ is forced to 0 at the inflow faces.
- **Fixed lattice time step.** Low winds run at a low lattice Mach number: accurate, but more steps per simulated
  second. Raising `u10Max` above 8 shortens dt.
- **WebGPU limits.** The 20 m grid needs a 246 MB storage buffer and ~0.7 GB of GPU memory. Browsers or adapters that
  expose only the default 128 MB binding limit must use 40 m; the error message says so.
- **Instantaneous fields.** `read()` returns instantaneous fields (LES-like turbulence). Average in the app if a steady
  picture is wanted.

## Re-running the validation

```
python tools/live/pack_live.py --proto <prototype dc folder>
node tools/live/run_headless.js "test=short&dx=30&steps=300&pre=2000&campus=A&tag=p30b&state1=1" --dump <dir> --playwright <node_modules/playwright-core>
python tools/live/validate_live.py short <dir> p30b --proto <prototype dc folder>        # add --ref-oclip 0 for the prototype outlet
node tools/live/run_headless.js "test=long&dx=30&spin=2200&avg=600&tag=L30c" --dump <dir> --playwright <...>
python tools/live/validate_live.py long <dir> L30c --proto <prototype dc folder>
node tools/live/run_headless.js "test=stab&dxs=20,30,40&steps=10000" ...               # also: bench, readlat, spinup, switchscan, dirscan
python tools/live/validate_live.py summary <dir> all <tags...> --out tests/live/validation_summary.json
```

Use Python with numba.cuda and run it isolated (`-I`); the driver adds the user site-packages for cv2 itself.
Headless Chrome needs `--enable-unsafe-webgpu --ignore-gpu-blocklist`, which the runner sets. The page must come from
http://localhost, because WebGPU requires a secure context.
