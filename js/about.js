// "About the model" sheet: method, calibration, limits, data and credits.
import { fmt } from './util.js';

export function aboutHTML(M) {
  const cal = M.calibration, B = M.baseline, chk = M.solver_checks;
  const mass = Object.values(chk.mass_conservation).map(v => (v * 100).toFixed(1));
  const heat = Object.values(chk.heat_budget_out_over_in);
  const src = M.sources.map(s => `<tr><td>${s.name}<br><span style="color:var(--ink-3);font-size:12.5px">${s.via}</span></td><td>${s.licence}</td></tr>`).join('');
  return `
  <button class="btn quiet close" aria-label="Close">Close ×</button>
  <div class="kicker"><span class="chap">About the model</span></div>
  <h2>A physics-based digital twin of data-center heat</h2>
  <p style="margin-bottom:6px"><button class="filmlink" data-film><i></i><span>Watch the two-minute film</span></button></p>
  <p>The Ashburn Heat Twin rebuilds a 6 × 6 km block of Ashburn, Loudoun County, Virginia (UTM 18N, 283 785–289 785 E, 4 317 795–4 323 795 N) from public data and simulates where the heat rejected by its data centers goes. It is a research prototype from the Department of Environmental Sciences at Emory University, made to show how GeoAI and physical simulation can be combined. It is not an environmental assessment of any facility.</p>

  <h3>The physics engine</h3>
  <p>The wind is solved with a three-dimensional lattice Boltzmann model (D3Q19, regularized BGK collision) with a Smagorinsky large-eddy closure plus a K-profile eddy viscosity for the daytime convective boundary layer. The warming of the air, θ, is carried by a second lattice Boltzmann equation (D3Q7) and feeds back on the flow through Boussinesq buoyancy.</p>
  <div class="eq"><div>f<sub>i</sub>(<b>x</b> + <b>c</b><sub>i</sub>Δt, t + Δt) = f<sub>i</sub><sup>eq</sup>(ρ, <b>u</b>) + (1 − 1/τ) f<sub>i</sub><sup>neq</sup> + F<sub>i</sub></div>
  <div>g<sub>j</sub>(<b>x</b> + <b>c</b><sub>j</sub>Δt, t + Δt) = g<sub>j</sub> − [g<sub>j</sub> − g<sub>j</sub><sup>eq</sup>(θ, <b>u</b>)] / τ<sub>θ</sub> + w<sub>j</sub>S</div>
  <div style="font-size:14px;color:var(--ink-2);margin-top:4px">τ = τ<sub>0</sub> + 3(ν<sub>K</sub> + ν<sub>sgs</sub>)Δt/Δx²,&nbsp; F<sub>z</sub> = ρ g θ / T<sub>0</sub>,&nbsp; S = rooftop heat / (ρ c<sub>p</sub> Δx)</div></div>
  <ul>
    <li>Terrain (USGS 3DEP) and buildings (footprints with heights) are solid cells with half-way bounce-back; the first air layer feels a log-law drag with roughness from ESA WorldCover.</li>
    <li>Inflow follows the wind observed at Washington Dulles (KIAD) at the Landsat overpass, with a power-law profile; outflow faces have a pressure outlet and a viscous sponge; the lid is free-slip with an absorbing layer.</li>
    <li>Heat enters the first air cell above each operational data-center roof, in proportion to roof area.</li>
    <li>600 × 600 × 64 cells at 10 m (23 million cells), run on one GPU at about 950 million cell updates per second. Mass imbalance ${mass.join(' / ')} %; heat leaving the box over heat injected ${heat.join(' / ')}.</li>
  </ul>

  <h3>Checked against Landsat</h3>
  <p>The simulated near-ground warming was combined with land cover, roof type and greenness in a regression on Landsat 8/9 surface temperature (30 m). Two calibration days had opposite winds (30 July 2023 from 030°, 16 July 2024 from 190°), so a real downwind plume would switch sides between them. On those days R² is ${cal.r2_cal.map(v => v.toFixed(2)).join(' and ')}; on the held-out day (25 August 2024) R² is ${cal.r2_val.toFixed(2)} with an RMSE of ${cal.rmse_val.toFixed(2)} °C.</p>
  <p>The plume term is not detected (coefficient ${cal.plume_term.toFixed(3)}, 90% interval ${cal.plume_term_90.map(v => v.toFixed(2)).join(' to ')}). Read the other way round, the satellite bounds the heat that reaches the ground-level air at no more than about ${fmt(cal.F_90[1], 0)} W per m² of roof (one-sided 95%). The archived runs therefore use 400 W/m² for existing roofs, about ${fmt(B.heat_MW, 0)} MW in the box: an upper-bound case.</p>
  <p class="note">Data-center roofs read cooler than other large flat roofs once roof darkness is accounted for, likely because of reflective membranes and shading by rooftop equipment. The satellite therefore cannot be used to estimate heat rejection from roof temperature alone.</p>

  <h3>Scenarios</h3>
  <p>A hypothetical campus of three 14 m data halls (14.8 ha of roof, 120 MW, so 810 W/m²) was placed at three sites: inside the existing cluster (A), on an open field next to neighbourhoods (B) and in woodland near two streams (C). Each run starts from the converged baseline; metrics are taken from the difference between the scenario and the baseline.</p>

  <h3>The live solver</h3>
  <p>The live mode runs a WebGPU port of the same kernel (same collision, forcing, closures and boundaries) on a 20, 30 or 40 m grid in your browser.</p>
  <ul>
    <li><b>Checked against the reference.</b> On the same 30 m grid, the 10-minute mean near-ground warming matches the numba/CUDA solver to a relative error of 4 × 10⁻⁵ (pattern correlation above 0.999999). Single steps agree to float precision. The heat budget closes to 98.6 %.</li>
    <li><b>Speed.</b> About 4 billion cell updates per second on a desktop GPU (RTX 4090), roughly a thousand times faster than real time at 30 m.</li>
    <li><b>Two safeguards that the offline prototype lacks.</b> They keep the solver stable across the whole range of the controls:
      <ul>
        <li>The outflow faces block backflow.</li>
        <li>When the wind turns by more than 30°, the flow restarts from the new wind profile, while the warm air is kept.</li>
      </ul>
    </li>
    <li><b>Coarser first air layer.</b> It is one grid cell deep (20–40 m), so the live field is smoother than the 10 m runs. Against the 10 m baseline averaged to 30 m: correlation 0.97, mean warming about 15 % lower. The live mode is meant for exploring “what if” questions, not for numbers.</li>
  </ul>

  <h3>Limits</h3>
  <ul>
    <li>Daytime, convective conditions only; no night-time inversions, no moisture, no evaporative cooling plumes.</li>
    <li>Heat is assumed to leave as sensible heat at roof level, which is an upper bound for sites with evaporative cooling.</li>
    <li>There is no facility-level power data. All operational roofs share one flux, and campus numbers are arbitrary groupings; operators are not identified.</li>
    <li>Data-center heights are partly assumed (14 m where the footprint height estimate was implausible); land cover is the 2021 map.</li>
    <li>Trees are drawn from the NAIP canopy as model trees for orientation; the vertical scale is exaggerated 2.5 times by default.</li>
  </ul>

  <h3>Data and licences</h3>
  <table>${src}</table>
  <p style="font-size:13px;color:var(--ink-3)">Building footprints and OpenStreetMap-derived layers are © their contributors under the ODbL. Code: MIT licence. Fonts: Newsreader, Source Sans 3 and Barlow (SIL Open Font License). Rendering: three.js.</p>

  <h3>Credits</h3>
  <p>Xiao Huang, Department of Environmental Sciences, Emory University. Prototype built for a department seminar in November 2026.</p>
  <p style="font-size:13px;color:var(--ink-3)">Please cite as: Huang, X. (2026). <i>Ashburn Heat Twin: a lattice-Boltzmann digital twin of data-center heat in Loudoun County, Virginia</i>. https://github.com/alanhuang116/ashburn-heat-twin</p>`;
}
