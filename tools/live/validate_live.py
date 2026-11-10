#!/usr/bin/env python3
"""Numerical validation of the live WebGPU solver (js/live/) against the prototype's numba.cuda reference.

The reference kernel is generated, unmodified, by the prototype's lbm_kernel_gen.gen_source() and driven like
lbm3d.Sim.run (python-float scalar arguments, int16 kt/ktop, uint8 flag, [q][cell] float32 populations) on the
*JS-built* domain and the *JS* initial state dumped by tests/live/validate.html. The only optional change is the
outflow-sponge width (16 cells in the prototype = 160 m at 10 m): with --nsw the two literal 16s of that rule are
replaced in the generated source (asserted to be exactly two), so production settings (round(160/dx)) can be compared.

usage
  python tools/live/validate_live.py short  <dumpdir> <tag> --proto <prototype>      # bit-level parity run
  python tools/live/validate_live.py long   <dumpdir> <tag> --proto <prototype>      # 25 Aug 2024 case, means
  python tools/live/validate_live.py domain <dumpdir> <tag> --proto <prototype>      # JS domain vs Python mirror
  python tools/live/validate_live.py spinup <dumpdir> <tag> [<tag> ...]              # settling times
  python tools/live/validate_live.py summary <dumpdir> <name> <tag> [<tag> ...] --out <file>   # collect reports
Writes <dumpdir>/<tag>_<mode>_report.json and prints it.
"""
import argparse, json, math, os, site, sys, time
if site.getusersitepackages() not in sys.path:
    sys.path.insert(0, site.getusersitepackages())
import numpy as np

RHO_CP = 1.16 * 1005.0


# ------------------------------------------------------------------ reference kernel
def reference_kernel(proto, nsw=16, oclip=False):
    """lbm_kernel_gen.gen_source(), unmodified except (optionally, each asserted):
    nsw != 16 : outflow-sponge width (the two literal 16s of the sponge rule)
    oclip     : the live solver's backflow-clipped outlets (same math as js/live/kernel.js, P.oclip != 0)"""
    import re
    code = os.path.join(proto, 'code')
    sys.path.insert(0, code)
    import lbm_kernel_gen as G
    src = G.gen_source()
    if nsw != 16:
        a, b = "    if dmin < 16:", "        rr = (16.0 - dmin)/16.0"
        assert src.count(a) == 1 and src.count(b) == 1, 'sponge rule not found exactly once'
        src = src.replace(a, f"    if dmin < {int(nsw)}:").replace(b, f"        rr = ({float(nsw)} - dmin)/{float(nsw)}")
    if oclip:
        a = "    usqn = uxn*uxn + uyn*uyn + uzn*uzn\n"
        assert src.count(a) == 1
        src = src.replace(a, a + "    uxc = uxn; uyc = uyn\n"
                          "    if i == 0 and inflow_w == 0 and uxc > 0.0: uxc = 0.0\n"
                          "    if i == nx-1 and inflow_e == 0 and uxc < 0.0: uxc = 0.0\n"
                          "    if j == 0 and inflow_s == 0 and uyc > 0.0: uyc = 0.0\n"
                          "    if j == ny-1 and inflow_n == 0 and uyc < 0.0: uyc = 0.0\n"
                          "    usqc = uxc*uxc + uyc*uyc + uzn*uzn\n")
        pat = re.compile(r"( +)cu = (.*)\n( +)(f\d+) = (fA\[\d+\*N \+ n\]) \+ ([0-9.]+)\*\(1\.0 - rn\)\*\(1\.0 \+ 3\.0\*cu \+ 4\.5\*cu\*cu - 1\.5\*usqn\)\n")
        def sub(m):
            cuc = m.group(2).replace('uxn', 'uxc').replace('uyn', 'uyc')
            return (f"{m.group(1)}cu = {m.group(2)}\n{m.group(1)}cuc = {cuc}\n"
                    f"{m.group(3)}{m.group(4)} = {m.group(5)} + {m.group(6)}*((1.0 + 3.0*cuc + 4.5*cuc*cuc - 1.5*usqc) - rn*(1.0 + 3.0*cu + 4.5*cu*cu - 1.5*usqn))\n")
        src, nrep = pat.subn(sub, src)
        assert nrep == 18, f'outflow lines patched: {nrep}'
    ns = {}
    exec(compile(src, f'<lbm_kernel_gen.gen_source nsw={nsw} oclip={oclip}>', 'exec'), ns)
    return ns['lbm_step'], G


class RefSim:
    """lbm3d.Sim driven from a JS dump (domain, parameters, state)."""

    def __init__(self, ddir, tag, proto, nsw=None, state='0', oclip=None):
        from numba import cuda
        self.cuda = cuda
        m = json.load(open(os.path.join(ddir, f'{tag}_domain.json')))
        self.meta = m
        nx, ny, nz = m['nx'], m['ny'], m['nz']
        self.nx, self.ny, self.nz, self.N = nx, ny, nz, nx * ny * nz
        rd = lambda nm, dt: np.fromfile(os.path.join(ddir, f'{tag}_{nm}.bin'), dt)
        self.kt = rd('kt', np.int16).reshape(ny, nx)
        self.ktop = rd('ktop', np.int16).reshape(ny, nx)
        self.cd = rd('cd', np.float32); self.q = rd('q', np.float32)
        self.uin = rd('uin', np.float32); self.vin = rd('vin', np.float32); self.kbg = rd('kbg', np.float32)
        p = m['params']
        nsw = p['nsw'] if nsw is None else nsw
        oclip = bool(p.get('oclip', 0)) if oclip is None else oclip
        self.lbm_step, self.G = reference_kernel(proto, nsw, oclip)
        self.nsw, self.oclip = nsw, oclip
        flag = np.zeros((nz, ny, nx), np.uint8)
        kk = np.arange(nz)[:, None, None]
        flag[kk < self.kt[None].astype(np.int32)] = 1
        flag[(kk >= self.kt[None].astype(np.int32)) & (kk < self.ktop[None].astype(np.int32))] = 2
        self.flag = flag
        self.fluid = flag == 0
        f0 = rd(f'f{state}', np.float32); g0 = rd(f'g{state}', np.float32)
        assert f0.size == 19 * self.N and g0.size == 7 * self.N
        self.f0 = f0
        td = cuda.to_device
        self.d_flag = td(flag.ravel()); self.d_kt = td(self.kt.ravel()); self.d_ktop = td(self.ktop.ravel())
        self.d_cd = td(self.cd); self.d_q = td(self.q); self.d_kbg = td(self.kbg); self.d_uin = td(self.uin); self.d_vin = td(self.vin)
        self.fA = td(f0); self.fB = td(f0.copy()); self.gA = td(g0); self.gB = td(g0.copy())
        self.theta = cuda.device_array(self.N, np.float32); self.uo = cuda.device_array(self.N, np.float32)
        self.vo = cuda.device_array(self.N, np.float32); self.wo = cuda.device_array(self.N, np.float32)
        self.acc = td(np.zeros(4 * self.N, np.float32)); self.nacc = 0
        # scalar arguments exactly as lbm3d.Sim.run passes them: python floats / ints
        self.args = (float(p['tau0']), float(p['cs2smag']), float(p['gbeta']), float(p['prt']), float(p['kfloor']),
                     int(m['inflow']['w']), int(m['inflow']['e']), int(m['inflow']['s']), int(m['inflow']['n']),
                     float(p['momfac']), float(p['srcscale']))
        self.nsp, self.sigmax = int(p['nsp']), float(p['sigmax'])
        self.tpb = 128; self.blocks = (self.N + self.tpb - 1) // self.tpb
        self.steps = 0

    def run(self, nsteps, out_last=False, acc=False):
        a = self.args
        for s in range(nsteps):
            do_out = 1 if (out_last and s == nsteps - 1) else 0
            do_acc = 1 if acc else 0
            self.lbm_step[self.blocks, self.tpb](self.fA, self.fB, self.gA, self.gB, self.d_flag, self.d_kt, self.d_ktop, self.d_cd,
                self.d_q, self.d_kbg, self.d_uin, self.d_vin, self.theta, self.uo, self.vo, self.wo, self.acc,
                self.nx, self.ny, self.nz, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8], a[9], a[10], do_out, do_acc, self.nsp, self.sigmax)
            self.fA, self.fB = self.fB, self.fA
            self.gA, self.gB = self.gB, self.gA
            self.nacc += do_acc
            self.steps += 1
        self.cuda.synchronize()

    def outputs(self):
        sh = (self.nz, self.ny, self.nx)
        return [x.copy_to_host().reshape(sh) for x in (self.theta, self.uo, self.vo, self.wo)]

    def mean(self):
        sh = (self.nz, self.ny, self.nx)
        return self.acc.copy_to_host().reshape(4, *sh) / max(self.nacc, 1)

    def surface(self, field3d):
        jj, ii = np.meshgrid(np.arange(self.ny), np.arange(self.nx), indexing='ij')
        return field3d[self.ktop.astype(np.int64), jj, ii]


def init_state_numpy(rs):
    """the initial state exactly as lbm3d.Sim.__init__ computes it (numpy float32), for comparison with resetFlow"""
    C19, W19 = rs.G.C19, rs.G.W19
    kt = rs.kt.astype(np.int64)
    zagl = np.clip(np.arange(rs.nz)[:, None, None] - kt[None], 0, rs.nz - 1)
    U = rs.uin[zagl]; V = rs.vin[zagl]
    f = np.empty((19, rs.nz, rs.ny, rs.nx), np.float32)
    usq = U * U + V * V
    for q, (cx, cy, cz) in enumerate(C19):
        cu = cx * U + cy * V
        f[q] = W19[q] * (1 + 3 * cu + 4.5 * cu * cu - 1.5 * usq)
    return f.ravel()


def rel_l2(a, b):
    a = np.asarray(a, np.float64); b = np.asarray(b, np.float64)
    den = np.linalg.norm(b)
    return float(np.linalg.norm(a - b) / den) if den > 0 else float(np.linalg.norm(a - b))


def cmp(a, b):
    a = np.asarray(a, np.float64); b = np.asarray(b, np.float64)
    return dict(rel_l2=rel_l2(a, b), max_abs=float(np.abs(a - b).max()), ref_absmax=float(np.abs(b).max()))


FN = lambda a: np.ascontiguousarray(a[..., ::-1, :])        # solver (j = 0 south) <-> north-up


# ------------------------------------------------------------------ modes
def ref_oclip(a):
    return None if a.ref_oclip == 'auto' else a.ref_oclip == '1'


def mode_short(a):
    rs = RefSim(a.dump, a.tag, a.proto, nsw=a.nsw, oclip=ref_oclip(a))
    res = json.load(open(os.path.join(a.dump, f'{a.tag}_result.json'))) if os.path.exists(os.path.join(a.dump, f'{a.tag}_result.json')) else {}
    steps = int(res.get('steps', a.steps))
    rep = dict(tag=a.tag, steps=steps, pre=res.get('pre'), nsw_reference=rs.nsw, nsw_webgpu=rs.meta['params']['nsw'],
               oclip_reference=rs.oclip, oclip_webgpu=bool(rs.meta['params'].get('oclip', 0)),
               grid=[rs.nx, rs.ny, rs.nz], dx=rs.meta['dx'], inflow=rs.meta['inflow'], fluid_cells=int(rs.fluid.sum()))
    if res.get('pre', 0) == 0:
        fi = init_state_numpy(rs)
        d = np.abs(fi.astype(np.float64) - rs.f0.astype(np.float64))
        ulp = d / np.maximum(np.abs(np.spacing(fi)), 1e-30)
        rep['init_state_vs_lbm3d'] = dict(max_abs=float(d.max()), max_ulp=float(ulp.max()), frac_bit_identical=float((d == 0).mean()))
    t0 = time.time()
    rs.run(steps, out_last=True)
    rep['ref_wall_s'] = time.time() - t0
    th, u, v, w = rs.outputs()
    fl = rs.fluid
    dbg = np.fromfile(os.path.join(a.dump, f'{a.tag}_dbg.bin'), np.float32).reshape(rs.nz, rs.ny, rs.nx, 4)
    rep['step_outputs_all_fluid_cells'] = {nm: cmp(dbg[..., c][fl], ref[fl]) for c, (nm, ref) in enumerate([('theta', th), ('ux', u), ('uy', v), ('uz', w)])}
    # uz is small (|uz| << |u|), so its relative L2 is no precision measure: also give its rms error over the rms speed
    sp_rms = float(np.sqrt((u[fl].astype(np.float64) ** 2 + v[fl] ** 2 + w[fl] ** 2).mean()))
    rep['step_outputs_all_fluid_cells']['uz']['rms_err_over_rms_speed'] = float(np.sqrt(((dbg[..., 3][fl] - w[fl]).astype(np.float64) ** 2).mean()) / sp_rms)
    # state after `steps` steps
    if os.path.exists(os.path.join(a.dump, f'{a.tag}_f1.bin')):
        f1 = np.fromfile(os.path.join(a.dump, f'{a.tag}_f1.bin'), np.float32).reshape(19, -1)
        g1 = np.fromfile(os.path.join(a.dump, f'{a.tag}_g1.bin'), np.float32).reshape(7, -1)
        fr = rs.fA.copy_to_host().reshape(19, -1); gr = rs.gA.copy_to_host().reshape(7, -1)
        m = fl.ravel()
        rep['state_f'] = cmp(f1[:, m], fr[:, m]); rep['state_g'] = cmp(g1[:, m], gr[:, m])
    # read(): pre-collision moments of the current state == what the reference outputs on the next step
    rs.run(1, out_last=True)
    th, u, v, w = rs.outputs()
    conv = rs.meta['conv']
    rd = lambda nm: np.fromfile(os.path.join(a.dump, f'{a.tag}_{nm}.bin'), np.float32).reshape(rs.ny, rs.nx)
    s_th, s_u, s_v = rd('surf'), rd('u'), rd('v')                         # north-up
    r_th, r_u, r_v = FN(rs.surface(th)), FN(rs.surface(u)) * conv, FN(rs.surface(v)) * conv
    rep['read_surface'] = dict(theta=cmp(s_th, r_th), u=cmp(s_u, r_u), v=cmp(s_v, r_v))
    nzv = res.get('info', {}).get('nzv') or int(round(420 / rs.meta['dx']))
    vol = np.fromfile(os.path.join(a.dump, f'{a.tag}_vol.bin'), np.uint8).reshape(nzv, rs.ny, rs.nx)
    thv = np.where(rs.fluid, th, 0)[:nzv]
    qv = np.clip(np.round(thv / 3.0 * 255), 0, 255).astype(np.int64)
    dq = np.abs(vol.astype(np.int64) - FN(qv))
    rep['read_volume_u8'] = dict(frac_equal=float((dq == 0).mean()), max_diff=int(dq.max()), nonzero=int((vol > 0).sum()))
    rep['theta_max_ref'] = float(th[fl].max()); rep['theta_max_webgpu_read'] = res.get('thetaMax')
    so = rep['step_outputs_all_fluid_cells']
    ok = all(v['rel_l2'] < 1e-4 for v in rep['read_surface'].values()) and all(so[k]['rel_l2'] < 1e-4 for k in ('theta', 'ux', 'uy'))
    rep['pass_rel_l2_lt_1e-4'] = bool(ok)
    rep['pass_note'] = 'criterion: near-surface theta/u/v from read() and full-field theta/ux/uy; uz reported, not gated'
    return rep


def surf_stats(th):
    return dict(max=float(np.nanmax(th)), mean=float(np.nanmean(th)), p99=float(np.nanpercentile(th, 99)))


def pattern_corr(a, b):
    a = np.asarray(a, np.float64).ravel(); b = np.asarray(b, np.float64).ravel()
    ok = np.isfinite(a) & np.isfinite(b)
    return float(np.corrcoef(a[ok], b[ok])[0, 1])


def heat_diag(mean, fluid, dx, q, dt, ktop):
    """run_case.py-style diagnostics on time-mean fields (physical units, solver orientation [z, j, i])"""
    th, u, v, w = mean
    def face(arr, sl):
        return float((arr[sl] * fluid[sl]).sum() * dx * dx)
    hw = face(th * u, (slice(None), slice(None), 0)); he = face(th * u, (slice(None), slice(None), -1))
    hs = face(th * v, (slice(None), 0, slice(None))); hn = face(th * v, (slice(None), -1, slice(None)))
    out_W = RHO_CP * (he - hw + hn - hs)
    in_W = float(q.astype(np.float64).sum() * RHO_CP * dx ** 3 / dt)
    content_MJ = float(np.where(fluid, th, 0).sum() * RHO_CP * dx ** 3 / 1e6)
    qw = face(u, (slice(None), slice(None), 0)); qe = face(u, (slice(None), slice(None), -1))
    qs = face(v, (slice(None), 0, slice(None))); qn = face(v, (slice(None), -1, slice(None)))
    thru = abs(qw) + abs(qe) + abs(qs) + abs(qn)
    return dict(heat_injected_MW=in_W / 1e6, heat_advected_out_MW=out_W / 1e6, heat_content_MJ=content_MJ,
                mass_imbalance_frac=abs(qw - qe + qs - qn) / max(thru, 1e-9))


def plume(th, ktop, dca_northup, dx):
    """plume_check.py on a north-up [z, row, col] mean theta"""
    nz, ny, nx = th.shape
    kk = np.arange(nz)[:, None, None]
    fluid = kk >= ktop[None]
    thp = np.where(fluid, np.clip(th, 0, None), 0)
    yy, xx = np.mgrid[0:ny, 0:nx]
    X = (xx + 0.5) * dx; Y = (ny - yy - 0.5) * dx
    sx = (dca_northup * X).sum() / dca_northup.sum(); sy = (dca_northup * Y).sum() / dca_northup.sum()
    tot = thp.sum(axis=0); cx = (tot * X).sum() / tot.sum(); cy = (tot * Y).sum() / tot.sum()
    zagl = (kk - ktop[None] + 0.5) * dx
    return dict(displacement_m=float(np.hypot(cx - sx, cy - sy)), bearing_deg=float((np.degrees(np.arctan2(cx - sx, cy - sy)) + 360) % 360),
                heat_weighted_height_agl_m=float((thp * zagl).sum() / thp.sum()))


def mode_long(a):
    rs = RefSim(a.dump, a.tag, a.proto, nsw=a.nsw, oclip=ref_oclip(a))
    res = json.load(open(os.path.join(a.dump, f'{a.tag}_result.json')))
    nspin, navg, dt, dx = res['nspin'], res['navg'], res['dt'], rs.meta['dx']
    conv = rs.meta['conv']
    t0 = time.time()
    rs.run(nspin)
    rs.run(navg, acc=True)
    wall = time.time() - t0
    mr = rs.mean(); mr[1:] *= conv
    acc = np.fromfile(os.path.join(a.dump, f'{a.tag}_acc.bin'), np.float32).reshape(rs.nz, rs.ny, rs.nx, 4)
    mw = np.moveaxis(acc, -1, 0).astype(np.float64) / navg; mw[1:] *= conv
    fl = rs.fluid
    s_ref = FN(rs.surface(mr[0])); s_web = FN(rs.surface(mw[0]))
    sm = np.fromfile(os.path.join(a.dump, f'{a.tag}_surfmean.bin'), np.float32).reshape(3, rs.ny, rs.nx)
    ktop_n = FN(rs.ktop).astype(np.int64)
    dca = np.fromfile(os.path.join(a.dump, f'{a.tag}_q.bin'), np.float32).reshape(rs.ny, rs.nx)  # source map (solver) ~ roof area
    dca_n = FN(dca)
    rep = dict(tag=a.tag, nspin=nspin, navg=navg, dt=dt, sim_s=(nspin + navg) * dt, avg_window_s=navg * dt, dx=dx,
               nsw_reference=rs.nsw, nsw_webgpu=rs.meta['params']['nsw'], oclip_reference=rs.oclip, oclip_webgpu=bool(rs.meta['params'].get('oclip', 0)),
               ref_wall_s=wall, ref_mlups=rs.N * (nspin + navg) / wall / 1e6,
               webgpu_wall_s=res.get('wall_s'))
    rep['surface_mean_theta'] = dict(pattern_corr=pattern_corr(s_web, s_ref), webgpu=surf_stats(s_web), numba=surf_stats(s_ref),
                                     rel_l2=rel_l2(s_web, s_ref), max_ratio=float(s_web.max() / s_ref.max()),
                                     sampled_read_vs_acc_corr=pattern_corr(sm[0], s_web))
    su_w, sv_w = FN(rs.surface(mw[1])), FN(rs.surface(mw[2])); su_r, sv_r = FN(rs.surface(mr[1])), FN(rs.surface(mr[2]))
    rep['surface_mean_wind'] = dict(u_corr=pattern_corr(su_w, su_r), v_corr=pattern_corr(sv_w, sv_r),
                                    speed_mean_webgpu=float(np.hypot(su_w, sv_w).mean()), speed_mean_numba=float(np.hypot(su_r, sv_r).mean()))
    rep['volume_mean_theta_corr'] = pattern_corr(np.where(fl, mw[0], np.nan), np.where(fl, mr[0], np.nan))
    rep['heat_budget_webgpu'] = heat_diag(mw, fl, dx, rs.q, dt, rs.ktop)
    rep['heat_budget_numba'] = heat_diag(mr, fl, dx, rs.q, dt, rs.ktop)
    rep['plume_webgpu'] = plume(FN(mw[0]), ktop_n, dca_n, dx); rep['plume_numba'] = plume(FN(mr[0]), ktop_n, dca_n, dx)
    met = rs.meta['met']; rep['wind_toward_deg'] = (met['dirFrom'] + 180) % 360
    # archived 10 m baseline (existing campuses at 400 W/m2, same weather), block-averaged to dx
    base = os.path.join(a.proto, 'sim', 'baseline')
    if os.path.exists(os.path.join(base, 'theta2m_mean.npy')) and abs(rs.meta['dx'] / 10 - round(rs.meta['dx'] / 10)) < 1e-9:
        r = int(round(dx / 10))
        b2 = np.load(os.path.join(base, 'theta2m_mean.npy')).astype(np.float64)
        bavg = b2.reshape(rs.ny, r, rs.nx, r).mean(axis=(1, 3))
        # same layer depth: 10 m 3-D mean theta averaged over the first r fluid cells above the surface, then block-averaged
        th3 = np.load(os.path.join(base, 'mean_theta.npy')).astype(np.float32)
        kt10 = np.load(os.path.join(base, 'ktop.npy')).astype(np.int64)
        jj, ii = np.meshgrid(np.arange(600), np.arange(600), indexing='ij')
        lay = np.mean([th3[np.minimum(kt10 + m, 63), jj, ii] for m in range(r)], axis=0)
        lavg = lay.reshape(rs.ny, r, rs.nx, r).mean(axis=(1, 3))
        rep['vs_10m_baseline'] = dict(
            first_cell_blockavg=dict(corr_webgpu=pattern_corr(s_web, bavg), corr_numba=pattern_corr(s_ref, bavg),
                                     peak_ratio_webgpu=float(s_web.max() / bavg.max()), peak_ratio_numba=float(s_ref.max() / bavg.max()),
                                     mean_ratio_webgpu=float(s_web.mean() / bavg.mean()), baseline_peak=float(bavg.max()), baseline_10m_peak=float(b2.max())),
            same_layer_0_to_dx_blockavg=dict(corr_webgpu=pattern_corr(s_web, lavg), peak_ratio_webgpu=float(s_web.max() / lavg.max()),
                                             mean_ratio_webgpu=float(s_web.mean() / lavg.mean()), baseline_peak=float(lavg.max())))
        np.save(os.path.join(a.dump, f'{a.tag}_cmp_surfaces.npy'), np.stack([s_web, s_ref, bavg, lavg]).astype(np.float32))
    return rep


def mode_domain(a):
    """JS-built domain vs an independent numpy mirror of build_domain on the same static10 data, and the mirror at
    dx = 10 m on the unquantized inputs vs the prototype's own lbm3d.build_domain."""
    m = json.load(open(os.path.join(a.dump, f'{a.tag}_domain.json')))
    nx, ny, dx = m['nx'], m['ny'], m['dx']
    here = os.path.dirname(os.path.abspath(__file__))
    live = os.path.normpath(os.path.join(here, '..', '..', 'data', 'live'))
    meta = json.load(open(os.path.join(live, 'static10.json')))
    import gzip
    raw = gzip.decompress(open(os.path.join(live, 'static10.bin.gz'), 'rb').read())
    P = meta['planes']; n = 360000
    lo = np.frombuffer(raw, np.uint8, n, P['dem']['offset']).astype(np.int64)
    hi = np.frombuffer(raw, np.uint8, n, P['dem']['offset'] + n).astype(np.int64)
    d = (lo | (hi << 8)); d = np.where(d >= 32768, d - 65536, d).reshape(600, 600)
    demcm = np.cumsum(d, axis=1)
    lc = np.frombuffer(raw, np.uint8, n, P['lc']['offset']).reshape(600, 600)
    f25 = np.frombuffer(raw, np.uint8, n, P['frac25']['offset']).astype(np.int64).reshape(600, 600)
    hcm = np.frombuffer(raw, '<u2', n, P['hmean']['offset']).astype(np.int64).reshape(600, 600)
    d4 = np.frombuffer(raw, np.uint8, n, P['dca4']['offset']).astype(np.int64).reshape(600, 600)
    z0tab = {int(k): v for k, v in meta['z0_by_class'].items()}

    def mirror(dem_sum_fn, cnt, hsum, dca, lcodes, dxm, r, nz_req):
        z0 = np.vectorize(lambda c: z0tab.get(int(c), meta['z0_default']))(lcodes).astype(np.float64)
        cd10 = (0.4 / np.log((0.5 * dxm) / z0)) ** 2
        B = lambda x: x.reshape(600 // r, r, 600 // r, r).sum(axis=(1, 3))
        dem = dem_sum_fn(B)
        c = B(cnt); frac = c / (25 * r * r); hmean = np.where(c > 0, B(hsum) / np.maximum(c, 1) / 100, 0)
        cd = B(cd10) / (r * r)
        zbase = math.floor(dem.min()) - dxm
        kt = np.maximum(np.ceil((dem - zbase) / dxm - 0.5).astype(np.int64), 1)
        nb = np.where(frac >= 0.5, np.round(hmean / dxm), 0).astype(np.int64)
        return dict(dem=dem, kt=kt, nb=nb, ktop=kt + nb, cd=cd.astype(np.float32), zbase=zbase, dca=B(dca))
    r = int(round(dx / 10))
    mm = mirror(lambda B: B(demcm) / (100 * r * r), f25, f25 * hcm, d4 * 4, lc, dx, r, m['nz'])
    kt_js = FN(np.fromfile(os.path.join(a.dump, f'{a.tag}_kt.bin'), np.int16).reshape(ny, nx)).astype(np.int64)
    ktop_js = FN(np.fromfile(os.path.join(a.dump, f'{a.tag}_ktop.bin'), np.int16).reshape(ny, nx)).astype(np.int64)
    cd_js = FN(np.fromfile(os.path.join(a.dump, f'{a.tag}_cd.bin'), np.float32).reshape(ny, nx))
    rep = dict(tag=a.tag, dx=dx, zbase_js=m['zbase'], zbase_mirror=mm['zbase'], kt_equal=bool((kt_js == mm['kt']).all()),
               ktop_equal=bool((ktop_js == mm['ktop']).all()), ktop_mismatch=int((ktop_js != mm['ktop']).sum()),
               cd_max_rel=float(np.abs(cd_js.astype(np.float64) / mm['cd'] - 1).max()), kt_range=[int(mm['kt'].min()), int(mm['kt'].max())],
               ktop_max=int(mm['ktop'].max()), building_columns=int((mm['nb'] > 0).sum()), nz=m['nz'],
               note='campus-free columns only if the dump had campuses' if m.get('campusInfo') else 'no campuses in this dump')
    # the mirror at 10 m on the original inputs vs lbm3d.build_domain (the prototype's own function)
    try:
        sys.path.insert(0, os.path.join(a.proto, 'code'))
        import lbm3d
        dom = lbm3d.build_domain(nz=64)
        dem10 = np.load(os.path.join(a.proto, 'data', 'dem.npy')).astype(np.float64)
        z = np.load(os.path.join(a.proto, 'sim', '_domain_base.npz'))
        cnt10 = np.round(z['frac'] * 25).astype(np.int64)
        m10 = mirror(lambda B: B(dem10), cnt10, cnt10 * np.round(z['hmean'].astype(np.float64) * 100), z['dcarea'], lc, 10.0, 1, 64)
        rep['mirror_10m_vs_lbm3d_build_domain'] = dict(
            zbase=[m10['zbase'], dom['zbase']], kt_equal=bool((FN(dom['kt']) == m10['kt']).all()),
            nb_mismatch=int((FN(dom['nb']) != m10['nb']).sum()), cd_max_rel=float(np.abs(FN(dom['cd']).astype(np.float64) / m10['cd'] - 1).max()))
    except Exception as e:      # lbm3d needs cv2/numba; skip if unavailable
        rep['mirror_10m_vs_lbm3d_build_domain'] = f'skipped: {e}'
    return rep


def settle_time(t, x, tol, tail_frac=0.25):
    """first time after which the 60-s running mean of x stays within tol (relative) of its mean over the last
    tail_frac of the record"""
    t = np.asarray(t); x = np.asarray(x, np.float64)
    ref = x[t >= t[-1] * (1 - tail_frac)].mean()
    w = max(1, int(round(60 / max(np.median(np.diff(t)), 1e-9))))
    rm = np.convolve(x, np.ones(w) / w, mode='same')
    bad = np.where(np.abs(rm - ref) > tol * abs(ref))[0]
    bad = bad[bad < len(t) - w // 2]
    return float(t[bad[-1] + 1]) if len(bad) else float(t[0]), float(ref)


def mode_spinup(a):
    out = {}
    for tag in a.tags:
        ts = json.load(open(os.path.join(a.dump, f'{tag}_ts.json')))
        res = json.load(open(os.path.join(a.dump, f'{tag}_result.json')))
        t = np.array([r['t'] for r in ts]); dt = res['dt']
        rec = dict(dt=dt, T=float(t[-1]))
        for key, tol in (('windMean', 0.02), ('ke', 0.03), ('heatMJ', 0.05), ('surfMean', 0.05)):
            ts_, ref = settle_time(t, [r[key] for r in ts], tol)
            rec[key] = dict(settle_s=ts_, settle_steps=int(round(ts_ / dt)), final=ref, tol=tol)
        out[tag] = rec
    return out


def mode_summary(a):
    """collect <tag>_*report.json and <tag>_result.json of the given tags (first tag = output name)"""
    import glob
    out = {}
    for tag in a.tags[1:]:
        rec = {}
        for f in sorted(glob.glob(os.path.join(a.dump, f'{tag}_*report.json')) + glob.glob(os.path.join(a.dump, f'{tag}_result.json'))):
            key = os.path.basename(f)[len(tag) + 1:-5]
            d = json.load(open(f))
            if key == 'result':
                d = {k: v for k, v in d.items() if k not in ('stack',)}
            rec[key] = d
        out[tag] = rec
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['short', 'long', 'domain', 'spinup', 'summary'])
    ap.add_argument('dump')
    ap.add_argument('tags', nargs='+')
    ap.add_argument('--proto', default=None)
    ap.add_argument('--nsw', type=int, default=None, help='outflow-sponge width of the reference (default: the WebGPU run\'s)')
    ap.add_argument('--steps', type=int, default=300)
    ap.add_argument('--ref-oclip', choices=['auto', '0', '1'], default='auto', help='reference outlet: auto = as the WebGPU run')
    ap.add_argument('--suffix', default='', help='appended to the report name')
    ap.add_argument('--out', default=None, help='output path (default <dump>/<tag>_<mode>_report.json)')
    a = ap.parse_args()
    a.tag = a.tags[0]
    fn = dict(short=mode_short, long=mode_long, domain=mode_domain, spinup=mode_spinup, summary=mode_summary)[a.mode]
    rep = fn(a)
    path = a.out or os.path.join(a.dump, f'{a.tag}_{a.mode}{a.suffix}_report.json')
    json.dump(rep, open(path, 'w'), indent=1)
    print(json.dumps(rep, indent=1))
    print('->', path)


if __name__ == '__main__':
    main()
