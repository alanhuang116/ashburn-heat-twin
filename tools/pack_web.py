#!/usr/bin/env python3
"""Pack the web data for the Ashburn Heat Twin from the prototype's input data and archived solver runs.

  python tools/pack_web.py --proto <prototype folder with data/ and sim/> --assets <render-asset folder> --out data

Everything is written as gzip-compressed little-endian arrays (*.bin.gz, decoded in the browser with
DecompressionStream) plus one manifest (data/manifest.json) that records dtype, shape, scale and provenance.
Scene frame (metres): x = UTM18N easting - 286780, z = -(northing - 4320800) (+z = south), y = elevation - 83.53166.
All 2-D grids are north-up (row 0 = north edge z = -2995), column 0 = west edge (x = -2995).
"""
import argparse, gzip, json, math, os, struct
import numpy as np
from PIL import Image

ORIGIN_E, ORIGIN_N = 286780.0, 4320800.0
BASE_ELEV = 83.53166198730469
X0, X1, Z0, Z1 = -2995.0, 3005.0, -2995.0, 3005.0
UTM_BOX = (283785.0, 4317795.0, 289785.0, 4323795.0)


def loc(e, n):
    return e - ORIGIN_E, -(n - ORIGIN_N)


class Packer:
    def __init__(self, out):
        self.out = out
        self.files = {}
        os.makedirs(out, exist_ok=True)

    def put(self, name, arr, **meta):
        """write arr (any numpy dtype) as gzip'd raw bytes; record dtype/shape + extra meta (scale, offset, nodata)."""
        path = os.path.join(self.out, name)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        a = np.ascontiguousarray(arr)
        if a.dtype.byteorder == '>':
            a = a.byteswap().newbyteorder()
        with open(path, 'wb') as fh:
            with gzip.GzipFile(fileobj=fh, mode='wb', compresslevel=9, mtime=0) as f:
                f.write(a.tobytes())
        rec = dict(dtype=str(a.dtype), shape=list(a.shape), bytes=os.path.getsize(path))
        rec.update(meta)
        self.files[name] = rec
        print(f'  {name:42s} {str(a.dtype):8s} {str(list(a.shape)):18s} {rec["bytes"] / 1e6:6.2f} MB')
        return rec


def block_mean(a, f):
    h, w = a.shape[-2:]
    sh = a.shape[:-2] + (h // f, f, w // f, f)
    return a.reshape(sh).mean(axis=(-3, -1))


def q16(a, scale, nodata=None):
    """float -> uint16 with value = q * scale (nan -> nodata)"""
    v = np.round(np.nan_to_num(a, nan=0.0) / scale)
    v = np.clip(v, 0, 65534 if nodata is not None else 65535).astype(np.uint16)
    if nodata is not None:
        v[~np.isfinite(a)] = nodata
    return v


def q8(a, vmax):
    return np.clip(np.round(np.nan_to_num(a, nan=0.0) / vmax * 255.0), 0, 255).astype(np.uint8)


# ------------------------------------------------------------------ geometry helpers
def dp_simplify(pts, tol):
    """Douglas-Peucker on an (n,2) array"""
    if len(pts) < 3:
        return pts
    a, b = pts[0], pts[-1]
    ab = b - a
    L = np.hypot(*ab)
    if L < 1e-9:
        d = np.hypot(*(pts - a).T)
    else:
        d = np.abs(np.cross(ab, pts - a)) / L
    i = int(np.argmax(d))
    if d[i] > tol:
        left = dp_simplify(pts[:i + 1], tol)
        right = dp_simplify(pts[i:], tol)
        return np.vstack([left[:-1], right])
    return np.vstack([a, b])


def clip_polyline(pts, pad=40.0):
    """split a polyline into pieces whose points lie inside the box (+pad)"""
    inside = (pts[:, 0] > X0 - pad) & (pts[:, 0] < X1 + pad) & (pts[:, 1] > Z0 - pad) & (pts[:, 1] < Z1 + pad)
    pieces, cur = [], []
    for p, ok in zip(pts, inside):
        if ok:
            cur.append(p)
        elif cur:
            if len(cur) > 1:
                pieces.append(np.array(cur))
            cur = []
    if len(cur) > 1:
        pieces.append(np.array(cur))
    return pieces


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--proto', required=True, help='prototype folder containing data/ and sim/')
    ap.add_argument('--assets', required=True, help='render-asset folder (buildings.json, trees.bin, sim/*_lines.bin, meta.json)')
    ap.add_argument('--out', default=os.path.join(os.path.dirname(__file__), '..', 'data'))
    args = ap.parse_args()
    D = os.path.join(args.proto, 'data')
    S = os.path.join(args.proto, 'sim')
    A = args.assets
    P = Packer(os.path.normpath(args.out))
    M = {}  # manifest

    summary = json.load(open(os.path.join(S, 'summary.json')))
    ameta = json.load(open(os.path.join(A, 'meta.json')))

    # ---------------------------------------------------------------- terrain, land cover, streams
    print('terrain')
    dem = np.load(os.path.join(D, 'dem.npy')).astype(np.float64)          # 600x600 m NAVD88 north-up
    yloc = dem - BASE_ELEV
    P.put('terrain.bin.gz', np.round(yloc * 100).astype(np.int16), scale=0.01, unit='m (scene y = elevation - 83.53166)')
    lc = np.load(os.path.join(D, 'landcover.npy')).astype(np.uint8)
    P.put('landcover.bin.gz', lc, note='ESA WorldCover 2021 class codes')
    st = np.load(os.path.join(S, '_stream10.npy')).astype(np.uint8)
    P.put('streams10.bin.gz', (st > 0).astype(np.uint8), note='stream-cell mask (10 m) used for stream-length metrics')

    def ground(x, z):
        c = int(np.clip((x - X0) / 10.0, 0, 599)); r = int(np.clip((z - Z0) / 10.0, 0, 599))
        return float(yloc[r, c])

    # ---------------------------------------------------------------- imagery (NAIP 2023, 1 m -> 4096 / 1024 px)
    print('imagery')
    Image.MAX_IMAGE_PIXELS = None
    im = Image.open(os.path.join(D, 'naip_rgb.jpg')).convert('RGB')
    small = np.asarray(im.resize((1500, 1500), Image.BILINEAR), np.float32) / 255.0
    lo = np.percentile(small.reshape(-1, 3), 0.4, axis=0)
    hi = np.percentile(small.reshape(-1, 3), 99.6, axis=0)
    for size, q, name in ((4096, 82, 'imagery_4k.jpg'), (1024, 80, 'imagery_1k.jpg')):
        a = np.asarray(im.resize((size, size), Image.LANCZOS), np.float32) / 255.0
        a = np.clip((a - lo) / (hi - lo), 0, 1) ** 0.92
        Image.fromarray((a * 255 + 0.5).astype(np.uint8)).save(os.path.join(P.out, name), quality=q, optimize=True, progressive=True)
        P.files[name] = dict(bytes=os.path.getsize(os.path.join(P.out, name)), shape=[size, size, 3],
                             note='USDA NAIP 2023-10-11 true colour, percentile-stretched')
        print(f'  {name:42s} {os.path.getsize(os.path.join(P.out, name)) / 1e6:6.2f} MB')

    # ---------------------------------------------------------------- buildings (+ campus id from the 10 m campus raster)
    print('buildings')
    cid = np.load(os.path.join(S, 'campus_id_10m.npy'))
    B = json.load(open(os.path.join(A, 'buildings.json')))['b']
    out_b = []
    for b in B:
        ring = np.array(b['r'][0]).reshape(-1, 2)
        cx, cz = ring.mean(0)
        r_ = int(np.clip((cz - Z0) / 10, 0, 599)); c_ = int(np.clip((cx - X0) / 10, 0, 599))
        camp = -1
        if b['dc']:
            win = cid[max(0, r_ - 2):r_ + 3, max(0, c_ - 2):c_ + 3].ravel()
            win = win[win >= 0]
            if win.size:
                camp = int(np.bincount(win).argmax())
        rec = dict(r=[[round(v, 1) for v in rr] for rr in b['r']], y0=round(b['y0'], 2), y1=round(b['y1'], 2), h=round(b['h'], 1),
                   a=int(b['a']))
        if b['dc']:
            rec['dc'] = 1
            rec['c'] = camp
        if b.get('res'):
            rec['res'] = 1
        out_b.append(rec)
    with open(os.path.join(P.out, 'buildings.json'), 'w') as f:
        json.dump({'b': out_b}, f, separators=(',', ':'))
    P.files['buildings.json'] = dict(bytes=os.path.getsize(os.path.join(P.out, 'buildings.json')), n=len(out_b),
                                     note='footprint rings [x,z,...] (scene m); y0 base, y1 roof (scene y); dc=data centre, c=campus id, res=residential')
    print('  buildings.json', len(out_b), os.path.getsize(os.path.join(P.out, 'buildings.json')) / 1e6, 'MB')

    # ---------------------------------------------------------------- trees (architectural-model crowns), thinned
    print('trees')
    t = np.fromfile(os.path.join(A, 'trees.bin'), dtype=np.float32).reshape(-1, 8)
    core = (t[:, 0] > X0 + 3) & (t[:, 0] < X1 - 3) & (t[:, 2] > Z0 + 3) & (t[:, 2] < Z1 - 3)
    t = t[core]
    rng = np.random.default_rng(7)
    # thin to one crown per 7 m cell (largest wins), then cap
    cell = (np.floor((t[:, 0] - X0) / 7).astype(np.int64) * 10000 + np.floor((t[:, 2] - Z0) / 7).astype(np.int64))
    order = np.lexsort((-t[:, 3], cell))
    keep = np.ones(len(order), bool)
    keep[1:] = cell[order][1:] != cell[order][:-1]
    t = t[order[keep]]
    if len(t) > 110000:
        t = t[rng.choice(len(t), 110000, replace=False)]
    lum = 0.299 * t[:, 5] + 0.587 * t[:, 6] + 0.114 * t[:, 7]
    tone = np.clip((lum - np.percentile(lum, 2)) / (np.percentile(lum, 98) - np.percentile(lum, 2)), 0, 1)
    rec = np.zeros(len(t), dtype=[('x', '<i2'), ('z', '<i2'), ('y', '<i2'), ('r', 'u1'), ('h', 'u1'), ('tone', 'u1'), ('pad', 'u1')])
    rec['x'] = np.round(t[:, 0] * 5); rec['z'] = np.round(t[:, 2] * 5); rec['y'] = np.round(t[:, 1] * 100)
    rec['r'] = np.clip(np.round(t[:, 3] * 10), 10, 255); rec['h'] = np.clip(np.round(t[:, 4] * 4), 8, 255)
    rec['tone'] = np.round(tone * 255)
    P.put('trees.bin.gz', rec.view(np.uint8), n=int(len(t)), record='int16 x*5, int16 z*5, int16 y*100, u8 crown_r*10, u8 height*4, u8 tone, u8 pad')

    # ---------------------------------------------------------------- OSM waterways + major roads (scene coords)
    print('osm lines')
    from pyproj import Transformer
    tr = Transformer.from_crs('EPSG:4326', 'EPSG:32618', always_xy=True)
    osm = json.load(open(os.path.join(D, 'raw', 'osm', 'osm_aoi.json')))['elements']
    ctx = json.load(open(os.path.join(D, 'raw', 'osm', 'osm_aoi_context.json')))['elements']

    def way_pts(e):
        g = e.get('geometry') or []
        if len(g) < 2:
            return None
        E, N = tr.transform([p['lon'] for p in g], [p['lat'] for p in g])
        x, z = loc(np.array(E), np.array(N))
        return np.stack([x, z], 1)

    water_lines, water_polys, roads = [], [], []
    for e in osm:
        tg = e.get('tags', {})
        if e['type'] != 'way':
            continue
        if tg.get('waterway') in ('stream', 'river'):
            p = way_pts(e)
            if p is None:
                continue
            for piece in clip_polyline(p):
                s = dp_simplify(piece, 2.0)
                water_lines.append(dict(n=tg.get('name', ''), k=tg['waterway'], p=np.round(s, 1).ravel().tolist()))
        elif tg.get('natural') == 'water':
            p = way_pts(e)
            if p is None or len(p) < 4:
                continue
            if (p[:, 0].max() < X0 or p[:, 0].min() > X1 or p[:, 1].max() < Z0 or p[:, 1].min() > Z1):
                continue
            s = dp_simplify(p, 1.5)
            water_polys.append(np.round(s, 1).ravel().tolist())
    rank = {'motorway': 0, 'trunk': 1, 'primary': 1, 'secondary': 2}
    for e in ctx:
        tg = e.get('tags', {})
        hw = tg.get('highway')
        if e['type'] != 'way' or hw not in rank:
            continue
        p = way_pts(e)
        if p is None:
            continue
        for piece in clip_polyline(p, pad=10):
            s = dp_simplify(piece, 3.0)
            roads.append(dict(k=rank[hw], n=tg.get('ref') or tg.get('name', ''), p=np.round(s, 1).ravel().tolist()))
    lines = dict(water=water_lines, ponds=water_polys, roads=roads)
    with open(os.path.join(P.out, 'lines.json'), 'w') as f:
        json.dump(lines, f, separators=(',', ':'))
    P.files['lines.json'] = dict(bytes=os.path.getsize(os.path.join(P.out, 'lines.json')),
                                 note='OpenStreetMap waterways (stream/river), ponds, motorway/primary/secondary roads; scene coords')
    print('  lines.json', len(water_lines), 'water', len(water_polys), 'ponds', len(roads), 'roads', os.path.getsize(os.path.join(P.out, 'lines.json')) / 1e6, 'MB')

    # ---------------------------------------------------------------- labels
    print('labels')
    LA = json.load(open(os.path.join(args.proto, 'figs', 'overlays', 'labels_aoi.json')))['labels']
    labels = []
    keep_text = {'Ashburn': ('place', 1), 'Sterling': ('place', 2), 'Moorefield': ('place', 2), 'Dulles Town Center': ('place', 2),
                 'Loudoun Valley Estates': ('place', 3), 'Broad Run': ('stream', 1), 'Beaverdam Run': ('stream', 2), 'Cabin Branch': ('stream', 1),
                 'VA-267 Dulles Greenway': ('road', 2), 'VA-28 Sully Rd': ('road', 2), 'Loudoun County Pkwy': ('road', 2), 'Waxpool Rd': ('road', 3),
                 'Pacific Blvd': ('road', 3), '1757 Golf Club': ('landmark', 3)}
    seen = set()
    for l in LA:
        if l['text'] not in keep_text or l['text'] in seen:
            continue
        seen.add(l['text'])
        x, z = loc(l['x_utm'], l['y_utm'])
        x = float(np.clip(x, X0 + 60, X1 - 60)); z = float(np.clip(z, Z0 + 60, Z1 - 60))
        kind, prio = keep_text[l['text']]
        labels.append(dict(text=l['text'], kind=kind, prio=prio, x=round(x, 1), z=round(z, 1), y=round(ground(x, z), 1), src=l['source']))
    # Dulles airport lies ~8 km south: direction tag at the south edge
    ex, ez = loc(287936.0, 4316600.0)
    labels.append(dict(text='Dulles Airport', kind='direction', prio=1, x=round(ex, 1), z=round(ez, 1), y=0.0,
                       src='IEM ASOS station KIAD; OSM aerodrome'))

    # ---------------------------------------------------------------- campuses (existing) + scenario sites
    print('campuses')
    cm = json.load(open(os.path.join(S, 'campus_metrics.json')))
    byid = {r['campus_id']: r for r in cm['campuses'] if r.get('kind') == 'existing'}
    camps = []
    for c in ameta['campuses']:
        r = byid.get(c['id'], {})
        camps.append(dict(id=c['id'], x=c['x'], z=c['z'], y=c['y'], bbox=c['bbox'], roof_m2=c['area_m2'], n=c['n'], op=c['op'],
                          mw=r.get('heat_rejection_MW'), a05=r.get('area_gt_0p5K_ha'), a025=r.get('area_gt_0p25K_ha'), a1=r.get('area_gt_1K_ha'),
                          peak=r.get('peak_downwind_K'), homes025=r.get('residential_bldgs_gt_0p25K'), homes05=r.get('residential_bldgs_gt_0p5K'),
                          homes01=r.get('residential_bldgs_gt_0p1K'), nearest_home=r.get('nearest_home_m'), stream05=r.get('stream_km_gt_0p5K'),
                          forest05=r.get('forest_ha_gt_0p5K'), tracer=('c%d' % c['id']) in np.load(os.path.join(S, 'campus_tracers', 'tracers_existing_baseline.npz')).files))
    scen_geo = json.load(open(os.path.join(A, 'scenarios.json')))
    sm = json.load(open(os.path.join(S, 'scenario_metrics.json')))
    sites = {}
    for k in ('A', 'B', 'C'):
        g = scen_geo[k]; m = sm['scenario_' + k]
        sites[k] = dict(label=g['label'], center=g['center'], bbox=g['bbox'], ground=g['ground'],
                        halls=[dict(r=h['r'][0], y0=h['y0'], y1=h['y1'], h=h['h']) for h in g['halls']],
                        mw=m['new_heat_MW'], roof_ha=round(m['new_roof_ha'], 2), flux=round(m['new_flux_wm2'], 1),
                        a05=m['area_gt_0p5K_ha'], a1=m['area_gt_1K_ha'], a025=m['area_gt_0p25K_ha'],
                        homes025=m['residential_bldgs_gt_0p25K'], homes05=m['residential_bldgs_gt_0p5K'],
                        peak=round(m['peak_dT_outside_footprint_K'], 2), stream05=m['stream_length_gt_0p5K_km'], stream025=m['stream_length_gt_0p25K_km'],
                        forest05=m['tree_area_gt_0p5K_ha'], intake=m['existing_dc_bldgs_intake_warmed_gt_0p5K'],
                        intake_campuses=m['existing_campuses_intake_warmed_gt_0p5K'],
                        down01=round(m['mean_dT_downwind_0_1km_K'], 2), down12=round(m['mean_dT_downwind_1_2km_K'], 2), down23=round(m['mean_dT_downwind_2_3km_K'], 2))

    # ---------------------------------------------------------------- archived runs (10 m solver)
    print('runs')
    VOLMAX = 4.0
    runs = {}
    for case in ('baseline', 'scenario_A', 'scenario_B', 'scenario_C'):
        d = os.path.join(S, case)
        meta = json.load(open(os.path.join(d, 'meta.json')))
        t2 = np.load(os.path.join(d, 'theta2m_mean.npy')).astype(np.float32)
        th = np.load(os.path.join(d, 'mean_theta.npy'), mmap_mode='r')[:40].astype(np.float32)   # 40 levels x 600 x 600 (10 m)
        ktop = np.load(os.path.join(d, 'ktop.npy')).astype(np.int32)
        # zero inside solids so trilinear sampling fades to 0 at the ground
        kk = np.arange(40)[:, None, None]
        th = np.where(kk >= ktop[None], th, 0.0)
        th = np.nan_to_num(th, nan=0.0)
        vol = block_mean(th, 2)                                                                   # 40 x 300 x 300 (20 m x 10 m)
        u2 = np.load(os.path.join(d, 'u2m_mean.npy')).astype(np.float32)
        v2 = np.load(os.path.join(d, 'v2m_mean.npy')).astype(np.float32)
        u4, v4 = block_mean(np.nan_to_num(u2), 4), block_mean(np.nan_to_num(v2), 4)            # 150 x 150 (40 m)
        rec = dict(met=meta.get('met'), t2m_max=float(np.nanmax(t2)), t2m_p99=float(np.nanpercentile(t2, 99)), vol_max=float(vol.max()))
        P.put(f'runs/{case}/t2m.bin.gz', q16(t2, 1e-4), scale=1e-4, unit='K', note='near-surface (0-10 m layer) warming, time mean of last 600 s')
        P.put(f'runs/{case}/vol.bin.gz', q8(vol, VOLMAX), scale=VOLMAX / 255, unit='K', levels_m='zbase 50 m NAVD88, dz 10 m, 40 levels',
              note='time-mean potential-temperature anomaly, 20 m horizontal')
        P.put(f'runs/{case}/wind.bin.gz', np.round(np.stack([u4, v4]) * 1000).astype(np.int16), scale=0.001, unit='m/s',
              note='near-surface time-mean wind [u east, v north], 40 m')
        if case != 'baseline':
            dt = np.load(os.path.join(d, 'delta_theta2m_mean.npy')).astype(np.float32)
            P.put(f'runs/{case}/dt2m.bin.gz', q16(np.clip(dt, 0, None), 1e-4), scale=1e-4, unit='K', note='scenario minus baseline, near-surface')
        # streamlines through the mean wind (from the render assets): int32 n, per line int32 m, f32 seed, m x (x, y, z, tau, warm)
        lp = os.path.join(A, 'sim', f'{case}_lines.bin')
        if os.path.exists(lp):
            buf = open(lp, 'rb').read(); o = 0
            n = struct.unpack_from('<i', buf, o)[0]; o += 4
            pts_all, heads = [], []
            for i in range(n):
                m_, seed = struct.unpack_from('<if', buf, o); o += 8
                arr = np.frombuffer(buf, dtype='<f4', count=m_ * 5, offset=o).reshape(m_, 5); o += m_ * 20
                arr = arr[::2] if m_ > 40 else arr
                heads.append(len(arr)); pts_all.append(arr)
            P_ = np.concatenate(pts_all)
            q = np.zeros((len(P_), 4), np.int16)
            q[:, 0] = np.round(P_[:, 0] * 4); q[:, 1] = np.round(P_[:, 1] * 20); q[:, 2] = np.round(P_[:, 2] * 4)
            q[:, 3] = np.round(np.clip(P_[:, 4], 0, 3) * 10000)
            tau = np.round(np.clip(P_[:, 3], 0, 6000) * 10).astype(np.uint16)
            P.put(f'runs/{case}/lines.bin.gz', np.concatenate([np.array(heads, np.uint16).view(np.uint8), q.view(np.uint8).ravel(), tau.view(np.uint8)]),
                  n_lines=len(heads), n_pts=int(len(P_)), record='u16 lengths[n_lines]; int16 (x*4, y*20, z*4, warm*1e4)[n_pts]; u16 tau*10[n_pts]')
        runs[case] = rec

    # switch-on frames of the baseline (all existing campuses switched on at t = 0)
    print('frames')
    bm = json.load(open(os.path.join(S, 'baseline', 'meta.json')))
    FR2 = list(range(0, 140, 2))
    f2 = np.stack([block_mean(np.nan_to_num(np.load(os.path.join(S, 'baseline', f'theta2m_{i:04d}.npy')).astype(np.float32)), 2) for i in FR2])
    P.put('runs/baseline/frames2d.bin.gz', q8(f2, 3.0), scale=3.0 / 255, unit='K', dt_s=bm['frame_dt_s'] * 2, n=len(FR2),
          note='near-surface warming after switch-on, 20 m, every 2nd solver frame')
    FR3 = list(range(0, 47, 3))
    ktop = np.load(os.path.join(S, 'baseline', 'ktop.npy')).astype(np.int32)
    kt2 = np.minimum.reduce([ktop[0::2, 0::2], ktop[1::2, 0::2], ktop[0::2, 1::2], ktop[1::2, 1::2]])
    f3 = []
    for i in FR3:
        a = np.nan_to_num(np.load(os.path.join(S, 'baseline', f'theta_{i:04d}.npy')).astype(np.float32))   # 40 x 300 x 300
        a = np.where(np.arange(40)[:, None, None] >= kt2[None], a, 0.0)
        f3.append(block_mean(a, 2))                                                                          # 40 x 150 x 150
    f3 = np.stack(f3)
    P.put('runs/baseline/frames3d.bin.gz', q8(f3, 3.0), scale=3.0 / 255, unit='K', dt_s=bm['frame3d_dt_s'] * 3, n=len(FR3),
          note='3-D warming after switch-on, 40 m x 10 m, every 3rd solver 3-D frame')

    # ---------------------------------------------------------------- Landsat LST (observed) + calibrated model LST
    print('landsat')
    lm = json.load(open(os.path.join(D, 'lst_meta.json')))
    cal = summary['calibration']
    landsat = []
    for d8 in ('20230730', '20240716', '20240825', '20240622'):
        obs = np.load(os.path.join(D, f'lst_{d8}.npy')).astype(np.float32)
        P.put(f'landsat/lst_{d8}.bin.gz', np.where(np.isfinite(obs), np.round(obs * 100), -32768).astype(np.int16), scale=0.01, unit='degC', nodata=-32768)
        ent = dict(date=f'{d8[:4]}-{d8[4:6]}-{d8[6:]}', id=d8, mean=float(np.nanmean(obs)), p5=float(np.nanpercentile(obs, 5)), p95=float(np.nanpercentile(obs, 95)))
        mp = os.path.join(S, 'calib', f'lst_sim_{d8}.npy')
        if os.path.exists(mp):
            sim = np.load(mp).astype(np.float32)
            mask = np.load(os.path.join(S, 'calib', f'lst_mask_{d8}.npy'))
            sim = np.where(mask & np.isfinite(sim), sim, np.nan)
            P.put(f'landsat/sim_{d8}.bin.gz', np.where(np.isfinite(sim), np.round(sim * 100), -32768).astype(np.int16), scale=0.01, unit='degC', nodata=-32768)
            ok = np.isfinite(sim) & np.isfinite(obs)
            o_, s_ = obs[ok], sim[ok]
            lo_, hi_ = float(np.percentile(o_, 0.5)), float(np.percentile(o_, 99.5))
            H, xe, ye = np.histogram2d(o_, s_, bins=48, range=[[lo_, hi_], [lo_, hi_]])
            ent['sim'] = True
            ent['hist'] = dict(lo=lo_, hi=hi_, n=48, counts=H.astype(int).ravel().tolist())
            met = cal['metrics'].get(d8)
            if met:
                ent['role'] = met['role']; ent['r2'] = met['full']['r2']; ent['rmse'] = met['full']['rmse']
        for x in summary['data']['landsat']:
            if x.get('date', '').replace('-', '') == d8:
                ent['platform'] = x.get('platform'); ent['scene'] = x.get('scene'); ent['utc'] = x.get('utc')
                ent['metar'] = x.get('kiad_metar')
        run = summary['runs'].get('calib_' + d8)
        if run:
            ent['met'] = run['met']
        landsat.append(ent)

    # ---------------------------------------------------------------- per-campus attribution tracers (baseline mean flow)
    print('tracers')
    z = np.load(os.path.join(S, 'campus_tracers', 'tracers_existing_baseline.npz'))
    keys = [k for k in z.files]
    tr_stack, tr_meta = [], []
    for k in keys:
        a = block_mean(np.nan_to_num(z[k].astype(np.float32)), 2)
        vmax = float(max(a.max(), 1e-3))
        tr_stack.append(q8(a, vmax)); tr_meta.append(dict(key=k, id=int(k[1:]), vmax=vmax))
    P.put('campus/tracers.bin.gz', np.stack(tr_stack), order=[m['key'] for m in tr_meta], vmax=[m['vmax'] for m in tr_meta],
          unit='K', note='near-surface warming caused by each campus alone (source-tagged tracers), 20 m; c9999 = small campuses lumped')
    P.put('campus/campus_id.bin.gz', cid.astype(np.int8), note='campus id per 10 m cell (-1 none)')

    # ---------------------------------------------------------------- manifest
    base = summary['scenarios']['baseline']
    M = dict(
        title='Ashburn Heat Twin',
        version=1,
        frame=dict(origin_utm=[ORIGIN_E, ORIGIN_N], crs='EPSG:32618', base_elev=BASE_ELEV, box=[X0, X1, Z0, Z1], utm_box=list(UTM_BOX),
                   grid=dict(n=600, dx=10.0), zbase=50.0, lonlat_box=summary['aoi']['bbox_lonlat_approx']),
        files=P.files,
        labels=labels,
        campuses=camps,
        sites=sites,
        runs=runs,
        landsat=landsat,
        baseline=dict(heat_MW=base['heat_existing_MW'], flux=base['F_existing_wm2'], a05=base['area_gt_0p5K_ha'], a1=base['area_gt_1K_ha'],
                      a2=base['area_gt_2K_ha'], homes05=base['residential_bldgs_gt_0p5K'], max=base['max_anomaly_K'], p99=base['p99_anomaly_K'],
                      stream05=base['stream_length_gt_0p5K_km'], stream_total=base['stream_length_total_km'], dc_roof_mean=base['mean_anomaly_over_dc_roofs_K']),
        facts=dict(n_dc_buildings=211, n_campuses=54, dc_roof_ha=215, buildings=len(out_b), residential=int(sum(1 for b in out_b if b.get('res')))),
        calibration=dict(r2_cal=[cal['metrics']['20230730']['full']['r2'], cal['metrics']['20240716']['full']['r2']],
                         rmse_cal=[cal['metrics']['20230730']['full']['rmse'], cal['metrics']['20240716']['full']['rmse']],
                         r2_val=cal['metrics']['20240825']['full']['r2'], rmse_val=cal['metrics']['20240825']['full']['rmse'],
                         plume_term=cal['g_theta'], plume_term_90=cal['g_theta_90'], F_median=cal['F_median_wm2'], F_90=cal['F_90'],
                         text=summary['calibration_interpretation']),
        solver=summary['solver'],
        solver_checks={k: summary['solver_checks'][k] for k in ('mass_conservation', 'heat_budget_out_over_in')},
        sources=[
            dict(name='USDA NAIP 2023 orthoimagery (Virginia, 11 Oct 2023)', licence='public domain', via='Microsoft Planetary Computer'),
            dict(name='USGS 3DEP 1/3 arc-second DEM', licence='public domain', via='Microsoft Planetary Computer'),
            dict(name='Landsat 8/9 Collection 2 Level-2 surface temperature (USGS)', licence='public domain', via='Microsoft Planetary Computer'),
            dict(name='ESA WorldCover 2021 v200', licence='CC BY 4.0', via='Microsoft Planetary Computer'),
            dict(name='Microsoft Global ML Building Footprints', licence='ODbL 1.0', via='github.com/microsoft/GlobalMLBuildingFootprints'),
            dict(name='OpenStreetMap (data-center tags, waterways, roads, places)', licence='ODbL 1.0, © OpenStreetMap contributors', via='Overpass API'),
            dict(name='KIAD ASOS METAR (Washington Dulles)', licence='public', via='Iowa Environmental Mesonet'),
            dict(name='ERA5 boundary-layer height', licence='CC BY 4.0', via='Open-Meteo archive API'),
            dict(name='US Census Bureau cartographic boundaries 2023', licence='public domain', via='census.gov'),
        ],
    )
    with open(os.path.join(P.out, 'manifest.json'), 'w') as f:
        json.dump(M, f, indent=1)
    tot = sum(v.get('bytes', 0) for v in P.files.values())
    print(f'total {tot / 1e6:.1f} MB in {len(P.files)} files')


if __name__ == '__main__':
    main()
