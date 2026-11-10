#!/usr/bin/env python3
"""Pack the 10 m static inputs of the live (in-browser WebGPU) solver.

    python tools/live/pack_live.py --proto <prototype folder with data/ and sim/> [--out data/live]

Writes
  data/live/static10.bin.gz  gzip'd little-endian planes (decoded in the browser with DecompressionStream)
  data/live/static10.json    layout, encodings, grid conventions, provenance, checksums

Planes (all 600 x 600, 10 m, north-up: row 0 = north edge z = -2995, col 0 = west edge x = -2995):
  dem     terrain, m NAVD88, quantized to 1 cm (q = round(dem*100)); stored as row-wise first differences
          (int16, first column relative to 0), split into a low-byte plane followed by a high-byte plane
          (the byte planes gzip ~30 % better than interleaved int16). Max abs error 5 mm.
  lc      ESA WorldCover 2021 class code (uint8), the roughness input (z0 table in static10.json)
  frac25  building cover fraction x 25 (uint8, exact: frac is a count of covered 2 m sub-pixels / 25)
  hmean   mean building height over covered sub-pixels, cm (uint16), max abs error 5 mm
  dca4    operational data-center roof area / 4 m^2 (uint8, exact: area is a count of 2 m sub-pixels x 4 m^2)
"""
import argparse, gzip, hashlib, json, os
import numpy as np

Z0_BY_CLASS = {10: 1.0, 20: 0.3, 30: 0.03, 40: 0.05, 50: 0.5, 60: 0.01, 80: 0.0002, 90: 0.05}   # = lbm3d.Z0_BY_CLASS
Z0_DEFAULT = 0.1


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument('--proto', required=True, help='prototype folder containing data/ and sim/')
    ap.add_argument('--out', default=os.path.normpath(os.path.join(here, '..', '..', 'data', 'live')))
    a = ap.parse_args()
    D = os.path.join(a.proto, 'data'); S = os.path.join(a.proto, 'sim')
    os.makedirs(a.out, exist_ok=True)

    dem = np.load(os.path.join(D, 'dem.npy')).astype(np.float64)
    lc = np.load(os.path.join(D, 'landcover.npy')).astype(np.uint8)
    z = np.load(os.path.join(S, '_domain_base.npz'))
    frac, hmean, dcarea = z['frac'].astype(np.float64), z['hmean'].astype(np.float64), z['dcarea'].astype(np.float64)
    aoi = json.load(open(os.path.join(D, 'aoi.json')))
    ny, nx = dem.shape
    assert dem.shape == lc.shape == frac.shape == hmean.shape == dcarea.shape == (600, 600)

    # ---- quantize (and check the lossless ones really are)
    qd = np.round(dem * 100.0).astype(np.int64)
    assert qd.min() >= 0 and qd.max() < 32768
    d = np.diff(np.concatenate([np.zeros((ny, 1), np.int64), qd], axis=1), axis=1)
    assert d.min() >= -32768 and d.max() < 32768
    du = d.astype(np.int16).view(np.uint16)
    dem_planes = np.concatenate([(du & 0xFF).astype(np.uint8).ravel(), (du >> 8).astype(np.uint8).ravel()])
    f25 = np.round(frac * 25.0).astype(np.int64)
    assert np.abs(f25 / 25.0 - frac).max() < 1e-6 and f25.max() <= 25
    hq = np.round(hmean * 100.0).astype(np.int64)
    assert hq.min() >= 0 and hq.max() < 65536
    if not ((hq > 0) == (f25 > 0)).all():
        print('note: cells where (hmean > 0) != (frac > 0):', int(((hq > 0) != (f25 > 0)).sum()))
    d4 = np.round(dcarea / 4.0).astype(np.int64)
    assert np.abs(d4 * 4.0 - dcarea).max() < 1e-6 and d4.max() <= 25

    planes = [
        ('dem', dem_planes.tobytes(), dict(dtype='int16', encoding='rowdelta-byteplanes', scale=0.01, units='m NAVD88',
                                           note='q[r,c] = sum_{c\'<=c} delta[r,c\']; value = q*0.01; low-byte plane then high-byte plane')),
        ('lc', lc.tobytes(), dict(dtype='uint8', encoding='raw', units='ESA WorldCover class')),
        ('frac25', f25.astype(np.uint8).tobytes(), dict(dtype='uint8', encoding='raw', scale=1 / 25, units='fraction')),
        ('hmean', hq.astype('<u2').tobytes(), dict(dtype='uint16', encoding='raw', scale=0.01, units='m')),
        ('dca4', d4.astype(np.uint8).tobytes(), dict(dtype='uint8', encoding='raw', scale=4.0, units='m2 per 10 m cell')),
    ]
    blob = bytearray(); layout = {}
    for name, b, meta in planes:
        while len(blob) % 4: blob.append(0)
        layout[name] = dict(offset=len(blob), bytes=len(b), **meta)
        blob += b
    gzpath = os.path.join(a.out, 'static10.bin.gz')
    with open(gzpath, 'wb') as fh:
        with gzip.GzipFile(fileobj=fh, mode='wb', compresslevel=9, mtime=0) as g:
            g.write(bytes(blob))

    # checksums of the decoded values (the browser decoder can verify itself against these)
    dem_dec = qd / 100.0
    chk = dict(dem_sum_cm=int(qd.sum()), lc_sum=int(lc.astype(np.int64).sum()), frac25_sum=int(f25.sum()),
               hmean_sum_cm=int(hq.sum()), dca4_sum=int(d4.sum()))
    meta = dict(
        version=1,
        what='10 m static inputs of the live WebGPU solver (js/live/): terrain, land cover (roughness), building cover '
             'fraction, mean building height, operational data-center roof area',
        file='static10.bin.gz', raw_bytes=len(blob), gz_bytes=os.path.getsize(gzpath),
        sha1_raw=hashlib.sha1(bytes(blob)).hexdigest(),
        grid=dict(nrows=ny, ncols=nx, dx_m=10.0, orientation='north-up: row 0 = north edge (scene z = -2995), col 0 = west edge (scene x = -2995)',
                  scene_x0=-2995.0, scene_z0=-2995.0, utm_bbox=[aoi['xmin'], aoi['ymin'], aoi['xmax'], aoi['ymax']], crs=aoi['crs']),
        scene=dict(origin_utm=[286780.0, 4320800.0], base_elev_m=83.53166198730469,
                   frame='x = easting - 286780, z = -(northing - 4320800) (+z south), y = elevation - 83.53166'),
        planes=layout,
        z0_by_class={str(k): v for k, v in Z0_BY_CLASS.items()}, z0_default=Z0_DEFAULT,
        checksums=chk,
        stats=dict(dem_min=float(dem_dec.min()), dem_max=float(dem_dec.max()), dcarea_total_m2=float(d4.sum() * 4.0),
                   dem_max_abs_err_m=float(np.abs(dem_dec - dem).max()), hmean_max_abs_err_m=float(np.abs(hq / 100.0 - hmean).max())),
        provenance=dict(dem='USGS 3DEP 1/3" (prototype data/dem.npy)', lc='ESA WorldCover 2021 v200 (data/landcover.npy)',
                        frac_hmean_dcarea='sim/_domain_base.npz (lbm3d.rasterize_buildings of data/buildings.geojson at 2 m sub-pixels)'),
    )
    json.dump(meta, open(os.path.join(a.out, 'static10.json'), 'w'), indent=1)
    print(f'wrote {gzpath}: raw {len(blob)/1e6:.2f} MB -> gz {meta["gz_bytes"]/1e6:.3f} MB')
    print(json.dumps(meta['stats'], indent=1))


if __name__ == '__main__':
    main()
