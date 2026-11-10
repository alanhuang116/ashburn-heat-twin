// The maquette: a 6 x 6 km terrain block of Ashburn with buildings, trees, streams and roads, rendered as a
// museum-style physical model. Fields (warming, Landsat, tracers) are draped on the ground and roofs; the 3-D
// warming volume is ray-marched in a composite pass that respects the scene depth.
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/OrbitControls.js';
import { clamp, lerp, Tweens, easeInOut, loadArray, loadImage, rampTexture, RAMP_WARM, RAMP_LST, fetchJSON } from './util.js';

export const X0 = -2995, X1 = 3005, Z0 = -2995, Z1 = 3005, SIZE = 6000, N10 = 600;
export const COL = { bg: '#ECE9E1', paper: '#F7F5EF', ink: '#1B1F27', navy: '#012169', gold: '#F2A900', terra: '#B5432C', steel: '#5B7FA6' };

const halfTex = (data, w, h, { nearest = false } = {}) => {
  const hd = new Uint16Array(data.length);
  for (let i = 0; i < data.length; i++) hd[i] = THREE.DataUtils.toHalfFloat(data[i]);
  const t = new THREE.DataTexture(hd, w, h, THREE.RedFormat, THREE.HalfFloatType);
  t.minFilter = t.magFilter = nearest ? THREE.NearestFilter : THREE.LinearFilter;
  t.wrapS = t.wrapT = THREE.ClampToEdgeWrapping; t.flipY = false; t.needsUpdate = true;
  return t;
};
export { halfTex };

export function setHalf(tex, data) {
  const hd = tex.image.data;
  for (let i = 0; i < data.length; i++) hd[i] = THREE.DataUtils.toHalfFloat(data[i]);
  tex.needsUpdate = true;
}

// GLSL shared by ground, roofs and trees
const FIELD_GLSL = /* glsl */`
uniform sampler2D uImg; uniform float uPhoto; uniform float uVE;
uniform sampler2D uField; uniform sampler2D uField2; uniform float uFieldOn; uniform float uFieldMode; uniform float uFieldScale; uniform float uFieldScale2;
uniform float uFieldMax; uniform float uFieldAlpha; uniform float uSwipeOn; uniform float uSwipeX;
uniform sampler2D uRampW; uniform sampler2D uRampL; uniform vec2 uLstRange; uniform float uIsoOn; uniform vec4 uIsoLv;
uniform vec3 uPaper; uniform vec3 uShade; uniform vec3 uSage; uniform vec3 uInk; uniform vec3 uGold; uniform vec3 uNavy;
uniform float uMapTone;
varying vec3 vWP;
vec2 fieldUV(vec3 p) { return vec2((p.x - (${X0.toFixed(1)})) / ${SIZE.toFixed(1)}, (p.z - (${Z0.toFixed(1)})) / ${SIZE.toFixed(1)}); }
float fieldAt(vec2 uv, bool second) { return second ? texture2D(uField2, uv).r * uFieldScale2 : texture2D(uField, uv).r * uFieldScale; }
vec3 mapBase(vec2 uv) {
  vec3 img = texture2D(uImg, uv).rgb;
  float l = dot(img, vec3(0.2126, 0.7152, 0.0722));
  float p = pow(max(l, 0.0), 0.4545);
  vec3 m = mix(uShade, uPaper, smoothstep(0.10, 0.80, p));
  float veg = clamp((img.g - img.r) * 9.0 + 0.15, 0.0, 1.0) * (1.0 - smoothstep(0.45, 0.75, p));
  m = mix(m, m * uSage, veg * 0.85);
  vec3 photo = pow(img, vec3(0.94)) * 1.04;
  return mix(m, photo, uPhoto);
}
// returns rgb = overlay colour, a = overlay weight; iso = isotherm line strength
vec4 fieldColor(vec3 p, out float iso) {
  iso = 0.0;
  if (uFieldOn < 0.5) return vec4(0.0);
  vec2 uv = fieldUV(p);
  bool second = uSwipeOn > 0.5 && p.x > uSwipeX;
  float th = fieldAt(uv, second);
  if (uFieldMode < 0.5) {
    vec4 rc = texture2D(uRampW, vec2(clamp(th / uFieldMax, 0.0, 1.0), 0.5));
    float fw = max(fwidth(th), 1e-4);
    for (int i = 0; i < 4; i++) { float L = uIsoLv[i]; if (L > 0.0) iso = max(iso, 1.0 - smoothstep(0.35 * fw, 1.35 * fw, abs(th - L))); }
    iso *= uIsoOn;
    return vec4(rc.rgb, rc.a * uFieldAlpha);
  }
  if (th < -100.0) return vec4(uPaper * 0.92, 0.35);   // no data (cloud)
  float t = (th - uLstRange.x) / (uLstRange.y - uLstRange.x);
  vec4 rc = texture2D(uRampL, vec2(clamp(t, 0.0, 1.0), 0.5));
  return vec4(rc.rgb, 0.93 * uFieldAlpha);
}
`;

export class Maquette {
  constructor(canvas, manifest) {
    this.M = manifest; this.canvas = canvas; this.VE = 2.5; this.tweens = new Tweens();
    const r = this.renderer = new THREE.WebGLRenderer({ canvas, antialias: false, powerPreference: 'high-performance', stencil: false });
    r.setClearColor(COL.bg, 1); r.outputColorSpace = THREE.SRGBColorSpace;
    r.shadowMap.enabled = true; r.shadowMap.type = THREE.PCFSoftShadowMap; r.shadowMap.autoUpdate = false;
    this.scene = new THREE.Scene(); this.scene.background = new THREE.Color(COL.bg);
    this.scene.fog = new THREE.Fog(COL.bg, 15000, 42000);
    this.camera = new THREE.PerspectiveCamera(28, 1, 40, 70000);
    this.world = new THREE.Group(); this.world.scale.y = this.VE; this.scene.add(this.world);
    this.quality = { scale: 1, steps: 96 };
    const dummy = halfTex(new Float32Array([0, 0, 0, 0]), 2, 2);
    this.U = {
      uImg: { value: null }, uPhoto: { value: 0 }, uVE: { value: this.VE },
      uField: { value: dummy }, uField2: { value: dummy }, uFieldOn: { value: 0 }, uFieldMode: { value: 0 }, uFieldScale: { value: 1 }, uFieldScale2: { value: 1 },
      uFieldMax: { value: 2.4 }, uFieldAlpha: { value: 1 }, uSwipeOn: { value: 0 }, uSwipeX: { value: 0 },
      uRampW: { value: rampTexture(RAMP_WARM) }, uRampL: { value: rampTexture(RAMP_LST) }, uLstRange: { value: new THREE.Vector2(30, 50) },
      uIsoOn: { value: 1 }, uIsoLv: { value: new THREE.Vector4(0.25, 0.5, 1.0, 2.0) },
      uPaper: { value: new THREE.Color('#F7F4EC') }, uShade: { value: new THREE.Color('#B9B2A3') }, uSage: { value: new THREE.Color('#C9D2B0') },
      uInk: { value: new THREE.Color(COL.ink) }, uGold: { value: new THREE.Color(COL.gold) }, uNavy: { value: new THREE.Color(COL.navy) },
      uMapTone: { value: 1 },
      uDcGlow: { value: 0 }, uDcTint: { value: 0 }, uHover: { value: -9 }, uSel: { value: -9 }, uRoofField: { value: 1 },
      uClear: { value: [new THREE.Vector4(1e9, 1e9, -1e9, -1e9), new THREE.Vector4(1e9, 1e9, -1e9, -1e9), new THREE.Vector4(1e9, 1e9, -1e9, -1e9)] },
      uTime: { value: 0 }
    };
    this.controls = new OrbitControls(this.camera, canvas);
    Object.assign(this.controls, { enableDamping: true, dampingFactor: 0.085, rotateSpeed: 0.55, zoomSpeed: 0.9, panSpeed: 0.8, minDistance: 500, maxDistance: 24000,
      maxPolarAngle: 1.36, minPolarAngle: 0.0, screenSpacePanning: false, zoomToCursor: true });
    this.controls.addEventListener('change', () => this.clampTarget());
    this._raycaster = new THREE.Raycaster();
  }

  // ------------------------------------------------------------------ loading + building
  async init(progress = () => {}) {
    const M = this.M;
    progress('terrain', 0.05);
    const [ter, lc, bjson, trees, lines, img1k] = await Promise.all([
      loadArray(M, 'terrain.bin.gz'), loadArray(M, 'landcover.bin.gz'), fetchJSON('data/buildings.json'),
      loadArray(M, 'trees.bin.gz'), fetchJSON('data/lines.json'), loadImage('data/imagery_1k.jpg')]);
    progress('model', 0.45);
    const H = new Float32Array(ter.data.length); for (let i = 0; i < H.length; i++) H[i] = ter.data[i] * 0.01;
    this.H = H; this.landcover = lc.data;
    let mn = 1e9, mx = -1e9; for (const v of H) { if (v < mn) mn = v; if (v > mx) mx = v; } this.hmin = mn; this.hmax = mx;
    this.U.uImg.value = this._imgTex(img1k);
    this._lights();
    this._terrain();
    this._sides();
    this._table();
    this.buildings = this._buildings(bjson.b);
    this.treeData = trees.data; this._trees();
    this._lines(lines);
    this._halls();
    this._composite();
    this.demTex = halfTex(this.H, N10, N10);
    this._section();
    this.resize();
    this.setView(this.views.home, 0);
    this.renderer.shadowMap.needsUpdate = true;
    progress('imagery', 0.8);
    // full-resolution imagery in the background
    loadImage('data/imagery_4k.jpg').then(im => { const t = this._imgTex(im); const old = this.U.uImg.value; this.U.uImg.value = t; old && old.dispose(); this.renderer.shadowMap.needsUpdate = true; }).catch(() => {});
  }

  _imgTex(im) {
    const t = new THREE.Texture(im); t.colorSpace = THREE.SRGBColorSpace; t.flipY = false;
    t.anisotropy = Math.min(8, this.renderer.capabilities.getMaxAnisotropy()); t.generateMipmaps = true;
    t.minFilter = THREE.LinearMipmapLinearFilter; t.magFilter = THREE.LinearFilter; t.wrapS = t.wrapT = THREE.ClampToEdgeWrapping; t.needsUpdate = true; return t;
  }

  heightAt(x, z) {   // local metres (scene y before vertical exaggeration)
    const fx = clamp((x - X0) / 10 - 0.5, 0, N10 - 1.001), fz = clamp((z - Z0) / 10 - 0.5, 0, N10 - 1.001);
    const j = Math.floor(fx), i = Math.floor(fz), ax = fx - j, az = fz - i, H = this.H;
    return H[i * N10 + j] * (1 - ax) * (1 - az) + H[i * N10 + j + 1] * ax * (1 - az) + H[(i + 1) * N10 + j] * (1 - ax) * az + H[(i + 1) * N10 + j + 1] * ax * az;
  }
  landcoverAt(x, z) { const c = Math.floor((x - X0) / 10), r = Math.floor((z - Z0) / 10); if (c < 0 || r < 0 || c >= N10 || r >= N10) return 0; return this.landcover[r * N10 + c]; }

  _lights() {
    this.hemi = new THREE.HemisphereLight('#FFFFFF', '#CBC4B4', 1.55); this.scene.add(this.hemi);
    const s = this.sun = new THREE.DirectionalLight('#FFF4E6', 2.35);
    s.position.set(-0.52, 0.78, -0.36).normalize().multiplyScalar(12000); s.target.position.set(0, 0, 0);
    s.castShadow = true; s.shadow.mapSize.set(4096, 4096);
    const sc = s.shadow.camera; sc.left = -4300; sc.right = 4300; sc.top = 4300; sc.bottom = -4300; sc.near = 2000; sc.far = 24000;
    s.shadow.bias = -0.00035; s.shadow.normalBias = 0.6; s.shadow.radius = 2.5;
    this.scene.add(s); this.scene.add(s.target);
  }

  _material(kind) {
    // Lambert + injected field colouring
    const m = new THREE.MeshLambertMaterial({ color: '#ffffff' });
    const U = this.U;
    m.onBeforeCompile = sh => {
      Object.assign(sh.uniforms, U);
      sh.vertexShader = sh.vertexShader
        .replace('#include <common>', `#include <common>\nvarying vec3 vWP;\n${kind === 'bld' ? 'attribute vec4 aInfo; varying vec4 vInfo;' : ''}${kind === 'tree' ? 'uniform vec4 uClear[3]; attribute float aTone; varying float vTone;' : ''}`)
        .replace('#include <begin_vertex>', `#include <begin_vertex>\n${kind === 'bld' ? 'vInfo = aInfo;' : ''}${kind === 'tree' ? `vTone = aTone;
#ifdef USE_INSTANCING
{ vec3 o0 = (instanceMatrix * vec4(0.0, 0.0, 0.0, 1.0)).xyz; for (int i = 0; i < 3; i++) { vec4 r = uClear[i]; if (o0.x > r.x && o0.x < r.z && o0.z > r.y && o0.z < r.w) transformed *= 0.0; } }
#endif` : ''}`)
        .replace('#include <worldpos_vertex>', `#include <worldpos_vertex>
{ vec4 wp4 = vec4(transformed, 1.0);
#ifdef USE_INSTANCING
  wp4 = instanceMatrix * wp4;
#endif
  wp4 = modelMatrix * wp4; vWP = wp4.xyz; }`);
      let frag = sh.fragmentShader.replace('#include <common>', `#include <common>\n${FIELD_GLSL}\n${kind === 'bld' ? 'varying vec4 vInfo; uniform float uDcGlow; uniform float uDcTint; uniform float uHover; uniform float uSel; uniform float uRoofField;' : ''}${kind === 'tree' ? 'varying float vTone;' : ''}`);
      if (kind === 'ground') frag = frag.replace('#include <color_fragment>', `#include <color_fragment>
{ vec2 uv = fieldUV(vWP); vec3 col = mapBase(uv); float iso; vec4 fc = fieldColor(vWP, iso);
  col = mix(col, fc.rgb, fc.a); col = mix(col, uInk, iso * 0.62);
  if (uSwipeOn > 0.5) { float d = abs(vWP.x - uSwipeX); float w = fwidth(vWP.x) * 1.6; col = mix(col, uInk, (1.0 - smoothstep(0.0, w, d)) * 0.9); }
  diffuseColor.rgb = col; }`);
      if (kind === 'bld') frag = frag.replace('#include <color_fragment>', `#include <color_fragment>
{ float kind = vInfo.x; float roof = vInfo.y; float camp = vInfo.z;
  vec3 c = kind > 1.5 ? vec3(0.86, 0.885, 0.93) : (kind > 0.5 ? vec3(0.93, 0.905, 0.865) : vec3(0.905, 0.895, 0.875));
  if (kind > 2.5) c = vec3(0.98, 0.955, 0.90);
  c *= (0.97 + 0.06 * vInfo.w);
  if (roof < 0.5) c *= 0.92;
  if (kind > 1.5 && kind < 2.5) { vec3 nv = roof > 0.5 ? vec3(0.085, 0.125, 0.30) : vec3(0.055, 0.085, 0.22); c = mix(c, nv, uDcTint * 0.86); if (roof > 0.5) c = mix(c, uGold * 1.02, uDcGlow * 0.9); }
  if (abs(camp - uHover) < 0.5) c = mix(c, uNavy * 2.2 + 0.08, 0.45);
  if (abs(camp - uSel) < 0.5) c = roof > 0.5 ? mix(c, uGold, 0.92) : mix(c, vec3(0.04, 0.06, 0.17), 0.8);
  if (roof > 0.5 && uRoofField > 0.5) { float iso; vec4 fc = fieldColor(vWP, iso); c = mix(c, fc.rgb, fc.a * 0.55); }
  if (uPhoto > 0.5 && roof > 0.5 && kind < 2.5) { vec3 im = texture2D(uImg, fieldUV(vWP)).rgb; c = mix(c, im * 1.15 + 0.03, 0.55); }
  diffuseColor.rgb = c; }`);
      if (kind === 'tree') frag = frag.replace('#include <color_fragment>', `#include <color_fragment>
{ vec3 a = mix(vec3(0.42, 0.50, 0.36), vec3(0.62, 0.68, 0.52), vTone); vec3 b = mix(vec3(0.16, 0.24, 0.12), vec3(0.34, 0.42, 0.22), vTone);
  vec3 c = mix(a, b, uPhoto);
  float iso; vec4 fc = fieldColor(vWP, iso); c = mix(c, fc.rgb * 0.92, fc.a * 0.55);
  diffuseColor.rgb = c; }`);
      sh.fragmentShader = frag;
    };
    m.customProgramCacheKey = () => 'maq_' + kind;
    return m;
  }

  _terrain() {
    const n = N10, H = this.H;
    const corner = (i, j) => { let s = 0, c = 0; for (const [a, b] of [[i - 1, j - 1], [i - 1, j], [i, j - 1], [i, j]]) if (a >= 0 && b >= 0 && a < n && b < n) { s += H[a * n + b]; c++; } return s / c; };
    const V = n + 1, pos = new Float32Array(V * V * 3), nor = new Float32Array(V * V * 3);
    const hc = new Float32Array(V * V);
    for (let i = 0; i <= n; i++) for (let j = 0; j <= n; j++) hc[i * V + j] = corner(i, j);
    this.hc = hc;
    for (let i = 0; i <= n; i++) for (let j = 0; j <= n; j++) {
      const k = i * V + j; pos[k * 3] = X0 + j * 10; pos[k * 3 + 1] = hc[k]; pos[k * 3 + 2] = Z0 + i * 10;
      const hx = (hc[i * V + Math.min(n, j + 1)] - hc[i * V + Math.max(0, j - 1)]) / 20, hz = (hc[Math.min(n, i + 1) * V + j] - hc[Math.max(0, i - 1) * V + j]) / 20;
      const l = Math.hypot(hx, 1, hz); nor[k * 3] = -hx / l; nor[k * 3 + 1] = 1 / l; nor[k * 3 + 2] = -hz / l;
    }
    const idx = new Uint32Array(n * n * 6); let o = 0;
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) { const a = i * V + j, b = a + 1, c = a + V, d = c + 1; idx[o++] = a; idx[o++] = c; idx[o++] = b; idx[o++] = b; idx[o++] = c; idx[o++] = d; }
    const g = new THREE.BufferGeometry(); g.setAttribute('position', new THREE.BufferAttribute(pos, 3)); g.setAttribute('normal', new THREE.BufferAttribute(nor, 3)); g.setIndex(new THREE.BufferAttribute(idx, 1));
    g.computeBoundingSphere(); g.computeBoundingBox();
    this.groundMat = this._material('ground');
    const m = this.terrain = new THREE.Mesh(g, this.groundMat); m.receiveShadow = true; m.castShadow = true; m.name = 'terrain';
    this.world.add(m);
  }

  _sides() {
    if (this.sideMesh) { this.world.remove(this.sideMesh); this.sideMesh.geometry.dispose(); }
    const V = N10 + 1, hc = this.hc, base = this.hmin - 240 / this.VE;
    this.baseY = base;
    const pos = [], nor = [], uvy = [];
    const edge = (pts, nx, nz) => {
      for (let k = 0; k < pts.length - 1; k++) {
        const [x1, z1, y1] = pts[k], [x2, z2, y2] = pts[k + 1];
        const q = [[x1, base, z1], [x2, base, z2], [x2, y2, z2], [x1, y1, z1]];
        const tri = [0, 1, 2, 0, 2, 3];
        // make the face point outward
        // triangle (0,1,2) has normal (-dz, 0, dx)*h; keep it if that points outward
        const ddx = x2 - x1, ddz = z2 - z1; const cr = -ddz * nx + ddx * nz;
        const order = cr > 0 ? tri : [0, 2, 1, 0, 3, 2];
        for (const t of order) { pos.push(...q[t]); nor.push(nx, 0, nz); }
      }
    };
    const N = [], S = [], W = [], E = [];
    for (let j = 0; j <= N10; j++) { N.push([X0 + j * 10, Z0, hc[j]]); S.push([X0 + j * 10, Z1, hc[N10 * V + j]]); }
    for (let i = 0; i <= N10; i++) { W.push([X0, Z0 + i * 10, hc[i * V]]); E.push([X1, Z0 + i * 10, hc[i * V + N10]]); }
    edge(N, 0, -1); edge(S, 0, 1); edge(W, -1, 0); edge(E, 1, 0);
    const g = new THREE.BufferGeometry(); g.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3)); g.setAttribute('normal', new THREE.Float32BufferAttribute(nor, 3));
    if (!this.sideMat) {
      this.sideMat = new THREE.MeshLambertMaterial({ color: '#ffffff' });
      const U = this.U;
      this.sideMat.onBeforeCompile = sh => {
        sh.uniforms.uVE = U.uVE; sh.uniforms.uBase = this._uBase = this._uBase || { value: base }; sh.uniforms.uTop = { value: this.hmax };
        sh.vertexShader = sh.vertexShader.replace('#include <common>', '#include <common>\nvarying float vY;').replace('#include <begin_vertex>', '#include <begin_vertex>\nvY = position.y;');
        sh.fragmentShader = sh.fragmentShader.replace('#include <common>', '#include <common>\nvarying float vY; uniform float uBase; uniform float uVE;')
          .replace('#include <color_fragment>', `#include <color_fragment>
{ float d = -vY * uVE;  // depth below the datum in exaggerated metres
  vec3 c0 = vec3(0.86, 0.83, 0.77), c1 = vec3(0.80, 0.765, 0.70), c2 = vec3(0.74, 0.70, 0.635), c3 = vec3(0.68, 0.64, 0.575);
  float yy = (vY - uBase) * uVE;           // height above the block base
  vec3 c = yy > 150.0 ? c0 : (yy > 95.0 ? c1 : (yy > 45.0 ? c2 : c3));
  float seam = 1.0 - smoothstep(0.0, 1.6, abs(mod(yy, 1e6) - 150.0)) ; c *= 1.0 - 0.0 * seam;
  diffuseColor.rgb = c; }`);
      };
    }
    if (this._uBase) this._uBase.value = base;
    const m = this.sideMesh = new THREE.Mesh(g, this.sideMat); m.castShadow = true; m.receiveShadow = false;
    this.world.add(m);
  }

  _table() {
    // the table the model stands on: receives the model's shadow, plus a soft contact shadow
    const cv = document.createElement('canvas'); cv.width = cv.height = 512; const c = cv.getContext('2d');
    const g = c.createRadialGradient(256, 256, 120, 256, 256, 256); g.addColorStop(0, 'rgba(40,36,30,0.55)'); g.addColorStop(0.55, 'rgba(40,36,30,0.18)'); g.addColorStop(1, 'rgba(40,36,30,0)');
    c.fillStyle = g; c.fillRect(0, 0, 512, 512);
    const tex = new THREE.CanvasTexture(cv); tex.colorSpace = THREE.SRGBColorSpace;
    const ao = new THREE.Mesh(new THREE.PlaneGeometry(10400, 10400), new THREE.MeshBasicMaterial({ map: tex, transparent: true, depthWrite: false, opacity: 0.62, fog: false }));
    ao.rotation.x = -Math.PI / 2; this.scene.add(ao); this.contactShadow = ao;
    ao.position.x = 260; ao.position.z = 220;
    this._placeTable();
  }
  _placeTable() { const y = this.baseY * this.VE - 0.5; this.contactShadow.position.y = y + 0.3; }

  _buildings(list) {
    const groups = { 0: this._bucket(), 1: this._bucket() };
    const campusOf = [];
    const seedOf = s => { const x = Math.sin(s * 12.9898) * 43758.5453; return x - Math.floor(x); };
    let bi = 0;
    for (const b of list) {
      const kind = b.dc ? 2 : (b.res ? 1 : 0); const camp = b.dc ? (b.c ?? -1) : -5;
      this._extrude(groups[0], b.r, b.y0 - 1.5, b.y1, kind, camp, seedOf(bi++));
    }
    const mat = this.bldMat = this._material('bld');
    const mesh = this._bucketMesh(groups[0], mat); mesh.name = 'buildings'; mesh.castShadow = mesh.receiveShadow = true;
    this.world.add(mesh);
    // residential centroids (for counting homes inside warming contours)
    this.homes = []; this.dcRoofs = [];
    for (const b of list) {
      const r = b.r[0]; let sx = 0, sz = 0; const n = r.length / 2; for (let k = 0; k < n; k++) { sx += r[k * 2]; sz += r[k * 2 + 1]; }
      if (b.res) this.homes.push([sx / n, sz / n]);
    }
    this.bList = list;
    return mesh;
  }
  _bucket() { return { pos: [], nor: [], info: [], idx: [] }; }
  _extrude(G, rings, y0, y1, kind, camp, seed) {
    const base = G.pos.length / 3;
    // walls
    rings.forEach((ring, ri) => {
      const n = ring.length / 2; let A = 0;
      for (let i = 0; i < n; i++) { const j = (i + 1) % n; A += ring[i * 2] * ring[j * 2 + 1] - ring[j * 2] * ring[i * 2 + 1]; }
      const sgn = (A > 0 ? 1 : -1) * (ri === 0 ? 1 : -1);
      for (let i = 0; i < n; i++) {
        const j = (i + 1) % n; const ax = ring[i * 2], az = ring[i * 2 + 1], bx = ring[j * 2], bz = ring[j * 2 + 1];
        const dx = bx - ax, dz = bz - az, L = Math.hypot(dx, dz); if (L < 0.05) continue;
        // outward normal: rotate the edge clockwise (in the x,z plane) for a counter-clockwise outer ring; holes reversed via sgn
        const nx = sgn * dz / L, nz = -sgn * dx / L;
        const v = G.pos.length / 3;
        G.pos.push(ax, y0, az, bx, y0, bz, bx, y1, bz, ax, y1, az);
        for (let k = 0; k < 4; k++) { G.nor.push(nx, 0, nz); G.info.push(kind, 0, camp, seed); }
        // triangle (v, v+1, v+2) has face normal (-dz, 0, dx); flip the winding if that is not outward
        const facing = (-dz * nx + dx * nz) > 0;
        if (facing) G.idx.push(v, v + 1, v + 2, v, v + 2, v + 3); else G.idx.push(v, v + 2, v + 1, v, v + 3, v + 2);
      }
    });
    // roof
    const toV = ring => { const a = []; for (let i = 0; i < ring.length; i += 2) a.push(new THREE.Vector2(ring[i], ring[i + 1])); return a; };
    const contour = toV(rings[0]); const holes = rings.slice(1).map(toV);
    let tris;
    try { tris = THREE.ShapeUtils.triangulateShape(contour, holes); } catch (e) { tris = []; }
    const all = contour.concat(...holes);
    const v0 = G.pos.length / 3;
    for (const p of all) { G.pos.push(p.x, y1, p.y); G.nor.push(0, 1, 0); G.info.push(kind, 1, camp, seed); }
    for (const t of tris) {
      // roof faces must point up: triangle (a,b,c) in x,z with z south -> check orientation
      const a = all[t[0]], b = all[t[1]], c = all[t[2]];
      const cr = (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x);
      if (cr < 0) G.idx.push(v0 + t[0], v0 + t[1], v0 + t[2]); else G.idx.push(v0 + t[0], v0 + t[2], v0 + t[1]);
    }
  }
  _bucketMesh(G, mat) {
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.Float32BufferAttribute(G.pos, 3)); g.setAttribute('normal', new THREE.Float32BufferAttribute(G.nor, 3));
    g.setAttribute('aInfo', new THREE.Float32BufferAttribute(G.info, 4)); g.setIndex(G.idx.length > 65535 ? new THREE.Uint32BufferAttribute(G.idx, 1) : new THREE.Uint16BufferAttribute(G.idx, 1));
    g.computeBoundingSphere(); g.computeBoundingBox();
    return new THREE.Mesh(g, mat);
  }

  _trees() {
    const d = this.treeData, dv = new DataView(d.buffer, d.byteOffset, d.byteLength); const n = d.byteLength / 10;
    const geo = new THREE.IcosahedronGeometry(1, 1);
    // soften normals toward the sphere normal (felt balls)
    const mat = this.treeMat = this._material('tree');
    const mesh = this.trees = new THREE.InstancedMesh(geo, mat, n);
    const tone = new Float32Array(n);
    this._treeRec = new Float32Array(n * 5);
    for (let i = 0; i < n; i++) {
      const o = i * 10;
      const x = dv.getInt16(o, true) / 5, z = dv.getInt16(o + 2, true) / 5, y = dv.getInt16(o + 4, true) / 100, r = dv.getUint8(o + 6) / 10, h = dv.getUint8(o + 7) / 4;
      tone[i] = dv.getUint8(o + 8) / 255;
      this._treeRec.set([x, y, z, r, h], i * 5);
    }
    geo.setAttribute('aTone', new THREE.InstancedBufferAttribute(tone, 1));
    this._placeTrees();
    mesh.castShadow = true; mesh.receiveShadow = true; mesh.frustumCulled = false; mesh.name = 'trees';
    this.world.add(mesh);
  }
  _placeTrees() {
    const R = this._treeRec, n = R.length / 5, m4 = new THREE.Matrix4(), q = new THREE.Quaternion(), s = new THREE.Vector3(), t = new THREE.Vector3();
    const ve = this.VE, ex = Math.min(ve, 2.2);
    for (let i = 0; i < n; i++) {
      const x = R[i * 5], y = R[i * 5 + 1], z = R[i * 5 + 2], r = R[i * 5 + 3], h = R[i * 5 + 4];
      const cr = clamp(r * 1.15, 2.4, 6.8), cv = cr * (0.95 + 0.25 * clamp((h - 8) / 14)) * ex * 0.62;
      const yg = this.heightAt(x, z);
      t.set(x, yg + (cv * 0.82) / ve, z); s.set(cr, cv / ve, cr);
      m4.compose(t, q, s); this.trees.setMatrixAt(i, m4);
    }
    this.trees.instanceMatrix.needsUpdate = true;
  }

  _lines(L) {
    // streams + ponds (steel) and major roads (light ribbons) draped on the terrain
    const ribbon = (list, width, lift) => {
      const pos = [], idx = [];
      for (const item of list) {
        const p = item.p || item; const n = p.length / 2; if (n < 2) continue;
        const w = typeof width === 'function' ? width(item) : width;
        const v0 = pos.length / 3;
        for (let k = 0; k < n; k++) {
          const x = p[k * 2], z = p[k * 2 + 1];
          const xa = p[Math.max(0, k - 1) * 2], za = p[Math.max(0, k - 1) * 2 + 1], xb = p[Math.min(n - 1, k + 1) * 2], zb = p[Math.min(n - 1, k + 1) * 2 + 1];
          let tx = xb - xa, tz = zb - za; const l = Math.hypot(tx, tz) || 1; tx /= l; tz /= l;
          const nx = -tz * w / 2, nz = tx * w / 2;
          const cx = clamp(x, X0, X1), cz = clamp(z, Z0, Z1);
          const y = this.heightAt(cx, cz) + lift;
          pos.push(clamp(x + nx, X0, X1), y, clamp(z + nz, Z0, Z1), clamp(x - nx, X0, X1), y, clamp(z - nz, Z0, Z1));
          if (k < n - 1) { const a = v0 + k * 2; idx.push(a, a + 1, a + 2, a + 1, a + 3, a + 2); }
        }
      }
      const g = new THREE.BufferGeometry(); g.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3)); g.setIndex(idx); g.computeVertexNormals();
      return g;
    };
    const waterMat = new THREE.MeshLambertMaterial({ color: '#7E9CBC', polygonOffset: true, polygonOffsetFactor: -2, polygonOffsetUnits: -4, side: THREE.DoubleSide });
    const streams = new THREE.Mesh(ribbon(L.water, it => it.k === 'river' ? 16 : (it.n ? 9 : 5), 0.35), waterMat); streams.receiveShadow = true;
    // ponds
    const pondPos = [];
    for (const ring of L.ponds) {
      const pts = []; for (let k = 0; k < ring.length; k += 2) pts.push(new THREE.Vector2(clamp(ring[k], X0, X1), clamp(ring[k + 1], Z0, Z1)));
      let tris; try { tris = THREE.ShapeUtils.triangulateShape(pts, []); } catch (e) { continue; }
      let ys = 0; for (const p of pts) ys += this.heightAt(p.x, p.y); const y = Math.min(...pts.map(p => this.heightAt(p.x, p.y))) + 0.3;
      for (const t of tris) for (const k of [t[0], t[2], t[1]]) pondPos.push(pts[k].x, y, pts[k].y);
    }
    const pg = new THREE.BufferGeometry(); pg.setAttribute('position', new THREE.Float32BufferAttribute(pondPos, 3)); pg.computeVertexNormals();
    const ponds = new THREE.Mesh(pg, waterMat);
    const roadMat = this.roadMat = new THREE.MeshLambertMaterial({ color: '#FFFFFF', transparent: true, opacity: 0.78, polygonOffset: true, polygonOffsetFactor: -1, polygonOffsetUnits: -2, depthWrite: false, side: THREE.DoubleSide });
    const roads = new THREE.Mesh(ribbon(L.roads, it => [20, 13, 9][it.k] || 8, 0.25), roadMat); roads.receiveShadow = true;
    this.water = new THREE.Group(); this.water.add(streams, ponds); this.world.add(this.water);
    this.roads = roads; this.world.add(roads);
    this.lineData = L;
  }

  _halls() {
    // hypothetical campuses A / B / C (three 14 m data halls each), shown only with their scenario
    this.halls = {};
    const mat = this.bldMat;
    for (const [k, s] of Object.entries(this.M.sites)) {
      const G = this._bucket();
      for (const h of s.halls) this._extrude(G, [h.r], h.y0 - 1.5, h.y1, 3, -7, 0.5);
      const m = this._bucketMesh(G, mat); m.visible = false; m.castShadow = m.receiveShadow = true; m.name = 'hall_' + k;
      this.world.add(m); this.halls[k] = m;
    }
  }
  showSite(k) {
    for (const [kk, m] of Object.entries(this.halls)) m.visible = kk === k;
    const cl = this.U.uClear.value;
    cl.forEach(v => v.set(1e9, 1e9, -1e9, -1e9));
    if (k && this.M.sites[k]) { const b = this.M.sites[k].bbox; cl[0].set(b[0] - 12, b[1] - 12, b[2] + 12, b[3] + 12); }
    this.renderer.shadowMap.needsUpdate = true;
  }

  setVE(v) {
    this.VE = v; this.world.scale.y = v; this.U.uVE.value = v;
    this._sides(); this._placeTable(); this._placeTrees(); this.renderer.shadowMap.needsUpdate = true;
  }
  setPhoto(on) {
    this.U.uPhoto.value = on ? 1 : 0; this.roads.visible = !on; this.water.visible = true;
  }

  // ------------------------------------------------------------------ composite pass: scene colour + depth -> volume ray-march -> tone map
  _composite() {
    const dt = new THREE.DepthTexture(4, 4); dt.type = THREE.UnsignedIntType;
    this.rt = new THREE.WebGLRenderTarget(4, 4, { type: THREE.HalfFloatType, samples: 4, depthTexture: dt, colorSpace: THREE.LinearSRGBColorSpace });
    const empty = new THREE.Data3DTexture(new Uint8Array(8), 2, 2, 2); empty.format = THREE.RedFormat; empty.needsUpdate = true;
    this.CU = {
      tColor: { value: this.rt.texture }, tDepth: { value: dt },
      uInvProj: { value: new THREE.Matrix4() }, uInvView: { value: new THREE.Matrix4() }, uCam: { value: new THREE.Vector3() },
      uVolA: { value: empty }, uVolB: { value: empty }, uVolMix: { value: 0 }, uVolOn: { value: 0 }, uVolScale: { value: 1 }, uVolDens: { value: 1 },
      uBoxMin: { value: new THREE.Vector3() }, uBoxMax: { value: new THREE.Vector3() }, uVE: this.U.uVE,
      uRamp: this.U.uRampW, uRampMax: { value: 2.4 }, uSteps: { value: 96 }, uFrame: { value: 0 }, uExpo: { value: 1.0 },
      uRes: { value: new THREE.Vector2(1, 1) }, uVolLo: { value: 0.08 }
    };
    const mat = new THREE.ShaderMaterial({
      uniforms: this.CU, depthTest: false, depthWrite: false,
      vertexShader: 'varying vec2 vUv; void main(){ vUv = uv; gl_Position = vec4(position.xy, 0.0, 1.0); }',
      fragmentShader: /* glsl */`
precision highp float; precision highp sampler3D;
varying vec2 vUv;
uniform sampler2D tColor; uniform sampler2D tDepth; uniform mat4 uInvProj; uniform mat4 uInvView; uniform vec3 uCam;
uniform sampler3D uVolA; uniform sampler3D uVolB; uniform float uVolMix; uniform float uVolOn; uniform float uVolScale; uniform float uVolDens;
uniform vec3 uBoxMin; uniform vec3 uBoxMax; uniform sampler2D uRamp; uniform float uRampMax; uniform float uSteps; uniform float uFrame; uniform float uExpo; uniform vec2 uRes; uniform float uVolLo;
vec3 neutral(vec3 color) {   // Khronos PBR Neutral tone mapping
  const float startCompression = 0.8 - 0.04; const float desaturation = 0.15;
  float x = min(color.r, min(color.g, color.b)); float offset = x < 0.08 ? x - 6.25 * x * x : 0.04; color -= offset;
  float peak = max(color.r, max(color.g, color.b)); if (peak < startCompression) return color;
  const float d = 1.0 - startCompression; float newPeak = 1.0 - d * d / (peak + d - startCompression); color *= newPeak / peak;
  float g = 1.0 - 1.0 / (desaturation * (peak - newPeak) + 1.0); return mix(color, newPeak * vec3(1.0), g);
}
vec3 toSRGB(vec3 c) { return mix(c * 12.92, 1.055 * pow(c, vec3(1.0 / 2.4)) - 0.055, step(0.0031308, c)); }
float hash(vec2 p) { return fract(sin(dot(p, vec2(12.9898, 78.233)) + uFrame * 0.618) * 43758.5453); }
void main() {
  vec3 col = texture2D(tColor, vUv).rgb;
  if (uVolOn > 0.5) {
    float d = texture2D(tDepth, vUv).r;
    vec4 ndc = vec4(vUv * 2.0 - 1.0, d * 2.0 - 1.0, 1.0);
    vec4 vp = uInvProj * ndc; vp /= vp.w; vec3 wp = (uInvView * vp).xyz;
    vec4 fp = uInvProj * vec4(vUv * 2.0 - 1.0, 1.0, 1.0); fp /= fp.w; vec3 far = (uInvView * fp).xyz;
    vec3 ro = uCam, rd = normalize(far - ro);
    float tScene = d >= 0.999999 ? 1e9 : length(wp - ro);
    vec3 inv = 1.0 / rd; vec3 t0 = (uBoxMin - ro) * inv, t1 = (uBoxMax - ro) * inv;
    vec3 tmin = min(t0, t1), tmax = max(t0, t1);
    float ta = max(max(tmin.x, tmin.y), max(tmin.z, 0.0)), tb = min(min(tmax.x, tmax.y), tmax.z);
    tb = min(tb, tScene);
    if (tb > ta) {
      float n = uSteps; float dt = (tb - ta) / n; float t = ta + dt * hash(gl_FragCoord.xy);
      vec3 acc = vec3(0.0); float A = 0.0; vec3 sz = uBoxMax - uBoxMin;
      for (int i = 0; i < 160; i++) {
        if (float(i) >= n || A > 0.985) break;
        vec3 p = ro + rd * t; vec3 q = (p - uBoxMin) / sz;          // q.x east, q.y up, q.z south
        vec3 tc = vec3(q.x, q.z, q.y);
        float th = mix(texture(uVolA, tc).r, texture(uVolB, tc).r, uVolMix) * uVolScale;
        vec3 eg = min(q, 1.0 - q);
        th *= smoothstep(0.0, 0.06, eg.x) * smoothstep(0.0, 0.06, eg.z) * smoothstep(0.0, 0.14, eg.y);
        if (th > uVolLo) {
          float w = smoothstep(uVolLo, uVolLo + 0.9, th);
          float sigma = uVolDens * w * min(th, 1.6) * 0.0036;
          float a = 1.0 - exp(-sigma * dt);
          vec4 rc = texture2D(uRamp, vec2(clamp(th / uRampMax, 0.0, 1.0), 0.5));
          vec3 c = mix(rc.rgb, vec3(1.0, 0.93, 0.80), 0.18) * (0.92 + 0.35 * q.y);
          acc += (1.0 - A) * a * c; A += (1.0 - A) * a;
        }
        t += dt;
      }
      col = col * (1.0 - A) + acc;
    }
  }
  col = neutral(col * uExpo);
  gl_FragColor = vec4(toSRGB(clamp(col, 0.0, 1.0)), 1.0);
}`
    });
    this.compQuad = new THREE.Mesh(new THREE.PlaneGeometry(2, 2), mat); this.compQuad.frustumCulled = false;
    this.compScene = new THREE.Scene(); this.compScene.add(this.compQuad); this.compCam = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);
  }

  _section() {
    const CU = this.CU;
    this.SU = { uVolA: CU.uVolA, uVolB: CU.uVolB, uVolMix: CU.uVolMix, uVolScale: CU.uVolScale, uRamp: CU.uRamp, uRampMax: CU.uRampMax,
      uBoxMinL: { value: new THREE.Vector3() }, uBoxMaxL: { value: new THREE.Vector3(1, 1, 1) }, uDem: { value: this.demTex },
      uPaper: { value: new THREE.Color('#FBFAF6') }, uInk: this.U.uInk, uSecMax: { value: 1.0 }, uTopL: { value: 166 } };
    const mat = new THREE.ShaderMaterial({ uniforms: this.SU, side: THREE.DoubleSide, transparent: true, depthWrite: false,
      vertexShader: 'varying vec3 vL; void main(){ vL = position; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }',
      fragmentShader: `precision highp float; precision highp sampler3D;
        uniform sampler3D uVolA; uniform sampler3D uVolB; uniform float uVolMix; uniform float uVolScale; uniform vec3 uBoxMinL; uniform vec3 uBoxMaxL;
        uniform sampler2D uRamp; uniform float uRampMax; uniform float uSecMax; uniform float uTopL; uniform sampler2D uDem; uniform vec3 uPaper; uniform vec3 uInk; varying vec3 vL;
        void main(){
          vec3 q = clamp((vL - uBoxMinL) / (uBoxMaxL - uBoxMinL), 0.0, 1.0); vec3 tc = vec3(q.x, q.z, q.y);
          float th = mix(texture(uVolA, tc).r, texture(uVolB, tc).r, uVolMix) * uVolScale;
          vec4 rc = texture2D(uRamp, vec2(clamp(th / uSecMax, 0.0, 1.0), 0.5));
          float wa = smoothstep(0.015, 0.12, th) * max(rc.a, 0.45); vec3 col = mix(uPaper, rc.rgb, wa); float alpha = 0.16 + 0.8 * wa;
          float g = texture2D(uDem, vec2((vL.x - (${X0.toFixed(1)})) / ${SIZE.toFixed(1)}, (vL.z - (${Z0.toFixed(1)})) / ${SIZE.toFixed(1)})).r;
          float h = vL.y - g; float fw = max(fwidth(h), 1e-3); float grid = 0.0;
          for (int k = 1; k <= 3; k++) grid = max(grid, 1.0 - smoothstep(0.4 * fw, 1.4 * fw, abs(h - 100.0 * float(k))));
          float ft = max(fwidth(th), 1e-4); float iso = 0.0; vec4 L = vec4(0.1, 0.25, 0.5, 1.0);
          for (int i = 0; i < 4; i++) iso = max(iso, 1.0 - smoothstep(0.4 * ft, 1.4 * ft, abs(th - L[i])));
          col = mix(col, uInk, grid * 0.35); col = mix(col, uInk, iso * 0.55); alpha = max(alpha, max(grid * 0.45, iso * 0.6));
          float edge = 1.0 - smoothstep(0.0, 2.0 * fwidth(vL.y), abs(vL.y - uTopL)); col = mix(col, uInk, edge * 0.8); alpha = max(alpha, edge * 0.85);
          gl_FragColor = vec4(col * 0.98, alpha);
        }` });
    this.section = new THREE.Mesh(new THREE.BufferGeometry(), mat); this.section.visible = false; this.section.frustumCulled = false; this.section.renderOrder = 1;
    this.world.add(this.section);
  }
  /** vertical section card along a line through (cx, cz), oriented the way the wind blows (dirTo, compass degrees) */
  setSection(cx, cz, dirTo, len = 5600, on = true) {
    const sec = this.section; sec.visible = on; if (!on) return;
    const t = THREE.MathUtils.degToRad(dirTo), dx = Math.sin(t), dz = -Math.cos(t);
    let ta = -len / 2, tb = len / 2;
    for (const [o, d, lo, hi] of [[cx, dx, X0 + 2, X1 - 2], [cz, dz, Z0 + 2, Z1 - 2]]) { if (Math.abs(d) < 1e-6) continue; let t1 = (lo - o) / d, t2 = (hi - o) / d; if (t1 > t2) [t1, t2] = [t2, t1]; ta = Math.max(ta, t1); tb = Math.min(tb, t2); }
    const n = 240, pos = [], idx = [], top = 230 - 33.5; this.SU.uTopL.value = top;
    for (let i = 0; i <= n; i++) { const u = ta + (tb - ta) * i / n, x = cx + dx * u, z = cz + dz * u; pos.push(x, this.heightAt(x, z) - 0.6, z, x, top, z); if (i < n) { const a = i * 2; idx.push(a, a + 1, a + 2, a + 1, a + 3, a + 2); } }
    const g = new THREE.BufferGeometry(); g.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3)); g.setIndex(idx);
    sec.geometry.dispose(); sec.geometry = g;
    this.sectionLine = { x0: cx + dx * ta, z0: cz + dz * ta, x1: cx + dx * tb, z1: cz + dz * tb, top };
  }
  setVolume(volA, volB, mix, scale, box) {
    const CU = this.CU;
    if (!volA) { CU.uVolOn.value = 0; return; }
    CU.uVolA.value = volA; CU.uVolB.value = volB || volA; CU.uVolMix.value = mix || 0; CU.uVolScale.value = scale; CU.uVolOn.value = 1;
    this.volBox = box;   // local {x0,x1,y0,y1,z0,z1}
    if (this.SU) { this.SU.uBoxMinL.value.set(box.x0, box.y0, box.z0); this.SU.uBoxMaxL.value.set(box.x1, box.y1, box.z1); }
  }

  resize() {
    const w = this.canvas.clientWidth || window.innerWidth, h = this.canvas.clientHeight || window.innerHeight;
    const pr = Math.min(window.devicePixelRatio || 1, 2) * this.quality.scale;
    this.renderer.setPixelRatio(pr); this.renderer.setSize(w, h, false);
    this.camera.aspect = w / h; this.applyOffset(w, h);
    const W = Math.round(w * pr), Hh = Math.round(h * pr);
    this.rt.setSize(W, Hh); this.CU.uRes.value.set(W, Hh);
    this.width = w; this.height = h;
  }

  /** shift the projection centre so the model is centred in the free area right of the side panel */
  applyOffset(w = this.width, h = this.height) {
    const shift = this.panelShift ?? 0;
    if (shift && w > 900) this.camera.setViewOffset(w, h, -shift, 0, w, h); else this.camera.clearViewOffset();
    this.camera.updateProjectionMatrix();
  }
  render(now) {
    const c = this.camera, CU = this.CU;
    this.U.uTime.value = now / 1000;
    c.updateMatrixWorld();
    CU.uInvProj.value.copy(c.projectionMatrixInverse); CU.uInvView.value.copy(c.matrixWorld); CU.uCam.value.copy(c.position);
    if (this.volBox) { const b = this.volBox, v = this.VE; CU.uBoxMin.value.set(b.x0, b.y0 * v, b.z0); CU.uBoxMax.value.set(b.x1, b.y1 * v, b.z1); }
    CU.uFrame.value = (CU.uFrame.value + 1) % 64;
    const r = this.renderer;
    r.setRenderTarget(this.rt); r.render(this.scene, c);
    r.setRenderTarget(null); r.render(this.compScene, this.compCam);
  }

  // ------------------------------------------------------------------ camera
  get views() {
    return {
      home: { target: [60, 0, 180], dist: 12900, az: 204, el: 42 },
      top: { target: [5, 0, 5], dist: 11800, az: 180, el: 86 },
      cluster: { target: [-60, 0, -300], dist: 5600, az: 212, el: 36 },
      plume: { target: [150, 60, -700], dist: 9400, az: 116, el: 31 },
      lst: { target: [30, 0, 40], dist: 13600, az: 182, el: 57 },
      sites: { target: [-160, 0, 120], dist: 11200, az: 198, el: 45 },
      live: { target: [40, 0, 160], dist: 12400, az: 205, el: 47 }
    };
  }
  viewToPose(v) {
    const az = THREE.MathUtils.degToRad(v.az), el = THREE.MathUtils.degToRad(v.el);
    const t = new THREE.Vector3(v.target[0], v.target[1] * this.VE, v.target[2]);
    const p = new THREE.Vector3(Math.sin(az) * Math.cos(el), Math.sin(el), -Math.cos(az) * Math.cos(el)).multiplyScalar(v.dist).add(t);
    return { p, t };
  }
  setView(v, ms = 1800, done) {
    const { p, t } = this.viewToPose(v);
    if (!ms) { this.camera.position.copy(p); this.controls.target.copy(t); this.controls.update(); return; }
    const p0 = this.camera.position.clone(), t0 = this.controls.target.clone();
    // travel along a gentle arc: interpolate target linearly, camera in spherical coordinates about the target
    const s0 = new THREE.Spherical().setFromVector3(p0.clone().sub(t0)), s1 = new THREE.Spherical().setFromVector3(p.clone().sub(t));
    let dth = s1.theta - s0.theta; if (dth > Math.PI) dth -= 2 * Math.PI; if (dth < -Math.PI) dth += 2 * Math.PI;
    this.tweens.cancelAll(); this.controls.enabled = false;
    const sp = new THREE.Spherical();
    this.tweens.add(ms, u => {
      const tt = t0.clone().lerp(t, u);
      sp.set(lerp(s0.radius, s1.radius, u) * (1 + 0.08 * Math.sin(Math.PI * u)), lerp(s0.phi, s1.phi, u), s0.theta + dth * u);
      this.camera.position.setFromSpherical(sp).add(tt); this.controls.target.copy(tt); this.camera.lookAt(tt);
    }, () => { this.controls.enabled = true; this.controls.update(); done && done(); });
  }
  clampTarget() {
    const t = this.controls.target; t.x = clamp(t.x, X0 - 400, X1 + 400); t.z = clamp(t.z, Z0 - 400, Z1 + 400); t.y = clamp(t.y, -200, 800);
  }

  // ------------------------------------------------------------------ picking
  ndc(clientX, clientY) { const r = this.canvas.getBoundingClientRect(); return new THREE.Vector2(((clientX - r.left) / r.width) * 2 - 1, -((clientY - r.top) / r.height) * 2 + 1); }
  rayFrom(clientX, clientY) { this._raycaster.setFromCamera(this.ndc(clientX, clientY), this.camera); return this._raycaster.ray; }
  /** heightfield intersection (fast); returns local point {x, y, z} or null */
  pickGround(clientX, clientY) {
    const ray = this.rayFrom(clientX, clientY), o = ray.origin, d = ray.direction, ve = this.VE;
    const ymax = (this.hmax + 2) * ve, ymin = (this.hmin - 2) * ve;
    // clip to the box
    let ta = 0, tb = 1e6;
    for (const [oo, dd, lo, hi] of [[o.x, d.x, X0, X1], [o.y, d.y, ymin, ymax], [o.z, d.z, Z0, Z1]]) {
      if (Math.abs(dd) < 1e-9) { if (oo < lo || oo > hi) return null; continue; }
      let t1 = (lo - oo) / dd, t2 = (hi - oo) / dd; if (t1 > t2) [t1, t2] = [t2, t1]; ta = Math.max(ta, t1); tb = Math.min(tb, t2);
    }
    if (tb <= ta) return null;
    const step = 6; let tPrev = ta, prevAbove = true;
    for (let t = ta; t <= tb + step; t += step) {
      const tt = Math.min(t, tb); const x = o.x + d.x * tt, y = o.y + d.y * tt, z = o.z + d.z * tt;
      const above = y > this.heightAt(x, z) * ve;
      if (!above) {
        let a = tPrev, b = tt; for (let k = 0; k < 18; k++) { const m = (a + b) / 2; const xm = o.x + d.x * m, ym = o.y + d.y * m, zm = o.z + d.z * m; if (ym > this.heightAt(xm, zm) * ve) a = m; else b = m; }
        const x2 = o.x + d.x * b, z2 = o.z + d.z * b; return { x: x2, y: this.heightAt(x2, z2), z: z2 };
      }
      tPrev = tt; if (tt >= tb) break;
    }
    return null;
  }
  pickBuilding(clientX, clientY) {
    this._raycaster.setFromCamera(this.ndc(clientX, clientY), this.camera);
    const hit = this._raycaster.intersectObject(this.buildings, false)[0];
    if (!hit) return null;
    const ia = this.buildings.geometry.getAttribute('aInfo'); const vi = hit.face.a;
    return { kind: ia.getX(vi), campus: ia.getZ(vi), point: hit.point, distance: hit.distance };
  }
  /** true if the section card hides the local point (x, y, z) from the camera */
  hiddenBySection(x, y, z) {
    if (!this.section || !this.section.visible) return false;
    const p = new THREE.Vector3(x, y * this.VE, z), o = this.camera.position, d = p.clone().sub(o); const L = d.length(); d.normalize();
    this._raycaster.set(o, d); this._raycaster.far = L - 5;
    const hit = this._raycaster.intersectObject(this.section, false).length > 0; this._raycaster.far = Infinity; return hit;
  }
  project(x, y, z) {   // local -> screen px (y local metres)
    const v = new THREE.Vector3(x, y * this.VE, z).project(this.camera);
    return { x: (v.x * 0.5 + 0.5) * this.width, y: (-v.y * 0.5 + 0.5) * this.height, z: v.z, visible: v.z > -1 && v.z < 1 };
  }
}
