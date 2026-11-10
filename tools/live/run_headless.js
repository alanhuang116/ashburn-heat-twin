#!/usr/bin/env node
// Headless WebGPU test runner for the live solver.
//   node tools/live/run_headless.js "<query>" [--dump <dir>] [--timeout <s>] [--chrome <exe>] [--playwright <path>]
// e.g. node tools/live/run_headless.js "test=short&dx=30&steps=300" --dump C:/tmp/live_val
// Serves the repository root on http://localhost:<port> (WebGPU needs a secure context), opens
// tests/live/validate.html?<query> in headless Chrome with WebGPU enabled, relays the console, and waits for the
// page to set window.__result. The page writes binary/JSON dumps through POST /__dump/<name> into --dump.
'use strict';
const http = require('http');
const fs = require('fs');
const path = require('path');
const os = require('os');

const args = process.argv.slice(2);
const opt = (k, d) => { const i = args.indexOf(k); return i >= 0 ? args[i + 1] : d; };
const query = args.find(a => !a.startsWith('--') && args[args.indexOf(a) - 1]?.startsWith('--') !== true) || 'test=smoke';
const DUMP = path.resolve(opt('--dump', path.join(os.tmpdir(), 'live_val')));
const TIMEOUT = +opt('--timeout', 1800) * 1000;
const CHROME = opt('--chrome', 'C:/Program Files/Google/Chrome/Application/chrome.exe');
const PW = opt('--playwright', process.env.PLAYWRIGHT_CORE || 'playwright-core');
const ROOT = path.resolve(__dirname, '..', '..');
fs.mkdirSync(DUMP, { recursive: true });

const MIME = { '.html': 'text/html', '.js': 'text/javascript', '.mjs': 'text/javascript', '.json': 'application/json', '.css': 'text/css',
  '.gz': 'application/gzip', '.bin': 'application/octet-stream', '.png': 'image/png', '.jpg': 'image/jpeg', '.wasm': 'application/wasm' };

const server = http.createServer((req, res) => {
  const url = new URL(req.url, 'http://localhost');
  if (url.pathname.startsWith('/__dump/')) {
    const name = path.basename(decodeURIComponent(url.pathname.slice(8)));
    const file = path.join(DUMP, name);
    if (req.method === 'POST') {
      const out = fs.createWriteStream(file, { flags: url.searchParams.get('append') === '1' ? 'a' : 'w' });
      req.pipe(out);
      out.on('finish', () => { res.writeHead(200); res.end('ok'); });
      out.on('error', e => { res.writeHead(500); res.end(String(e)); });
      return;
    }
    if (!fs.existsSync(file)) { res.writeHead(404); res.end('no ' + name); return; }
    res.writeHead(200, { 'content-type': 'application/octet-stream' });
    fs.createReadStream(file).pipe(res);
    return;
  }
  let p = path.normalize(decodeURIComponent(url.pathname)).replace(/^[\\/]+/, '');
  const file = path.join(ROOT, p);
  if (!file.startsWith(ROOT) || !fs.existsSync(file) || fs.statSync(file).isDirectory()) { res.writeHead(404); res.end('not found'); return; }
  // like GitHub Pages: .gz files are served raw (no Content-Encoding), the client gunzips them
  res.writeHead(200, { 'content-type': MIME[path.extname(file)] || 'application/octet-stream', 'cache-control': 'no-store' });
  fs.createReadStream(file).pipe(res);
});

(async () => {
  let chromium;
  try { ({ chromium } = require(PW)); } catch (e) {
    console.error(`cannot load playwright-core (${PW}); pass --playwright <path to node_modules/playwright-core>`); process.exit(2);
  }
  await new Promise(r => server.listen(0, '127.0.0.1', r));
  const port = server.address().port;
  const browser = await chromium.launch({ executablePath: CHROME, headless: true, args: ['--enable-unsafe-webgpu', '--ignore-gpu-blocklist'] });
  const page = await browser.newPage();
  page.on('console', m => console.log('[page]', m.text()));
  page.on('pageerror', e => console.log('[page error]', e.message));
  const t0 = Date.now();
  await page.goto(`http://localhost:${port}/tests/live/validate.html?${query}`);
  let result;
  try {
    await page.waitForFunction(() => window.__result !== undefined, null, { timeout: TIMEOUT, polling: 500 });
    result = await page.evaluate(() => window.__result);
  } catch (e) {
    result = { ok: false, error: 'timeout or page failure: ' + e.message };
  }
  const name = (new URLSearchParams(query).get('tag') || new URLSearchParams(query).get('test') || 'run') + '_result.json';
  fs.writeFileSync(path.join(DUMP, name), JSON.stringify(result, null, 1));
  console.log(`[runner] ${query} finished in ${((Date.now() - t0) / 1000).toFixed(1)} s -> ${path.join(DUMP, name)}`);
  console.log(JSON.stringify(result, null, 1).slice(0, 4000));
  await browser.close();
  server.close();
  process.exit(result && result.ok ? 0 : 1);
})().catch(e => { console.error('runner error', e); process.exit(1); });
