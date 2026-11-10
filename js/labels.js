// Map annotations: small paper tags on a hairline leader, anchored to 3-D points; collision-aware, priority-ordered.
export class Labels {
  constructor(root, Q) {
    this.root = root; this.Q = Q; this.items = []; this.visible = true;
  }
  /** list: [{ id, text, sub, kind, x, y, z, prio, cls, below, stem }] (local metres) */
  set(list) {
    const keep = new Map(this.items.map(it => [it.id, it]));
    const next = [];
    for (const d of list) {
      let it = keep.get(d.id);
      if (!it) {
        const el = document.createElement('div'); el.className = 'lab lab-' + (d.kind || 'place') + (d.cls ? ' ' + d.cls : '');
        el.innerHTML = `<span class="tag"></span><i class="stem"></i><b class="dot"></b>`;
        this.root.appendChild(el); it = { el, id: d.id, alpha: 0 };
        requestAnimationFrame(() => el.classList.add('in'));
      }
      keep.delete(d.id);
      const tag = it.el.querySelector('.tag');
      const html = d.html || (escapeHtml(d.text) + (d.sub ? `<small>${escapeHtml(d.sub)}</small>` : ''));
      if (tag.innerHTML !== html) tag.innerHTML = html;
      Object.assign(it, d); it.el.classList.toggle('below', !!d.below);
      next.push(it);
    }
    for (const it of keep.values()) { it.el.classList.remove('in'); const el = it.el; setTimeout(() => el.remove(), 400); }
    this.items = next.sort((a, b) => (a.prio ?? 5) - (b.prio ?? 5));
    this._sizes = null;
  }
  clear() { this.set([]); }
  setVisible(v) { this.visible = v; this.root.style.opacity = v ? 1 : 0; }
  update() {
    if (!this.visible) return;
    const Q = this.Q, W = Q.width, H = Q.height, placed = [];
    const margin = 8;
    // keep tags clear of the floating panels (side rail, legend, timeline)
    this.reserved = [];
    for (const sel of ['#rail', '#legend', '#timeline', '#probe']) { const el = document.querySelector(sel); if (el && !el.hidden && el.offsetParent !== null) { const r = el.getBoundingClientRect(); this.reserved.push({ x0: r.left - 6, x1: r.right + 6, y0: r.top - 6, y1: r.bottom + 6 }); } }
    for (const it of this.items) {
      const p = Q.project(it.x, it.y ?? Q.heightAt(it.x, it.z), it.z);
      const stem = it.stem ?? 26;
      let show = p.visible && p.x > 10 && p.x < W - 10 && p.y > 60 && p.y < H - 20 && (it.below || p.y - stem - (it.h || 22) > 84);
      if (show && Q.section && Q.section.visible && Q.hiddenBySection(it.x, it.y ?? Q.heightAt(it.x, it.z), it.z)) show = false;
      if (!it.w || it._text !== it.el.textContent) { it.w = it.el.querySelector('.tag').offsetWidth || 80; it.h = it.el.querySelector('.tag').offsetHeight || 22; it._text = it.el.textContent; }
      const below = !!it.below;
      const box = { x0: p.x - it.w / 2 - margin, x1: p.x + it.w / 2 + margin, y0: below ? p.y + stem : p.y - stem - it.h - margin, y1: below ? p.y + stem + it.h + margin : p.y - stem };
      if (show && it.avoid !== false) for (const b of placed) if (box.x0 < b.x1 && box.x1 > b.x0 && box.y0 < b.y1 && box.y1 > b.y0) { show = false; break; }
      for (const r of (this.reserved || [])) if (show && box.x0 < r.x1 && box.x1 > r.x0 && box.y0 < r.y1 && box.y1 > r.y0) show = false;
      if (show) placed.push(box);
      it.el.style.transform = `translate(${p.x.toFixed(1)}px, ${p.y.toFixed(1)}px)`;
      it.el.style.setProperty('--stem', stem + 'px');
      it.el.classList.toggle('hide', !show);
    }
  }
}
export const escapeHtml = s => String(s ?? '').replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
