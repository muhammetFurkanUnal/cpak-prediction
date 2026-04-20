// ── Shared coordinate tooltip (body-level to avoid overflow:hidden clipping) ──
const _dotTooltip = document.createElement('div');
_dotTooltip.className = 'dot-tooltip';
document.body.appendChild(_dotTooltip);

// ── DotHandle ─────────────────────────────────────────────────────────────────
class DotHandle {
  constructor(container, viewer, key, ix, iy, color) {
    this.container = container;
    this.viewer    = viewer;
    this.key       = key;
    this.ix        = ix;
    this.iy        = iy;
    this.color     = color;
    this.onMove    = null;
    this._el       = null;
    this._build();
    this._makeDraggable();
  }

  _build() {
    const el = document.createElement('div');
    el.className = 'dot-handle';
    el.style.setProperty('--dc', this.color);

    el.addEventListener('mouseenter', e => {
      _dotTooltip.textContent = `${Math.round(this.ix)}, ${Math.round(this.iy)}`;
      _dotTooltip.classList.add('visible');
      this._positionTooltip(e);
    });
    el.addEventListener('mousemove', e => this._positionTooltip(e));
    el.addEventListener('mouseleave', () => _dotTooltip.classList.remove('visible'));

    this.container.appendChild(el);
    this._el = el;
    this.reposition();
  }

  _positionTooltip(e) {
    _dotTooltip.style.left = `${e.clientX + 14}px`;
    _dotTooltip.style.top  = `${e.clientY - 32}px`;
  }

  reposition() {
    const sx = this.viewer.tx + this.ix * this.viewer.scale;
    const sy = this.viewer.ty + this.iy * this.viewer.scale;
    this._el.style.left = `${sx}px`;
    this._el.style.top  = `${sy}px`;
  }

  screenPos() {
    return {
      x: this.viewer.tx + this.ix * this.viewer.scale,
      y: this.viewer.ty + this.iy * this.viewer.scale,
    };
  }

  _makeDraggable() {
    const el = this._el;
    let startMX, startMY, startIX, startIY;

    el.addEventListener('mousedown', e => {
      if (e.button !== 0) return;
      e.stopPropagation();
      e.preventDefault();
      _dotTooltip.classList.remove('visible');
      startMX = e.clientX; startMY = e.clientY;
      startIX = this.ix;   startIY = this.iy;
      el.classList.add('dragging');

      const onMove = e => {
        this.ix = startIX + (e.clientX - startMX) / this.viewer.scale;
        this.iy = startIY + (e.clientY - startMY) / this.viewer.scale;
        this.reposition();
        if (this.onMove) this.onMove();
      };
      const onUp = () => {
        el.classList.remove('dragging');
        window.removeEventListener('mousemove', onMove);
        window.removeEventListener('mouseup',   onUp);
      };
      window.addEventListener('mousemove', onMove);
      window.addEventListener('mouseup',   onUp);
    });

    el.addEventListener('dblclick', e => e.stopPropagation());
  }

  remove() { this._el?.remove(); this._el = null; }
  getImagePos() { return { ix: this.ix, iy: this.iy }; }
}

// ── AxesOverlay ───────────────────────────────────────────────────────────────
// Manages SVG lines, dot handles, angle labels, and the reset button.
class AxesOverlay {
  constructor(container, viewer, config) {
    this.container = container;
    this.viewer    = viewer;
    this.imgHeight = config.imgHeight;
    this.lw        = config.lw;

    // Deep-copy originals for reset
    this._origDots   = JSON.parse(JSON.stringify(config.dots));
    this._origLabels = JSON.parse(JSON.stringify(config.labels));

    this.dots            = {};
    this.labels          = [];
    this.labelMap        = {};   // key → AngleLabel, populated in _buildLabels
    this._svgLines       = {};
    this._lineDefs       = null;
    this.svg             = null;
    this.resetBtn        = null;
    this._modified       = false;
    this.onAnglesChange  = null; // callback(cpakResult)

    this._buildSvg();
    this._buildDots(config.dots);
    this._buildLabels(config.labels);
    this._buildResetBtn();
    this._updateLines();

    viewer.onTransformChange = () => {
      Object.values(this.dots).forEach(d => d.reposition());
      this.labels.forEach(l => l.reposition());
      this._updateLines();
    };
  }

  _buildSvg() {
    this._lineDefs = [
      { key: 'femurCondyle', from: 'femurLateral',    to: 'femurMedial',       color: '#00e676', ext: false },
      { key: 'tibiaPlat',    from: 'tibiaLateral',    to: 'tibiaMedial',        color: '#00e676', ext: false },
      { key: 'ankle',        from: 'ankleLateral',    to: 'ankleMedial',        color: '#00e676', ext: false },
      { key: 'ldfaAxis',     from: 'femurHead',       to: 'femurNotch',         color: '#22d3ee', ext: true  },
      { key: 'mptaAxis',     from: 'finalAnkleMiddle',to: 'tibiaIntercondiler', color: '#fb923c', ext: true  },
    ];

    const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
    svg.style.cssText = 'position:absolute;inset:0;width:100%;height:100%;pointer-events:none;z-index:5;overflow:visible;';

    for (const def of this._lineDefs) {
      const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
      line.setAttribute('stroke', def.color);
      line.setAttribute('stroke-linecap', 'round');
      svg.appendChild(line);
      this._svgLines[def.key] = line;
    }

    this.container.insertBefore(svg, this.container.firstChild);
    this.svg = svg;
  }

  _buildDots(dots) {
    const colors = {
      femurHead:          '#00e676',
      femurLateral:       '#4488ff',
      femurMedial:        '#4488ff',
      femurNotch:         '#22d3ee',
      tibiaLateral:       '#4488ff',
      tibiaMedial:        '#4488ff',
      tibiaIntercondiler: '#fb923c',
      ankleLateral:       '#4488ff',
      ankleMedial:        '#4488ff',
      finalAnkleMiddle:   '#ffdd00',
    };
    for (const [key, color] of Object.entries(colors)) {
      if (!dots[key]) continue;
      const dot = new DotHandle(this.container, this.viewer, key, dots[key].ix, dots[key].iy, color);
      dot.onMove = () => { this._updateLines(); this._markModified(); this._updateAngles(); };
      this.dots[key] = dot;
    }
  }

  _buildLabels(labels) {
    this.labels = labels.map(pos => {
      const lbl = new AngleLabel(this.container, this.viewer, pos);
      lbl.onMove = () => this._markModified();
      if (pos.key) this.labelMap[pos.key] = lbl;
      return lbl;
    });
  }

  _buildResetBtn() {
    const btn = document.createElement('button');
    btn.className = 'axes-reset-btn';
    btn.innerHTML = `<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round"><path d="M3 12a9 9 0 1 0 9-9 9.75 9.75 0 0 0-6.74 2.74L3 8"/><path d="M3 3v5h5"/></svg> Reset Positions`;
    btn.style.display = 'none';
    btn.addEventListener('click', () => this.reset());
    btn.addEventListener('mousedown', e => e.stopPropagation());
    btn.addEventListener('dblclick',  e => e.stopPropagation());
    this.container.appendChild(btn);
    this.resetBtn = btn;
  }

  _markModified() {
    if (!this._modified) {
      this._modified = true;
      this.resetBtn.style.display = '';
    }
  }

  _updateLines() {
    const extPx = this.imgHeight * 0.03;
    const sw    = Math.max(0.5, this.lw * this.viewer.scale);

    for (const def of this._lineDefs) {
      const aDot = this.dots[def.from];
      const bDot = this.dots[def.to];
      if (!aDot || !bDot) continue;

      const pa = aDot.screenPos();
      let   pb;

      if (def.ext) {
        const dx  = bDot.ix - aDot.ix;
        const dy  = bDot.iy - aDot.iy;
        const len = Math.hypot(dx, dy) || 1;
        const exIx = bDot.ix + (dx / len) * extPx;
        const exIy = bDot.iy + (dy / len) * extPx;
        pb = {
          x: this.viewer.tx + exIx * this.viewer.scale,
          y: this.viewer.ty + exIy * this.viewer.scale,
        };
      } else {
        pb = bDot.screenPos();
      }

      const el = this._svgLines[def.key];
      el.setAttribute('x1', pa.x);  el.setAttribute('y1', pa.y);
      el.setAttribute('x2', pb.x);  el.setAttribute('y2', pb.y);
      el.setAttribute('stroke-width', sw);
    }
  }

  _updateAngles() {
    const angles = computeAngles(this.dots);
    if (!angles) return;
    const result = classifyCPAK(angles.ldfa, angles.mpta);

    if (this.labelMap['ldfa']) this.labelMap['ldfa'].updateText(`LDFA  ${result.ldfa.toFixed(1)}°`);
    if (this.labelMap['mpta']) this.labelMap['mpta'].updateText(`MPTA  ${result.mpta.toFixed(1)}°`);

    if (this.onAnglesChange) this.onAnglesChange(result);
  }

  reset() {
    for (const [key, orig] of Object.entries(this._origDots)) {
      const d = this.dots[key];
      if (d) { d.ix = orig.ix; d.iy = orig.iy; d.reposition(); }
    }
    for (let i = 0; i < this.labels.length; i++) {
      const orig = this._origLabels[i];
      const lbl  = this.labels[i];
      if (orig && lbl) { lbl.ix = orig.ix; lbl.iy = orig.iy; lbl.reposition(); }
    }
    this._updateLines();
    this._modified = false;
    this.resetBtn.style.display = 'none';
  }

  remove() {
    Object.values(this.dots).forEach(d => d.remove());
    this.labels.forEach(l => l.remove());
    this.svg?.remove();
    this.resetBtn?.remove();
    if (this.viewer) this.viewer.onTransformChange = null;
  }

  // Returns dot positions as {key: {x,y}} for canvas drawing (download)
  getDotPositions() {
    const out = {};
    for (const [key, dot] of Object.entries(this.dots)) out[key] = { x: dot.ix, y: dot.iy };
    return out;
  }

  getLabelPositions() {
    return this.labels.map(l => l.getImagePosition());
  }
}

// ── AngleLabel ────────────────────────────────────────────────────────────────
class AngleLabel {
  constructor(container, viewer, labelData) {
    this.container = container;
    this.viewer    = viewer;
    this.text      = labelData.text;
    this.color     = labelData.color;
    this.ix        = labelData.ix;
    this.iy        = labelData.iy;
    this.onMove    = null;
    this._el       = null;
    this._build();
    this._makeDraggable();
  }

  _build() {
    const el = document.createElement('div');
    el.className = 'angle-label';
    el.style.setProperty('--lc', this.color);
    el.innerHTML = `
      <svg class="angle-label-grip" width="8" height="12" viewBox="0 0 8 12" fill="currentColor">
        <circle cx="2" cy="2"  r="1.1"/><circle cx="6" cy="2"  r="1.1"/>
        <circle cx="2" cy="6"  r="1.1"/><circle cx="6" cy="6"  r="1.1"/>
        <circle cx="2" cy="10" r="1.1"/><circle cx="6" cy="10" r="1.1"/>
      </svg>
      <span>${this.text}</span>`;
    this.container.appendChild(el);
    this._el = el;
    this.reposition();
  }

  reposition() {
    const sx = this.viewer.tx + this.ix * this.viewer.scale;
    const sy = this.viewer.ty + this.iy * this.viewer.scale;
    this._el.style.left = `${sx}px`;
    this._el.style.top  = `${sy}px`;
  }

  _makeDraggable() {
    const el = this._el;
    let startMX, startMY, startIX, startIY;

    el.addEventListener('mousedown', e => {
      if (e.button !== 0) return;
      e.stopPropagation();
      e.preventDefault();
      startMX = e.clientX; startMY = e.clientY;
      startIX = this.ix;   startIY = this.iy;
      el.classList.add('dragging');

      const onMove = e => {
        this.ix = startIX + (e.clientX - startMX) / this.viewer.scale;
        this.iy = startIY + (e.clientY - startMY) / this.viewer.scale;
        this.reposition();
        if (this.onMove) this.onMove();
      };
      const onUp = () => {
        el.classList.remove('dragging');
        window.removeEventListener('mousemove', onMove);
        window.removeEventListener('mouseup',   onUp);
      };
      window.addEventListener('mousemove', onMove);
      window.addEventListener('mouseup',   onUp);
    });

    el.addEventListener('dblclick', e => e.stopPropagation());
  }

  updateText(text) {
    this.text = text;
    if (this._el) this._el.querySelector('span').textContent = text;
  }

  remove() { this._el?.remove(); this._el = null; }

  getImagePosition() {
    return { ix: this.ix, iy: this.iy, text: this.text, color: this.color };
  }
}
