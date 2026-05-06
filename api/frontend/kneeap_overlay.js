// KneeAP rendering — mirrors notebooks/lib/kneeap_inference.draw_lines in JS so
// drawing happens in the browser instead of the server.
//
// Toggle KNEEAP_DOTS_DRAGGABLE to true to make every keypoint draggable.
// Drag handlers re-fit anatomical axes on move; JLCA is NOT recomputed live yet.
const KNEEAP_DOTS_DRAGGABLE = false;

// ── Bodypart indices (mirror kneeap_inference.py, 0-based) ─────────────────
const KAP_NOTCH         = 0;
const KAP_F_LAT_JOINT   = 1;
const KAP_F_MED_JOINT   = 8;
const KAP_T_INTER_LAT   = 15;
const KAP_T_LAT_JOINT   = 16;
const KAP_T_INTER_MED   = 23;

const KAP_F_LAT_CHAIN = [1, 5, 3, 6, 4, 7, 2];
const KAP_F_MED_CHAIN = [8, 12, 10, 13, 11, 14, 9];
const KAP_T_LAT_CHAIN = [16, 20, 18, 21, 19, 22, 17];
const KAP_T_MED_CHAIN = [23, 27, 25, 28, 26, 29, 24];

// Colors — RGB equivalents of the BGR tuples used in draw_lines (Python/cv2).
const KAP_COLOR_F_LAT      = 'rgb(0,200,255)';   // cv2 BGR(255,200,0)
const KAP_COLOR_F_MED      = 'rgb(0,100,255)';   // cv2 BGR(255,100,0)
const KAP_COLOR_T_LAT      = 'rgb(255,200,0)';   // cv2 BGR(0,200,255)
const KAP_COLOR_T_MED      = 'rgb(255,100,0)';   // cv2 BGR(0,100,255)
const KAP_COLOR_NOTCH      = 'rgb(0,0,255)';     // cv2 BGR(255,0,0)
const KAP_COLOR_INTER      = 'rgb(255,0,0)';     // cv2 BGR(0,0,255)
const KAP_COLOR_JOINT      = 'rgb(0,255,0)';
const KAP_COLOR_FEMUR_AXIS = 'rgb(170,90,170)';  // muted purple
const KAP_COLOR_TIBIA_AXIS = 'rgb(170,170,90)';  // muted teal/khaki

const KAP_CHAIN_GROUPS = [
  { chain: KAP_F_LAT_CHAIN, color: KAP_COLOR_F_LAT },
  { chain: KAP_F_MED_CHAIN, color: KAP_COLOR_F_MED },
  { chain: KAP_T_LAT_CHAIN, color: KAP_COLOR_T_LAT },
  { chain: KAP_T_MED_CHAIN, color: KAP_COLOR_T_MED },
];

// ── Geometry helpers ───────────────────────────────────────────────────────

function _kapChainMidpoints(latChain, medChain, kps) {
  const out = [];
  for (let i = 0; i < latChain.length; i++) {
    const a = kps[latChain[i]], b = kps[medChain[i]];
    out.push({ x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 });
  }
  return out;
}

// 2D total least-squares line fit (PCA).
// Returns { vx, vy, x0, y0 } — unit direction and a point on the line.
function _kapFitLine(pts) {
  const n = pts.length;
  let mx = 0, my = 0;
  for (const p of pts) { mx += p.x; my += p.y; }
  mx /= n; my /= n;

  let sxx = 0, syy = 0, sxy = 0;
  for (const p of pts) {
    const dx = p.x - mx, dy = p.y - my;
    sxx += dx * dx; syy += dy * dy; sxy += dx * dy;
  }
  // Larger eigenvalue of [[sxx, sxy], [sxy, syy]]
  const tr = sxx + syy;
  const det = sxx * syy - sxy * sxy;
  const disc = Math.max(0, (tr * tr) / 4 - det);
  const lambda = tr / 2 + Math.sqrt(disc);

  let vx, vy;
  if (Math.abs(sxy) > 1e-9) {
    vx = lambda - syy;
    vy = sxy;
  } else if (sxx >= syy) {
    vx = 1; vy = 0;
  } else {
    vx = 0; vy = 1;
  }
  const m = Math.hypot(vx, vy) || 1;
  return { vx: vx / m, vy: vy / m, x0: mx, y0: my };
}

function _kapAxisEndpoints(latChain, medChain, refPoint, takeTop, kps) {
  const mids = _kapChainMidpoints(latChain, medChain, kps);
  mids.sort((a, b) => a.y - b.y);
  const chosen = takeTop ? mids.slice(0, 4) : mids.slice(-4);
  if (chosen.length < 2) return null;

  const { vx, vy, x0, y0 } = _kapFitLine(chosen);
  const proj = p => (p.x - x0) * vx + (p.y - y0) * vy;
  const tMids = chosen.map(proj);
  const tRef  = proj(refPoint);
  const tMean = tMids.reduce((s, t) => s + t, 0) / tMids.length;
  const dir   = tRef >= tMean ? 1 : -1;
  const tFar  = dir > 0 ? Math.min(...tMids) : Math.max(...tMids);
  const overshoot = Math.abs(tRef - tMean) * 0.08;
  const t1 = tFar;
  const t2 = tRef + dir * overshoot;
  return {
    p1: { x: x0 + vx * t1, y: y0 + vy * t1 },
    p2: { x: x0 + vx * t2, y: y0 + vy * t2 },
  };
}

// All annotation primitives in image-pixel space.
function buildKneeApAnnotations(kps) {
  const notch    = { x: kps[KAP_NOTCH].x, y: kps[KAP_NOTCH].y };
  const interMid = {
    x: (kps[KAP_T_INTER_LAT].x + kps[KAP_T_INTER_MED].x) / 2,
    y: (kps[KAP_T_INTER_LAT].y + kps[KAP_T_INTER_MED].y) / 2,
  };

  const chains = KAP_CHAIN_GROUPS.map(g => ({
    color: g.color,
    points: g.chain.map(i => ({ x: kps[i].x, y: kps[i].y })),
  }));

  const jointLines = [
    { from: { x: kps[KAP_F_LAT_JOINT].x, y: kps[KAP_F_LAT_JOINT].y },
      to:   { x: kps[KAP_F_MED_JOINT].x, y: kps[KAP_F_MED_JOINT].y },
      color: KAP_COLOR_JOINT },
    { from: { x: kps[KAP_T_LAT_JOINT].x, y: kps[KAP_T_LAT_JOINT].y },
      to:   { x: kps[KAP_T_INTER_MED].x, y: kps[KAP_T_INTER_MED].y },
      color: KAP_COLOR_JOINT },
  ];

  const axes = [
    { ..._kapAxisEndpoints(KAP_F_LAT_CHAIN, KAP_F_MED_CHAIN, notch, true, kps),
      color: KAP_COLOR_FEMUR_AXIS },
    { ..._kapAxisEndpoints(KAP_T_LAT_CHAIN, KAP_T_MED_CHAIN, interMid, false, kps),
      color: KAP_COLOR_TIBIA_AXIS },
  ];

  const dots = [];
  for (const g of KAP_CHAIN_GROUPS) {
    for (const i of g.chain) dots.push({ x: kps[i].x, y: kps[i].y, color: g.color });
  }
  dots.push({ x: notch.x,                 y: notch.y,                 color: KAP_COLOR_NOTCH });
  dots.push({ x: kps[KAP_T_INTER_LAT].x,  y: kps[KAP_T_INTER_LAT].y,  color: KAP_COLOR_INTER });
  dots.push({ x: kps[KAP_T_INTER_MED].x,  y: kps[KAP_T_INTER_MED].y,  color: KAP_COLOR_INTER });

  return { chains, jointLines, axes, dots };
}

// Bake annotations onto a 2D context — used for download.
function drawKneeApAnnotationsOnContext(ctx, kps, lw) {
  const { chains, jointLines, axes, dots } = buildKneeApAnnotations(kps);
  ctx.lineCap = 'round';

  for (const c of chains) {
    ctx.strokeStyle = c.color;
    ctx.lineWidth = lw;
    for (let i = 0; i < c.points.length - 1; i++) {
      ctx.beginPath();
      ctx.moveTo(c.points[i].x, c.points[i].y);
      ctx.lineTo(c.points[i + 1].x, c.points[i + 1].y);
      ctx.stroke();
    }
  }
  for (const jl of jointLines) {
    ctx.strokeStyle = jl.color;
    ctx.lineWidth = lw * 2;
    ctx.beginPath();
    ctx.moveTo(jl.from.x, jl.from.y);
    ctx.lineTo(jl.to.x, jl.to.y);
    ctx.stroke();
  }
  for (const ax of axes) {
    if (!ax.p1 || !ax.p2) continue;
    ctx.strokeStyle = ax.color;
    ctx.lineWidth = lw;
    ctx.beginPath();
    ctx.moveTo(ax.p1.x, ax.p1.y);
    ctx.lineTo(ax.p2.x, ax.p2.y);
    ctx.stroke();
  }
  const r = Math.max(2, lw * 1.4);
  for (const d of dots) {
    ctx.fillStyle = d.color;
    ctx.beginPath();
    ctx.arc(d.x, d.y, r, 0, Math.PI * 2);
    ctx.fill();
  }
}

// Build a raw-image canvas (no annotations baked) for kneeap.
async function buildKneeApVisCanvas(file) {
  return new Promise((resolve, reject) => {
    const url = URL.createObjectURL(file);
    const img = new Image();
    img.onload = () => {
      const canvas = document.createElement('canvas');
      canvas.width  = img.naturalWidth;
      canvas.height = img.naturalHeight;
      canvas.getContext('2d').drawImage(img, 0, 0);
      const lw = Math.max(1, Math.round(img.naturalWidth / 400));
      URL.revokeObjectURL(url);
      resolve({ canvas, lw, imgHeight: img.naturalHeight });
    };
    img.onerror = () => { URL.revokeObjectURL(url); reject(new Error('kneeap canvas: image decode failed')); };
    img.src = url;
  });
}

// Overlay class — SVG lines + DOM dots, repositioned with the viewer.
class KneeApOverlay {
  constructor(container, viewer, config) {
    this.container = container;
    this.viewer    = viewer;
    this.kps       = config.keypoints;
    this.lw        = config.lw;
    this.imgHeight = config.imgHeight;
    this._svg      = null;
    this._svgEls   = [];   // {el, a:{x,y}, b:{x,y}, w}
    this._dotEls   = [];   // {el, x, y, color}
    this._build();
    viewer.onTransformChange = () => this._reposition();
  }

  _build() {
    const svgNS = 'http://www.w3.org/2000/svg';
    const svg = document.createElementNS(svgNS, 'svg');
    svg.style.cssText = 'position:absolute;inset:0;width:100%;height:100%;pointer-events:none;z-index:5;overflow:visible;';
    this.container.insertBefore(svg, this.container.firstChild);
    this._svg = svg;

    const ann = buildKneeApAnnotations(this.kps);

    const addLine = (a, b, color, w) => {
      const ln = document.createElementNS(svgNS, 'line');
      ln.setAttribute('stroke', color);
      ln.setAttribute('stroke-linecap', 'round');
      svg.appendChild(ln);
      this._svgEls.push({ el: ln, a, b, w });
    };

    for (const c of ann.chains) {
      for (let i = 0; i < c.points.length - 1; i++) {
        addLine(c.points[i], c.points[i + 1], c.color, this.lw);
      }
    }
    for (const jl of ann.jointLines) {
      addLine(jl.from, jl.to, jl.color, this.lw * 2);
    }
    for (const ax of ann.axes) {
      if (!ax.p1 || !ax.p2) continue;
      addLine(ax.p1, ax.p2, ax.color, this.lw);
    }

    for (const d of ann.dots) {
      const el = document.createElement('div');
      el.className = 'dot-handle';
      el.style.setProperty('--dc', d.color);
      // Smaller than the 13px cpak dots — chains have many close-together points.
      el.style.width = '7px';
      el.style.height = '7px';
      el.style.borderWidth = '1px';
      if (!KNEEAP_DOTS_DRAGGABLE) {
        el.style.pointerEvents = 'none';
        el.style.cursor = 'default';
      }
      this.container.appendChild(el);
      this._dotEls.push({ el, x: d.x, y: d.y });
      // Drag handler intentionally not attached. Flip KNEEAP_DOTS_DRAGGABLE
      // and wire up DotHandle-style listeners here when dragging is desired.
    }

    this._reposition();
  }

  _reposition() {
    const { tx, ty, scale } = this.viewer;
    const sw = Math.max(0.5, this.lw * scale);
    for (const ln of this._svgEls) {
      ln.el.setAttribute('x1', tx + ln.a.x * scale);
      ln.el.setAttribute('y1', ty + ln.a.y * scale);
      ln.el.setAttribute('x2', tx + ln.b.x * scale);
      ln.el.setAttribute('y2', ty + ln.b.y * scale);
      ln.el.setAttribute('stroke-width', ln.w * scale);
    }
    for (const d of this._dotEls) {
      d.el.style.left = `${tx + d.x * scale}px`;
      d.el.style.top  = `${ty + d.y * scale}px`;
    }
  }

  remove() {
    this._svg?.remove();
    for (const d of this._dotEls) d.el.remove();
    this._svgEls = [];
    this._dotEls = [];
    this._svg = null;
    if (this.viewer) this.viewer.onTransformChange = null;
  }

  getKeypoints() { return this.kps; }
}
