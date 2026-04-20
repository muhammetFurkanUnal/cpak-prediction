// Gaussian elimination for a 3x3 linear system Ax = b
function solve3x3(A, b) {
  const M = A.map((row, i) => [...row, b[i]]);
  for (let col = 0; col < 3; col++) {
    let maxRow = col;
    for (let row = col + 1; row < 3; row++) {
      if (Math.abs(M[row][col]) > Math.abs(M[maxRow][col])) maxRow = row;
    }
    [M[col], M[maxRow]] = [M[maxRow], M[col]];
    for (let row = col + 1; row < 3; row++) {
      const f = M[row][col] / M[col][col];
      for (let k = col; k <= 3; k++) M[row][k] -= f * M[col][k];
    }
  }
  const x = [0, 0, 0];
  for (let i = 2; i >= 0; i--) {
    x[i] = M[i][3];
    for (let j = i + 1; j < 3; j++) x[i] -= M[i][j] * x[j];
    x[i] /= M[i][i];
  }
  return x;
}

// Least-squares circle fit through pts[{x,y}] → returns centre {x,y}
function circleCenter(pts) {
  const n = pts.length;
  const A = pts.map(p => [p.x, p.y, 1]);
  const b = pts.map(p => p.x * p.x + p.y * p.y);
  let AtA = [[0,0,0],[0,0,0],[0,0,0]];
  let Atb = [0,0,0];
  for (let i = 0; i < n; i++) {
    for (let r = 0; r < 3; r++) {
      Atb[r] += A[i][r] * b[i];
      for (let c = 0; c < 3; c++) AtA[r][c] += A[i][r] * A[i][c];
    }
  }
  const c = solve3x3(AtA, Atb);
  return { x: c[0] / 2, y: c[1] / 2 };
}

// Derive anatomical points from raw keypoints array (index matches model output)
function computeAxes(kps) {
  const p   = id => ({ x: kps[id].x, y: kps[id].y });
  const mid = (a, b) => ({ x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 });

  // Femur head: weighted average of circle fit (kp2-6) and kp7
  const mc      = circleCenter([p(2), p(3), p(4), p(5), p(6)]);
  const kp7     = p(7);
  const femurHead = { x: (mc.x + 3 * kp7.x) / 4, y: (mc.y + 3 * kp7.y) / 4 };

  const femurLateral       = p(8);
  const femurMedial        = p(9);
  const femurNotch         = p(10);
  const tibiaLateral       = p(15);
  const tibiaMedial        = p(16);
  const tibiaIntercondiler = p(21);
  const ankleLateral       = p(24);
  const ankleMedial        = p(25);
  const ankleModelMiddle   = p(26);

  // Ankle midpoint: average of geometric mid and model-predicted middle
  const ankleAxMiddle    = mid(ankleLateral, ankleMedial);
  const finalAnkleMiddle = mid(ankleAxMiddle, ankleModelMiddle);

  return {
    femurHead, femurLateral, femurMedial, femurNotch,
    tibiaLateral, tibiaMedial, tibiaIntercondiler,
    ankleLateral, ankleMedial, finalAnkleMiddle,
  };
}

// Draw axes lines + dots onto an existing 2D context (used for download baking).
// axes: { femurHead:{x,y}, femurLateral:{x,y}, ... } — all in image natural-pixel space.
function drawAxesOnContext(ctx, axes, lw, imgHeight) {
  const dot = (pt, color, r = lw * 1.4) => {
    ctx.beginPath();
    ctx.arc(pt.x, pt.y, r, 0, Math.PI * 2);
    ctx.fillStyle = color;
    ctx.fill();
  };
  const line = (a, b, color) => {
    ctx.beginPath();
    ctx.moveTo(a.x, a.y);
    ctx.lineTo(b.x, b.y);
    ctx.strokeStyle = color;
    ctx.lineWidth   = lw;
    ctx.stroke();
  };

  const GREEN  = '#00e676';
  const BLUE   = '#4488ff';
  const YELLOW = '#ffdd00';
  const LDFA_C = '#22d3ee';
  const MPTA_C = '#fb923c';

  line(axes.femurLateral,  axes.femurMedial,  GREEN);
  line(axes.tibiaLateral,  axes.tibiaMedial,  GREEN);
  line(axes.ankleLateral,  axes.ankleMedial,  GREEN);

  const ext    = imgHeight * 0.03;
  const extend = (a, b) => {
    const len = Math.hypot(b.x - a.x, b.y - a.y) || 1;
    return { x: b.x + (b.x - a.x) / len * ext, y: b.y + (b.y - a.y) / len * ext };
  };

  line(axes.femurHead,        extend(axes.femurHead,        axes.femurNotch),         LDFA_C);
  line(axes.finalAnkleMiddle, extend(axes.finalAnkleMiddle, axes.tibiaIntercondiler), MPTA_C);

  dot(axes.femurHead,        GREEN);
  dot(axes.femurLateral,     BLUE);
  dot(axes.femurMedial,      BLUE);
  dot(axes.tibiaLateral,     BLUE);
  dot(axes.tibiaMedial,      BLUE);
  dot(axes.ankleLateral,     BLUE);
  dot(axes.ankleMedial,      BLUE);
  dot(axes.finalAnkleMiddle, YELLOW);
}

// Compute initial label positions in image-pixel space (mirrors former canvas label placement)
function getDefaultLabelPositions(axes, metrics, lw, imgHeight) {
  const vOff = imgHeight * 0.04;

  const femurMidX  = (axes.femurLateral.x + axes.femurMedial.x) / 2;
  const femurMidY  = (axes.femurLateral.y + axes.femurMedial.y) / 2;
  const ldfaOutLen = Math.hypot(axes.femurLateral.x - femurMidX, axes.femurLateral.y - femurMidY) || 1;
  const ldfaHOff   = ((axes.femurLateral.x - femurMidX) / ldfaOutLen) * lw * 22;

  const tibiaMidX  = (axes.tibiaLateral.x + axes.tibiaMedial.x) / 2;
  const tibiaMidY  = (axes.tibiaLateral.y + axes.tibiaMedial.y) / 2;
  const mptaOutLen = Math.hypot(axes.tibiaMedial.x - tibiaMidX, axes.tibiaMedial.y - tibiaMidY) || 1;
  const mptaHOff   = ((axes.tibiaMedial.x - tibiaMidX) / mptaOutLen) * lw * 22;

  return [
    {
      text:  `LDFA  ${metrics.femur_mech_angle_notch.toFixed(1)}°`,
      color: '#22d3ee',
      ix:    axes.femurLateral.x + ldfaHOff,
      iy:    axes.femurLateral.y - vOff,
    },
    {
      text:  `MPTA  ${metrics.tibia_mech_angle_inter.toFixed(1)}°`,
      color: '#fb923c',
      ix:    axes.tibiaMedial.x + mptaHOff,
      iy:    axes.tibiaMedial.y + vOff,
    },
  ];
}

// Draw labels onto an existing canvas context (used when baking labels into download)
function drawLabelsOnCanvas(ctx, labelData, lw) {
  const fs = Math.max(11, lw * 9);
  ctx.font = `bold ${fs}px sans-serif`;
  for (const { ix, iy, text, color } of labelData) {
    const pad = fs * 0.35;
    const w   = ctx.measureText(text).width + pad * 2;
    const h   = fs + pad * 2;
    ctx.fillStyle = 'rgba(0,0,0,0.55)';
    ctx.beginPath();
    ctx.roundRect(ix - pad, iy - fs, w, h, fs * 0.3);
    ctx.fill();
    ctx.fillStyle = color;
    ctx.fillText(text, ix, iy);
  }
}

// Build image-only canvas + all initial overlay positions.
// Returns { canvas (raw image, no axes drawn), dotPositions, labelPositions, lw, imgHeight }
async function buildVisCanvas(file, keypoints, metrics) {
  return new Promise((resolve, reject) => {
    const url = URL.createObjectURL(file);
    const img = new Image();
    img.onload = () => {
      // Canvas contains only the raw image — axes/dots drawn as HTML/SVG overlays
      const canvas = document.createElement('canvas');
      canvas.width  = img.naturalWidth;
      canvas.height = img.naturalHeight;
      canvas.getContext('2d').drawImage(img, 0, 0);

      const axes = computeAxes(keypoints);
      const lw   = Math.max(1, Math.round(img.naturalWidth / 400));

      // Convert axes {x,y} → dotPositions {key: {ix,iy}} (same coordinate space)
      const dotPositions = {};
      for (const [key, val] of Object.entries(axes)) {
        dotPositions[key] = { ix: val.x, iy: val.y };
      }

      const labelPositions = getDefaultLabelPositions(axes, metrics, lw, img.naturalHeight);

      URL.revokeObjectURL(url);
      resolve({ canvas, dotPositions, labelPositions, lw, imgHeight: img.naturalHeight });
    };
    img.onerror = () => reject(new Error('Failed to load image for canvas drawing'));
    img.src = url;
  });
}
