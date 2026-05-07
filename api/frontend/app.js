const API = '';

// ── Theme ──────────────────────────────────────────────────────────────────
const themeBtn = document.getElementById('theme-btn');

const SUN_ICON  = `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="5"/><line x1="12" y1="1" x2="12" y2="3"/><line x1="12" y1="21" x2="12" y2="23"/><line x1="4.22" y1="4.22" x2="5.64" y2="5.64"/><line x1="18.36" y1="18.36" x2="19.78" y2="19.78"/><line x1="1" y1="12" x2="3" y2="12"/><line x1="21" y1="12" x2="23" y2="12"/><line x1="4.22" y1="19.78" x2="5.64" y2="18.36"/><line x1="18.36" y1="5.64" x2="19.78" y2="4.22"/></svg>`;
const MOON_ICON = `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/></svg>`;

function applyTheme(theme) {
  if (theme === 'light') {
    document.documentElement.setAttribute('data-theme', 'light');
    themeBtn.innerHTML = MOON_ICON;
  } else {
    document.documentElement.removeAttribute('data-theme');
    themeBtn.innerHTML = SUN_ICON;
  }
  localStorage.setItem('cpak-theme', theme);
}

themeBtn.addEventListener('click', () => {
  applyTheme(document.documentElement.getAttribute('data-theme') === 'light' ? 'dark' : 'light');
});

applyTheme(localStorage.getItem('cpak-theme') || 'dark');

// ── State ──────────────────────────────────────────────────────────────────
let selectedFile       = null;
let visCanvas          = null;   // raw image canvas (no axes drawn — used as download base)
let visLw              = 1;
let visImgHeight       = 0;
let currentViewer      = null;
let currentAxesOverlay = null;   // single-knee overlay (legacy, single-mode only)
let currentAxesOverlays = [];    // dual-knee overlays (length 0 in single mode, 2 in dual)
let cpakPanelOpen      = false;
let kneeapPanelOpen    = false;
let modelKinds         = {};     // { modelName: 'cpak' | 'kneeap' }
let inferenceMode      = localStorage.getItem('cpak-mode') || 'dual';   // 'dual' | 'single'

// ── DOM ────────────────────────────────────────────────────────────────────
const appEl          = document.querySelector('.app');
const uploadZone     = document.getElementById('upload-zone');
const fileInput      = document.getElementById('file-input');
const previewImg     = document.getElementById('preview-img');
const previewName    = document.getElementById('preview-name');
const uploadPh       = document.getElementById('upload-placeholder');
const modelSelect    = document.getElementById('model-select');
const runBtn         = document.getElementById('run-btn');
const clearBtn       = document.getElementById('clear-btn');
const panelClearBtn  = document.getElementById('panel-clear-btn');
const btnSpinner     = document.getElementById('btn-spinner');
const btnIcon        = document.getElementById('btn-icon');
const btnLabel       = document.getElementById('btn-label');
const topGrid        = document.getElementById('top-grid');
const resultsSection = document.getElementById('results-section');
const resultsGrid    = document.getElementById('results-grid');
const visWrap        = document.getElementById('vis-wrap');
const dlVis          = document.getElementById('dl-vis');
const toast          = document.getElementById('toast');
const navLoadBtn     = document.getElementById('nav-load-btn');
const fabInput       = document.getElementById('fab-input');
const cpakPanel      = document.getElementById('cpak-panel');
const cpakTab        = document.getElementById('cpak-tab');
const cpakTypeValue  = document.getElementById('cpak-type-value');
const cpakLdfa       = document.getElementById('cpak-ldfa');
const cpakMpta       = document.getElementById('cpak-mpta');
const cpakAhka       = document.getElementById('cpak-ahka');
const cpakJlo        = document.getElementById('cpak-jlo');
const cpakAhkaCat    = document.getElementById('cpak-ahka-cat');
const cpakJloCat     = document.getElementById('cpak-jlo-cat');
const cpakMatrix     = document.getElementById('cpak-matrix');
const kneeapPanel    = document.getElementById('kneeap-panel');
const kneeapTab      = document.getElementById('kneeap-tab');
const kneeapJlcaVal  = document.getElementById('kneeap-jlca-value');
const modeToggle     = document.getElementById('mode-toggle');

function currentKind() { return modelKinds[modelSelect.value] || 'cpak'; }

// ── Mode toggle ─────────────────────────────────────────────────────────────
function applyMode(mode) {
  inferenceMode = mode === 'single' ? 'single' : 'dual';
  for (const btn of modeToggle.querySelectorAll('.mode-opt')) {
    const active = btn.dataset.mode === inferenceMode;
    btn.classList.toggle('active', active);
    btn.setAttribute('aria-selected', active ? 'true' : 'false');
  }
  localStorage.setItem('cpak-mode', inferenceMode);
}
modeToggle.addEventListener('click', e => {
  const btn = e.target.closest('.mode-opt');
  if (!btn) return;
  applyMode(btn.dataset.mode);
});
applyMode(inferenceMode);

// ── Health check ───────────────────────────────────────────────────────────
async function checkHealth() {
  try {
    const r = await fetch(`${API}/health`);
    if (!r.ok) throw new Error();
  } catch { /* server offline — no visible pill, ignore */ }
}

// ── Models ─────────────────────────────────────────────────────────────────
async function loadModels() {
  try {
    const data = await (await fetch(`${API}/models`)).json();
    // Normalize: backend returns [{name, kind}], older builds may return [string]
    const items = data.models.map(m =>
      typeof m === 'string' ? { name: m, kind: 'cpak' } : m);
    modelKinds = Object.fromEntries(items.map(m => [m.name, m.kind]));
    modelSelect.innerHTML = items.length
      ? items.map(m => `<option value="${m.name}" data-kind="${m.kind}">${m.name}</option>`).join('')
      : '<option value="">No models found</option>';
    updateRunBtn();
  } catch {
    modelSelect.innerHTML = '<option value="">Failed to load models</option>';
  }
}

// ── File upload ────────────────────────────────────────────────────────────
function handleFile(file) {
  if (!file || !file.type.startsWith('image/')) { showToast('Please select a valid image file.'); return; }
  selectedFile = file;
  previewImg.src = URL.createObjectURL(file);
  previewImg.classList.add('visible');
  uploadPh.style.display = 'none';
  previewName.textContent = file.name;
  uploadZone.classList.add('has-image');
  updateRunBtn();
}

fileInput.addEventListener('change', e => { handleFile(e.target.files[0]); fileInput.value = ''; });

uploadZone.addEventListener('click', (e) => {
  if (e.target === fileInput) return;
  fileInput.value = '';
  fileInput.click();
});

// Nav "Load Image" button — picks file and immediately re-runs inference
fabInput.addEventListener('change', e => {
  const file = e.target.files[0];
  fabInput.value = '';
  if (!file || !file.type.startsWith('image/')) { showToast('Please select a valid image file.'); return; }
  selectedFile = file;
  runInference();
});

uploadZone.addEventListener('dragover',  e => { e.preventDefault(); uploadZone.classList.add('drag-over'); });
uploadZone.addEventListener('dragleave', () => uploadZone.classList.remove('drag-over'));
uploadZone.addEventListener('drop', e => { e.preventDefault(); uploadZone.classList.remove('drag-over'); handleFile(e.dataTransfer.files[0]); });

// ── Clear ──────────────────────────────────────────────────────────────────
function doClear() {
  selectedFile = null; fileInput.value = '';
  previewImg.src = ''; previewImg.classList.remove('visible');
  uploadZone.classList.remove('has-image');
  uploadPh.style.display = ''; previewName.textContent = '';
  topGrid.style.display = '';
  navLoadBtn.classList.remove('visible');
  closeCpakPanel();
  closeKneeapPanel();
  resultsSection.style.display = 'none';
  resultsGrid.classList.remove('visible');
  appEl.classList.remove('map-view');
  destroyAxesOverlays();
  currentViewer      = null;
  visCanvas          = null;
  updateRunBtn();
}
clearBtn.addEventListener('click', doClear);
panelClearBtn.addEventListener('click', doClear);

// ── Run state ──────────────────────────────────────────────────────────────
function updateRunBtn() { runBtn.disabled = !selectedFile || !modelSelect.value; }
modelSelect.addEventListener('change', updateRunBtn);

// ── Inference ──────────────────────────────────────────────────────────────
runBtn.addEventListener('click', runInference);

async function runInference() {
  if (!selectedFile || !modelSelect.value) return;
  const model = modelSelect.value;
  const kind  = currentKind();
  setLoading(true);
  resultsSection.style.display = 'block';
  showImgLoading(visWrap);

  try {
    if (inferenceMode === 'dual') {
      if (kind === 'kneeap') {
        await runDualKneeapInference(model);
      } else {
        await runDualCpakInference(model);
      }
    } else {
      if (kind === 'kneeap') {
        await runKneeapInference(model);
      } else {
        await runCpakInference(model);
      }
    }
  } catch (err) {
    showToast('Inference failed: ' + err.message);
    clearImgLoading(visWrap);
  } finally {
    setLoading(false);
  }
}

async function runCpakInference(model) {
  setPanelMode('single');
  const jsonResp = await postImage(`${API}/infer/${model}`, selectedFile);
  if (!jsonResp.ok) throw new Error(await jsonResp.text());

  const data = await jsonResp.json();
  const { canvas, dotPositions, labelPositions, lw, imgHeight } =
    await buildVisCanvas(selectedFile, data.keypoints, data.metrics);
  visCanvas    = canvas;
  visLw        = lw;
  visImgHeight = imgHeight;

  appEl.classList.add('map-view');
  topGrid.style.display = 'none';
  navLoadBtn.classList.add('visible');
  closeKneeapPanel();

  const blob = await new Promise((res, rej) =>
    canvas.toBlob(b => b ? res(b) : rej(new Error('toBlob failed')), 'image/png'));
  renderImage(blob, selectedFile.name);

  requestAnimationFrame(() => {
    resultsGrid.classList.add('visible');
    if (currentViewer) {
      currentViewer._fitToContainer();
      destroyAxesOverlays();
      currentAxesOverlay = new AxesOverlay(visWrap, currentViewer, {
        dots: dotPositions, labels: labelPositions, lw, imgHeight,
      });
      currentAxesOverlays = [currentAxesOverlay];

      const initAngles = computeAngles(currentAxesOverlay.dots);
      if (initAngles) {
        updateCpakPanel(classifyCPAK(initAngles.ldfa, initAngles.mpta));
      }
      openCpakPanel();

      currentAxesOverlay.onAnglesChange = result => updateCpakPanel(result);
    }
  });
}

async function runKneeapInference(model) {
  setPanelMode('single');
  // Backend only supplies keypoints + JLCA; all drawing happens here.
  const jsonResp = await postImage(`${API}/infer/${model}`, selectedFile);
  if (!jsonResp.ok) throw new Error(await jsonResp.text());
  const data = await jsonResp.json();

  const { canvas, lw, imgHeight } = await buildKneeApVisCanvas(selectedFile);
  visCanvas    = canvas;
  visLw        = lw;
  visImgHeight = imgHeight;

  appEl.classList.add('map-view');
  topGrid.style.display = 'none';
  navLoadBtn.classList.add('visible');
  closeCpakPanel();

  const blob = await new Promise((res, rej) =>
    canvas.toBlob(b => b ? res(b) : rej(new Error('toBlob failed')), 'image/png'));
  renderImage(blob, selectedFile.name);

  requestAnimationFrame(() => {
    resultsGrid.classList.add('visible');
    if (currentViewer) {
      currentViewer._fitToContainer();
      destroyAxesOverlays();
      currentAxesOverlay = new KneeApOverlay(visWrap, currentViewer, {
        keypoints: data.keypoints,
        lw,
        imgHeight,
      });
      currentAxesOverlays = [currentAxesOverlay];
    }
    updateKneeapPanel(data.metrics.jlca);
    openKneeapPanel();
  });
}

function updateKneeapPanel(jlca) {
  kneeapJlcaVal.textContent = jlca.toFixed(2) + '°';
}

function postImage(url, file) {
  const fd = new FormData();
  fd.append('image', file);
  return fetch(url, { method: 'POST', body: fd });
}

// ── Dual inference (stubs — render wired in next steps) ────────────────────
async function runDualCpakInference(model) {
  setPanelMode('dual');
  const resp = await postImage(`${API}/infer/${model}/dual`, selectedFile);
  if (!resp.ok) throw new Error(await resp.text());
  const data = await resp.json();

  const { canvas, lw, imgHeight, sides } = await buildDualVisCanvas(
    selectedFile, data.left.keypoints, data.left.metrics,
    data.right.keypoints, data.right.metrics,
  );
  visCanvas    = canvas;
  visLw        = lw;
  visImgHeight = imgHeight;

  appEl.classList.add('map-view');
  topGrid.style.display = 'none';
  navLoadBtn.classList.add('visible');
  closeKneeapPanel();

  const blob = await new Promise((res, rej) =>
    canvas.toBlob(b => b ? res(b) : rej(new Error('toBlob failed')), 'image/png'));
  renderImage(blob, selectedFile.name);

  requestAnimationFrame(() => {
    resultsGrid.classList.add('visible');
    if (currentViewer) {
      currentViewer._fitToContainer();
      destroyAxesOverlays();
      for (const side of sides) {
        const overlay = new AxesOverlay(visWrap, currentViewer, {
          dots: side.dots, labels: side.labels, lw, imgHeight, side: side.key,
        });
        overlay.onAnglesChange = result => updateCpakDualPanel(side.key, result);
        currentAxesOverlays.push(overlay);
      }
      // Initial CPAK results for both sides
      for (const overlay of currentAxesOverlays) {
        const a = computeAngles(overlay.dots);
        if (a) updateCpakDualPanel(overlay.side, classifyCPAK(a.ldfa, a.mpta));
      }
      openCpakPanel();
    }
  });
}

async function runDualKneeapInference(model) {
  setPanelMode('dual');
  const resp = await postImage(`${API}/infer/${model}/dual`, selectedFile);
  if (!resp.ok) throw new Error(await resp.text());
  const data = await resp.json();

  const { canvas, lw, imgHeight } = await buildKneeApVisCanvas(selectedFile);
  visCanvas    = canvas;
  visLw        = lw;
  visImgHeight = imgHeight;

  appEl.classList.add('map-view');
  topGrid.style.display = 'none';
  navLoadBtn.classList.add('visible');
  closeCpakPanel();

  const blob = await new Promise((res, rej) =>
    canvas.toBlob(b => b ? res(b) : rej(new Error('toBlob failed')), 'image/png'));
  renderImage(blob, selectedFile.name);

  requestAnimationFrame(() => {
    resultsGrid.classList.add('visible');
    if (currentViewer) {
      currentViewer._fitToContainer();
      destroyAxesOverlays();
      for (const sideKey of ['left', 'right']) {
        const overlay = new KneeApOverlay(visWrap, currentViewer, {
          keypoints: data[sideKey].keypoints,
          lw,
          imgHeight,
        });
        overlay.side = sideKey;
        currentAxesOverlays.push(overlay);
      }
    }
    updateKneeapDualPanel(data.left.metrics.jlca, data.right.metrics.jlca);
    openKneeapPanel();
  });
}

function destroyAxesOverlays() {
  currentAxesOverlay?.remove();
  currentAxesOverlay = null;
  for (const o of currentAxesOverlays) o.remove();
  currentAxesOverlays = [];
}

// ── Render ─────────────────────────────────────────────────────────────────
function renderImage(blob, filename) {
  const stem = filename.replace(/\.[^.]+$/, '');
  mountViewer(visWrap, URL.createObjectURL(blob));
  dlVis.style.display = '';
  dlVis.onclick = () => downloadWithLabels(stem);
}

function downloadWithLabels(stem) {
  const dc  = document.createElement('canvas');
  dc.width  = visCanvas.width;
  dc.height = visCanvas.height;
  const ctx = dc.getContext('2d');
  ctx.drawImage(visCanvas, 0, 0);
  for (const overlay of currentAxesOverlays) {
    if (overlay instanceof KneeApOverlay) {
      drawKneeApAnnotationsOnContext(ctx, overlay.getKeypoints(), visLw);
    } else if (overlay) {
      drawAxesOnContext(ctx, overlay.getDotPositions(), visLw, visImgHeight);
      drawLabelsOnCanvas(ctx, overlay.getLabelPositions(), visLw);
    }
  }
  dc.toBlob(blob => downloadBlob(blob, `${stem}_axes.png`), 'image/png');
}

function mountViewer(container, imgUrl) {
  container.innerHTML = '';
  container.classList.add('has-image');
  const img = document.createElement('img');
  img.className = 'viewer-img';
  img.src = imgUrl;
  container.appendChild(img);
  currentViewer = new ImageViewer(container);
  currentViewer.mountImage(img);
}

function showImgLoading(wrap) {
  wrap.innerHTML = `<div class="img-loading"><div class="img-spinner"></div></div>`;
}
function clearImgLoading(wrap) {
  wrap.innerHTML = `<div class="img-placeholder"><span>Failed to load</span></div>`;
}
function downloadBlob(blob, name) {
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = name; a.click();
}

// ── Loading ────────────────────────────────────────────────────────────────
function setLoading(on) {
  runBtn.disabled          = on;
  btnSpinner.style.display = on ? 'block' : 'none';
  btnIcon.style.display    = on ? 'none'  : 'block';
  btnLabel.textContent     = on ? 'Running…' : 'Run Inference';
}

// ── Toast ──────────────────────────────────────────────────────────────────
let toastTimer;
function showToast(msg) {
  toast.textContent = msg;
  toast.classList.add('show');
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => toast.classList.remove('show'), 4000);
}

// ── CPAK Panel ─────────────────────────────────────────────────────────────
function buildCpakMatrix(activeType) {
  cpakMatrix.innerHTML = '';
  for (let r = 0; r < 3; r++) {
    for (let c = 0; c < 3; c++) {
      const type = CPAK_MATRIX[r][c];
      const cell = document.createElement('div');
      cell.className = 'cpak-matrix-cell' + (type === activeType ? ' active' : '');
      cell.innerHTML = `<span class="cell-type">${type}</span>`;
      cpakMatrix.appendChild(cell);
    }
  }
}

function setCpakCatClass(el, cat) {
  el.className = 'cpak-cat';
  const c = cat.toLowerCase();
  if      (c === 'varus')          el.classList.add('varus');
  else if (c === 'valgus')         el.classList.add('valgus');
  else if (c === 'neutral')        el.classList.add('neutral');
  else if (c.includes('proximal')) el.classList.add('proximal');
  else if (c.includes('distal'))   el.classList.add('distal');
}

function updateCpakPanel(result) {
  cpakTypeValue.textContent = result.cpakType;
  cpakLdfa.textContent      = result.ldfa.toFixed(1) + '°';
  cpakMpta.textContent      = result.mpta.toFixed(1) + '°';
  cpakAhka.textContent      = (result.ahka >= 0 ? '+' : '') + result.ahka.toFixed(1) + '°';
  cpakJlo.textContent       = result.jlo.toFixed(1) + '°';
  cpakAhkaCat.textContent   = result.ahkaCat;
  cpakJloCat.textContent    = result.jloCat;
  setCpakCatClass(cpakAhkaCat, result.ahkaCat);
  setCpakCatClass(cpakJloCat,  result.jloCat);
  buildCpakMatrix(result.cpakType);
}

function setPanelMode(mode) {
  // Toggles the .dual class on both cpak and kneeap panels so the right
  // single/dual section is visible.
  const dual = mode === 'dual';
  cpakPanel.classList.toggle('dual', dual);
  kneeapPanel.classList.toggle('dual', dual);
}

function updateCpakDualPanel(side, result) {
  const prefix = side === 'left' ? 'l' : 'r';
  document.getElementById(`cpak-${prefix}-type`).textContent     = result.cpakType;
  document.getElementById(`cpak-${prefix}-ldfa`).textContent     = result.ldfa.toFixed(1) + '°';
  document.getElementById(`cpak-${prefix}-mpta`).textContent     = result.mpta.toFixed(1) + '°';
  document.getElementById(`cpak-${prefix}-ahka`).textContent     =
    (result.ahka >= 0 ? '+' : '') + result.ahka.toFixed(1) + '°';
  document.getElementById(`cpak-${prefix}-jlo`).textContent      = result.jlo.toFixed(1) + '°';
  const ahkaCat = document.getElementById(`cpak-${prefix}-ahka-cat`);
  const jloCat  = document.getElementById(`cpak-${prefix}-jlo-cat`);
  ahkaCat.textContent = result.ahkaCat;
  jloCat.textContent  = result.jloCat;
  setCpakCatClass(ahkaCat, result.ahkaCat);
  setCpakCatClass(jloCat,  result.jloCat);

  // Highlight the side that just updated
  for (const block of cpakPanel.querySelectorAll('.cpak-side-block')) {
    block.classList.toggle('active', block.dataset.side === side);
  }
}

function updateKneeapDualPanel(leftJlca, rightJlca) {
  document.getElementById('kneeap-l-jlca').textContent = leftJlca.toFixed(2) + '°';
  document.getElementById('kneeap-r-jlca').textContent = rightJlca.toFixed(2) + '°';
}

function openCpakPanel() {
  cpakPanelOpen = true;
  cpakPanel.classList.add('open');
}

function closeCpakPanel() {
  cpakPanelOpen = false;
  cpakPanel.classList.remove('open');
}

cpakTab.addEventListener('click', () => cpakPanelOpen ? closeCpakPanel() : openCpakPanel());

// ── KneeAP Panel ───────────────────────────────────────────────────────────
function openKneeapPanel() {
  kneeapPanelOpen = true;
  kneeapPanel.classList.add('open');
}
function closeKneeapPanel() {
  kneeapPanelOpen = false;
  kneeapPanel.classList.remove('open');
}
kneeapTab.addEventListener('click', () => kneeapPanelOpen ? closeKneeapPanel() : openKneeapPanel());

// ── Init ───────────────────────────────────────────────────────────────────
checkHealth();
loadModels();
setInterval(checkHealth, 10000);
