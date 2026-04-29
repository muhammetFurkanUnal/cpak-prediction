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
let currentAxesOverlay = null;
let cpakPanelOpen      = false;

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
const modelInfo      = document.getElementById('model-info');
const modelNameDisp  = document.getElementById('model-name-display');
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
    modelSelect.innerHTML = data.models.length
      ? data.models.map(m => `<option value="${m}">${m}</option>`).join('')
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
  resultsSection.style.display = 'none';
  resultsGrid.classList.remove('visible');
  appEl.classList.remove('map-view');
  currentAxesOverlay?.remove();
  currentAxesOverlay = null;
  currentViewer      = null;
  visCanvas          = null;
  updateRunBtn();
}
clearBtn.addEventListener('click', doClear);
panelClearBtn.addEventListener('click', doClear);

// ── Run state ──────────────────────────────────────────────────────────────
function updateRunBtn() { runBtn.disabled = !selectedFile || !modelSelect.value; }
modelSelect.addEventListener('change', () => {
  updateRunBtn();
  if (modelSelect.value) { modelInfo.style.display = 'block'; modelNameDisp.textContent = modelSelect.value; }
});

// ── Inference ──────────────────────────────────────────────────────────────
runBtn.addEventListener('click', runInference);

async function runInference() {
  if (!selectedFile || !modelSelect.value) return;
  const model = modelSelect.value;
  setLoading(true);
  resultsSection.style.display = 'block';
  showImgLoading(visWrap);

  try {
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

    const blob = await new Promise((res, rej) =>
      canvas.toBlob(b => b ? res(b) : rej(new Error('toBlob failed')), 'image/png'));
    renderImage(blob, selectedFile.name);

    requestAnimationFrame(() => {
      resultsGrid.classList.add('visible');
      if (currentViewer) {
        currentViewer._fitToContainer();
        currentAxesOverlay?.remove();
        currentAxesOverlay = new AxesOverlay(visWrap, currentViewer, {
          dots: dotPositions, labels: labelPositions, lw, imgHeight,
        });

        // Compute initial CPAK, populate panel and open it
        const initAngles = computeAngles(currentAxesOverlay.dots);
        if (initAngles) {
          updateCpakPanel(classifyCPAK(initAngles.ldfa, initAngles.mpta));
        }
        openCpakPanel();

        currentAxesOverlay.onAnglesChange = result => updateCpakPanel(result);
      }
    });
  } catch (err) {
    showToast('Inference failed: ' + err.message);
    clearImgLoading(visWrap);
  } finally {
    setLoading(false);
  }
}

function postImage(url, file) {
  const fd = new FormData();
  fd.append('image', file);
  return fetch(url, { method: 'POST', body: fd });
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
  if (currentAxesOverlay) {
    drawAxesOnContext(ctx, currentAxesOverlay.getDotPositions(), visLw, visImgHeight);
    drawLabelsOnCanvas(ctx, currentAxesOverlay.getLabelPositions(), visLw);
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

function openCpakPanel() {
  cpakPanelOpen = true;
  cpakPanel.classList.add('open');
}

function closeCpakPanel() {
  cpakPanelOpen = false;
  cpakPanel.classList.remove('open');
}

cpakTab.addEventListener('click', () => cpakPanelOpen ? closeCpakPanel() : openCpakPanel());

// ── Init ───────────────────────────────────────────────────────────────────
checkHealth();
loadModels();
setInterval(checkHealth, 10000);
