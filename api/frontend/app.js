const API = '';

// ── State ──────────────────────────────────────────────────────────────────
let selectedFile   = null;
let visBlob        = null;
let currentViewer  = null;

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
  updateRunBtn();
}

fileInput.addEventListener('change', e => handleFile(e.target.files[0]));

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
  uploadPh.style.display = ''; previewName.textContent = '';
  topGrid.style.display = '';
  navLoadBtn.classList.remove('visible');
  resultsSection.style.display = 'none';
  resultsGrid.classList.remove('visible');
  appEl.classList.remove('map-view');
  currentViewer = null;
  visBlob = null;
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
    visBlob = await buildVisBlob(selectedFile, data.keypoints, data.metrics);

    appEl.classList.add('map-view');
    topGrid.style.display = 'none';
    navLoadBtn.classList.add('visible');

    renderImage(visBlob, selectedFile.name);

    requestAnimationFrame(() => {
      resultsGrid.classList.add('visible');
      if (currentViewer) currentViewer._fitToContainer();
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
function renderImage(visB, filename) {
  const stem = filename.replace(/\.[^.]+$/, '');
  mountViewer(visWrap, URL.createObjectURL(visB));
  dlVis.style.display = '';
  dlVis.onclick = () => downloadBlob(visB, `${stem}_axes.png`);
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

// ── Init ───────────────────────────────────────────────────────────────────
checkHealth();
loadModels();
setInterval(checkHealth, 10000);
