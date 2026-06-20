"use strict";
// cpak dev-gui — ön yüz. Framework yok; küçük bir hash-router + fetch + DOM.
// Veri yalnızca sunucudan okunur; eksik alanlar "mevcut değil" olarak gösterilir.

const enc = encodeURIComponent;

// ── Mini DOM yardımcısı ─────────────────────────────────────────────────────
function h(tag, attrs, ...kids) {
  const el = document.createElement(tag);
  if (attrs) {
    for (const [k, v] of Object.entries(attrs)) {
      if (v == null || v === false) continue;
      if (k === "class") el.className = v;
      else if (k === "html") el.innerHTML = v;
      else if (k === "text") el.textContent = v;
      else if (k.startsWith("on") && typeof v === "function") el.addEventListener(k.slice(2), v);
      else if (k === "dataset") Object.assign(el.dataset, v);
      else el.setAttribute(k, v);
    }
  }
  for (const kid of kids.flat()) {
    if (kid == null || kid === false) continue;
    el.append(kid.nodeType ? kid : document.createTextNode(String(kid)));
  }
  return el;
}

async function api(path) {
  const r = await fetch(path);
  if (!r.ok) throw new Error(r.status + " " + path);
  return r.json();
}

function mount(...nodes) {
  document.getElementById("app").replaceChildren(...nodes);
}

function tr(k, v) {
  return h("tr", {}, h("td", {}, k), h("td", {}, String(v)));
}

// ── Durum ───────────────────────────────────────────────────────────────────
const state = {
  models: [],
  byName: {},
  currentModel: null,
  mode: "model", // 'all' | 'model'
  images: [],
  imagesSig: null,
  allImages: null,
};

function selectModel(name, mode) {
  state.currentModel = name || null;
  if (mode) state.mode = mode;
  state.images = [];
  state.imagesSig = null;
  updateChip();
}

function setActiveNav(view) {
  document.querySelectorAll(".nav-btn").forEach((b) => b.classList.toggle("active", b.dataset.nav === view));
}

function updateChip() {
  const chip = document.getElementById("model-chip");
  if (!state.currentModel) {
    chip.classList.add("hidden");
    chip.replaceChildren();
    return;
  }
  chip.classList.remove("hidden");
  const m = state.byName[state.currentModel];
  const kids = [h("span", {}, "Model: "), h("b", {}, state.currentModel)];
  if (m && m.has_test) kids.push(h("a", { href: "#/test/" + enc(m.name) }, "test analizi"));
  chip.replaceChildren(...kids);
}

async function ensureImages() {
  if (state.mode === "model" && state.currentModel) {
    const sig = "model:" + state.currentModel;
    if (state.imagesSig !== sig) {
      state.images = await api("/api/models/" + enc(state.currentModel) + "/images");
      state.imagesSig = sig;
    }
  } else {
    if (!state.allImages) state.allImages = await api("/api/images");
    state.images = state.allImages;
    state.imagesSig = "all";
  }
}

// ── Router ──────────────────────────────────────────────────────────────────
let keyHandler = null;
function removeKeyNav() {
  if (keyHandler) {
    window.removeEventListener("keydown", keyHandler);
    keyHandler = null;
  }
}

function route() {
  removeKeyNav();
  const parts = (location.hash || "#/models").replace(/^#\//, "").split("/");
  const view = parts[0] || "models";
  if (view === "browse") return renderBrowse();
  if (view === "image") return renderImage(decodeURIComponent(parts.slice(1).join("/")));
  if (view === "test") return renderTest(decodeURIComponent(parts.slice(1).join("/")));
  return renderModels();
}

// ── Görünüm: Modeller ───────────────────────────────────────────────────────
function modelCard(m) {
  const badges = [h("span", { class: "badge " + m.kind }, m.kind)];
  if (m.experimental) badges.push(h("span", { class: "badge exp" }, "deneysel"));
  badges.push(m.has_test ? h("span", { class: "badge ok" }, "test analizi var") : h("span", { class: "badge" }, "test yok"));
  if (!m.available) badges.push(h("span", { class: "badge warn" }, "inference klasörü yok"));

  const provenance = h(
    "details",
    { class: "provenance" },
    h("summary", {}, "veri kaynağı (" + m.metric_files.length + " dosya)"),
    h(
      "div",
      { class: "files" },
      h("div", { class: "kv small" }, h("span", {}, "inference"), h("span", { class: "mono" }, m.inference_dir)),
      h("div", { class: "kv small" }, h("span", {}, "MPTA/LDFA kaynağı"), h("span", { class: "mono" }, m.angle_source || "—")),
      ...m.metric_files.map((f) => h("div", { class: "mono small" }, "• " + f)),
      m.weights_path ? h("div", { class: "kv small" }, h("span", {}, "ağırlık"), h("span", { class: "mono" }, m.weights_path)) : null
    )
  );

  const actions = h(
    "div",
    { class: "actions" },
    h(
      "button",
      {
        class: "btn-primary",
        disabled: !m.available,
        onclick: () => {
          selectModel(m.name, "model");
          location.hash = "#/browse";
        },
      },
      "Görüntülere göz at"
    ),
    m.has_test
      ? h("button", { onclick: () => (location.hash = "#/test/" + enc(m.name)) }, "Test analizi")
      : h("button", { disabled: true, title: "Bu model için diskte test analizi yok" }, "Test analizi")
  );

  return h(
    "div",
    { class: "card" },
    h("h3", {}, m.name),
    h("div", { class: "row" }, ...badges),
    h("div", { class: "kv" }, h("span", {}, "grafik tipi"), h("span", {}, m.graphic_label)),
    h("div", { class: "kv" }, h("span", {}, "çıktı görüntüsü"), h("span", {}, String(m.output_count))),
    h("div", { class: "kv" }, h("span", {}, "ağırlık dosyası"), h("span", {}, m.weights_present ? "var" : "yok")),
    provenance,
    actions
  );
}

function renderModels() {
  setActiveNav("models");
  updateChip();
  mount(h("h2", { class: "view-title" }, "Modeller"), h("div", { class: "cards" }, ...state.models.map(modelCard)));
}

// ── Görünüm: Göz at (grid) ──────────────────────────────────────────────────
function tile(item) {
  const key = item.key;
  const hasOut =
    "has_png" in item
      ? item.has_png
      : state.currentModel
      ? (item.models_with_output || []).includes(state.currentModel)
      : (item.models_with_output || []).length > 0;

  const img = h("img", {
    class: "thumb",
    loading: "lazy",
    src: "/img/source/" + enc(key),
    alt: item.display_id,
    onerror: function () {
      this.replaceWith(h("div", { class: "thumb missing" }, "görsel yok"));
    },
  });

  return h(
    "button",
    { class: "tile", onclick: () => (location.hash = "#/image/" + enc(key)) },
    img,
    h(
      "div",
      { class: "cap" },
      h("span", { class: "id" }, item.display_id),
      h("span", { class: "dot " + (hasOut ? "has" : "no"), title: hasOut ? "çıktı var" : "çıktı yok" })
    )
  );
}

async function renderBrowse() {
  setActiveNav("browse");
  updateChip();

  const modelSelect = h(
    "select",
    {
      onchange: (e) => {
        selectModel(e.target.value || null);
        renderBrowse();
      },
    },
    h("option", { value: "" }, "(model seçilmedi)"),
    ...state.models.map((m) => h("option", { value: m.name }, m.name))
  );

  const seg = h(
    "div",
    { class: "seg" },
    h(
      "button",
      {
        class: state.mode === "all" ? "active" : "",
        onclick: () => {
          state.mode = "all";
          state.imagesSig = null;
          renderBrowse();
        },
      },
      "Tüm dataset"
    ),
    h(
      "button",
      {
        class: state.mode === "model" ? "active" : "",
        onclick: () => {
          if (!state.currentModel) {
            alert("Önce bir model seç.");
            return;
          }
          state.mode = "model";
          state.imagesSig = null;
          renderBrowse();
        },
      },
      "Bu modelin görüntüleri"
    )
  );

  const toolbar = h(
    "div",
    { class: "toolbar" },
    h("label", {}, "Model"),
    modelSelect,
    seg,
    h("span", { id: "count", class: "muted small" }, "…")
  );

  mount(
    h("h2", { class: "view-title" }, "Görüntüler"),
    toolbar,
    h("div", { id: "grid", class: "grid" }, h("div", { class: "loading" }, "Yükleniyor…"))
  );
  modelSelect.value = state.currentModel || "";

  try {
    await ensureImages();
  } catch (e) {
    document.getElementById("grid").replaceChildren(h("div", { class: "empty-box" }, "Liste yüklenemedi: " + e.message));
    return;
  }
  document.getElementById("count").textContent = state.images.length + " görüntü";
  document.getElementById("grid").replaceChildren(...state.images.map(tile));
}

// ── Görünüm: Tek görüntü detayı ─────────────────────────────────────────────
function gtBlock(rec) {
  if (!rec.ground_truth.length) {
    return h("div", { class: "block" }, h("h4", {}, "Ground truth (CSV)"), h("div", { class: "muted" }, "Bu görüntü için CSV satırı yok."));
  }
  const body = [];
  rec.ground_truth.forEach((g, i) => {
    if (rec.ground_truth.length > 1) body.push(h("div", { class: "muted small" }, "satır " + (i + 1)));
    body.push(
      h(
        "table",
        { class: "kv-table" },
        g.mpta != null ? tr("MPTA", g.mpta) : null,
        g.ldfa != null ? tr("LDFA", g.ldfa) : null,
        g.diff != null ? tr("MPTA-LDFA", g.diff) : null,
        g.sum != null ? tr("MPTA+LDFA", g.sum) : null,
        g.dizilim ? tr("DİZİLİM", g.dizilim) : null,
        g.eklem ? tr("EKLEM", g.eklem) : null,
        g.siniflama ? tr("SINIFLAMA", g.siniflama) : null
      )
    );
  });
  return h("div", { class: "block" }, h("h4", {}, "Ground truth (CSV)"), ...body);
}

function predBlock(rec, modelName, out) {
  if (!modelName) {
    return h("div", { class: "block" }, h("h4", {}, "Model tahmini"), h("div", { class: "muted" }, "Model seçilmedi."));
  }
  const m = state.byName[modelName];
  if (!out) {
    return h(
      "div",
      { class: "block" },
      h("h4", {}, "Model tahmini — " + modelName),
      h("div", { class: "note" }, "Bu model bu görüntü için çıktı üretmemiş.")
    );
  }
  const t = h("table", { class: "kv-table" });
  if (out.mpta != null && out.ldfa != null) {
    [
      tr("MPTA", out.mpta),
      tr("LDFA", out.ldfa),
      tr("MPTA-LDFA", out.diff),
      tr("MPTA+LDFA", out.sum),
      tr("DİZİLİM", out.dizilim),
      tr("EKLEM", out.eklem),
      tr("SINIFLAMA", out.siniflama),
    ].forEach((r) => t.append(r));
  } else {
    t.append(h("tr", {}, h("td", { colspan: 2 }, h("span", { class: "note" }, "MPTA/LDFA bu modelde mevcut değil."))));
  }
  if (out.extra) for (const [k, v] of Object.entries(out.extra)) t.append(tr(k, v));

  const srcNote = m && m.angle_source ? h("div", { class: "muted small" }, "kaynak: " + m.inference_dir + "/" + m.angle_source) : null;
  return h("div", { class: "block" }, h("h4", {}, "Model tahmini — " + modelName), t, srcNote);
}

function compareBlock(rec) {
  const names = Object.keys(rec.outputs).sort();
  if (names.length < 2) return null;
  const cards = names.map((n) => {
    const o = rec.outputs[n];
    const metrics =
      o.mpta != null ? "MPTA " + o.mpta + " · LDFA " + o.ldfa : o.extra && o.extra.jlca != null ? "jlca " + o.extra.jlca : "açı yok";
    const im = o.has_png
      ? h("img", { loading: "lazy", src: "/img/inference/" + enc(n) + "/" + enc(rec.key) })
      : h("div", { class: "thumb missing" }, "görsel yok");
    return h(
      "div",
      {
        class: "c",
        onclick: () => {
          selectModel(n);
          renderImage(rec.key);
        },
      },
      im,
      h("div", { class: "lab" }, h("b", {}, n), h("span", { class: "muted" }, metrics))
    );
  });
  return h("div", { class: "block" }, h("h4", {}, "Modeller arası karşılaştırma"), h("div", { class: "compare" }, ...cards));
}

async function renderImage(key) {
  setActiveNav("browse");
  updateChip();
  try {
    await ensureImages();
  } catch (e) {
    /* liste yoksa gezinme kapalı kalır */
  }

  let rec;
  try {
    rec = await api("/api/images/" + enc(key));
  } catch (e) {
    mount(h("div", { class: "empty-box" }, "Görüntü bulunamadı: " + key));
    return;
  }

  const list = state.images;
  const idx = list.findIndex((x) => x.key === key);
  const prevKey = idx > 0 ? list[idx - 1].key : null;
  const nextKey = idx >= 0 && idx < list.length - 1 ? list[idx + 1].key : null;

  const modelName = state.currentModel;
  const out = modelName ? rec.outputs[modelName] : null;
  const hasAnno = !!(out && out.has_png);

  const stageImg = h("img", { alt: rec.display_id });
  const empty = h("div", { class: "empty hidden" });
  function setStage(which) {
    if (which === "anno" && hasAnno) {
      stageImg.src = "/img/inference/" + enc(modelName) + "/" + enc(key);
      stageImg.classList.remove("hidden");
      empty.classList.add("hidden");
    } else if (rec.has_source) {
      stageImg.src = "/img/source/" + enc(key);
      stageImg.classList.remove("hidden");
      empty.classList.add("hidden");
    } else {
      stageImg.classList.add("hidden");
      empty.classList.remove("hidden");
      empty.textContent = "Bu görüntü için dosya yok.";
    }
  }

  const annoBtn = h("button", { disabled: !hasAnno, onclick: () => mark("anno") }, "Annotasyonlu çıktı");
  const srcBtn = h("button", { disabled: !rec.has_source, onclick: () => mark("source") }, "Kaynak görüntü");
  function mark(which) {
    setStage(which);
    annoBtn.classList.toggle("active", which === "anno");
    srcBtn.classList.toggle("active", which === "source");
  }

  const nav = h(
    "div",
    { class: "navbtns" },
    h("button", { disabled: !prevKey, onclick: () => prevKey && (location.hash = "#/image/" + enc(prevKey)) }, "‹"),
    h("button", { disabled: !nextKey, onclick: () => nextKey && (location.hash = "#/image/" + enc(nextKey)) }, "›")
  );

  const stageBar = h(
    "div",
    { class: "stage-bar" },
    h("button", { onclick: () => (location.hash = "#/browse") }, "‹ Grid"),
    annoBtn,
    srcBtn,
    h("span", { class: "muted small" }, (idx >= 0 ? idx + 1 : "?") + " / " + list.length)
  );

  const panel = h("div", { class: "panel" }, ...[predBlock(rec, modelName, out), gtBlock(rec), compareBlock(rec)].filter(Boolean));

  mount(
    h(
      "h2",
      { class: "view-title" },
      rec.display_id + "  ",
      h("span", { class: "badge " + (rec.expected_model_kind || "") }, rec.graphic_label)
    ),
    stageBar,
    h("div", { class: "detail" }, h("div", { class: "stage" }, stageImg, empty, nav), panel)
  );

  mark(hasAnno ? "anno" : "source");

  keyHandler = (e) => {
    if (e.key === "ArrowLeft" && prevKey) location.hash = "#/image/" + enc(prevKey);
    else if (e.key === "ArrowRight" && nextKey) location.hash = "#/image/" + enc(nextKey);
    else if (e.key === "Escape") location.hash = "#/browse";
  };
  window.addEventListener("keydown", keyHandler);
}

// ── Görünüm: Test analizi ───────────────────────────────────────────────────
function metricCard(title, big, sub) {
  return h("div", { class: "metric" }, h("div", { class: "t" }, title), h("div", { class: "big" }, String(big)), sub ? h("div", { class: "muted small" }, sub) : null);
}

function sidToKey(sid, grafi) {
  const n = String(sid).toLowerCase().replace(/[._\s]/g, "").match(/^(\d+)([lr])$/);
  return n ? n[1] + "~" + n[2] + "~" + grafi : null;
}

function distGroup(name, group, grafi) {
  const entries = Object.entries(group);
  const counts = entries.map(([, v]) => (v && typeof v === "object" && "count" in v ? v.count : null)).filter((c) => c != null);
  const max = Math.max(1, ...counts);
  const rows = [];
  for (const [label, v] of entries) {
    if (v && typeof v === "object" && "count" in v) {
      const samples = Array.isArray(v.samples) ? v.samples : [];
      const row = h(
        "div",
        { class: "bar-row" + (samples.length ? " expandable" : "") },
        h("span", {}, label),
        h("div", { class: "bar-track" }, h("div", { class: "bar-fill", style: "width:" + ((v.count / max) * 100).toFixed(1) + "%" })),
        h("span", { class: "muted" }, v.count + "  " + (v.percentage || ""))
      );
      rows.push(row);
      if (samples.length) {
        const box = h(
          "div",
          { class: "samples hidden" },
          ...samples.map((sid) => {
            const key = sidToKey(sid, grafi);
            return key ? h("a", { href: "#/image/" + enc(key) }, sid) : h("span", { class: "muted" }, sid);
          })
        );
        row.addEventListener("click", () => box.classList.toggle("hidden"));
        rows.push(box);
      }
    } else {
      rows.push(h("div", { class: "bar-row" }, h("span", {}, label), h("span", { class: "muted", style: "grid-column:2/4" }, JSON.stringify(v))));
    }
  }
  return h("div", { class: "section" }, h("h3", {}, name + " hata dağılımı"), h("div", { class: "bars" }, ...rows));
}

const CHART_LABEL = {
  distribution_plot: "Hata dağılımı",
  femur_comparison_graphs: "Femur karşılaştırma",
  tibia_comparison_graphs: "Tibia karşılaştırma",
};

function perImageTable(per) {
  const rows = Object.entries(per).map(([key, r]) => Object.assign({ key }, r));
  const cols = [
    { k: "id", label: "id", t: "s" },
    { k: "f_gt", label: "F gt", t: "n" },
    { k: "f_pred", label: "F pred", t: "n" },
    { k: "f_err", label: "F err", t: "n" },
    { k: "t_gt", label: "T gt", t: "n" },
    { k: "t_pred", label: "T pred", t: "n" },
    { k: "t_err", label: "T err", t: "n" },
    { k: "gt_type", label: "GT tip", t: "s" },
    { k: "pred_type", label: "Tah tip", t: "s" },
    { k: "type_correct", label: "doğru?", t: "b" },
  ];
  let sortK = "f_err",
    sortDir = -1,
    onlyWrong = false;
  const tbody = h("tbody", {});
  const fmt = (v, t) => (v == null ? "—" : t === "n" ? (typeof v === "number" ? v.toFixed(2) : v) : t === "b" ? (v ? "✓" : "✗") : v);

  function render() {
    let data = rows.slice();
    if (onlyWrong) data = data.filter((r) => !r.type_correct);
    data.sort((a, b) => {
      const x = a[sortK],
        y = b[sortK];
      if (x == null) return 1;
      if (y == null) return -1;
      return x < y ? -sortDir : x > y ? sortDir : 0;
    });
    tbody.replaceChildren(
      ...data.map((r) =>
        h(
          "tr",
          { class: r.type_correct ? "" : "wrong", style: "cursor:pointer", onclick: () => (location.hash = "#/image/" + enc(r.key)) },
          ...cols.map((c) => h("td", {}, fmt(r[c.k], c.t)))
        )
      )
    );
  }

  const thead = h(
    "thead",
    {},
    h(
      "tr",
      {},
      ...cols.map((c) =>
        h(
          "th",
          {
            onclick: () => {
              if (sortK === c.k) sortDir *= -1;
              else {
                sortK = c.k;
                sortDir = 1;
              }
              render();
            },
          },
          c.label
        )
      )
    )
  );

  const filter = h(
    "label",
    {},
    h("input", { type: "checkbox", onchange: (e) => { onlyWrong = e.target.checked; render(); } }),
    " yalnız yanlış sınıflama"
  );
  render();
  return h(
    "div",
    { class: "section" },
    h("h3", {}, "Görüntü-başı sonuçlar (" + rows.length + ")"),
    h("div", { class: "filters" }, filter, h("span", { class: "muted small" }, "satıra tıkla → görüntüye git")),
    h("div", { class: "table-wrap" }, h("table", { class: "data" }, thead, tbody))
  );
}

async function renderTest(model) {
  setActiveNav(null);
  if (state.currentModel !== model) selectModel(model);
  updateChip();

  let t;
  try {
    t = await api("/api/models/" + enc(model) + "/test");
  } catch (e) {
    mount(h("div", { class: "empty-box" }, "Model bulunamadı: " + model));
    return;
  }

  if (!t.has_test) {
    mount(
      h("h2", { class: "view-title" }, "Test analizi — " + model),
      h("div", { class: "empty-box" }, "Bu model için diskte test analizi mevcut değil.", h("div", { class: "path mono" }, "bakılan yol: " + t.looked_at))
    );
    return;
  }

  const grafi = (state.byName[model] && state.byName[model].grafi) || "1";
  const met = t.metrics || {};
  const cards = [];
  if (met.femur) cards.push(metricCard("Femur MAE", met.femur.MAE, "RMSE " + met.femur.RMSE + " · R² " + met.femur.R2 + " · n=" + met.femur.sample_count));
  if (met.tibia) cards.push(metricCard("Tibia MAE", met.tibia.MAE, "RMSE " + met.tibia.RMSE + " · R² " + met.tibia.R2 + " · n=" + met.tibia.sample_count));
  if (met.classification)
    cards.push(metricCard("Sınıflandırma", met.classification.accuracy + "%", met.classification.correct + "/" + met.classification.total + " doğru"));

  const sections = [];
  if (t.distribution) for (const [name, group] of Object.entries(t.distribution)) sections.push(distGroup(name, group, grafi));

  const charts = (t.charts || []).map((c) =>
    h("figure", {}, h("img", { loading: "lazy", src: "/img/test/" + enc(model) + "/" + enc(c) }), h("figcaption", {}, CHART_LABEL[c] || c))
  );

  mount(
    h("h2", { class: "view-title" }, "Test analizi — " + model, "  ", h("span", { class: "muted small" }, t.test_dir)),
    h("div", { class: "metric-cards" }, ...cards),
    ...sections,
    charts.length ? h("div", { class: "section" }, h("h3", {}, "Görsel analiz"), h("div", { class: "charts" }, ...charts)) : null,
    t.per_image && Object.keys(t.per_image).length ? perImageTable(t.per_image) : null
  );
}

// ── Başlat ──────────────────────────────────────────────────────────────────
async function init() {
  try {
    state.models = await api("/api/models");
    state.byName = Object.fromEntries(state.models.map((m) => [m.name, m]));
  } catch (e) {
    mount(h("div", { class: "empty-box" }, "Sunucuya bağlanılamadı: " + e.message));
    return;
  }
  window.addEventListener("hashchange", route);
  if (!location.hash) location.hash = "#/models";
  else route();
}

document.addEventListener("DOMContentLoaded", init);
