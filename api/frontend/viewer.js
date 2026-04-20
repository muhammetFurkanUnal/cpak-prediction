// ImageViewer — zoom (scroll), pan (drag + momentum), double-click reset, fullscreen
class ImageViewer {
  constructor(container) {
    this.container = container;
    this.img       = null;
    this.scale     = 1;
    this.tx        = 0;
    this.ty        = 0;
    this.dragging  = false;
    this.lastX     = 0;
    this.lastY     = 0;
    this.vx        = 0;
    this.vy        = 0;
    this._rafId    = null;
    this.MIN_SCALE        = 0.1;
    this.MAX_SCALE        = 20;
    this.onTransformChange = null;
    this.toolbar   = null;
    this.zoomBadge = null;
  }

  mountImage(img) {
    this.img = img;
    if (img.complete && img.naturalWidth) {
      this._fitToContainer();
    } else {
      img.onload = () => this._fitToContainer();
    }
    this._buildToolbar();
    this._buildZoomBadge();
    this._attachEvents();
  }

  _fitToContainer() {
    const cw = this.container.clientWidth  || 600;
    const ch = this.container.clientHeight || 800;
    const iw = this.img.naturalWidth;
    const ih = this.img.naturalHeight;
    const scaleX   = cw / iw;
    const scaleY   = ch / ih;
    this.MIN_SCALE = Math.min(scaleX, scaleY); // can't zoom out past initial fit
    this.scale     = this.MIN_SCALE;
    this.tx        = (cw - iw * this.scale) / 2;
    this.ty        = (ch - ih * this.scale) / 2;
    this._applyTransform();
    this._updateBadge();
  }

  _buildToolbar() {
    this.toolbar = document.createElement('div');
    this.toolbar.className = 'viewer-toolbar';
    this.toolbar.innerHTML = `
      <button class="icon-btn" data-action="zoom-in"  title="Zoom in">
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/><line x1="11" y1="8" x2="11" y2="14"/><line x1="8" y1="11" x2="14" y2="11"/></svg>
      </button>
      <button class="icon-btn" data-action="zoom-out" title="Zoom out">
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/><line x1="8" y1="11" x2="14" y2="11"/></svg>
      </button>
      <button class="icon-btn" data-action="reset"    title="Reset view (double-click)">
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round"><path d="M3 12a9 9 0 1 0 9-9 9.75 9.75 0 0 0-6.74 2.74L3 8"/><path d="M3 3v5h5"/></svg>
      </button>
      <div class="toolbar-sep"></div>
      <button class="icon-btn" data-action="fullscreen" title="Fullscreen">
        <svg class="icon-enter-fs" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round"><path d="M8 3H5a2 2 0 0 0-2 2v3m18 0V5a2 2 0 0 0-2-2h-3m0 18h3a2 2 0 0 0 2-2v-3M3 16v3a2 2 0 0 0 2 2h3"/></svg>
        <svg class="icon-exit-fs"  width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" style="display:none"><path d="M8 3v3a2 2 0 0 1-2 2H3m18 0h-3a2 2 0 0 1-2-2V3m0 18v-3a2 2 0 0 1 2-2h3M3 16h3a2 2 0 0 1 2 2v3"/></svg>
      </button>
    `;
    this.toolbar.addEventListener('mousedown', e => e.stopPropagation());
    this.toolbar.addEventListener('dblclick',  e => e.stopPropagation());
    this.toolbar.addEventListener('click', e => {
      const btn = e.target.closest('[data-action]');
      if (!btn) return;
      const cx = this.container.clientWidth  / 2;
      const cy = this.container.clientHeight / 2;
      switch (btn.dataset.action) {
        case 'zoom-in':    this._zoom(1.25, cx, cy); break;
        case 'zoom-out':   this._zoom(0.8,  cx, cy); break;
        case 'reset':      this._fitToContainer(); break;
        case 'fullscreen': this._toggleFullscreen(); break;
      }
    });
    this.container.appendChild(this.toolbar);
    document.addEventListener('fullscreenchange',       () => this._syncFsIcons());
    document.addEventListener('webkitfullscreenchange', () => this._syncFsIcons());
  }

  _buildZoomBadge() {
    this.zoomBadge = document.createElement('div');
    this.zoomBadge.className = 'zoom-badge';
    this.container.appendChild(this.zoomBadge);
  }

  _attachEvents() {
    const c = this.container;

    // Wheel → zoom around cursor
    c.addEventListener('wheel', e => {
      e.preventDefault();
      const rect   = c.getBoundingClientRect();
      const cx     = e.clientX - rect.left;
      const cy     = e.clientY - rect.top;
      const factor = e.deltaY < 0 ? 1.12 : 1 / 1.12;
      this._zoom(factor, cx, cy);
    }, { passive: false });

    // Mouse drag → pan with momentum
    c.addEventListener('mousedown', e => {
      if (e.button !== 0) return;
      this._stopMomentum();
      this.dragging = true;
      this.lastX = e.clientX;
      this.lastY = e.clientY;
      this.vx = 0; this.vy = 0;
      c.classList.add('dragging');
    });
    window.addEventListener('mousemove', e => {
      if (!this.dragging) return;
      const dx = e.clientX - this.lastX;
      const dy = e.clientY - this.lastY;
      this.vx = dx; this.vy = dy;
      this.tx += dx; this.ty += dy;
      this.lastX = e.clientX; this.lastY = e.clientY;
      this._applyTransform();
    });
    window.addEventListener('mouseup', () => {
      if (!this.dragging) return;
      this.dragging = false;
      c.classList.remove('dragging');
      this._startMomentum();
    });

    // Touch — single finger pan, two-finger pinch zoom
    let lastTouches = null;
    c.addEventListener('touchstart', e => {
      this._stopMomentum();
      this.vx = 0; this.vy = 0;
      lastTouches = e.touches;
    }, { passive: true });
    c.addEventListener('touchmove', e => {
      e.preventDefault();
      if (e.touches.length === 1 && lastTouches?.length === 1) {
        const dx = e.touches[0].clientX - lastTouches[0].clientX;
        const dy = e.touches[0].clientY - lastTouches[0].clientY;
        this.vx = dx; this.vy = dy;
        this.tx += dx; this.ty += dy;
        this._applyTransform();
      } else if (e.touches.length === 2 && lastTouches?.length === 2) {
        const prevDist = Math.hypot(lastTouches[0].clientX - lastTouches[1].clientX, lastTouches[0].clientY - lastTouches[1].clientY);
        const currDist = Math.hypot(e.touches[0].clientX - e.touches[1].clientX, e.touches[0].clientY - e.touches[1].clientY);
        const rect = c.getBoundingClientRect();
        const cx = ((e.touches[0].clientX + e.touches[1].clientX) / 2) - rect.left;
        const cy = ((e.touches[0].clientY + e.touches[1].clientY) / 2) - rect.top;
        this._zoom(currDist / prevDist, cx, cy);
      }
      lastTouches = e.touches;
    }, { passive: false });
    c.addEventListener('touchend', () => {
      lastTouches = null;
      this._startMomentum();
    }, { passive: true });

    // Double-click → reset
    c.addEventListener('dblclick', () => this._fitToContainer());
  }

  _zoom(factor, cx, cy) {
    const newScale = Math.min(this.MAX_SCALE, Math.max(this.MIN_SCALE, this.scale * factor));
    const ratio    = newScale / this.scale;
    this.tx    = cx - (cx - this.tx) * ratio;
    this.ty    = cy - (cy - this.ty) * ratio;
    this.scale = newScale;
    this._applyTransform();
    this._updateBadge();
  }

  _clampTranslation() {
    if (!this.img) return;
    const cw = this.container.clientWidth;
    const ch = this.container.clientHeight;
    const iw = this.img.naturalWidth  * this.scale;
    const ih = this.img.naturalHeight * this.scale;

    // Each axis: if image fits inside container, center it; otherwise clamp so
    // image edges never move past container edges (no empty space, no sliding off).
    if (iw <= cw) {
      this.tx = (cw - iw) / 2;
    } else {
      this.tx = Math.min(this.tx, 0);
      this.tx = Math.max(this.tx, cw - iw);
    }

    if (ih <= ch) {
      this.ty = (ch - ih) / 2;
    } else {
      this.ty = Math.min(this.ty, 0);
      this.ty = Math.max(this.ty, ch - ih);
    }
  }

  _startMomentum() {
    const FRICTION = 0.88;
    const tick = () => {
      this.vx *= FRICTION;
      this.vy *= FRICTION;
      if (Math.abs(this.vx) < 0.3 && Math.abs(this.vy) < 0.3) return;
      this.tx += this.vx;
      this.ty += this.vy;
      this._applyTransform();
      this._rafId = requestAnimationFrame(tick);
    };
    this._rafId = requestAnimationFrame(tick);
  }

  _stopMomentum() {
    if (this._rafId) { cancelAnimationFrame(this._rafId); this._rafId = null; }
  }

  _applyTransform() {
    if (!this.img) return;
    this._clampTranslation();
    this.img.style.transform = `translate(${this.tx}px, ${this.ty}px) scale(${this.scale})`;
    // Keep grid cell size between 20–80px on screen
    let gs = 40 * this.scale;
    while (gs > 80) gs /= 2;
    while (gs < 20) gs *= 2;
    const ox = ((this.tx % gs) + gs) % gs;
    const oy = ((this.ty % gs) + gs) % gs;
    this.container.style.setProperty('--grid-size', `${gs}px`);
    this.container.style.setProperty('--grid-ox',   `${ox}px`);
    this.container.style.setProperty('--grid-oy',   `${oy}px`);
    if (this.onTransformChange) this.onTransformChange();
  }

  _updateBadge() {
    if (this.zoomBadge) {
      this.zoomBadge.textContent = `${Math.round(this.scale * 100)}%`;
    }
  }

  _toggleFullscreen() {
    const c = this.container;
    if (!document.fullscreenElement) {
      (c.requestFullscreen || c.webkitRequestFullscreen || c.mozRequestFullScreen).call(c)
        .then(() => { setTimeout(() => this._fitToContainer(), 50); }).catch(() => {});
    } else {
      (document.exitFullscreen || document.webkitExitFullscreen || document.mozCancelFullScreen).call(document);
    }
  }

  _syncFsIcons() {
    if (!this.toolbar) return;
    const inside = document.fullscreenElement === this.container;
    this.toolbar.querySelector('.icon-enter-fs').style.display = inside ? 'none' : '';
    this.toolbar.querySelector('.icon-exit-fs').style.display  = inside ? ''     : 'none';
    if (inside) setTimeout(() => this._fitToContainer(), 50);
  }
}
