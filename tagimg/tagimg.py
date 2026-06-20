"""Minimal cross-platform image tagger.

Pick a folder, create tags via the "New tag" button, then choose one from
the dropdown and apply it to the selected thumbnails. Save the result as
JSON (the file also stores every tag ever created). Columns per row 1-5.

Usage:
    python3 tagimg.py
"""

import json
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, simpledialog, ttk

from PIL import Image, ImageTk


IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tif", ".tiff"}
SEL_COLOR = "#3a7"
BG_COLOR = "#222"
FG_COLOR = "#ddd"


class TagImg:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        root.title("tagimg")
        root.geometry("1100x750")

        self.folder: Path | None = None
        self.image_paths: list[Path] = []
        self.selected: set[Path] = set()
        self.tags: dict[str, list[str]] = {}
        self.thumbs: dict[Path, ImageTk.PhotoImage] = {}
        self.cards: dict[Path, tk.Frame] = {}
        self.tag_labels: dict[Path, tk.Label] = {}
        self.columns = tk.IntVar(value=4)
        self.available_tags: list[str] = []
        self.tag_choice = tk.StringVar()
        self.filtered_paths: list[Path] = []
        self.filter_vars: dict[str, tk.BooleanVar] = {}
        self.filter_exclude = tk.BooleanVar(value=False)
        self._resize_job: str | None = None
        self._last_thumb: int = 0
        self._dirty = False

        self._setup_styles()
        self._build_toolbar()
        self._build_filter_bar()
        self._build_grid()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _setup_styles(self) -> None:
        style = ttk.Style()
        style.configure("Apply.TButton", foreground="#2ea043",
                        font=("TkDefaultFont", 12, "bold"), padding=(4, 0))
        style.configure("Remove.TButton", foreground="#d33",
                        font=("TkDefaultFont", 12, "bold"), padding=(4, 0))
        style.configure("New.TButton", foreground="#3a8af0",
                        font=("TkDefaultFont", 14, "bold"), padding=(4, 0))

    def _build_toolbar(self) -> None:
        bar = ttk.Frame(self.root, padding=6)
        bar.pack(side=tk.TOP, fill=tk.X)

        ttk.Button(bar, text="Open folder", command=self.open_folder).pack(side=tk.LEFT)

        ttk.Label(bar, text="  Columns:").pack(side=tk.LEFT)
        cols = ttk.Combobox(
            bar, width=3, values=[1, 2, 3, 4, 5],
            textvariable=self.columns, state="readonly",
        )
        cols.pack(side=tk.LEFT)
        cols.bind("<<ComboboxSelected>>", lambda _e: self._force_refresh())

        ttk.Label(bar, text="  Tag:").pack(side=tk.LEFT)
        self.tag_combo = ttk.Combobox(
            bar, textvariable=self.tag_choice, state="readonly", width=22,
        )
        self.tag_combo.pack(side=tk.LEFT)
        ttk.Button(bar, text="✓", width=3, style="Apply.TButton",
                   command=self.apply_tag).pack(side=tk.LEFT, padx=4)
        ttk.Button(bar, text="🗑", width=3, style="Remove.TButton",
                   command=self.remove_tag).pack(side=tk.LEFT)
        ttk.Button(bar, text="+", width=3, style="New.TButton",
                   command=self.new_tag).pack(side=tk.LEFT, padx=4)

        ttk.Button(bar, text="Select all", command=self.select_all).pack(side=tk.LEFT)
        ttk.Button(bar, text="Clear selection", command=self.clear_selection).pack(side=tk.LEFT)
        ttk.Button(bar, text="Save JSON", command=self.save_json).pack(side=tk.RIGHT)

    def _build_filter_bar(self) -> None:
        bar = ttk.Frame(self.root, padding=(6, 0, 6, 6))
        bar.pack(side=tk.TOP, fill=tk.X)
        ttk.Label(bar, text="Filter:").pack(side=tk.LEFT)
        self.filter_cb_frame = ttk.Frame(bar)
        self.filter_cb_frame.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=4)
        ttk.Button(bar, text="Clear filter", command=self.clear_filter).pack(side=tk.RIGHT)
        ttk.Checkbutton(
            bar, text="Exclude", variable=self.filter_exclude,
            command=self._apply_filter,
        ).pack(side=tk.RIGHT, padx=4)

    def _build_grid(self) -> None:
        container = ttk.Frame(self.root)
        container.pack(fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(container, highlightthickness=0, bg=BG_COLOR)
        scroll = ttk.Scrollbar(container, orient=tk.VERTICAL, command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=scroll.set)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.inner = tk.Frame(self.canvas, bg=BG_COLOR)
        self.inner_window = self.canvas.create_window((0, 0), window=self.inner, anchor="nw")
        self.inner.bind(
            "<Configure>",
            lambda _e: self.canvas.configure(scrollregion=self.canvas.bbox("all")),
        )
        self.canvas.bind("<Configure>", self._on_canvas_resize)
        # Mousewheel: macOS / Windows use <MouseWheel>, Linux uses Button-4/5.
        self.canvas.bind_all("<MouseWheel>",
                             lambda e: self.canvas.yview_scroll(int(-e.delta / 120), "units"))
        self.canvas.bind_all("<Button-4>", lambda _e: self.canvas.yview_scroll(-1, "units"))
        self.canvas.bind_all("<Button-5>", lambda _e: self.canvas.yview_scroll(1, "units"))

    def _force_refresh(self) -> None:
        self._last_thumb = 0
        self.refresh_grid()

    def _on_canvas_resize(self, e: tk.Event) -> None:
        # Stretch inner frame to fill canvas width so no horizontal scroll is needed.
        self.canvas.itemconfigure(self.inner_window, width=e.width)
        if not self.image_paths:
            return
        # Debounce: only re-render after resize settles.
        if self._resize_job is not None:
            self.root.after_cancel(self._resize_job)
        self._resize_job = self.root.after(120, self.refresh_grid)

    def open_folder(self) -> None:
        d = filedialog.askdirectory(title="Image folder")
        if not d:
            return
        self.folder = Path(d)
        self.image_paths = sorted(
            p for p in self.folder.iterdir() if p.suffix.lower() in IMG_EXTS
        )
        self.selected.clear()
        self.tags = {p.name: [] for p in self.image_paths}
        self._load_existing_tags()
        self.thumbs.clear()
        self.filtered_paths = list(self.image_paths)
        self._apply_filter()
        self._dirty = False

    def _load_existing_tags(self) -> None:
        self.available_tags = []
        self._refresh_tag_choices()
        if self.folder is None:
            return
        json_path = self.folder / "tags.json"
        if not json_path.exists():
            return
        try:
            data = json.loads(json_path.read_text())
        except Exception:
            return
        if not isinstance(data, dict):
            return
        existing = data.get("tags", {})
        if isinstance(existing, dict):
            for name, tags in existing.items():
                if name in self.tags and isinstance(tags, list):
                    self.tags[name] = [t for t in tags if isinstance(t, str)]
        avail = data.get("available_tags", [])
        if isinstance(avail, list):
            self.available_tags = [t for t in avail if isinstance(t, str)]
        # Backfill: any tag found on an image but not in available_tags.
        for tags in self.tags.values():
            for t in tags:
                if t not in self.available_tags:
                    self.available_tags.append(t)
        self._refresh_tag_choices()

    def refresh_grid(self) -> None:
        self._resize_job = None
        cols = self.columns.get()
        self.root.update_idletasks()
        canvas_w = self.canvas.winfo_width()
        if canvas_w <= 1:
            # Canvas not yet realized; try again on next idle.
            self.root.after(50, self.refresh_grid)
            return
        cell_pad = 8
        thumb = max(80, (canvas_w - cell_pad * 2 * cols) // cols - 8)
        # Skip rebuild if thumb size hasn't meaningfully changed and grid is populated.
        if self.cards and thumb == self._last_thumb:
            return
        self._last_thumb = thumb

        for child in self.inner.winfo_children():
            child.destroy()
        self.cards.clear()
        self.tag_labels.clear()
        for c in range(cols):
            self.inner.columnconfigure(c, weight=1, uniform="col")

        for i, p in enumerate(self.filtered_paths):
            try:
                img = Image.open(p)
                img.thumbnail((thumb, thumb))
                photo = ImageTk.PhotoImage(img)
            except Exception:
                continue
            self.thumbs[p] = photo

            r, c = divmod(i, cols)
            card = tk.Frame(self.inner, bg=BG_COLOR, padx=4, pady=4)
            card.grid(row=r, column=c, padx=cell_pad, pady=cell_pad)
            img_lbl = tk.Label(card, image=photo, bg=BG_COLOR)
            img_lbl.pack()
            name_lbl = tk.Label(card, text=p.name, bg=BG_COLOR, fg=FG_COLOR,
                                wraplength=thumb, font=("TkDefaultFont", 9))
            name_lbl.pack()
            tag_lbl = tk.Label(card, text=", ".join(self.tags.get(p.name, [])),
                               bg=BG_COLOR, fg="#7df",
                               wraplength=thumb, font=("TkDefaultFont", 8))
            tag_lbl.pack()
            for w in (card, img_lbl, name_lbl, tag_lbl):
                w.bind("<Button-1>", lambda _e, path=p: self.toggle(path))
            self.cards[p] = card
            self.tag_labels[p] = tag_lbl
            self._update_card_style(p)

    def _update_card_style(self, p: Path) -> None:
        card = self.cards.get(p)
        if card is None:
            return
        color = SEL_COLOR if p in self.selected else BG_COLOR
        card.config(bg=color)
        for child in card.winfo_children():
            child.config(bg=color)

    def toggle(self, p: Path) -> None:
        if p in self.selected:
            self.selected.remove(p)
        else:
            self.selected.add(p)
        self._update_card_style(p)

    def select_all(self) -> None:
        for p in self.filtered_paths:
            if p not in self.selected:
                self.selected.add(p)
                self._update_card_style(p)

    def clear_selection(self) -> None:
        was = list(self.selected)
        self.selected.clear()
        for p in was:
            self._update_card_style(p)

    def new_tag(self) -> None:
        name = simpledialog.askstring("New tag", "Tag name:", parent=self.root)
        if not name:
            return
        name = name.strip()
        if not name or name in self.available_tags:
            return
        self.available_tags.append(name)
        self._refresh_tag_choices()
        self.tag_choice.set(name)
        self._dirty = True

    def _refresh_tag_choices(self) -> None:
        self.tag_combo["values"] = list(self.available_tags)
        if self.tag_choice.get() not in self.available_tags:
            self.tag_choice.set("")
        self._rebuild_filter_checkboxes()

    def _rebuild_filter_checkboxes(self) -> None:
        prev = {t: v.get() for t, v in self.filter_vars.items()}
        for child in self.filter_cb_frame.winfo_children():
            child.destroy()
        self.filter_vars.clear()
        for t in self.available_tags:
            var = tk.BooleanVar(value=prev.get(t, False))
            ttk.Checkbutton(
                self.filter_cb_frame, text=t, variable=var,
                command=self._apply_filter,
            ).pack(side=tk.LEFT, padx=2)
            self.filter_vars[t] = var

    def clear_filter(self) -> None:
        for var in self.filter_vars.values():
            var.set(False)
        self.filter_exclude.set(False)
        self._apply_filter()

    def _apply_filter(self) -> None:
        active = {t for t, var in self.filter_vars.items() if var.get()}
        if not active:
            self.filtered_paths = list(self.image_paths)
        elif self.filter_exclude.get():
            self.filtered_paths = [
                p for p in self.image_paths
                if not (set(self.tags.get(p.name, [])) & active)
            ]
        else:
            self.filtered_paths = [
                p for p in self.image_paths
                if set(self.tags.get(p.name, [])) & active
            ]
        self._force_refresh()

    def apply_tag(self) -> None:
        tag = self.tag_choice.get().strip()
        if not tag or not self.selected:
            return
        changed = False
        for p in self.selected:
            tags = self.tags.setdefault(p.name, [])
            if tag not in tags:
                tags.append(tag)
                changed = True
            lbl = self.tag_labels.get(p)
            if lbl is not None:
                lbl.config(text=", ".join(tags))
        if changed:
            self._dirty = True
        # If the applied tag is part of the active filter, re-evaluate visibility.
        active_filter = {t for t, var in self.filter_vars.items() if var.get()}
        if tag in active_filter:
            self._apply_filter()

    def remove_tag(self) -> None:
        tag = self.tag_choice.get().strip()
        if not tag or not self.selected:
            return
        changed = False
        for p in self.selected:
            tags = self.tags.get(p.name)
            if tags and tag in tags:
                tags.remove(tag)
                changed = True
            lbl = self.tag_labels.get(p)
            if lbl is not None:
                lbl.config(text=", ".join(self.tags.get(p.name, [])))
        if changed:
            self._dirty = True
        active_filter = {t for t, var in self.filter_vars.items() if var.get()}
        if tag in active_filter:
            self._apply_filter()

    def save_json(self) -> bool:
        if not self.folder:
            messagebox.showinfo("tagimg", "Open a folder first.")
            return False
        path = filedialog.asksaveasfilename(
            title="Save JSON",
            defaultextension=".json",
            filetypes=[("JSON", "*.json")],
            initialfile="tags.json",
        )
        if not path:
            return False
        data = {
            "folder": str(self.folder),
            "available_tags": list(self.available_tags),
            "tags": {p.name: self.tags.get(p.name, []) for p in self.image_paths},
        }
        Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False))
        self._dirty = False
        messagebox.showinfo("tagimg", f"Saved to {path}")
        return True

    def _on_close(self) -> None:
        if not self._dirty:
            self.root.destroy()
            return
        ans = messagebox.askyesnocancel(
            "tagimg", "There are unsaved changes. Save before closing?"
        )
        if ans is None:
            return
        if ans and not self.save_json():
            return
        self.root.destroy()


def main() -> None:
    root = tk.Tk()
    TagImg(root)
    root.mainloop()


if __name__ == "__main__":
    main()
