"""HTML gallery with a zoom/pan lightbox for the generated maps."""

from __future__ import annotations

import datetime as dt
import html
import logging
from pathlib import Path

from PIL import Image

log = logging.getLogger(__name__)


def _humanize_size(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} GB"


def _dimensions(p: Path) -> str:
    try:
        with Image.open(p) as img:
            return f"{img.width} × {img.height}"
    except Exception:
        return "?"


def _collect_heightmaps(out_dir: Path) -> list[dict]:
    previews = out_dir / "previews"
    rows = []
    for hm in sorted(out_dir.glob("*_height.png")):
        stem = hm.stem.replace("_height", "")
        color = out_dir / f"{stem}_color.png"
        forest = out_dir / f"{stem}_forest.png"
        annotated = out_dir / f"{stem}_annotated.png"
        thumb_candidates = [
            previews / f"{stem}_annotated_thumb.png",
            previews / f"{stem}_forest_thumb.png",
            previews / f"{stem}_thumb.png",
        ]
        thumb = next((t for t in thumb_candidates if t.exists()), None)
        main = annotated if annotated.exists() else forest if forest.exists() else color if color.exists() else hm
        rows.append({
            "name": stem,
            "main_rel": main.relative_to(out_dir).as_posix(),
            "thumb_rel": thumb.relative_to(out_dir).as_posix() if thumb else main.relative_to(out_dir).as_posix(),
            "height_rel": hm.relative_to(out_dir).as_posix(),
            "color_rel": color.relative_to(out_dir).as_posix() if color.exists() else None,
            "forest_rel": forest.relative_to(out_dir).as_posix() if forest.exists() else None,
            "annotated_rel": annotated.relative_to(out_dir).as_posix() if annotated.exists() else None,
            "dimensions": _dimensions(hm),
            "filesize": _humanize_size(hm.stat().st_size),
        })
    return rows


def _collect_landcover(out_dir: Path) -> list[dict]:
    previews = out_dir / "previews"
    rows = []
    for lc in sorted(out_dir.glob("*_landcover.png")):
        stem = lc.stem.replace("_landcover", "")
        thumb = previews / f"{stem}_landcover_thumb.png"
        rows.append({
            "name": stem,
            "main_rel": lc.relative_to(out_dir).as_posix(),
            "thumb_rel": (thumb if thumb.exists() else lc).relative_to(out_dir).as_posix(),
            "dimensions": _dimensions(lc),
            "filesize": _humanize_size(lc.stat().st_size),
        })
    return rows


_STYLE = """
body { font-family: -apple-system, "Segoe UI", sans-serif;
       background: #1a1a1a; color: #ddd; margin: 0; padding: 32px; }
h1 { margin: 0 0 4px 0; font-size: 28px; }
.subtitle { color: #888; margin-bottom: 32px; font-size: 13px; }
h2 { border-bottom: 1px solid #333; padding-bottom: 8px;
     margin-top: 40px; color: #eee; font-weight: 500; }
.grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
        gap: 16px; }
.card { background: #262626; border-radius: 8px; overflow: hidden;
        border: 1px solid #333; transition: border-color .15s; }
.card:hover { border-color: #4a4a4a; }
.card img { width: 100%; display: block; background: #0a0a0a;
            image-rendering: auto; cursor: zoom-in; }
.card-body { padding: 12px 16px; }
.card-title { font-weight: 600; margin: 0 0 4px 0; color: #fff; }
.meta { font-size: 12px; color: #888; margin: 2px 0; }
.links { margin-top: 8px; display: flex; gap: 12px; font-size: 12px; }
.links a { color: #6ab7ff; text-decoration: none; }
.links a:hover { text-decoration: underline; }
.empty { color: #666; font-style: italic; padding: 8px 0; }
.count { color: #888; font-size: 13px; font-weight: normal; margin-left: 8px; }

#lb { position: fixed; inset: 0; z-index: 1000; display: none;
      background: rgba(0,0,0,0.94); }
#lb.open { display: block; }
#lb-canvas { position: absolute; inset: 0; overflow: hidden;
             cursor: grab; user-select: none; }
#lb-canvas.dragging { cursor: grabbing; }
#lb-img { position: absolute; left: 50%; top: 50%; max-width: none;
          transform-origin: center center; pointer-events: none;
          image-rendering: auto; }
#lb-ctrl { position: absolute; top: 16px; right: 16px; z-index: 10;
           display: flex; gap: 8px; background: rgba(30,30,30,0.92);
           padding: 8px 12px; border-radius: 6px; color: #eee;
           font-size: 13px; align-items: center; }
#lb-ctrl button { background: #444; border: 1px solid #555;
                  color: #eee; padding: 4px 12px; border-radius: 4px;
                  cursor: pointer; font-size: 13px; }
#lb-ctrl button:hover { background: #555; }
#lb-zoom { min-width: 48px; text-align: center; color: #aaa;
           font-variant-numeric: tabular-nums; }
#lb-help { position: absolute; bottom: 16px; left: 16px; color: #888;
           font-size: 12px; background: rgba(30,30,30,0.8);
           padding: 6px 10px; border-radius: 4px; }
#lb-title { position: absolute; top: 16px; left: 16px; color: #fff;
            background: rgba(30,30,30,0.92); padding: 6px 12px;
            border-radius: 6px; font-size: 14px; font-weight: 500; }
"""

_LIGHTBOX_HTML = """
<div id="lb">
  <div id="lb-canvas"><img id="lb-img" alt=""></div>
  <div id="lb-title"></div>
  <div id="lb-ctrl">
    <button id="lb-out">−</button>
    <span id="lb-zoom">50%</span>
    <button id="lb-in">+</button>
    <button id="lb-reset">0</button>
    <button id="lb-close">×</button>
  </div>
  <div id="lb-help">Scroll = Zoom · Drag = Pan · 0 = Reset · Esc = Close · min 10% · max 300%</div>
</div>
"""

_LIGHTBOX_JS = """
(function() {
  const lb = document.getElementById('lb');
  const img = document.getElementById('lb-img');
  const canvas = document.getElementById('lb-canvas');
  const zoomLabel = document.getElementById('lb-zoom');
  const titleEl = document.getElementById('lb-title');
  const MIN_Z = 0.1, MAX_Z = 3.0, INITIAL_Z = 0.5;
  let z = INITIAL_Z, px = 0, py = 0;
  let dragging = false, lx = 0, ly = 0;

  function apply() {
    img.style.transform =
      'translate(calc(-50% + ' + px + 'px), calc(-50% + ' + py + 'px)) scale(' + z + ')';
    zoomLabel.textContent = Math.round(z * 100) + '%';
  }
  function reset() { z = INITIAL_Z; px = 0; py = 0; apply(); }
  function open(src, title) {
    img.src = src;
    titleEl.textContent = title || '';
    reset();
    lb.classList.add('open');
  }
  function close() { lb.classList.remove('open'); }

  canvas.addEventListener('wheel', function(e) {
    e.preventDefault();
    const factor = e.deltaY < 0 ? 1.15 : 1 / 1.15;
    const newZ = Math.max(MIN_Z, Math.min(MAX_Z, z * factor));
    const rect = canvas.getBoundingClientRect();
    const cx = e.clientX - rect.left - rect.width / 2;
    const cy = e.clientY - rect.top - rect.height / 2;
    px = cx + (px - cx) * (newZ / z);
    py = cy + (py - cy) * (newZ / z);
    z = newZ;
    apply();
  }, { passive: false });

  canvas.addEventListener('mousedown', function(e) {
    if (e.button !== 0) return;
    dragging = true; lx = e.clientX; ly = e.clientY;
    canvas.classList.add('dragging');
  });
  window.addEventListener('mousemove', function(e) {
    if (!dragging) return;
    px += e.clientX - lx; py += e.clientY - ly;
    lx = e.clientX; ly = e.clientY;
    apply();
  });
  window.addEventListener('mouseup', function() {
    dragging = false; canvas.classList.remove('dragging');
  });

  window.addEventListener('keydown', function(e) {
    if (!lb.classList.contains('open')) return;
    if (e.key === 'Escape') close();
    if (e.key === '0') reset();
    if (e.key === '+' || e.key === '=') {
      z = Math.min(MAX_Z, z * 1.2); apply();
    }
    if (e.key === '-') { z = Math.max(MIN_Z, z / 1.2); apply(); }
  });

  document.getElementById('lb-in').addEventListener('click', function() {
    z = Math.min(MAX_Z, z * 1.2); apply();
  });
  document.getElementById('lb-out').addEventListener('click', function() {
    z = Math.max(MIN_Z, z / 1.2); apply();
  });
  document.getElementById('lb-reset').addEventListener('click', reset);
  document.getElementById('lb-close').addEventListener('click', close);

  document.querySelectorAll('.card a[data-fullsrc]').forEach(function(a) {
    a.addEventListener('click', function(e) {
      e.preventDefault();
      open(a.getAttribute('data-fullsrc'), a.getAttribute('data-title') || '');
    });
  });
})();
"""


def _render_heightmap_card(e: dict) -> str:
    name = html.escape(e["name"])
    dims = html.escape(e["dimensions"])
    size = html.escape(e["filesize"])
    thumb = html.escape(e["thumb_rel"])
    main = html.escape(e["main_rel"])

    links = []
    if e.get("annotated_rel"):
        links.append(f'<a href="{html.escape(e["annotated_rel"])}" target="_blank">Annotated</a>')
    if e.get("forest_rel"):
        links.append(f'<a href="{html.escape(e["forest_rel"])}" target="_blank">Forest</a>')
    if e.get("color_rel"):
        links.append(f'<a href="{html.escape(e["color_rel"])}" target="_blank">Plain Color</a>')
    links.append(f'<a href="{html.escape(e["height_rel"])}" target="_blank">16-bit</a>')

    return (
        f'<div class="card">'
        f'<a href="{main}" data-fullsrc="{main}" data-title="{name} (heightmap)">'
        f'<img src="{thumb}" alt="{name}"></a>'
        f'<div class="card-body">'
        f'<div class="card-title">{name}</div>'
        f'<div class="meta">{dims}</div>'
        f'<div class="meta">{size}</div>'
        f'<div class="links">{" ".join(links)}</div>'
        f'</div></div>'
    )


def _render_simple_card(e: dict) -> str:
    name = html.escape(e["name"])
    dims = html.escape(e["dimensions"])
    size = html.escape(e["filesize"])
    main = html.escape(e["main_rel"])
    thumb = html.escape(e["thumb_rel"])
    return (
        f'<div class="card">'
        f'<a href="{main}" data-fullsrc="{main}" data-title="{name}">'
        f'<img src="{thumb}" alt="{name}"></a>'
        f'<div class="card-body">'
        f'<div class="card-title">{name}</div>'
        f'<div class="meta">{dims}</div>'
        f'<div class="meta">{size}</div>'
        f'</div></div>'
    )


def _render_section(title: str, rows: list[dict], renderer) -> str:
    count = f'<span class="count">({len(rows)})</span>' if rows else ""
    out = [f"<h2>{html.escape(title)}{count}</h2>"]
    if not rows:
        out.append('<div class="empty">No assets in this category yet.</div>')
        return "\n".join(out)
    out.append('<div class="grid">')
    out.extend(renderer(e) for e in rows)
    out.append("</div>")
    return "\n".join(out)


def build_html(out_dir: Path) -> str:
    now = dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    landcover = _collect_landcover(out_dir)
    heightmaps = _collect_heightmaps(out_dir)
    sections = [
        _render_section("Ground (ESA WorldCover 10m)", landcover, _render_simple_card),
        _render_section("Heightmaps", heightmaps, _render_heightmap_card),
    ]
    return (
        "<!DOCTYPE html>\n"
        '<html lang="en"><head><meta charset="utf-8">'
        "<title>maptool — Map Gallery</title>"
        f"<style>{_STYLE}</style></head><body>"
        "<h1>maptool — Map Gallery</h1>"
        f'<div class="subtitle">Generated: {now} — click an image for the lightbox</div>'
        + "\n".join(sections)
        + _LIGHTBOX_HTML
        + f"<script>{_LIGHTBOX_JS}</script>"
        + "</body></html>"
    )


def write_gallery(out_dir: Path) -> Path:
    path = out_dir / "gallery.html"
    out_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(build_html(out_dir), encoding="utf-8")
    log.info("Gallery -> %s", path)
    return path
