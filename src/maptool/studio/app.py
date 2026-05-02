"""Biome Studio — live editor for procedural top-down biome textures.

Carthage:Bellum-Punicum-style maps are hand-painted top-down — solid biome
masses with a hand-paint grain. This studio lets you edit the *parameters*
of each biome's procedural texture (two RGB colors + noise + brightness +
contrast), see the result instantly, and persist the parameters so the
regular ``maptool build`` pipeline picks them up.

No fal.ai, no API calls — purely numpy + scipy noise.
"""

from __future__ import annotations

import logging
from pathlib import Path

import gradio as gr
from PIL import Image

from maptool.assets.biomes import CATALOG_STANDARD
from maptool.assets.generator import asset_paths
from maptool.assets.procedural import BiomeParams, render_biome
from maptool.studio.preview import render_sample
from maptool.studio.state import StudioState, load_state

log = logging.getLogger(__name__)

TILE_PREVIEW_DIR_NAME = "studio_tiled"


def _tile_preview_path(state: StudioState, slug: str) -> Path:
    d = state.cache_root / TILE_PREVIEW_DIR_NAME
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{slug}.png"


def _hex_to_rgb(value: str) -> tuple[int, int, int]:
    s = value.lstrip("#")
    if len(s) != 6:
        raise ValueError(f"expected #RRGGBB, got {value!r}")
    return int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16)


def _rgb_to_hex(rgb: tuple[int, int, int]) -> str:
    return "#{:02x}{:02x}{:02x}".format(*rgb)


def _render_and_persist(
    state: StudioState,
    slug: str,
    params: BiomeParams,
) -> tuple[Path, Path]:
    """Render the biome texture, save it as the canonical biome cache, plus
    refresh the tiled preview file. Returns (texture_path, tile_preview_path)."""
    biome = next(b for b in CATALOG_STANDARD if b.slug == slug)
    asset = asset_paths(state.cache_root, biome)

    img = render_biome(params, size=1024, seed=state.seeds[slug])
    asset.image_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(asset.image_path, optimize=True)

    # Tiled 3x3 preview at 768 px
    cell = 256
    canvas = Image.new("RGB", (cell * 3, cell * 3))
    tile = img.resize((cell, cell), Image.Resampling.LANCZOS)
    for ty in range(3):
        for tx in range(3):
            canvas.paste(tile, (tx * cell, ty * cell))
    tile_path = _tile_preview_path(state, slug)
    canvas.save(tile_path, optimize=True)

    state.params[slug] = params
    state.save()
    return asset.image_path, tile_path


def _params_from_inputs(
    slug: str,
    base_hex: str,
    accent_hex: str,
    noise_scale: float,
    grain_scale: float,
    grain_weight: float,
    brightness: float,
    contrast: float,
) -> BiomeParams:
    return BiomeParams(
        slug=slug,
        base_color=_hex_to_rgb(base_hex),
        accent_color=_hex_to_rgb(accent_hex),
        noise_scale=float(noise_scale),
        grain_scale=float(grain_scale),
        grain_weight=float(grain_weight),
        brightness=float(brightness),
        contrast=float(contrast),
    )


def build_ui(cache_root: Path) -> gr.Blocks:
    state = load_state(cache_root)

    # Make sure every biome has an initial preview so the page isn't blank.
    for biome in CATALOG_STANDARD:
        try:
            _render_and_persist(state, biome.slug, state.params[biome.slug])
        except Exception as e:  # noqa: BLE001
            log.warning("initial render failed for %s: %s", biome.slug, e)

    # Compose an initial sample map so the page opens with real preview data.
    sample_path: Path | None = None
    try:
        sample_path = render_sample(cache_root)
    except Exception as e:  # noqa: BLE001
        log.warning("initial sample render failed: %s", e)

    with gr.Blocks(title="Maptool — Biome Studio") as app:
        gr.Markdown(
            "# Maptool — Biome Studio\n"
            "Tune biome textures live. Each slider release re-runs the full "
            "biome compositor on a synthetic 1024×512 sample (central peak, "
            "forest ring, grass band, coast strip, sea) so every biome is "
            "exercised and you see the map effect immediately. The same "
            "compositor runs on real regions when you `maptool build`."
        )

        with gr.Row():
            with gr.Column(scale=2):
                map_view = gr.Image(
                    value=str(sample_path) if sample_path else None,
                    label="Live map preview (synthetic sample)",
                    type="filepath",
                    interactive=False,
                    height=360,
                )
            with gr.Column(scale=1):
                refresh_btn = gr.Button("Refresh map", variant="primary")
                preview_status = gr.Markdown(
                    "*synthetic sample — every biome rendered in one frame*"
                )

        def _refresh_map() -> tuple[str | None, str]:
            try:
                p = render_sample(cache_root)
                return str(p), "*refreshed*"
            except Exception as e:  # noqa: BLE001
                log.exception("sample refresh failed")
                return None, f"*FAILED: {e!s}*"

        refresh_btn.click(fn=_refresh_map, inputs=[], outputs=[map_view, preview_status])

        for biome in CATALOG_STANDARD:
            slug = biome.slug
            params = state.params[slug]
            asset = asset_paths(cache_root, biome)

            with gr.Group():
                gr.Markdown(f"### {slug} — *{biome.description}*")
                with gr.Row():
                    with gr.Column(scale=2):
                        with gr.Row():
                            base_in = gr.ColorPicker(
                                value=_rgb_to_hex(params.base_color),
                                label="Base colour",
                            )
                            accent_in = gr.ColorPicker(
                                value=_rgb_to_hex(params.accent_color),
                                label="Accent colour",
                            )
                            seed_in = gr.Number(
                                value=state.seeds[slug],
                                label="Seed",
                                precision=0,
                            )
                        with gr.Row():
                            noise_in = gr.Slider(2.0, 60.0, value=params.noise_scale,
                                                 step=1.0, label="Noise scale")
                            grain_in = gr.Slider(0.5, 6.0, value=params.grain_scale,
                                                 step=0.1, label="Grain scale")
                            grainw_in = gr.Slider(0.0, 1.0, value=params.grain_weight,
                                                  step=0.02, label="Grain weight")
                        with gr.Row():
                            brt_in = gr.Slider(0.4, 1.6, value=params.brightness,
                                               step=0.02, label="Brightness")
                            con_in = gr.Slider(0.4, 2.0, value=params.contrast,
                                               step=0.02, label="Contrast")
                        save_btn = gr.Button("Save params", variant="primary")
                        status = gr.Markdown("")
                    with gr.Column(scale=1):
                        single_view = gr.Image(
                            value=str(asset.image_path) if asset.image_path.exists() else None,
                            label="Single tile (1024²)",
                            type="filepath",
                            interactive=False,
                            height=240,
                        )
                        tile_view = gr.Image(
                            value=str(_tile_preview_path(state, slug)),
                            label="Tiled 3×3 preview",
                            type="filepath",
                            interactive=False,
                            height=240,
                        )

                # Per-biome handlers, capturing slug.
                def _make_live(s: str):
                    def _go(base_hex, accent_hex, seed, noise_scale, grain_scale,
                            grain_weight, brightness, contrast):
                        try:
                            state.seeds[s] = int(seed)
                            params = _params_from_inputs(
                                s, base_hex, accent_hex, noise_scale,
                                grain_scale, grain_weight, brightness, contrast,
                            )
                            tex_path, tile_path = _render_and_persist(state, s, params)
                            map_path = render_sample(cache_root)
                            return (
                                str(tex_path),
                                str(tile_path),
                                "",
                                str(map_path) if map_path else gr.update(),
                            )
                        except Exception as e:  # noqa: BLE001
                            log.exception("live render failed for %s", s)
                            return gr.update(), gr.update(), f"*FAILED: {e!s}*", gr.update()
                    return _go

                live_inputs = [base_in, accent_in, seed_in,
                               noise_in, grain_in, grainw_in,
                               brt_in, con_in]
                live_outputs = [single_view, tile_view, status, map_view]
                live_handler = _make_live(slug)
                for ctrl in (noise_in, grain_in, grainw_in, brt_in, con_in):
                    ctrl.release(fn=live_handler, inputs=live_inputs, outputs=live_outputs)
                base_in.change(fn=live_handler, inputs=live_inputs, outputs=live_outputs)
                accent_in.change(fn=live_handler, inputs=live_inputs, outputs=live_outputs)
                seed_in.change(fn=live_handler, inputs=live_inputs, outputs=live_outputs)

                def _make_save(s: str):
                    def _go(base_hex, accent_hex, seed, noise_scale, grain_scale,
                            grain_weight, brightness, contrast):
                        params = _params_from_inputs(
                            s, base_hex, accent_hex, noise_scale,
                            grain_scale, grain_weight, brightness, contrast,
                        )
                        state.seeds[s] = int(seed)
                        state.params[s] = params
                        state.save()
                        return f"*saved → {state.params_path.name}*"
                    return _go

                save_btn.click(
                    fn=_make_save(slug),
                    inputs=live_inputs,
                    outputs=status,
                )

    return app


def launch(
    cache_root: Path,
    *,
    host: str = "127.0.0.1",
    port: int = 7860,
    share: bool = False,
) -> None:
    app = build_ui(cache_root)
    app.launch(
        server_name=host,
        server_port=port,
        share=share,
        inbrowser=True,
        theme=gr.themes.Soft(),
    )
