# maptool

Top-down painted terrain maps in the **Carthage / Total War Attila** campaign-map style. Driven by public DEM + landcover, painted with editable biome textures, with tree-sprite forests and a live Gradio studio.

```
SRTM heightmap (OpenTopography)
  + ESA WorldCover 10m landcover
  -> biome compositor (8 biomes: sea/coast/arid/grass/cliffs/forest/alpine_rock/snow)
  -> tree sprite scatter (lat-driven species, density falloff with elevation, drop shadows)
  -> bushes on shrubland
  -> hand-paint global noise overlay
  -> hillshade modulation
  -> HTML gallery
```

## Quick start (fresh clone)

```bash
git clone https://github.com/<user>/maptool.git
cd maptool
python -m venv .venv
.venv\Scripts\activate     # Windows
# source .venv/bin/activate    # Unix
pip install -e ".[dev]"
cp .env.example .env
# edit .env: paste OPENTOPO_API_KEY (https://portal.opentopography.org/myopentopo)
# optional: paste FAL_KEY for fal.ai-generated biome textures
```

The repo ships **default biome textures** (`.cache/biomes/`), **tree sprites** (`.cache/sprites/trees/`) and **biome parameters** (`.cache/biome_params.json`) so the studio works out of the box without any API keys.

## Use

```bash
# launch the live editor
maptool studio

# build a map for a region (downloads DEM once, caches it)
maptool build --region sicily --quality high --paint biome

# render-only (re-paint without re-downloading)
maptool render --region sicily --quality ultra

# every built-in region in one go
maptool build --all --quality high

# any bbox worldwide (south,north,west,east in decimal degrees)
maptool build --bbox 36,42,-6,3 --name iberia-south

# rebuild the gallery HTML and open it in a browser
maptool gallery --open

# tile pyramid render (overview + native-density tiles)
maptool tile --region sicily

# import existing aquila DEM tifs to skip downloads
maptool cache-import --from C:/Users/Till/Projects/aquila/assets/terrain/raw

# generate biome textures via fal.ai (optional, needs FAL_KEY)
maptool assets-generate

# list built-in regions
maptool list-regions
```

## Studio

`maptool studio` launches a Gradio web UI at <http://127.0.0.1:7860>.

Top: live preview of a synthetic test sample (central peak with snow + cliffs, forest ring, grass plains, scattered shrub patches, sea around) re-rendered through the full biome compositor + tree-sprite scatter on every slider release.

Per biome: base + accent colour pickers, noise-scale, grain-scale, grain-weight, brightness, contrast, seed. **Save params** writes them to `.cache/biome_params.json` so subsequent `maptool build` runs use the new look automatically.

## Output

```
out/
  <region>_height.png       # 16-bit heightmap, engine input
  <region>_landcover.png    # ESA WorldCover RGB
  <region>_landcover_codes.png  # categorical class IDs
  <region>_color.png        # painted base
  <region>_forest.png       # painted base + tree sprites + shadows
  <region>_meta.json
  previews/                 # 256 px thumbnails for the gallery
  gallery.html
```

## Architecture

```
src/maptool/
  cli.py              — argparse entry, every subcommand
  pipeline.py         — orchestration, Layout dataclass, render() / build_all()
  regions.py          — built-in BBoxes, tile subdivision
  sources/
    srtm.py           — OpenTopography SRTM/COP DEM downloader (windowed read)
    worldcover.py     — ESA WorldCover S3 COG mosaic
    osm.py            — (planned) OSM Overpass for rivers / roads / settlements
  assets/
    biomes.py         — biome catalog (8 slugs + descriptions)
    procedural.py     — per-biome procedural texture renderer
    fal_client.py     — optional fal.ai integration
    generator.py      — biome-texture generator + cache
  render/
    biome_compositor.py — per-pixel biome classification + texture sampling + hillshade
    climate_paint.py  — alternative procedural climate-paint mode
    forest.py         — older procedural canopy scatter (used by climate paint)
    tree_sprites.py   — image-sprite scatter for biome paint mode (lat-driven species)
    paint_noise.py    — multi-tier hand-paint noise
    palette.py        — climate elevation palettes
    annotate.py       — vector overlay (rivers etc.)
  studio/
    app.py            — Gradio biome studio
    state.py          — persistent biome params
    image_ops.py      — colour adjustments + tile preview
    preview.py        — synthetic-sample compositor for live preview
  gallery.py          — HTML gallery builder with lightbox
```

## Quality presets

| Preset | max-dim | DEM | use case |
|---|---|---|---|
| `standard` | 4096 | SRTMGL3 (90 m) | fast iteration, reuses aquila cache |
| `high` (default rec.) | 10240 | SRTMGL1 (30 m) | nice mid-zoom map |
| `ultra` | 16384 | SRTMGL1 (30 m) | ~130 MP, ~5-10 min/region |

## What's working today

- 8 biomes with editable per-biome textures (sea / coast / arid / grass / cliffs / forest / alpine_rock / snow)
- Latitude-driven single tree species per region (olive / oak / pine / dense_cluster bands)
- Tree sprite scatter with NW-sun drop shadows + density falloff with elevation
- Bush layer on shrubland
- Cliff biome between snow and forest
- Coastline sand band + sea/water mask from WorldCover
- Hillshade with optional pre-blur for low-res DEMs
- Stripe-processed compositor (memory-bounded at ultra resolution)
- Tile pyramid renderer for native-density per-tile output
- Live Gradio studio with synthetic test sample
- Multi-octave global hand-paint noise overlay

## Roadmap (planned)

See `docs/PLAN.md` (when committed). Highlights: rivers + roads + settlements via OSM, field-grid on cropland, foam at coast, 2-3 species per band, region picker in studio, A/B compare, preset save/load.

## License

MIT
