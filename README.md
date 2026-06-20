# infinity-terrain-radar

A modern, pure-Rust (except wgpu) terrain-radar stack. It packs world SRTM elevation into a single compact `.itr`
file and renders heading-up, altitude-referenced ground-map images with a headless
wgpu pipeline - ready to PNG-encode and push into the sim.

## Workspace

| Crate | Kind | Role |
|-------|------|------|
| `itr-format` | lib | The `.itr` container: tile codec + mmap reader + streaming writer. Shared contract. |
| `infinity-terrain-radar` | **rlib** | `.itr` reader + headless wgpu renderer → `image::RgbaImage` / PNG. **This is what the manager imports.** |
| `itr-build` | bin | Pack an SRTM tree into one `.itr`; inspect existing files. |
| `itr-debug` | bin | egui live preview with Lat / Lon / Range / Heading / Altitude controls. |

## The `.itr` format

A single file: a `360×180` sparse index of 1°×1° tiles. Each present tile is a
zstd frame of `R×R` u16 heights (north row first). Heights only — no water mask, no
stored hillshade. Water reads back as ordinary terrain, matching how the T-38 paints
the ground return of its weather radar. Full layout is documented in
`crates/itr-format/src/lib.rs`.

* `resolution` (R) is the per-tile sample count and the **main size lever**.
* Heights are quantised: `elevation_m = stored * height_resolution − 500`.
* Downsampling uses **max-pooling** — a terrain radar wants the worst-case (highest)
  obstacle in each cell, not the average.

### Size vs quality (measured)

At ~31 KB/tile (res 300, zstd 19) and 14,546 land tiles:

| Resolution | ≈ ground sample | Full-world `.itr` |
|-----------|-----------------|-------------------|
| 300² | ~360 m | ~450 MB |
| 600² | ~180 m | ~1.8 GB |

Pick `--resolution` to hit your size target (must divide 1200: 600, 400, 300, 240, 200, 150, 120, 100, 75, 60).

## Usage

Build a database from the SRTM tree (taclink's downloader layout — `<srtm>/<Region>/<TILE>.hgt.zip`):

```sh
cargo run --release -p itr-build -- build \
  --srtm C:\Users\Cameron\Documents\github\taclink\srtm \
  --out world.itr --resolution 300

cargo run --release -p itr-build -- info world.itr --coverage coverage.png
```

Preview interactively:

```sh
cargo run --release -p itr-debug
```

Render headlessly (the path the manager uses):

```rust
use infinity_terrain_radar::{Dataset, Renderer, FrameOptions, LatLon};

let ds = Dataset::open("world.itr".as_ref())?;
let mut r = Renderer::new(ds)?;
let png = r.render_png(&FrameOptions {
    width: 2048, height: 2048,
    center: LatLon { lat: 19.6, lon: -155.5 },
    range_nm: 40.0, heading: 0.0, altitude_ft: 5000.0,
})?;
```
