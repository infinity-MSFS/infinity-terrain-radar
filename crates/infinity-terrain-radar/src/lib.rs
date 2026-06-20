//! Infinity Terrain Radar: read a packed `.itr` terrain database and render a
//! heading-up, altitude-referenced ground-map image with a headless wgpu pipeline.
//!
//! The renderer owns its own wgpu device, so it runs anywhere (a Tauri backend
//! task, a CLI, the debug app) without sharing a surface or device. Output is a
//! plain [`image::RgbaImage`] — ready to PNG-encode and ship into the sim.

mod render;

pub use itr_format::{Dataset, LoadError};
pub use render::{FrameOptions, LatLon, Renderer, RendererError};
