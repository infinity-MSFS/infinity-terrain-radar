//! `itr-debug` — interactive preview for an `.itr` terrain database.
//!
//! Loads a database, renders it with the same headless pipeline the manager uses,
//! and exposes the classic controls: Lat / Lon / Range / Heading / Altitude.

use eframe::egui;
use infinity_terrain_radar::{Dataset, FrameOptions, LatLon, Renderer};

fn main() -> eframe::Result<()> {
	let options = eframe::NativeOptions {
		viewport: egui::ViewportBuilder::default().with_inner_size([1100.0, 820.0]),
		..Default::default()
	};
	eframe::run_native(
		"itr-debug",
		options,
		Box::new(|_cc| Ok(Box::<App>::default())),
	)
}

struct App {
	path: String,
	renderer: Option<Renderer>,
	status: String,

	lat: f32,
	lon: f32,
	range_nm: f32,
	heading: f32,
	altitude_ft: f32,
	hillshade: bool,

	texture: Option<egui::TextureHandle>,
	last_key: Option<FrameKey>,
	view_size: u32,
}

/// Snapshot used to decide when a re-render is needed.
#[derive(PartialEq, Clone, Copy)]
struct FrameKey {
	lat: u32,
	lon: u32,
	range: u32,
	heading: u32,
	alt: u32,
	size: u32,
	hillshade: bool,
}

impl Default for App {
	fn default() -> Self {
		Self {
			path: String::new(),
			renderer: None,
			status: "Open an .itr file to begin.".into(),
			lat: 36.1,
			lon: -112.1, // Grand Canyon — nice relief for a first look
			range_nm: 40.0,
			heading: 0.0,
			altitude_ft: 10_000.0,
			hillshade: true,
			texture: None,
			last_key: None,
			view_size: 768,
		}
	}
}

impl App {
	fn load(&mut self, path: std::path::PathBuf) {
		match Dataset::open(&path).map_err(|e| e.to_string()).and_then(|ds| {
			let tiles = ds.tile_count();
			let res = ds.resolution();
			Renderer::new(ds).map(|r| (r, tiles, res)).map_err(|e| e.to_string())
		}) {
			Ok((r, tiles, res)) => {
				self.path = path.display().to_string();
				self.renderer = Some(r);
				self.status = format!("Loaded {tiles} tiles @ {res}²");
				self.last_key = None;
			},
			Err(e) => {
				self.renderer = None;
				self.status = format!("Failed to load: {e}");
			},
		}
	}

	fn key(&self) -> FrameKey {
		// Quantise floats so sub-pixel jitter doesn't force re-renders.
		FrameKey {
			lat: (self.lat * 1e4) as i32 as u32,
			lon: (self.lon * 1e4) as i32 as u32,
			range: (self.range_nm * 10.0) as u32,
			heading: (self.heading * 10.0) as u32,
			alt: self.altitude_ft as u32,
			size: self.view_size,
			hillshade: self.hillshade,
		}
	}

	fn maybe_render(&mut self, ctx: &egui::Context) {
		let key = self.key();
		if self.renderer.is_none() || self.last_key == Some(key) {
			return;
		}
		let opt = FrameOptions {
			width: self.view_size,
			height: self.view_size,
			center: LatLon { lat: self.lat, lon: self.lon },
			range_nm: self.range_nm,
			heading: self.heading,
			altitude_ft: self.altitude_ft,
			hillshade: self.hillshade,
		};
		match self.renderer.as_mut().unwrap().render(&opt) {
			Ok(img) => {
				let color = egui::ColorImage::from_rgba_unmultiplied(
					[img.width() as usize, img.height() as usize],
					img.as_raw(),
				);
				self.texture = Some(ctx.load_texture("terrain", color, egui::TextureOptions::LINEAR));
				self.last_key = Some(key);
			},
			Err(e) => self.status = format!("Render error: {e}"),
		}
	}
}

impl eframe::App for App {
	fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
		egui::SidePanel::left("controls").min_width(240.0).show(ctx, |ui| {
			ui.heading("Terrain Radar");
			ui.separator();

			ui.horizontal(|ui| {
				if ui.button("Open .itr…").clicked() {
					if let Some(p) = rfd::FileDialog::new().add_filter("itr", &["itr"]).pick_file() {
						self.load(p);
					}
				}
			});
			ui.label(&self.status);
			ui.separator();

			egui::Grid::new("ctrls").num_columns(2).spacing([8.0, 6.0]).show(ui, |ui| {
				ui.label("Lat");
				ui.add(egui::DragValue::new(&mut self.lat).speed(0.05).range(-90.0..=90.0));
				ui.end_row();

				ui.label("Lon");
				ui.add(egui::DragValue::new(&mut self.lon).speed(0.05).range(-180.0..=180.0));
				ui.end_row();

				ui.label("Range (nm)");
				ui.add(egui::DragValue::new(&mut self.range_nm).speed(1.0).range(1.0..=320.0));
				ui.end_row();

				ui.label("Heading");
				ui.add(egui::DragValue::new(&mut self.heading).speed(1.0).range(0.0..=360.0));
				ui.end_row();

				ui.label("Altitude (ft)");
				ui.add(egui::DragValue::new(&mut self.altitude_ft).speed(100.0).range(0.0..=60_000.0));
				ui.end_row();

				ui.label("Hillshade");
				ui.checkbox(&mut self.hillshade, "");
				ui.end_row();
			});

			ui.separator();
			ui.label("Color = terrain elevation vs aircraft altitude:");
			ui.colored_label(egui::Color32::from_rgb(242, 0, 0), "red  > alt + 2000 ft");
			ui.colored_label(egui::Color32::from_rgb(194, 135, 25), "orange  > alt + 1000 ft");
			ui.colored_label(egui::Color32::from_rgb(245, 250, 2), "yellow  near alt");
			ui.colored_label(egui::Color32::from_rgb(48, 163, 76), "green  below");
		});

		self.maybe_render(ctx);

		egui::CentralPanel::default().show(ctx, |ui| {
			if let Some(tex) = &self.texture {
				ui.centered_and_justified(|ui| {
					ui.add(egui::Image::new(tex).max_size(egui::vec2(900.0, 900.0)));
				});
			} else {
				ui.centered_and_justified(|ui| ui.label("No terrain loaded."));
			}
		});
	}
}
