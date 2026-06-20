use std::collections::HashMap;

use bytemuck::{Pod, Zeroable};
use image::RgbaImage;
use itr_format::Dataset;

const M_PER_DEG_LAT: f32 = 111_320.0;
const NM_TO_M: f32 = 1852.0;
const MAX_TEX_DIM: u32 = 8192;

/// A polar coordinate in degrees.
#[derive(Copy, Clone, Debug)]
pub struct LatLon {
	pub lat: f32,
	pub lon: f32,
}

/// Everything needed to render one frame.
#[derive(Copy, Clone, Debug)]
pub struct FrameOptions {
	/// Output width in pixels.
	pub width: u32,
	/// Output height in pixels.
	pub height: u32,
	/// Aircraft position (rendered at the center of the frame).
	pub center: LatLon,
	/// Half the vertical extent of the frame, in nautical miles (center to top edge).
	pub range_nm: f32,
	/// Aircraft heading in degrees (screen-up points along this heading).
	pub heading: f32,
	/// Aircraft altitude in feet (the color ramp is referenced to this).
	pub altitude_ft: f32,
}

impl Default for FrameOptions {
	fn default() -> Self {
		Self {
			width: 1024,
			height: 1024,
			center: LatLon { lat: 0.0, lon: 0.0 },
			range_nm: 40.0,
			heading: 0.0,
			altitude_ft: 10_000.0,
		}
	}
}

#[derive(Debug)]
pub enum RendererError {
	NoAdapter,
	Device(wgpu::RequestDeviceError),
	Tile(std::io::Error),
}

impl std::fmt::Display for RendererError {
	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		match self {
			RendererError::NoAdapter => write!(f, "no suitable wgpu adapter"),
			RendererError::Device(e) => write!(f, "wgpu device error: {e}"),
			RendererError::Tile(e) => write!(f, "tile decode error: {e}"),
		}
	}
}
impl std::error::Error for RendererError {}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct Uniforms {
	center_lat: f32,
	center_lon: f32,
	cos_lat: f32,
	meters_per_pixel: f32,
	width: f32,
	height: f32,
	heading: f32,
	altitude_ft: f32,
	lon_min: f32,
	lat_max: f32,
	tile_res: f32,
	height_res: f32,
}

pub struct Renderer {
	ds: Dataset,
	res: u32,
	height_res: u16,
	device: wgpu::Device,
	queue: wgpu::Queue,
	pipeline: wgpu::RenderPipeline,
	bind_group_layout: wgpu::BindGroupLayout,
	uniform_buf: wgpu::Buffer,

	tile_cache: HashMap<(i32, i32), Option<Vec<u16>>>,
	height_tex: Option<(u32, u32, wgpu::Texture)>,
	target: Option<(u32, u32, wgpu::Texture)>,
}

impl Renderer {
	pub fn new(ds: Dataset) -> Result<Self, RendererError> {
		let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
			backends: wgpu::Backends::all(),
			..Default::default()
		});
		let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
			power_preference: wgpu::PowerPreference::HighPerformance,
			compatible_surface: None,
			force_fallback_adapter: false,
		}))
		.ok_or(RendererError::NoAdapter)?;

		let (device, queue) = pollster::block_on(adapter.request_device(
			&wgpu::DeviceDescriptor {
				label: Some("itr-renderer"),
				required_features: wgpu::Features::empty(),
				required_limits: wgpu::Limits::default(),
				memory_hints: wgpu::MemoryHints::Performance,
			},
			None,
		))
		.map_err(RendererError::Device)?;

		let res = ds.resolution();
		let height_res = ds.height_resolution();

		let uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
			label: Some("itr-uniforms"),
			size: std::mem::size_of::<Uniforms>() as u64,
			usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
			mapped_at_creation: false,
		});

		let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
			label: Some("itr-bgl"),
			entries: &[
				wgpu::BindGroupLayoutEntry {
					binding: 0,
					visibility: wgpu::ShaderStages::FRAGMENT,
					ty: wgpu::BindingType::Buffer {
						ty: wgpu::BufferBindingType::Uniform,
						has_dynamic_offset: false,
						min_binding_size: None,
					},
					count: None,
				},
				wgpu::BindGroupLayoutEntry {
					binding: 1,
					visibility: wgpu::ShaderStages::FRAGMENT,
					ty: wgpu::BindingType::Texture {
						sample_type: wgpu::TextureSampleType::Uint,
						view_dimension: wgpu::TextureViewDimension::D2,
						multisampled: false,
					},
					count: None,
				},
			],
		});

		let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
			label: Some("itr-terrain"),
			source: wgpu::ShaderSource::Wgsl(include_str!("shaders/terrain.wgsl").into()),
		});

		let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
			label: Some("itr-pl"),
			bind_group_layouts: &[&bind_group_layout],
			push_constant_ranges: &[],
		});

		let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
			label: Some("itr-pipeline"),
			layout: Some(&pipeline_layout),
			vertex: wgpu::VertexState {
				module: &shader,
				entry_point: "vs",
				buffers: &[],
				compilation_options: Default::default(),
			},
			fragment: Some(wgpu::FragmentState {
				module: &shader,
				entry_point: "fs",
				targets: &[Some(wgpu::ColorTargetState {
					format: wgpu::TextureFormat::Rgba8Unorm,
					blend: None,
					write_mask: wgpu::ColorWrites::ALL,
				})],
				compilation_options: Default::default(),
			}),
			primitive: wgpu::PrimitiveState::default(),
			depth_stencil: None,
			multisample: wgpu::MultisampleState::default(),
			multiview: None,
			cache: None,
		});

		Ok(Self {
			ds,
			res,
			height_res,
			device,
			queue,
			pipeline,
			bind_group_layout,
			uniform_buf,
			tile_cache: HashMap::new(),
			height_tex: None,
			target: None,
		})
	}

	pub fn dataset(&self) -> &Dataset { &self.ds }

	/// Render one frame to an RGBA image.
	pub fn render(&mut self, opt: &FrameOptions) -> Result<RgbaImage, RendererError> {
		let res = self.res;
		let cos_lat = opt.center.lat.to_radians().cos().max(1e-3);
		let mpp = (2.0 * opt.range_nm * NM_TO_M) / opt.height.max(1) as f32;

		let bounds = self.visible_tiles(opt, cos_lat, mpp);
		let cols = (bounds.lon_max - bounds.lon_min + 1) as u32;
		let rows = (bounds.lat_max - bounds.lat_min + 1) as u32;
		let aw = cols * res;
		let ah = rows * res;

		let assembled = self.assemble(&bounds, res)?;
		self.ensure_height_tex(aw, ah);
		self.upload_heights(aw, ah, &assembled);

		let uniforms = Uniforms {
			center_lat: opt.center.lat,
			center_lon: opt.center.lon,
			cos_lat,
			meters_per_pixel: mpp,
			width: opt.width as f32,
			height: opt.height as f32,
			heading: opt.heading.to_radians(),
			altitude_ft: opt.altitude_ft,
			lon_min: bounds.lon_min as f32,
			lat_max: bounds.lat_max as f32,
			tile_res: res as f32,
			height_res: self.height_res as f32,
		};
		self.queue.write_buffer(&self.uniform_buf, 0, bytemuck::bytes_of(&uniforms));

		self.ensure_target(opt.width, opt.height);
		self.draw_and_readback(opt.width, opt.height)
	}

	/// Convenience: render and PNG-encode in one shot.
	pub fn render_png(&mut self, opt: &FrameOptions) -> Result<Vec<u8>, RendererError> {
		let img = self.render(opt)?;
		let mut buf = std::io::Cursor::new(Vec::new());
		img.write_to(&mut buf, image::ImageFormat::Png)
			.map_err(|e| RendererError::Tile(std::io::Error::new(std::io::ErrorKind::Other, e)))?;
		Ok(buf.into_inner())
	}

	fn visible_tiles(&self, opt: &FrameOptions, cos_lat: f32, mpp: f32) -> TileBounds {
		let project = |px: f32, py: f32| -> (f32, f32) {
			let east_m = (px - opt.width as f32 * 0.5) * mpp;
			let north_m = (opt.height as f32 * 0.5 - py) * mpp;
			let h = opt.heading.to_radians();
			let (s, c) = (h.sin(), h.cos());
			let we = east_m * c + north_m * s;
			let wn = -east_m * s + north_m * c;
			let lat = opt.center.lat + wn / M_PER_DEG_LAT;
			let lon = opt.center.lon + we / (M_PER_DEG_LAT * cos_lat);
			(lat, lon)
		};

		let (w, hgt) = (opt.width as f32, opt.height as f32);
		let corners = [project(0.0, 0.0), project(w, 0.0), project(0.0, hgt), project(w, hgt)];
		let (mut min_lat, mut max_lat) = (f32::MAX, f32::MIN);
		let (mut min_lon, mut max_lon) = (f32::MAX, f32::MIN);
		for (la, lo) in corners {
			min_lat = min_lat.min(la);
			max_lat = max_lat.max(la);
			min_lon = min_lon.min(lo);
			max_lon = max_lon.max(lo);
		}

		// One tile of margin for bilinear lookups at the edges.
		let lat_min = (min_lat.floor() as i32 - 1).clamp(-90, 89);
		let mut lat_max = (max_lat.floor() as i32 + 1).clamp(-90, 89);
		let lon_min = (min_lon.floor() as i32 - 1).clamp(-180, 179);
		let mut lon_max = (max_lon.floor() as i32 + 1).clamp(-180, 179);

		// Keep the assembled texture within device limits.
		let max_tiles = (MAX_TEX_DIM / self.res) as i32;
		if lat_max - lat_min + 1 > max_tiles {
			lat_max = lat_min + max_tiles - 1;
		}
		if lon_max - lon_min + 1 > max_tiles {
			lon_max = lon_min + max_tiles - 1;
		}

		TileBounds {
			lat_min,
			lat_max,
			lon_min,
			lon_max,
		}
	}

	fn assemble(&mut self, b: &TileBounds, res: u32) -> Result<Vec<u16>, RendererError> {
		let r = res as usize;
		let cols = (b.lon_max - b.lon_min + 1) as usize;
		let rows = (b.lat_max - b.lat_min + 1) as usize;
		let aw = cols * r;
		// Absent tiles (open ocean) fill with sea level so they match real coastline
		// data and never form a hard seam against present tiles.
		let sea_level = itr_format::encode_height(0, self.height_res);
		let mut out = vec![sea_level; aw * rows * r];

		for tlat in b.lat_min..=b.lat_max {
			for tlon in b.lon_min..=b.lon_max {
				let tile = self.tile(tlat, tlon)?;
				let Some(tile) = tile else { continue };
				let col_block = (tlon - b.lon_min) as usize;
				let row_block = (b.lat_max - tlat) as usize; // north-up
				for ry in 0..r {
					let dst = (row_block * r + ry) * aw + col_block * r;
					out[dst..dst + r].copy_from_slice(&tile[ry * r..ry * r + r]);
				}
			}
		}
		Ok(out)
	}

	/// Cached tile fetch. `None` = absent from the database.
	fn tile(&mut self, lat: i32, lon: i32) -> Result<Option<&Vec<u16>>, RendererError> {
		if !self.tile_cache.contains_key(&(lat, lon)) {
			let decoded = match self.ds.read_tile(lat, lon) {
				Some(Ok(t)) => Some(t),
				Some(Err(e)) => return Err(RendererError::Tile(e)),
				None => None,
			};
			self.tile_cache.insert((lat, lon), decoded);
		}
		Ok(self.tile_cache.get(&(lat, lon)).unwrap().as_ref())
	}

	fn ensure_height_tex(&mut self, w: u32, h: u32) {
		if matches!(&self.height_tex, Some((tw, th, _)) if *tw == w && *th == h) {
			return;
		}
		let tex = self.device.create_texture(&wgpu::TextureDescriptor {
			label: Some("itr-heightmap"),
			size: wgpu::Extent3d {
				width: w,
				height: h,
				depth_or_array_layers: 1,
			},
			mip_level_count: 1,
			sample_count: 1,
			dimension: wgpu::TextureDimension::D2,
			format: wgpu::TextureFormat::R16Uint,
			usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
			view_formats: &[],
		});
		self.height_tex = Some((w, h, tex));
	}

	fn upload_heights(&self, w: u32, h: u32, data: &[u16]) {
		let tex = &self.height_tex.as_ref().unwrap().2;
		self.queue.write_texture(
			wgpu::ImageCopyTexture {
				texture: tex,
				mip_level: 0,
				origin: wgpu::Origin3d::ZERO,
				aspect: wgpu::TextureAspect::All,
			},
			bytemuck::cast_slice(data),
			wgpu::ImageDataLayout {
				offset: 0,
				bytes_per_row: Some(w * 2),
				rows_per_image: Some(h),
			},
			wgpu::Extent3d {
				width: w,
				height: h,
				depth_or_array_layers: 1,
			},
		);
	}

	fn ensure_target(&mut self, w: u32, h: u32) {
		if matches!(&self.target, Some((tw, th, _)) if *tw == w && *th == h) {
			return;
		}
		let tex = self.device.create_texture(&wgpu::TextureDescriptor {
			label: Some("itr-target"),
			size: wgpu::Extent3d {
				width: w,
				height: h,
				depth_or_array_layers: 1,
			},
			mip_level_count: 1,
			sample_count: 1,
			dimension: wgpu::TextureDimension::D2,
			format: wgpu::TextureFormat::Rgba8Unorm,
			usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
			view_formats: &[],
		});
		self.target = Some((w, h, tex));
	}

	fn draw_and_readback(&self, w: u32, h: u32) -> Result<RgbaImage, RendererError> {
		let target = &self.target.as_ref().unwrap().2;
		let target_view = target.create_view(&Default::default());
		let height_view = self.height_tex.as_ref().unwrap().2.create_view(&Default::default());

		let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
			label: Some("itr-bg"),
			layout: &self.bind_group_layout,
			entries: &[
				wgpu::BindGroupEntry {
					binding: 0,
					resource: self.uniform_buf.as_entire_binding(),
				},
				wgpu::BindGroupEntry {
					binding: 1,
					resource: wgpu::BindingResource::TextureView(&height_view),
				},
			],
		});

		let bytes_per_row = align_up(w * 4, wgpu::COPY_BYTES_PER_ROW_ALIGNMENT);
		let readback = self.device.create_buffer(&wgpu::BufferDescriptor {
			label: Some("itr-readback"),
			size: (bytes_per_row * h) as u64,
			usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
			mapped_at_creation: false,
		});

		let mut encoder = self
			.device
			.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("itr-enc") });
		{
			let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
				label: Some("itr-pass"),
				color_attachments: &[Some(wgpu::RenderPassColorAttachment {
					view: &target_view,
					resolve_target: None,
					ops: wgpu::Operations {
						load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
						store: wgpu::StoreOp::Store,
					},
				})],
				depth_stencil_attachment: None,
				timestamp_writes: None,
				occlusion_query_set: None,
			});
			pass.set_pipeline(&self.pipeline);
			pass.set_bind_group(0, &bind_group, &[]);
			pass.draw(0..3, 0..1);
		}
		encoder.copy_texture_to_buffer(
			wgpu::ImageCopyTexture {
				texture: target,
				mip_level: 0,
				origin: wgpu::Origin3d::ZERO,
				aspect: wgpu::TextureAspect::All,
			},
			wgpu::ImageCopyBuffer {
				buffer: &readback,
				layout: wgpu::ImageDataLayout {
					offset: 0,
					bytes_per_row: Some(bytes_per_row),
					rows_per_image: Some(h),
				},
			},
			wgpu::Extent3d {
				width: w,
				height: h,
				depth_or_array_layers: 1,
			},
		);
		self.queue.submit([encoder.finish()]);

		let slice = readback.slice(..);
		let (tx, rx) = std::sync::mpsc::channel();
		slice.map_async(wgpu::MapMode::Read, move |r| {
			let _ = tx.send(r);
		});
		self.device.poll(wgpu::Maintain::Wait);
		rx.recv()
			.unwrap()
			.map_err(|e| RendererError::Tile(std::io::Error::new(std::io::ErrorKind::Other, e)))?;

		let mapped = slice.get_mapped_range();
		let mut img = RgbaImage::new(w, h);
		let row_bytes = (w * 4) as usize;
		for y in 0..h as usize {
			let src = y * bytes_per_row as usize;
			img.as_mut()[y * row_bytes..(y + 1) * row_bytes].copy_from_slice(&mapped[src..src + row_bytes]);
		}
		drop(mapped);
		readback.unmap();
		Ok(img)
	}
}

struct TileBounds {
	lat_min: i32,
	lat_max: i32,
	lon_min: i32,
	lon_max: i32,
}

fn align_up(v: u32, align: u32) -> u32 { (v + align - 1) / align * align }
