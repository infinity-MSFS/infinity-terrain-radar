//! Heights for a client that colours its own picture — the DC-10 family's
//! EGPWS displays, which ask with `format: "elevation"`: a north-up square of
//! the database's own heights, GPU-free, on exactly the renderer's projection.
//!
//! ## The projection — the renderer's
//! Flat earth about the centre, as `terrain.wgsl` draws it: a pixel is
//! `2 · range_nm · 1852 / size` metres; north metres over 111 320 are degrees
//! of latitude, east metres over `111 320 · cos(lat₀)` degrees of longitude,
//! `lat₀` the centre's; row 0 north. A database sample covers `1/R` of a
//! degree, row 0 its tile's north edge — the shader's own lookup.
//!
//! ## The code — the client's `ELEVATION_ENCODING`
//! One `u16` a pixel, sent as a 16-bit greyscale PNG:
//! * `0` — no data (never sent: the database has none, a void is sea level);
//! * bit 15 — water;
//! * bits 0–14 — metres + [`ELEVATION_OFFSET_M`], clamped to `1..=32 767`.
//!
//! ## Water
//! The database carries no water flag. Open ocean is an ABSENT tile, and the
//! builder stores a void cell — SRTM's ocean — as exactly sea level. So a
//! pixel whose highest sample is exactly 0 m, or lies in an absent tile, is
//! water at 0 m; land below sea level (a polder, Death Valley) stays land.
//!
//! ## Max-pooled
//! A pixel is its footprint's HIGHEST sample, over a regular sub-grid of up to
//! [`MAX_SUB`]² samples — the builder's own rule one level up: a terrain
//! display must never lose a peak to a coarse pixel.

use std::collections::hash_map::Entry;
use std::collections::{HashMap, VecDeque};
use std::io;

use image::{ImageBuffer, Luma};
use itr_format::{decode_height, Dataset};

use crate::LatLon;

const M_PER_DEG_LAT: f64 = 111_320.0;
const NM_TO_M: f64 = 1852.0;
/// The narrowest `cos(lat)` the projection divides by.
const MIN_COS_LAT: f64 = 1e-3;

/// Added to a height in metres before it is coded.
pub const ELEVATION_OFFSET_M: i32 = 1000;
/// Set on a water sample.
pub const ELEVATION_WATER_BIT: u16 = 0x8000;
/// The most samples a side a pixel's footprint is pooled over.
pub const MAX_SUB: u32 = 16;
/// The most decoded tiles held (about 180 KB each at R = 300).
const CACHE_TILES: usize = 400;

/// One heights frame.
#[derive(Copy, Clone, Debug)]
pub struct HeightsOptions {
	/// Output edge in pixels; the frame is square.
	pub size: u32,
	/// The square's centre.
	pub center: LatLon,
	/// Half the square's side, nautical miles.
	pub range_nm: f32,
}

/// One pixel's ground.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum Ground {
	Land(i32),
	Water,
}

impl Ground {
	fn code(self) -> u16 {
		let m = match self {
			Ground::Land(m) => m,
			Ground::Water => 0,
		};
		let v = (m + ELEVATION_OFFSET_M).clamp(1, i32::from(!ELEVATION_WATER_BIT)) as u16;
		match self {
			Ground::Land(_) => v,
			Ground::Water => v | ELEVATION_WATER_BIT,
		}
	}
}

/// Samples heights frames off a [`Dataset`], keeping the tiles it decodes.
#[derive(Default)]
pub struct HeightSampler {
	cache: HashMap<(i32, i32), Option<Vec<u16>>>,
	order: VecDeque<(i32, i32)>,
}

/// The tile a point lies in, and the sample within it, as the shader
/// indexes them: row 0 the tile's north edge.
fn locate(lat: f64, lon: f64, res: usize) -> ((i32, i32), usize) {
	let lat = lat.clamp(-89.999_999, 89.999_999);
	let lon = (lon + 540.0).rem_euclid(360.0) - 180.0;
	let tlat = (lat.floor() as i32).clamp(-90, 89);
	let tlon = (lon.floor() as i32).clamp(-180, 179);
	let r = res as f64;
	let row = (((f64::from(tlat) + 1.0 - lat) * r) as usize).min(res - 1);
	let col = (((lon - f64::from(tlon)) * r) as usize).min(res - 1);
	((tlat, tlon), row * res + col)
}

/// A tile's longitude wrapped into the index's `-180..=179`.
fn wrap_lon(tlon: i32) -> i32 { (tlon + 180).rem_euclid(360) - 180 }

impl HeightSampler {
	pub fn new() -> Self { Self::default() }

	/// Decode tile `(lat, lon)` into the cache if it is not there.
	fn fetch(&mut self, ds: &Dataset, lat: i32, lon: i32) -> io::Result<()> {
		if let Entry::Vacant(slot) = self.cache.entry((lat, lon)) {
			let decoded = match ds.read_tile(lat, lon) {
				Some(t) => Some(t?),
				None => None,
			};
			slot.insert(decoded);
			self.order.push_back((lat, lon));
		}
		Ok(())
	}

	/// Forget the oldest tiles down to the cache's size — before a frame,
	/// never during one, so a frame that needs more holds them all.
	fn trim(&mut self) {
		while self.order.len() > CACHE_TILES {
			if let Some(old) = self.order.pop_front() {
				self.cache.remove(&old);
			}
		}
	}

	/// The database's height at `(lat, lon)`, metres — `None` in an absent
	/// tile (ocean).
	#[cfg(test)]
	fn height_m(&mut self, ds: &Dataset, lat: f64, lon: f64) -> io::Result<Option<i32>> {
		let res = ds.resolution() as usize;
		let ((tlat, tlon), i) = locate(lat, lon, res);
		self.fetch(ds, tlat, tlon)?;
		let tile = self.cache.get(&(tlat, tlon)).and_then(Option::as_ref);
		Ok(tile.and_then(|t| t.get(i)).map(|&s| decode_height(s, ds.height_resolution())))
	}

	/// The frame as coded samples, `size²`, row 0 north.
	pub fn sample(&mut self, ds: &Dataset, opt: &HeightsOptions) -> io::Result<Vec<u16>> {
		let size = opt.size.max(1);
		let res = ds.resolution().max(1) as usize;
		let hres = ds.height_resolution();
		let lat0 = f64::from(opt.center.lat);
		let lon0 = f64::from(opt.center.lon);
		let cos_lat = lat0.to_radians().cos().max(MIN_COS_LAT);
		let mpp = 2.0 * f64::from(opt.range_nm.max(0.01)) * NM_TO_M / f64::from(size);
		// Enough sub-samples to see every database sample a pixel covers.
		let sample_m = M_PER_DEG_LAT / res as f64;
		let sub = ((mpp / sample_m).ceil() as u32).clamp(1, MAX_SUB);
		let half = f64::from(size) * 0.5;

		// Every tile the frame touches — a tile's margin round it — decoded
		// before any is read.
		self.trim();
		let dlat = half * mpp / M_PER_DEG_LAT;
		let dlon = half * mpp / (M_PER_DEG_LAT * cos_lat);
		let lat_s = ((lat0 - dlat).floor() as i32 - 1).clamp(-90, 89);
		let lat_n = ((lat0 + dlat).floor() as i32 + 1).clamp(-90, 89);
		let lon_w = (lon0 - dlon).floor() as i32 - 1;
		let lon_e = ((lon0 + dlon).floor() as i32 + 1).min(lon_w + 359);
		for tlat in lat_s..=lat_n {
			for tlon in lon_w..=lon_e {
				self.fetch(ds, tlat, wrap_lon(tlon))?;
			}
		}

		let cache = &self.cache;
		// The tile last read, and its samples: consecutive sub-samples
		// nearly always share one.
		let mut memo: ((i32, i32), Option<&Vec<u16>>) = ((i32::MIN, i32::MIN), None);
		let mut out = Vec::with_capacity((size * size) as usize);
		for py in 0..size {
			for px in 0..size {
				let mut highest: Option<i32> = None;
				let mut water = false;
				for sy in 0..sub {
					for sx in 0..sub {
						let fx = f64::from(px) + (f64::from(sx) + 0.5) / f64::from(sub);
						let fy = f64::from(py) + (f64::from(sy) + 0.5) / f64::from(sub);
						let lat = lat0 + (half - fy) * mpp / M_PER_DEG_LAT;
						let lon = lon0 + (fx - half) * mpp / (M_PER_DEG_LAT * cos_lat);
						let (key, i) = locate(lat, lon, res);
						if memo.0 != key {
							memo = (key, cache.get(&key).and_then(Option::as_ref));
						}
						// An absent tile is open ocean, at sea level.
						let h = memo
							.1
							.and_then(|t| t.get(i))
							.map_or(0, |&s| decode_height(s, hres));
						water |= h == 0;
						highest = Some(highest.map_or(h, |m| m.max(h)));
					}
				}
				let ground = match highest {
					Some(h) if h > 0 || !water => Ground::Land(h),
					_ => Ground::Water,
				};
				out.push(ground.code());
			}
		}
		Ok(out)
	}

	/// [`Self::sample`] as a 16-bit greyscale PNG.
	pub fn sample_png(&mut self, ds: &Dataset, opt: &HeightsOptions) -> io::Result<Vec<u8>> {
		let size = opt.size.max(1);
		let samples = self.sample(ds, opt)?;
		let img: ImageBuffer<Luma<u16>, Vec<u16>> = ImageBuffer::from_raw(size, size, samples)
			.ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "heights buffer size"))?;
		let mut buf = io::Cursor::new(Vec::new());
		img.write_to(&mut buf, image::ImageFormat::Png).map_err(io::Error::other)?;
		Ok(buf.into_inner())
	}
}

#[cfg(test)]
mod tests {
	use super::*;
	use itr_format::{encode_height, Writer};

	/// A two-tile database: (40, -100) a ramp rising east from the sea,
	/// (40, -99) a plateau at 1 000 m with one 3 000 m peak sample; every
	/// other tile absent — ocean.
	fn dataset(name: &str) -> (Dataset, std::path::PathBuf) {
		let res: u16 = 8;
		let path = std::env::temp_dir().join(format!("itr-heights-{name}-{}.itr", std::process::id()));
		let mut w = Writer::create(&path, res, 1, 1).unwrap();
		let r = res as usize;
		let ramp: Vec<u16> = (0..r * r).map(|i| encode_height((i % r) as i32 * 100, 1)).collect();
		w.add_tile(40, -100, &ramp).unwrap();
		let mut plateau = vec![encode_height(1000, 1); r * r];
		plateau[3 * r + 4] = encode_height(3000, 1);
		w.add_tile(40, -99, &plateau).unwrap();
		w.finish().unwrap();
		(Dataset::open(&path).unwrap(), path)
	}

	fn decode(code: u16) -> (i32, bool) {
		(i32::from(code & !ELEVATION_WATER_BIT) - ELEVATION_OFFSET_M, code & ELEVATION_WATER_BIT != 0)
	}

	#[test]
	fn the_code_is_the_clients() {
		assert_eq!(Ground::Land(0).code(), 1000);
		assert_eq!(Ground::Land(1234).code(), 2234);
		assert_eq!(Ground::Water.code(), 1000 | 0x8000);
		assert_eq!(Ground::Land(-5000).code(), 1, "clamped, never 0 (no data)");
		assert_eq!(Ground::Land(40_000).code(), 0x7fff, "clamped below the water bit");
	}

	#[test]
	fn a_point_reads_the_sample_the_shader_reads() {
		let (ds, path) = dataset("point");
		let mut s = HeightSampler::new();
		// Column 5 of the ramp (500 m): 5/8 to 6/8 of the way east.
		assert_eq!(s.height_m(&ds, 40.5, -100.0 + 5.5 / 8.0).unwrap(), Some(500));
		// The peak: row 3 from the north edge, column 4.
		assert_eq!(s.height_m(&ds, 41.0 - 3.5 / 8.0, -99.0 + 4.5 / 8.0).unwrap(), Some(3000));
		assert_eq!(s.height_m(&ds, 10.0, 10.0).unwrap(), None, "absent: ocean");
		drop(ds);
		let _ = std::fs::remove_file(path);
	}

	#[test]
	fn a_frame_is_north_up_max_pooled_and_flags_the_sea() {
		let (ds, path) = dataset("frame");
		let mut s = HeightSampler::new();
		// A 4-pixel frame over both tiles, each pixel half a degree of
		// longitude (about 23 NM at 40°N) — wide enough to pool many samples.
		let opt = HeightsOptions {
			size: 4,
			center: LatLon { lat: 40.5, lon: -99.0 },
			range_nm: 46.0,
		};
		let f = s.sample(&ds, &opt).unwrap();
		assert_eq!(f.len(), 16);
		let row: Vec<(i32, bool)> = f[4..8].iter().map(|&c| decode(c)).collect();
		// West pixels: the ramp's highest; east: the plateau, and the pixel
		// holding the peak takes it.
		assert!(row.iter().all(|&(_, w)| !w), "{row:?}");
		assert!(row[0].0 >= 300 && row[0].0 <= 700, "{row:?}");
		assert!(row[1].0 >= 600, "{row:?}");
		assert!(row.iter().any(|&(m, _)| m == 3000), "the peak survives: {row:?}");
		// Far out to sea: every pixel water at 0 m.
		let sea = HeightsOptions {
			center: LatLon { lat: 10.0, lon: 10.0 },
			..opt
		};
		assert!(s.sample(&ds, &sea).unwrap().iter().all(|&c| decode(c) == (0, true)));
		// The ramp's own 0 m column is sea too — the builder's void.
		let shore = HeightsOptions {
			size: 1,
			center: LatLon { lat: 40.5, lon: -100.0 + 0.5 / 8.0 },
			range_nm: 0.5,
		};
		assert_eq!(decode(s.sample(&ds, &shore).unwrap()[0]), (0, true));
		drop(ds);
		let _ = std::fs::remove_file(path);
	}

	#[test]
	fn the_png_is_sixteen_bit_grey_and_round_trips() {
		let (ds, path) = dataset("png");
		let mut s = HeightSampler::new();
		let opt = HeightsOptions {
			size: 8,
			center: LatLon { lat: 40.5, lon: -99.5 },
			range_nm: 30.0,
		};
		let raw = s.sample(&ds, &opt).unwrap();
		let png = s.sample_png(&ds, &opt).unwrap();
		let img = image::load_from_memory_with_format(&png, image::ImageFormat::Png).unwrap();
		assert_eq!(img.color(), image::ColorType::L16);
		assert_eq!(img.to_luma16().into_raw(), raw);
		drop(ds);
		let _ = std::fs::remove_file(path);
	}
}
