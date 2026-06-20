//! The `.itr` (Infinity Terrain Radar) container format.
//!
//! A single file holding the whole landmass as a sparse grid of 1°×1° tiles.
//! Heights only — no water, no hillshade (hillshade is computed at render time,
//! water reads back as ordinary terrain, matching how the T-38 paints the ground
//! return of its weather radar).
//!
//! ## Layout
//! * `[0..4]`   magic `b"ITR1"`
//! * `[4..6]`   format version, u16 LE
//! * `[6..8]`   tile resolution `R` (side length in samples), u16 LE
//! * `[8..10]`  height resolution: meters per stored unit, u16 LE (usually 1)
//! * `[10..12]` flags, u16 LE (bit0 = tiles are zstd compressed)
//! * `[12..32]` reserved, must be zero
//! * `[32..32 + 360*180*8]` index: `360*180` u64 LE absolute file offsets, 0 = absent
//! * `[offset..]` per tile: `len: u32 LE` followed by a zstd frame of `R*R` u16 LE
//!   heights, row-major, **north row first** (matching SRTM `.hgt` order).
//!
//! Stored height `v` maps to elevation: `elevation_m = v * height_resolution - HEIGHT_OFFSET`.
//! Voids / no-data are stored as 0 (i.e. clamps to `-HEIGHT_OFFSET` m, treated as ground).

use std::{
	fs::File,
	io::{self, Seek, SeekFrom, Write},
	path::Path,
};

use memmap2::Mmap;

pub const MAGIC: [u8; 4] = *b"ITR1";
pub const VERSION: u16 = 1;

pub const GRID_W: usize = 360;
pub const GRID_H: usize = 180;
pub const GRID_LEN: usize = GRID_W * GRID_H;

pub const HEADER_LEN: usize = 32;
pub const INDEX_LEN: usize = GRID_LEN * 8;
pub const DATA_START: u64 = (HEADER_LEN + INDEX_LEN) as u64;

/// Added to every elevation before storing so the lowest land point (~-431 m) is
/// non-negative. `elevation_m = stored * height_resolution - HEIGHT_OFFSET`.
pub const HEIGHT_OFFSET: i32 = 500;

const FLAG_ZSTD: u16 = 1;

/// Maps an integer tile corner (its south-west / lowest lat & lon) to its slot in the index.
#[inline]
pub fn tile_index(lat: i32, lon: i32) -> usize {
	debug_assert!((-90..90).contains(&lat), "lat out of range: {lat}");
	debug_assert!((-180..180).contains(&lon), "lon out of range: {lon}");
	((lat + 90) as usize) * GRID_W + (lon + 180) as usize
}

#[inline]
pub fn index_to_tile(index: usize) -> (i32, i32) {
	let lat = (index / GRID_W) as i32 - 90;
	let lon = (index % GRID_W) as i32 - 180;
	(lat, lon)
}

/// Quantise a raw SRTM elevation (meters, may be void) into a stored unit.
#[inline]
pub fn encode_height(elevation_m: i32, height_resolution: u16) -> u16 {
	// SRTM void sentinel; treat as ground.
	let e = if elevation_m <= -1000 { -HEIGHT_OFFSET } else { elevation_m };
	let shifted = (e + HEIGHT_OFFSET).max(0) as u32;
	let q = (shifted + (height_resolution as u32 / 2)) / height_resolution as u32;
	q.min(u16::MAX as u32) as u16
}

/// Inverse of [`encode_height`].
#[inline]
pub fn decode_height(stored: u16, height_resolution: u16) -> i32 {
	stored as i32 * height_resolution as i32 - HEIGHT_OFFSET
}

#[derive(Copy, Clone, Debug)]
pub struct Header {
	pub version: u16,
	pub resolution: u16,
	pub height_resolution: u16,
	pub flags: u16,
}

/// Streaming writer. Append tiles in any order, then [`finish`](Writer::finish) to
/// flush the index. Tiles are zstd-compressed on the way out.
pub struct Writer {
	file: File,
	index: Vec<u64>,
	resolution: u16,
	height_resolution: u16,
	zstd_level: i32,
}

impl Writer {
	pub fn create(path: &Path, resolution: u16, height_resolution: u16, zstd_level: i32) -> io::Result<Self> {
		let mut file = File::create(path)?;
		// Reserve header + index; real index is written on finish().
		file.write_all(&vec![0u8; HEADER_LEN + INDEX_LEN])?;
		Ok(Self {
			file,
			index: vec![0; GRID_LEN],
			resolution,
			height_resolution,
			zstd_level,
		})
	}

	pub fn resolution(&self) -> u16 { self.resolution }

	/// `heights` must be `resolution * resolution`, row-major, north row first.
	pub fn add_tile(&mut self, lat: i32, lon: i32, heights: &[u16]) -> io::Result<()> {
		let expect = self.resolution as usize * self.resolution as usize;
		assert_eq!(heights.len(), expect, "tile has wrong sample count");

		let raw: &[u8] = bytemuck::cast_slice(heights);
		let compressed = zstd::bulk::compress(raw, self.zstd_level)?;

		let offset = self.file.seek(SeekFrom::End(0))?;
		self.file.write_all(&(compressed.len() as u32).to_le_bytes())?;
		self.file.write_all(&compressed)?;

		self.index[tile_index(lat, lon)] = offset;
		Ok(())
	}

	pub fn tile_count(&self) -> usize { self.index.iter().filter(|&&o| o != 0).count() }

	pub fn finish(mut self) -> io::Result<()> {
		let mut header = [0u8; HEADER_LEN];
		header[0..4].copy_from_slice(&MAGIC);
		header[4..6].copy_from_slice(&VERSION.to_le_bytes());
		header[6..8].copy_from_slice(&self.resolution.to_le_bytes());
		header[8..10].copy_from_slice(&self.height_resolution.to_le_bytes());
		header[10..12].copy_from_slice(&FLAG_ZSTD.to_le_bytes());

		self.file.seek(SeekFrom::Start(0))?;
		self.file.write_all(&header)?;

		let index_bytes: &[u8] = bytemuck::cast_slice(&self.index);
		self.file.write_all(index_bytes)?;
		self.file.flush()?;
		Ok(())
	}
}

#[derive(Debug)]
pub enum LoadError {
	Io(io::Error),
	BadMagic,
	BadVersion(u16),
	TooSmall,
}

impl std::fmt::Display for LoadError {
	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		match self {
			LoadError::Io(e) => write!(f, "io error: {e}"),
			LoadError::BadMagic => write!(f, "not an .itr file (bad magic)"),
			LoadError::BadVersion(v) => write!(f, "unsupported .itr version {v}"),
			LoadError::TooSmall => write!(f, "file too small / truncated"),
		}
	}
}

impl std::error::Error for LoadError {}
impl From<io::Error> for LoadError {
	fn from(e: io::Error) -> Self { LoadError::Io(e) }
}

/// Memory-mapped, read-only view of an `.itr` file.
pub struct Dataset {
	mmap: Mmap,
	header: Header,
	index: Vec<u64>,
}

impl Dataset {
	pub fn open(path: &Path) -> Result<Self, LoadError> {
		let file = File::open(path)?;
		let mmap = unsafe { Mmap::map(&file)? };
		if mmap.len() < HEADER_LEN + INDEX_LEN {
			return Err(LoadError::TooSmall);
		}
		if mmap[0..4] != MAGIC {
			return Err(LoadError::BadMagic);
		}
		let version = u16::from_le_bytes(mmap[4..6].try_into().unwrap());
		if version != VERSION {
			return Err(LoadError::BadVersion(version));
		}
		let header = Header {
			version,
			resolution: u16::from_le_bytes(mmap[6..8].try_into().unwrap()),
			height_resolution: u16::from_le_bytes(mmap[8..10].try_into().unwrap()),
			flags: u16::from_le_bytes(mmap[10..12].try_into().unwrap()),
		};
		let index = mmap[HEADER_LEN..HEADER_LEN + INDEX_LEN]
			.chunks_exact(8)
			.map(|c| u64::from_le_bytes(c.try_into().unwrap()))
			.collect();

		Ok(Self { mmap, header, index })
	}

	pub fn header(&self) -> Header { self.header }
	pub fn resolution(&self) -> u32 { self.header.resolution as u32 }
	pub fn height_resolution(&self) -> u16 { self.header.height_resolution }

	pub fn tile_exists(&self, lat: i32, lon: i32) -> bool { self.index[tile_index(lat, lon)] != 0 }

	pub fn tile_count(&self) -> usize { self.index.iter().filter(|&&o| o != 0).count() }

	/// Decompress a tile into `R*R` u16 heights (north row first). `None` if absent.
	pub fn read_tile(&self, lat: i32, lon: i32) -> Option<io::Result<Vec<u16>>> {
		let offset = self.index[tile_index(lat, lon)] as usize;
		if offset == 0 {
			return None;
		}
		Some(self.read_tile_at(offset))
	}

	fn read_tile_at(&self, offset: usize) -> io::Result<Vec<u16>> {
		let len = u32::from_le_bytes(self.mmap[offset..offset + 4].try_into().unwrap()) as usize;
		let frame = &self.mmap[offset + 4..offset + 4 + len];
		let samples = self.header.resolution as usize * self.header.resolution as usize;
		let raw = zstd::bulk::decompress(frame, samples * 2)?;
		Ok(bytemuck::cast_slice(&raw).to_vec())
	}
}
