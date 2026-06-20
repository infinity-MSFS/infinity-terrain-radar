//! `itr-build` — turn a tree of SRTM3 `.hgt.zip` tiles into one packed `.itr` file.
//!
//! Source layout (taclink's downloader): `<srtm>/<Region>/<TILE>.hgt.zip`, where each
//! zip holds a single `1201x1201` i16 **big-endian** `.hgt`, north row first.

use std::{
	io::Read,
	path::{Path, PathBuf},
	sync::{
		atomic::{AtomicUsize, Ordering},
		mpsc::sync_channel,
	},
	thread,
};

use anyhow::{bail, Context, Result};
use clap::{Parser, Subcommand};
use indicatif::{ProgressBar, ProgressStyle};
use itr_format::{encode_height, index_to_tile, Dataset, Writer};

/// SRTM3 tiles are 1201 samples per side; the last row/col is shared with the
/// neighbouring tile, so we treat 1200 as the usable extent for clean downsampling.
const SRTM_DIM: usize = 1201;
const SRTM_USABLE: usize = 1200;
const SRTM_VOID: i16 = -32768;

#[derive(Parser)]
#[command(name = "itr-build", about = "Build and inspect .itr terrain databases")]
struct Cli {
	#[command(subcommand)]
	cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
	/// Pack an SRTM tree into a single .itr file.
	Build(Build),
	/// Print stats about an existing .itr file (and optionally write a coverage PNG).
	Info(Info),
}

#[derive(Parser)]
struct Build {
	/// Directory containing the SRTM region subfolders of `*.hgt.zip`.
	#[arg(long, default_value = r"C:\Users\Cameron\Documents\github\taclink\srtm")]
	srtm: PathBuf,
	/// Output .itr path.
	#[arg(short, long)]
	out: PathBuf,
	/// Samples per side of each 1° tile in the output (downsample target). Must divide 1200 cleanly.
	#[arg(short, long, default_value_t = 600)]
	resolution: u16,
	/// Meters per stored height unit (coarser = smaller). 1 keeps full vertical detail.
	#[arg(long, default_value_t = 1)]
	height_resolution: u16,
	/// zstd compression level (1..=22). Higher = smaller + slower.
	#[arg(long, default_value_t = 19)]
	zstd_level: i32,
	/// Decode worker threads.
	#[arg(short, long, default_value_t = 0)]
	jobs: usize,
}

#[derive(Parser)]
struct Info {
	/// .itr file to inspect.
	file: PathBuf,
	/// Optional PNG path to write a 360x180 land-coverage map.
	#[arg(long)]
	coverage: Option<PathBuf>,
}

fn main() -> Result<()> {
	match Cli::parse().cmd {
		Cmd::Build(b) => build(b),
		Cmd::Info(i) => info(i),
	}
}

fn build(args: Build) -> Result<()> {
	if SRTM_USABLE % args.resolution as usize != 0 {
		bail!(
			"resolution {} must divide {} cleanly (try 600, 400, 300, 240, 200, 150, 120, 100, 75, 60)",
			args.resolution,
			SRTM_USABLE
		);
	}

	let files = collect_tiles(&args.srtm)?;
	if files.is_empty() {
		bail!("no .hgt.zip files found under {}", args.srtm.display());
	}
	println!(
		"Building {} from {} tiles @ {}²  (hres={}m, zstd={})",
		args.out.display(),
		files.len(),
		args.resolution,
		args.height_resolution,
		args.zstd_level
	);

	let jobs = if args.jobs == 0 {
		thread::available_parallelism().map(|n| n.get()).unwrap_or(4)
	} else {
		args.jobs
	};
	let resolution = args.resolution;
	let hres = args.height_resolution;

	let mut writer = Writer::create(&args.out, resolution, hres, args.zstd_level)?;

	let bar = ProgressBar::new(files.len() as u64);
	bar.set_style(
		ProgressStyle::with_template("{bar:40} {pos}/{len} tiles  {msg}")
			.unwrap()
			.progress_chars("=> "),
	);

	let next = AtomicUsize::new(0);
	// Bounded so parallel decoders apply backpressure on the single writer.
	let (tx, rx) = sync_channel::<(i32, i32, Vec<u16>)>(jobs * 2);

	thread::scope(|scope| -> Result<()> {
		for _ in 0..jobs {
			let tx = tx.clone();
			let next = &next;
			let files = &files;
			scope.spawn(move || loop {
				let i = next.fetch_add(1, Ordering::Relaxed);
				if i >= files.len() {
					break;
				}
				match decode_tile(&files[i], resolution, hres) {
					Ok(Some((lat, lon, heights))) => {
						let _ = tx.send((lat, lon, heights));
					},
					Ok(None) => {},
					Err(e) => eprintln!("skip {}: {e:#}", files[i].display()),
				}
			});
		}
		drop(tx); // workers hold their own clones; this lets rx finish.

		let mut written = 0usize;
		for (lat, lon, heights) in rx {
			writer.add_tile(lat, lon, &heights)?;
			written += 1;
			bar.inc(1);
			if written % 200 == 0 {
				bar.set_message(format!("{written} packed"));
			}
		}
		Ok(())
	})?;

	bar.finish_and_clear();
	let count = writer.tile_count();
	writer.finish()?;

	let size = std::fs::metadata(&args.out)?.len();
	println!(
		"Done: {} tiles, {:.1} MB ({:.2} GB)",
		count,
		size as f64 / 1e6,
		size as f64 / 1e9
	);
	Ok(())
}

/// Returns `(lat, lon, heights)` or `None` if the tile is entirely void.
fn decode_tile(path: &Path, resolution: u16, hres: u16) -> Result<Option<(i32, i32, Vec<u16>)>> {
	let (lat, lon) = parse_tile_name(path).with_context(|| format!("bad tile name {}", path.display()))?;

	let file = std::fs::File::open(path)?;
	let mut zip = zip::ZipArchive::new(file)?;
	let mut entry = zip.by_index(0)?;
	let mut raw = Vec::with_capacity(SRTM_DIM * SRTM_DIM * 2);
	entry.read_to_end(&mut raw)?;
	if raw.len() < SRTM_DIM * SRTM_DIM * 2 {
		bail!("short .hgt: {} bytes", raw.len());
	}

	let sample = |row: usize, col: usize| -> i16 {
		let i = (row * SRTM_DIM + col) * 2;
		i16::from_be_bytes([raw[i], raw[i + 1]])
	};

	let r = resolution as usize;
	let factor = SRTM_USABLE / r;
	let mut out = vec![0u16; r * r];
	let mut any_data = false;

	// Max-pool: a terrain radar wants the worst-case (highest) obstacle in each cell.
	for oy in 0..r {
		for ox in 0..r {
			let mut max_elev = i32::MIN;
			for sy in 0..factor {
				for sx in 0..factor {
					let v = sample(oy * factor + sy, ox * factor + sx);
					if v != SRTM_VOID {
						max_elev = max_elev.max(v as i32);
						any_data = true;
					}
				}
			}
			let elev = if max_elev == i32::MIN { -1000 } else { max_elev };
			out[oy * r + ox] = encode_height(elev, hres);
		}
	}

	Ok(any_data.then_some((lat, lon, out)))
}

/// `N00E006.hgt.zip` / `S17W069.hgt.zip` -> SW corner (lat, lon).
fn parse_tile_name(path: &Path) -> Option<(i32, i32)> {
	let name = path.file_name()?.to_str()?;
	let stem = name.split('.').next()?; // strip ".hgt.zip"
	let b = stem.as_bytes();
	if b.len() < 7 {
		return None;
	}
	let lat_sign = match b[0] {
		b'N' => 1,
		b'S' => -1,
		_ => return None,
	};
	let lat: i32 = stem.get(1..3)?.parse().ok()?;
	let lon_sign = match b[3] {
		b'E' => 1,
		b'W' => -1,
		_ => return None,
	};
	let lon: i32 = stem.get(4..7)?.parse().ok()?;
	Some((lat_sign * lat, lon_sign * lon))
}

fn collect_tiles(root: &Path) -> Result<Vec<PathBuf>> {
	let mut out = Vec::new();
	for region in std::fs::read_dir(root)? {
		let region = region?.path();
		if !region.is_dir() {
			continue;
		}
		for entry in std::fs::read_dir(&region)? {
			let p = entry?.path();
			if p.file_name().and_then(|n| n.to_str()).is_some_and(|n| n.ends_with(".hgt.zip")) {
				out.push(p);
			}
		}
	}
	out.sort();
	Ok(out)
}

fn info(args: Info) -> Result<()> {
	let ds = Dataset::open(&args.file).map_err(|e| anyhow::anyhow!("{e}"))?;
	let size = std::fs::metadata(&args.file)?.len();
	let count = ds.tile_count();
	println!("file:        {}", args.file.display());
	println!("size:        {:.1} MB ({:.2} GB)", size as f64 / 1e6, size as f64 / 1e9);
	println!("resolution:  {}² samples/tile", ds.resolution());
	println!("height res:  {} m/unit", ds.height_resolution());
	println!("tiles:       {count} present");
	println!("avg tile:    {:.1} KB", size as f64 / count.max(1) as f64 / 1e3);

	if let Some(path) = args.coverage {
		let mut img = image::GrayImage::new(itr_format::GRID_W as u32, itr_format::GRID_H as u32);
		for idx in 0..itr_format::GRID_LEN {
			let (lat, lon) = index_to_tile(idx);
			if ds.tile_exists(lat, lon) {
				// North up: row 0 = +89°.
				let px = (lon + 180) as u32;
				let py = (89 - lat) as u32;
				img.put_pixel(px, py, image::Luma([255]));
			}
		}
		img.save(&path)?;
		println!("coverage:    {}", path.display());
	}
	Ok(())
}
