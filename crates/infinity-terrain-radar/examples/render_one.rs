//! Headless smoke test: `cargo run -p infinity-terrain-radar --example render_one -- <db.itr> <lat> <lon> <range_nm> <alt_ft> <out.png>`

use infinity_terrain_radar::{Dataset, FrameOptions, LatLon, Renderer};

fn main() {
	let a: Vec<String> = std::env::args().collect();
	if a.len() < 7 {
		eprintln!("usage: render_one <db.itr> <lat> <lon> <range_nm> <alt_ft> <out.png>");
		std::process::exit(1);
	}
	let ds = Dataset::open(std::path::Path::new(&a[1])).expect("open db");
	println!("loaded {} tiles @ {}²", ds.tile_count(), ds.resolution());

	let mut r = Renderer::new(ds).expect("renderer");
	let opt = FrameOptions {
		width: 1024,
		height: 1024,
		center: LatLon {
			lat: a[2].parse().unwrap(),
			lon: a[3].parse().unwrap(),
		},
		range_nm: a[4].parse().unwrap(),
		altitude_ft: a[5].parse().unwrap(),
		heading: 0.0,
	};
	let t = std::time::Instant::now();
	let img = r.render(&opt).expect("render");
	println!("rendered {}x{} in {:?}", img.width(), img.height(), t.elapsed());
	img.save(&a[6]).expect("save png");
	println!("wrote {}", a[6]);
}
