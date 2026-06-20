// Terrain radar fragment shader.
// Heading-up, aircraft centered. Each pixel is projected onto a local tangent plane,
// converted to lat/lon, and sampled from an assembled heightmap of the visible tiles.
// Color is the classic TAWS ground-map ramp, referenced to aircraft altitude.

struct Uniforms {
    center_lat: f32,      // degrees
    center_lon: f32,      // degrees
    cos_lat: f32,         // cos(center_lat)
    meters_per_pixel: f32,
    width: f32,
    height: f32,
    heading: f32,         // radians, clockwise from north
    altitude_ft: f32,
    lon_min: f32,         // west edge of assembled heightmap, integer degrees
    lat_max: f32,         // north edge of assembled heightmap, integer degrees
    tile_res: f32,        // samples per tile side
    height_res: f32,      // meters per stored unit
};

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var heightmap: texture_2d<u32>;

const M_PER_DEG_LAT: f32 = 111320.0;
const HEIGHT_OFFSET: f32 = 500.0;
const FT_PER_M: f32 = 3.280839895;

const RED:       vec3<f32> = vec3<f32>(0.95, 0.00, 0.00);
const ORANGE:    vec3<f32> = vec3<f32>(0.76, 0.53, 0.10);
const YELLOW:    vec3<f32> = vec3<f32>(0.96, 0.98, 0.01);
const MED_GREEN: vec3<f32> = vec3<f32>(0.06, 0.36, 0.14);
const GREEN:     vec3<f32> = vec3<f32>(0.19, 0.64, 0.30);
const BLACK:     vec3<f32> = vec3<f32>(0.00, 0.00, 0.00);

@vertex
fn vs(@builtin(vertex_index) vid: u32) -> @builtin(position) vec4<f32> {
    let x = f32((vid << 1u) & 2u);
    let y = f32(vid & 2u);
    return vec4<f32>(x * 2.0 - 1.0, y * 2.0 - 1.0, 0.0, 1.0);
}

fn load(coord: vec2<i32>, dims: vec2<i32>) -> f32 {
    let c = clamp(coord, vec2<i32>(0, 0), dims - vec2<i32>(1, 1));
    return f32(textureLoad(heightmap, c, 0).x);
}

// Bilinear sample of the heightmap, returns stored units.
fn sample_stored(texel: vec2<f32>) -> f32 {
    let dims = vec2<i32>(textureDimensions(heightmap));
    let p = texel - vec2<f32>(0.5, 0.5);
    let p0 = floor(p);
    let f = p - p0;
    let i0 = vec2<i32>(p0);
    let c00 = load(i0, dims);
    let c10 = load(i0 + vec2<i32>(1, 0), dims);
    let c01 = load(i0 + vec2<i32>(0, 1), dims);
    let c11 = load(i0 + vec2<i32>(1, 1), dims);
    return mix(mix(c00, c10, f.x), mix(c01, c11, f.x), f.y);
}

// Lambertian hillshade computed in data space from the heightmap neighbours.
// Independent of zoom — it's a property of the terrain, not the view.
fn hillshade(texel: vec2<f32>) -> f32 {
    let hr = u.height_res;
    let mx = (M_PER_DEG_LAT * u.cos_lat) / u.tile_res; // meters per texel, east
    let my = M_PER_DEG_LAT / u.tile_res;               // meters per texel, north

    let zl = sample_stored(texel - vec2<f32>(1.0, 0.0)) * hr;
    let zr = sample_stored(texel + vec2<f32>(1.0, 0.0)) * hr;
    let zn = sample_stored(texel - vec2<f32>(0.0, 1.0)) * hr; // texel row up = north
    let zs = sample_stored(texel + vec2<f32>(0.0, 1.0)) * hr;

    let dzdx = (zr - zl) / (2.0 * mx);
    let dzdy = (zn - zs) / (2.0 * my); // positive northward

    let zenith = radians(45.0);
    let azimuth = radians(315.0); // light from the north-west
    let slope = atan(sqrt(dzdx * dzdx + dzdy * dzdy));
    var aspect = atan2(dzdy, -dzdx);
    if (aspect < 0.0) {
        aspect = aspect + 6.28318530718;
    }
    let hs = cos(zenith) * cos(slope) + sin(zenith) * sin(slope) * cos(azimuth - aspect);
    return clamp(hs, 0.0, 1.0);
}

fn color_for(elev_m: f32) -> vec3<f32> {
    let feet = elev_m * FT_PER_M;
    let alt = u.altitude_ft;
    if (feet - 2000.0 > alt) {
        return RED;
    } else if (feet - 1000.0 > alt) {
        return ORANGE;
    } else if (feet > alt - 500.0) {
        return YELLOW;
    } else if (feet + 1000.0 > alt) {
        return MED_GREEN;
    } else if (feet + 2000.0 > alt) {
        return GREEN;
    }
    return BLACK;
}

@fragment
fn fs(@builtin(position) frag: vec4<f32>) -> @location(0) vec4<f32> {
    // Pixel -> local east/north meters (screen up = forward).
    let east_m = (frag.x - u.width * 0.5) * u.meters_per_pixel;
    let north_m = (u.height * 0.5 - frag.y) * u.meters_per_pixel;

    // Rotate the screen so "up" points along the aircraft heading.
    let s = sin(u.heading);
    let c = cos(u.heading);
    let world_east = east_m * c + north_m * s;
    let world_north = -east_m * s + north_m * c;

    let lat = u.center_lat + world_north / M_PER_DEG_LAT;
    let lon = u.center_lon + world_east / (M_PER_DEG_LAT * u.cos_lat);

    // lat/lon -> texel in the north-up, west-left assembled heightmap.
    let col = (lon - u.lon_min) * u.tile_res;            // west->east
    let row = (u.lat_max + 1.0 - lat) * u.tile_res;      // north edge of grid is lat_max+1
    let texel = vec2<f32>(col, row);
    let stored = sample_stored(texel);

    let elev_m = stored * u.height_res - HEIGHT_OFFSET;
    let shade = mix(0.45, 1.0, hillshade(texel));
    return vec4<f32>(color_for(elev_m) * shade, 1.0);
}
