//! Throwaway prototype for the coloured-light map's ticket "How coloured light
//! looks" (.scratch/coloured-light/issues/03-how-coloured-light-looks.md). Not
//! crate code. `colour_look_build.sh` copies it into the probe worktree's
//! `examples/` for one build.
//!
//! The crate renders the world as it does on main: its sky light, sun gate and
//! white voxel light come from its own `sample_light`. The probe solves
//! coloured voxel light on the CPU with the crate's kernel
//! (`light_bench::KernelBox`), one box per chunk as the solver does, for each
//! candidate, uploads it as a 3D texture over the scene, and its material
//! swaps that in for the crate's voxel light. Sky light always stays the
//! crate's.
//!
//! Candidates (C / Shift+C):
//!
//! - today: the crate as it is, every lamp white.
//! - per-component levels: three runs, red, green and blue. Each lamp seeds a
//!   component at the whole level whose brightness under the falloff is its
//!   colour times its own, so a weaker component starts lower and ends sooner.
//!   Each component is the max over lamps.
//! - groups, max: a run per light colour. Each component is the max over runs
//!   of colour times brightness.
//! - groups, sum: the same, summed and capped at 1.
//! - groups, blend by brightness: the brightest run's level, in the runs'
//!   colours averaged with their brightness as weights.
//! - groups, blend by level: the same, with `0.7^(levels below the brightest)`
//!   as weights, which don't depend on the falloff curve.
//!
//! Group candidates are shown through the shader's per-component path: each
//! component's brightness is turned back into a level through the falloff, so
//! smoothing blends them as it blends levels. The shader leaves a cell out of
//! a blend when its brightest component is 0, as today's leaves out a 0.
//!
//! Keys: G glow colour, V view (look, light only, check), N night, F voxel
//! falloff, P precision, R voxel reach, M smoothing, H the panel, 1-6 places.
//! The right mouse button orbits, the middle one pans, the wheel zooms.
//!
//! Arguments: `shots out=<dir>` shoots every shot in `SHOTS` and exits.

use std::sync::Arc;

use bevy::{
    asset::{uuid_handle, RenderAssetUsages},
    core_pipeline::tonemapping::Tonemapping,
    image::{CompressedImageFormats, ImageAddressMode, ImageSampler, ImageSamplerDescriptor, ImageType},
    light::CascadeShadowConfigBuilder,
    mesh::MeshVertexBufferLayoutRef,
    pbr::{ExtendedMaterial, MaterialExtension, MaterialExtensionKey, MaterialExtensionPipeline},
    prelude::*,
    reflect::TypePath,
    render::{
        render_resource::{
            AsBindGroup, Extent3d, RenderPipelineDescriptor, ShaderType,
            SpecializedMeshPipelineError, TextureDimension, TextureFormat, TextureViewDescriptor,
            TextureViewDimension,
        },
        view::screenshot::{save_to_disk, Screenshot},
    },
    shader::{Shader, ShaderDefVal, ShaderRef},
};
use bevy_panorbit_camera::{PanOrbitCamera, PanOrbitCameraPlugin};
use bevy_voxel_world::{
    light_bench::KernelBox,
    prelude::*,
    rendering::{vertex_layout, VoxelWorldMaterial, VoxelWorldMaterialHandle},
};

const SHADER: Handle<Shader> = uuid_handle!("6b1f0c55-35c1-4d0e-9f0e-8a3d6f2c7c11");
const VOXEL_TEXTURE: Handle<Image> = uuid_handle!("6b1f0c55-35c1-4d0e-9f0e-8a3d6f2c7c12");
const COLOUR_TEXTURE: Handle<Image> = uuid_handle!("6b1f0c55-35c1-4d0e-9f0e-8a3d6f2c7c13");

// --- Lamps ------------------------------------------------------------------------

struct LampKind {
    name: &'static str,
    /// Linear RGB.
    colour: [f32; 3],
    level: u8,
}

const KINDS: [LampKind; 9] = [
    LampKind { name: "white", colour: [1.0, 1.0, 1.0], level: 15 },
    LampKind { name: "red", colour: [1.0, 0.12, 0.08], level: 15 },
    LampKind { name: "green", colour: [0.15, 1.0, 0.2], level: 15 },
    LampKind { name: "blue", colour: [0.15, 0.3, 1.0], level: 15 },
    LampKind { name: "orange", colour: [1.0, 0.45, 0.08], level: 15 },
    LampKind { name: "warm", colour: [1.0, 0.75, 0.45], level: 15 },
    LampKind { name: "magenta", colour: [0.85, 0.15, 1.0], level: 15 },
    LampKind { name: "teal", colour: [0.1, 0.75, 0.9], level: 15 },
    LampKind { name: "dim orange", colour: [1.0, 0.45, 0.08], level: 9 },
];
const WHITE: usize = 0;
const RED: usize = 1;
const GREEN: usize = 2;
const BLUE: usize = 3;
const ORANGE: usize = 4;
const WARM: usize = 5;
const MAGENTA: usize = 6;
const TEAL: usize = 7;
const DIM_ORANGE: usize = 8;

/// The first kind with the same light colour, so kinds sharing a colour share
/// a run.
fn colour_id(kind: usize) -> usize {
    (0..KINDS.len())
        .find(|&k| KINDS[k].colour == KINDS[kind].colour)
        .unwrap()
}

const LAMP_BASE: u8 = 20;

fn stage_x0(i: i32) -> i32 {
    52 + 13 * i
}

/// Every lamp in the scene.
const LAMPS: [(IVec3, usize); 17] = [
    // The cave: red, green and blue along the back wall, warm and magenta in the floor.
    (IVec3::new(-74, 45, -77), RED),
    (IVec3::new(-64, 45, -77), GREEN),
    (IVec3::new(-54, 45, -77), BLUE),
    (IVec3::new(-74, 39, -58), WARM),
    (IVec3::new(-58, 39, -58), MAGENTA),
    // The overhang: green and orange in the cliff under the shelf.
    (IVec3::new(-22, 44, -41), GREEN),
    (IVec3::new(-8, 44, -41), ORANGE),
    // The island: red and blue in its underside.
    (IVec3::new(18, 45, -7), RED),
    (IVec3::new(32, 45, -7), BLUE),
    // The stages: white; white, shown by the crate; white and red.
    (IVec3::new(52 + 5, 43, 4), WHITE),
    (IVec3::new(65 + 5, 43, 4), WHITE),
    (IVec3::new(78 + 2, 43, 4), WHITE),
    (IVec3::new(78 + 7, 43, 4), RED),
    // The trenches: orange, warm, teal and dim orange at their far ends.
    (IVec3::new(-62 + 1, 36, 9), ORANGE),
    (IVec3::new(-54 + 1, 36, 9), WARM),
    (IVec3::new(-46 + 1, 36, 9), TEAL),
    (IVec3::new(-38 + 1, 36, 9), DIM_ORANGE),
];

struct Flags {
    thin: bool,
    no_white: bool,
}

static FLAGS: std::sync::LazyLock<Flags> = std::sync::LazyLock::new(|| {
    let args: Vec<String> = std::env::args().collect();
    Flags {
        thin: args.iter().any(|a| a == "thin"),
        no_white: args.iter().any(|a| a == "nowhite"),
    }
});

/// The scene's lamps, less the third stage's white one under `nowhite`.
fn lamps() -> impl Iterator<Item = (IVec3, usize)> {
    LAMPS
        .into_iter()
        .filter(|&(p, _)| !(FLAGS.no_white && p == IVec3::new(78 + 2, 43, 4)))
}

/// The middle stage, which always shows the crate's light: the voxels whose
/// faces read it.
const TODAY_MIN: IVec3 = IVec3::new(64, 39, 4);
const TODAY_MAX: IVec3 = IVec3::new(76, 48, 15);

// --- The scene --------------------------------------------------------------------

const GRASS: u8 = 0;
const DIRT: u8 = 1;
const STONE: u8 = 2;
const PLASTER: u8 = 4;
const FLOOR: u8 = 5;

const GROUND: i32 = 40;
const HILL_TOP: i32 = 70;
const CLIFF: i32 = -40;

fn within(p: IVec3, min: IVec3, max: IVec3) -> bool {
    p.cmpge(min).all() && p.cmplt(max).all()
}

/// The material at `p`, or `None` for air. A pure function of position.
fn scene_voxel(p: IVec3) -> Option<u8> {
    for (at, kind) in lamps() {
        if p == at {
            return Some(LAMP_BASE + kind as u8);
        }
    }
    // The cave's boulder, hall and passage.
    if within(p, IVec3::new(-71, 40, -73), IVec3::new(-67, 43, -69)) {
        return Some(STONE);
    }
    if within(p, IVec3::new(-80, 40, -76), IVec3::new(-48, 52, -52))
        || within(p, IVec3::new(-66, 40, -52), IVec3::new(-63, 44, -40))
    {
        return None;
    }
    // Trenches, lined with stone.
    for i in 0..4 {
        let x0 = -62 + 8 * i;
        if within(p, IVec3::new(x0, 34, 10), IVec3::new(x0 + 3, 40, 40)) {
            return None;
        }
        if within(p, IVec3::new(x0 - 1, 33, 9), IVec3::new(x0 + 4, 40, 41)) {
            return Some(STONE);
        }
    }
    // The overhang's shelf.
    if within(p, IVec3::new(-30, 49, -40), IVec3::new(0, 53, -26)) {
        return Some(STONE);
    }
    // The floating island.
    if within(p, IVec3::new(10, 45, -22), IVec3::new(40, 51, 8)) {
        return Some(match p.y {
            50 => GRASS,
            47..=49 => DIRT,
            _ => STONE,
        });
    }
    // Three stages open to +z, their back walls two thick at z = 3 and 4, so a lamp in one
    // lights only the stage: plaster walls and roof, a plank floor.
    for i in 0..3 {
        let x0 = stage_x0(i);
        if within(p, IVec3::new(x0, 39, 5), IVec3::new(x0 + 10, 47, 15)) {
            return if p.y == 39 { Some(FLOOR) } else { None };
        }
        let back = if FLAGS.thin { 4 } else { 3 };
        if within(p, IVec3::new(x0 - 1, 40, back), IVec3::new(x0 + 11, 48, 15)) {
            return Some(PLASTER);
        }
    }
    let top = if p.z < CLIFF { HILL_TOP } else { GROUND };
    if p.y >= top {
        return None;
    }
    Some(if p.y == top - 1 {
        GRASS
    } else if p.y >= top - 4 {
        DIRT
    } else {
        STONE
    })
}

fn texture_indices(material: u8) -> [u32; 3] {
    match material {
        GRASS => [0, 1, 2],
        DIRT => [2, 2, 2],
        PLASTER => [4, 4, 4],
        FLOOR => [16, 16, 16],
        m if m >= LAMP_BASE => [13, 13, 13],
        _ => [3, 3, 3],
    }
}

// --- The world --------------------------------------------------------------------

#[derive(Resource, Clone)]
struct ColourWorld {
    light: LightPropagation,
}

impl Default for ColourWorld {
    fn default() -> Self {
        Self {
            light: LightPropagation::default(),
        }
    }
}

impl VoxelWorldConfig for ColourWorld {
    type MaterialIndex = u8;
    type ChunkUserBundle = ();

    fn voxel_lookup_delegate(&self) -> VoxelLookupDelegate<Self::MaterialIndex> {
        Box::new(|_, _, _| {
            Box::new(|position, _| match scene_voxel(position) {
                Some(m) => WorldVoxel::Solid(m),
                None => WorldVoxel::Air,
            })
        })
    }

    fn chunk_spawn_strategy(&self) -> ChunkSpawnStrategy {
        ChunkSpawnStrategy::Close
    }

    fn chunk_despawn_strategy(&self) -> ChunkDespawnStrategy {
        ChunkDespawnStrategy::FarAway
    }

    fn chunk_fill_hint(&self, key: ChunkKey) -> FillHint {
        if key.position.y + key.extent() <= 0 {
            FillHint::Uniform(WorldVoxel::Solid(STONE))
        } else if key.position.y >= 3 {
            FillHint::Uniform(WorldVoxel::Air)
        } else {
            FillHint::Unknown
        }
    }

    fn texture_index_mapper(&self) -> Arc<dyn Fn(Self::MaterialIndex) -> [u32; 3] + Send + Sync> {
        Arc::new(texture_indices)
    }

    fn light_propagation(&self) -> Option<LightPropagation> {
        Some(self.light.clone())
    }

    fn emissive_materials(&self) -> Vec<(Self::MaterialIndex, Emission)> {
        (0..KINDS.len())
            .map(|k| (LAMP_BASE + k as u8, Emission::level(KINDS[k].level as f32)))
            .collect()
    }
}

// --- The material -----------------------------------------------------------------

#[derive(ShaderType, Clone, Debug, Default)]
struct Params {
    box_min: IVec4,
    box_size: IVec4,
    today_min: IVec4,
    today_max: IVec4,
    candidate: u32,
    glow: u32,
    view: u32,
    smoothed: u32,
    emissive_glow: f32,
    voxel_brightness: f32,
    pad0: f32,
    pad1: f32,
}

#[derive(Asset, AsBindGroup, TypePath, Debug, Clone)]
struct ColourExt {
    #[texture(100, dimension = "2d_array")]
    #[sampler(101)]
    voxels_texture: Handle<Image>,
    #[texture(300, dimension = "3d", sample_type = "u_int")]
    colour: Handle<Image>,
    #[uniform(301)]
    params: Params,
}

impl MaterialExtension for ColourExt {
    fn vertex_shader() -> ShaderRef {
        SHADER.into()
    }

    fn fragment_shader() -> ShaderRef {
        SHADER.into()
    }

    fn enable_prepass() -> bool {
        false
    }

    fn specialize(
        _pipeline: &MaterialExtensionPipeline,
        descriptor: &mut RenderPipelineDescriptor,
        layout: &MeshVertexBufferLayoutRef,
        _key: MaterialExtensionKey<Self>,
    ) -> Result<(), SpecializedMeshPipelineError> {
        if descriptor
            .vertex
            .shader_defs
            .contains(&ShaderDefVal::Bool("PREPASS_PIPELINE".into(), true))
        {
            return Ok(());
        }
        let vertex_layout = layout.0.get_layout(&vertex_layout())?;
        descriptor.vertex.buffers = vec![vertex_layout];
        Ok(())
    }
}

type ColourMaterial = ExtendedMaterial<StandardMaterial, ColourExt>;

// --- The texture the probe's light goes in ----------------------------------------

const BOX_MIN: IVec3 = IVec3::new(-112, 0, -112);
const BOX_SIZE: IVec3 = IVec3::new(240, 96, 176);
/// Levels are stored times this, so 15 fits a u16.
const FIXED: f32 = 4096.0;

fn texel(p: IVec3) -> Option<usize> {
    let q = p - BOX_MIN;
    if q.cmplt(IVec3::ZERO).any() || q.cmpge(BOX_SIZE).any() {
        return None;
    }
    Some((q.x + q.y * BOX_SIZE.x + q.z * BOX_SIZE.x * BOX_SIZE.y) as usize)
}

// --- Solving ----------------------------------------------------------------------

/// Which voxels are open, over the box and the margin kernel boxes read.
struct Open {
    min: IVec3,
    size: IVec3,
    bits: Vec<u64>,
}

impl Open {
    fn new() -> Self {
        let min = BOX_MIN - IVec3::splat(64);
        let size = BOX_SIZE + IVec3::splat(128);
        let count = (size.x * size.y * size.z) as usize;
        let mut bits = vec![0u64; count.div_ceil(64)];
        for z in 0..size.z {
            for y in 0..size.y {
                for x in 0..size.x {
                    if scene_voxel(min + IVec3::new(x, y, z)).is_none() {
                        let i = (x + y * size.x + z * size.x * size.y) as usize;
                        bits[i / 64] |= 1 << (i % 64);
                    }
                }
            }
        }
        Self { min, size, bits }
    }

    fn at(&self, p: IVec3) -> bool {
        let q = p - self.min;
        if q.cmplt(IVec3::ZERO).any() || q.cmpge(self.size).any() {
            return scene_voxel(p).is_none();
        }
        let i = (q.x + q.y * self.size.x + q.z * self.size.x * self.size.y) as usize;
        self.bits[i / 64] >> (i % 64) & 1 == 1
    }
}

/// The brightness of each whole level under a falloff, as `LevelCurve` holds
/// it, level 0 black.
fn table(linear: bool, to_zero: bool) -> [f32; 16] {
    std::array::from_fn(|level| to_zero_or(linear, to_zero, level as u8))
}

/// The brightness of `level` under the default curve, the linear one, or the
/// default with level 1's brightness taken off and stretched back to 1 at the
/// top, so the last lit level is black and light fades out instead of
/// stopping at 0.7^14.
fn to_zero_or(linear: bool, to_zero: bool, level: u8) -> f32 {
    let exp = |l: u8| (l..15).fold(1.0f32, |b, _| b * 0.7);
    if level == 0 {
        0.0
    } else if linear {
        level as f32 / 15.0
    } else if to_zero {
        (exp(level) - exp(1)) / (1.0 - exp(1))
    } else {
        exp(level)
    }
}

/// `LevelCurve::at`.
fn at(t: &[f32; 16], level: f32) -> f32 {
    if level <= 0.0 {
        return 0.0;
    }
    let level = level.min(15.0);
    let low = level.floor() as usize;
    let high = (low + 1).min(15);
    let f = level - low as f32;
    t[low] * (1.0 - f) + t[high] * f
}

/// The level whose brightness is `b`, between whole levels as `at` reads them.
fn inverse(t: &[f32; 16], b: f32) -> f32 {
    if b <= 0.0 {
        return 0.0;
    }
    if b >= t[15] {
        return 15.0;
    }
    for low in 0..15 {
        if b < t[low + 1] {
            return low as f32 + (b - t[low]) / (t[low + 1] - t[low]);
        }
    }
    15.0
}

/// The whole level a lamp of `kind` seeds component `k` at, under the
/// per-component candidate, or 0 for none.
fn component_seed(kind: usize, k: usize, t: &[f32; 16]) -> u8 {
    let lamp = &KINDS[kind];
    inverse(t, lamp.colour[k] * at(t, lamp.level as f32)).round() as u8
}

#[derive(Clone, Copy, PartialEq, Debug)]
struct SolveKey {
    p256: bool,
    reach: u32,
    linear: bool,
    to_zero: bool,
}

struct ChunkRuns {
    key: IVec3,
    /// A run per light colour: the colour's id and the level at each voxel.
    groups: Vec<(usize, Vec<u8>)>,
    /// A run per component, empty where no lamp seeds it.
    components: [Vec<u8>; 3],
}

struct Solved {
    key: SolveKey,
    chunks: Vec<ChunkRuns>,
}

fn run(open: &Open, key: SolveKey, origin: IVec3, sources: &[(IVec3, u8)]) -> Vec<u8> {
    let reach = key.reach as i32;
    let box_min = origin - IVec3::splat(reach);
    let local: Vec<(IVec3, u8)> = sources.iter().map(|&(p, l)| (p - box_min, l)).collect();
    let mut kernel = KernelBox::new(key.reach, |p| open.at(box_min + p), &local);
    if key.p256 {
        kernel = kernel.at_256_levels();
    }
    kernel.spread(LightShape::Round);
    let mut levels = vec![0u8; 32 * 32 * 32];
    for z in 0..32u32 {
        for y in 0..32u32 {
            for x in 0..32u32 {
                levels[(x + y * 32 + z * 1024) as usize] = kernel.level(UVec3::new(x, y, z));
            }
        }
    }
    levels
}

fn solve(open: &Open, key: SolveKey) -> Solved {
    let started = std::time::Instant::now();
    let t = table(key.linear, key.to_zero);
    let reach = key.reach as i32;
    let mut chunks = Vec::new();
    let mut runs = 0;
    let first = BOX_MIN.div_euclid(IVec3::splat(32));
    let last = (BOX_MIN + BOX_SIZE - IVec3::ONE).div_euclid(IVec3::splat(32));
    for cz in first.z..=last.z {
        for cy in first.y..=last.y {
            for cx in first.x..=last.x {
                let chunk = IVec3::new(cx, cy, cz);
                let origin = chunk * 32;
                let lamps: Vec<(IVec3, usize)> = lamps()
                    .filter(|(p, _)| {
                        within(
                            *p,
                            origin - IVec3::splat(reach),
                            origin + IVec3::splat(32 + reach),
                        )
                    })
                    .collect();
                if lamps.is_empty() {
                    continue;
                }
                let mut colours: Vec<usize> = lamps.iter().map(|&(_, k)| colour_id(k)).collect();
                colours.sort();
                colours.dedup();
                let groups = colours
                    .into_iter()
                    .map(|colour| {
                        let sources: Vec<(IVec3, u8)> = lamps
                            .iter()
                            .filter(|&&(_, k)| colour_id(k) == colour)
                            .map(|&(p, k)| (p, KINDS[k].level))
                            .collect();
                        runs += 1;
                        (colour, run(open, key, origin, &sources))
                    })
                    .collect();
                let components = std::array::from_fn(|c| {
                    let sources: Vec<(IVec3, u8)> = lamps
                        .iter()
                        .map(|&(p, k)| (p, component_seed(k, c, &t)))
                        .filter(|&(_, l)| l > 0)
                        .collect();
                    if sources.is_empty() {
                        Vec::new()
                    } else {
                        runs += 1;
                        run(open, key, origin, &sources)
                    }
                });
                chunks.push(ChunkRuns { key: chunk, groups, components });
            }
        }
    }
    println!(
        "solved {:?}: {} chunks, {} runs, {:.0} ms",
        key,
        chunks.len(),
        runs,
        started.elapsed().as_secs_f64() * 1000.0
    );
    Solved { key, chunks }
}

const CANDIDATES: [&str; 6] = [
    "today (the crate, every lamp white)",
    "per-component levels",
    "groups, max per component",
    "groups, sum",
    "groups, blend by brightness",
    "groups, blend by level",
];

const CANDIDATE_NOTES: [&str; 6] = [
    "The crate's own light: each lamp at its level, in white.",
    "Three runs, red, green and blue. A lamp seeds each component at the level whose brightness is its colour times its own, so weaker components start lower and end sooner. Max per component.",
    "A run per light colour, so hue and reach hold. Each component is the max over runs of colour x brightness.",
    "A run per light colour. Each component is the sum over runs of colour x brightness, capped at 1.",
    "A run per light colour. The brightest run's level, in the runs' colours averaged with brightness as weights.",
    "A run per light colour. The brightest run's level, in the runs' colours averaged with 0.7^(levels below the brightest) as weights, which ignore the falloff.",
];

/// The light at each voxel of the box, as levels times `FIXED`, red, green,
/// blue and a lamp's own level in alpha. A lamp's cell holds its own colour
/// as levels, which the shader reads for a lamp's neighbours and its glow.
fn compose(solved: &Solved, candidate: usize, data: &mut [u16]) {
    data.fill(0);
    let div = if solved.key.p256 { 17.0 } else { 1.0 };
    let t = table(solved.key.linear, solved.key.to_zero);
    let colours: Vec<[f32; 3]> = KINDS.iter().map(|k| k.colour).collect();
    let mix = |groups: &[(usize, f32)]| -> [f32; 3] {
        if groups.is_empty() {
            return [0.0; 3];
        }
        let brightest = groups.iter().map(|g| g.1).fold(0.0, f32::max);
        [0, 1, 2].map(|c| inverse(&t, bright(groups, c, candidate, &t, &colours, brightest)))
    };
    for chunk in &solved.chunks {
        let origin = chunk.key * 32;
        let mut groups: Vec<(usize, f32)> = Vec::with_capacity(8);
        for z in 0..32 {
            for y in 0..32 {
                for x in 0..32 {
                    let local = IVec3::new(x, y, z);
                    let Some(i) = texel(origin + local) else { continue };
                    let li = (x + y * 32 + z * 1024) as usize;
                    let rgb = match candidate {
                        0 => {
                            let l = chunk.groups.iter().map(|g| g.1[li]).max().unwrap_or(0) as f32 / div;
                            [l; 3]
                        }
                        1 => [0, 1, 2].map(|c| {
                            chunk.components[c].get(li).copied().unwrap_or(0) as f32 / div
                        }),
                        _ => {
                            groups.clear();
                            for (id, levels) in &chunk.groups {
                                if levels[li] > 0 {
                                    groups.push((*id, levels[li] as f32 / div));
                                }
                            }
                            mix(&groups)
                        }
                    };
                    for c in 0..3 {
                        data[i * 4 + c] = (rgb[c] * FIXED).round() as u16;
                    }
                }
            }
        }
    }
    for (p, kind) in lamps() {
        let Some(i) = texel(p) else { continue };
        let lamp = &KINDS[kind];
        let level = lamp.level as f32;
        let own = match candidate {
            0 => [level; 3],
            1 => [0, 1, 2].map(|c| component_seed(kind, c, &t) as f32),
            _ => [0, 1, 2].map(|c| inverse(&t, lamp.colour[c] * at(&t, level))),
        };
        for c in 0..3 {
            data[i * 4 + c] = (own[c] * FIXED).round() as u16;
        }
        data[i * 4 + 3] = (level * FIXED).round() as u16;
    }
}

/// One component's brightness where runs of light colours `groups` (colour
/// id and level) meet, under a group candidate.
fn bright(
    groups: &[(usize, f32)],
    c: usize,
    candidate: usize,
    t: &[f32; 16],
    colours: &[[f32; 3]],
    brightest: f32,
) -> f32 {
    match candidate {
        2 => groups
            .iter()
            .map(|&(id, l)| colours[id][c] * at(t, l))
            .fold(0.0, f32::max),
        3 => groups
            .iter()
            .map(|&(id, l)| colours[id][c] * at(t, l))
            .sum::<f32>()
            .min(1.0),
        _ => {
            let weight = |l: f32| {
                if candidate == 4 {
                    at(t, l)
                } else {
                    0.7f32.powf(brightest - l)
                }
            };
            let total: f32 = groups.iter().map(|&(_, l)| weight(l)).sum();
            let colour: f32 = groups
                .iter()
                .map(|&(id, l)| colours[id][c] * weight(l))
                .sum::<f32>()
                / total.max(1e-9);
            colour * at(t, brightest)
        }
    }
}

// --- Places -----------------------------------------------------------------------

const PLACES: [(&str, Vec3, Vec3); 7] = [
    ("cave", Vec3::new(-64.0, 49.0, -53.0), Vec3::new(-64.0, 43.0, -74.0)),
    ("overhang", Vec3::new(-15.0, 43.0, -10.0), Vec3::new(-15.0, 46.0, -38.0)),
    ("island", Vec3::new(25.0, 41.5, 12.0), Vec3::new(25.0, 42.5, -12.0)),
    ("stages", Vec3::new(70.0, 48.0, 50.0), Vec3::new(70.0, 43.0, 6.0)),
    ("trenches", Vec3::new(-49.0, 64.0, 34.0), Vec3::new(-49.0, 36.0, 13.0)),
    ("overview", Vec3::new(0.0, 125.0, 135.0), Vec3::new(0.0, 40.0, -20.0)),
    ("stages, outside", Vec3::new(104.0, 60.0, 30.0), Vec3::new(80.0, 44.0, 8.0)),
];

fn orbit_at(orbit: &mut PanOrbitCamera, eye: Vec3, target: Vec3) {
    let offset = eye - target;
    let radius = offset.length();
    let yaw = offset.x.atan2(offset.z);
    let pitch = (offset.y / radius).asin();
    orbit.focus = target;
    orbit.target_focus = target;
    orbit.radius = Some(radius);
    orbit.target_radius = radius;
    orbit.yaw = Some(yaw);
    orbit.target_yaw = yaw;
    orbit.pitch = Some(pitch);
    orbit.target_pitch = pitch;
    orbit.force_update = true;
}

// --- State ------------------------------------------------------------------------

#[derive(Resource)]
struct Probe {
    candidate: usize,
    glow: usize,
    view: usize,
    night: bool,
    linear: bool,
    to_zero: bool,
    p256: bool,
    reach32: bool,
    smoothed: bool,
    bright: bool,
    tonemap: usize,
    panel: bool,
    place: usize,
    open: Open,
    solved: Option<Solved>,
    composed: Option<(SolveKey, usize)>,
    data: Vec<u16>,
}

impl Probe {
    fn key(&self) -> SolveKey {
        SolveKey {
            p256: self.p256,
            reach: if self.reach32 { 32 } else { 14 },
            linear: self.linear,
            to_zero: self.to_zero,
        }
    }
}

const TONEMAPS: [(&str, Tonemapping); 4] = [
    ("TonyMcMapface (Bevy's default)", Tonemapping::TonyMcMapface),
    ("none", Tonemapping::None),
    ("AgX", Tonemapping::AgX),
    ("Reinhard on luminance", Tonemapping::ReinhardLuminance),
];

const GLOWS: [&str; 3] = ["the material's own colour (today)", "the light colour", "both: light colour x material"];
const VIEWS: [&str; 3] = ["look", "light only (white surfaces)", "check: probe's white against the crate's, red where they differ"];

#[derive(Component)]
struct Sun;

#[derive(Component)]
struct Hud;

struct Shot {
    name: &'static str,
    place: usize,
    candidate: usize,
    glow: usize,
    night: bool,
    linear: bool,
    p256: bool,
    reach32: bool,
    view: usize,
    bright: bool,
    tonemap: usize,
    to_zero: bool,
}

const fn shot(name: &'static str, place: usize, candidate: usize) -> Shot {
    Shot { name, place, candidate, glow: 1, night: true, linear: false, p256: false, reach32: false, view: 0, bright: false, tonemap: 0, to_zero: false }
}

const SHOTS: &[Shot] = &[
    Shot { view: 2, ..shot("check_cave", 0, 0) },
    Shot { view: 2, ..shot("check_stages", 3, 0) },
    shot("cave_0_today", 0, 0),
    shot("cave_1_component", 0, 1),
    shot("cave_2_max", 0, 2),
    shot("cave_3_sum", 0, 3),
    shot("cave_4_blend", 0, 4),
    shot("cave_5_blendlevel", 0, 5),
    Shot { bright: true, ..shot("bright_cave_1_component", 0, 1) },
    Shot { bright: true, ..shot("bright_cave_2_max", 0, 2) },
    Shot { bright: true, ..shot("bright_cave_3_sum", 0, 3) },
    Shot { bright: true, ..shot("bright_cave_4_blend", 0, 4) },
    shot("overhang_0_today", 1, 0),
    shot("overhang_1_component", 1, 1),
    shot("overhang_2_max", 1, 2),
    shot("overhang_3_sum", 1, 3),
    shot("overhang_4_blend", 1, 4),
    shot("island_0_today", 2, 0),
    shot("island_1_component", 2, 1),
    shot("island_2_max", 2, 2),
    shot("island_3_sum", 2, 3),
    shot("island_4_blend", 2, 4),
    shot("stages_1_component", 3, 1),
    shot("stages_2_max", 3, 2),
    shot("stages_3_sum", 3, 3),
    shot("stages_4_blend", 3, 4),
    Shot { glow: 0, ..shot("stages_2_max_glow_material", 3, 2) },
    Shot { glow: 2, ..shot("stages_2_max_glow_both", 3, 2) },
    shot("trenches_0_today", 4, 0),
    shot("trenches_1_component", 4, 1),
    shot("trenches_2_max", 4, 2),
    Shot { bright: true, ..shot("bright_trenches_1_component", 4, 1) },
    Shot { bright: true, ..shot("bright_trenches_2_max", 4, 2) },
    Shot { view: 1, ..shot("light_trenches_1_component", 4, 1) },
    Shot { view: 1, ..shot("light_trenches_2_max", 4, 2) },
    Shot { glow: 0, ..shot("cave_2_max_glow_material", 0, 2) },
    Shot { glow: 2, ..shot("cave_2_max_glow_both", 0, 2) },
    Shot { night: false, ..shot("day_cave_2_max", 0, 2) },
    Shot { night: false, ..shot("day_overhang_1_component", 1, 1) },
    Shot { night: false, ..shot("day_overhang_2_max", 1, 2) },
    Shot { night: false, ..shot("day_island_1_component", 2, 1) },
    Shot { night: false, ..shot("day_island_2_max", 2, 2) },
    Shot { night: false, ..shot("day_stages_2_max", 3, 2) },
    Shot { linear: true, ..shot("linear_trenches_0_today", 4, 0) },
    Shot { linear: true, ..shot("linear_trenches_1_component", 4, 1) },
    Shot { linear: true, ..shot("linear_trenches_2_max", 4, 2) },
    Shot { linear: true, ..shot("linear_cave_1_component", 0, 1) },
    Shot { linear: true, ..shot("linear_cave_2_max", 0, 2) },
    Shot { linear: true, ..shot("linear_cave_3_sum", 0, 3) },
    Shot { linear: true, ..shot("linear_cave_4_blend", 0, 4) },
    Shot { linear: true, ..shot("linear_cave_5_blendlevel", 0, 5) },
    Shot { reach32: true, ..shot("r32_trenches_1_component", 4, 1) },
    Shot { reach32: true, ..shot("r32_trenches_2_max", 4, 2) },
    Shot { reach32: true, p256: true, ..shot("r32_256_trenches_1_component", 4, 1) },
    Shot { reach32: true, p256: true, ..shot("r32_256_trenches_2_max", 4, 2) },
    Shot { reach32: true, p256: true, ..shot("r32_256_cave_2_max", 0, 2) },
    Shot { reach32: true, ..shot("r32_cave_2_max", 0, 2) },
    shot("outside_0_today", 6, 0),
    shot("outside_2_max", 6, 2),
    Shot { view: 1, ..shot("outside_0_today_light", 6, 0) },
    shot("tm0_cave_2_max", 0, 2),
    Shot { tonemap: 1, ..shot("tm1_cave_2_max", 0, 2) },
    Shot { tonemap: 2, ..shot("tm2_cave_2_max", 0, 2) },
    Shot { tonemap: 3, ..shot("tm3_cave_2_max", 0, 2) },
    shot("tm0_outside_2_max", 6, 2),
    Shot { tonemap: 1, ..shot("tm1_outside_2_max", 6, 2) },
    Shot { tonemap: 2, ..shot("tm2_outside_2_max", 6, 2) },
    Shot { tonemap: 3, ..shot("tm3_outside_2_max", 6, 2) },
    shot("tm0_stages_2_max", 3, 2),
    Shot { tonemap: 1, ..shot("tm1_stages_2_max", 3, 2) },
    Shot { tonemap: 2, ..shot("tm2_stages_2_max", 3, 2) },
    Shot { tonemap: 3, ..shot("tm3_stages_2_max", 3, 2) },
    shot("seam_max", 6, 2),
    Shot { view: 1, ..shot("seam_max_light", 6, 2) },
    Shot { to_zero: true, ..shot("seam_tozero", 6, 2) },
    Shot { to_zero: true, view: 1, ..shot("seam_tozero_light", 6, 2) },
    Shot { view: 1, ..shot("seam_today_light", 6, 0) },
    shot("zero_cave_0_today", 0, 0),
    Shot { to_zero: true, ..shot("zero_cave_0_today_tozero", 0, 0) },
    shot("zero_trenches_0_today", 4, 0),
    Shot { to_zero: true, ..shot("zero_trenches_0_today_tozero", 4, 0) },
    shot("zero_cave_2_max", 0, 2),
    Shot { to_zero: true, ..shot("zero_cave_2_max_tozero", 0, 2) },
];

#[derive(Resource)]
struct Shots {
    out: String,
    only: Option<String>,
    index: usize,
    frames: u32,
    wait: u32,
    taken: bool,
}

// --- Systems ----------------------------------------------------------------------

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let shots = args.iter().any(|a| a == "shots");
    let out = args
        .iter()
        .find_map(|a| a.strip_prefix("out=").map(str::to_string))
        .unwrap_or_else(|| "colour_look_shots".into());
    let only = args.iter().find_map(|a| a.strip_prefix("only=").map(str::to_string));

    let material = ColourMaterial {
        base: StandardMaterial {
            reflectance: 0.05,
            metallic: 0.05,
            perceptual_roughness: 0.95,
            ..default()
        },
        extension: ColourExt {
            voxels_texture: VOXEL_TEXTURE,
            colour: COLOUR_TEXTURE,
            params: Params::default(),
        },
    };

    let mut app = App::new();
    app.add_plugins((
        DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "coloured light probe".into(),
                resolution: (1400, 860).into(),
                ..default()
            }),
            ..default()
        }),
        PanOrbitCameraPlugin,
        MaterialPlugin::<ColourMaterial>::default(),
    ))
    .add_plugins(VoxelWorldPlugin::with_config(ColourWorld::default()).with_material(material))
    .insert_resource(ClearColor(Color::srgb(0.02, 0.02, 0.03)))
    .add_systems(Startup, setup)
    .add_systems(Update, (keys, apply, push_params, hud).chain());
    if shots {
        std::fs::create_dir_all(&out).unwrap();
        app.insert_resource(Shots { out, only, index: 0, frames: 0, wait: 400, taken: false });
        app.add_systems(Update, take_shots.before(keys));
    }
    app.world_mut()
        .resource_mut::<Assets<Shader>>()
        .insert(
            SHADER.id(),
            Shader::from_wesl(
                include_str!("../../../probes/colour_look.wesl"),
                "embedded://colour_look/colour_look.wesl",
            ),
        )
        .unwrap();
    app.run();
}

fn setup(mut commands: Commands, mut images: ResMut<Assets<Image>>) {
    let mut image = Image::from_buffer(
        include_bytes!("../assets/mountain_house_texture.png"),
        ImageType::MimeType("image/png"),
        CompressedImageFormats::default(),
        true,
        ImageSampler::Descriptor(ImageSamplerDescriptor {
            address_mode_u: ImageAddressMode::Repeat,
            address_mode_v: ImageAddressMode::Repeat,
            address_mode_w: ImageAddressMode::Repeat,
            ..ImageSamplerDescriptor::nearest()
        }),
        RenderAssetUsages::default(),
    )
    .unwrap();
    image.reinterpret_stacked_2d_as_array(18).unwrap();
    image.texture_view_descriptor = Some(TextureViewDescriptor {
        dimension: Some(TextureViewDimension::D2Array),
        ..default()
    });
    images.insert(VOXEL_TEXTURE.id(), image).unwrap();

    let count = (BOX_SIZE.x * BOX_SIZE.y * BOX_SIZE.z) as usize;
    let colour = Image::new(
        Extent3d {
            width: BOX_SIZE.x as u32,
            height: BOX_SIZE.y as u32,
            depth_or_array_layers: BOX_SIZE.z as u32,
        },
        TextureDimension::D3,
        vec![0u8; count * 8],
        TextureFormat::Rgba16Uint,
        RenderAssetUsages::default(),
    );
    images.insert(COLOUR_TEXTURE.id(), colour).unwrap();

    let started = std::time::Instant::now();
    let open = Open::new();
    println!("open voxels in {:.0} ms", started.elapsed().as_secs_f64() * 1000.0);
    commands.insert_resource(Probe {
        candidate: 2,
        glow: 1,
        view: 0,
        night: true,
        linear: false,
        to_zero: false,
        p256: false,
        reach32: false,
        smoothed: true,
        bright: false,
        tonemap: 0,
        panel: true,
        place: 0,
        open,
        solved: None,
        composed: None,
        data: vec![0; count * 4],
    });

    let mut orbit = PanOrbitCamera {
        button_orbit: MouseButton::Right,
        button_pan: MouseButton::Middle,
        ..default()
    };
    let (_, eye, target) = PLACES[0];
    orbit_at(&mut orbit, eye, target);
    commands.spawn((
        Camera3d::default(),
        Transform::from_translation(eye).looking_at(target, Vec3::Y),
        orbit,
        VoxelWorldCamera::<ColourWorld>::default(),
    ));
    commands.spawn((
        DirectionalLight {
            color: Color::srgb(0.98, 0.95, 0.86),
            shadow_maps_enabled: true,
            ..default()
        },
        Transform::default().looking_at(Vec3::new(-0.35, -0.7, -0.6), Vec3::Y),
        CascadeShadowConfigBuilder {
            maximum_distance: 260.0,
            ..default()
        }
        .build(),
        Sun,
    ));
    commands.insert_resource(GlobalAmbientLight {
        color: Color::srgb(0.9, 0.94, 1.0),
        brightness: 600.0,
        affects_lightmapped_meshes: true,
    });
    commands.spawn((
        Text::new(""),
        TextFont {
            font_size: FontSize::Px(14.0),
            ..default()
        },
        TextColor(Color::WHITE),
        BackgroundColor(Color::srgba(0.0, 0.0, 0.0, 0.6)),
        Node {
            position_type: PositionType::Absolute,
            left: Val::Px(8.0),
            top: Val::Px(8.0),
            max_width: Val::Px(560.0),
            padding: UiRect::all(Val::Px(6.0)),
            ..default()
        },
        Hud,
    ));
}

fn keys(
    input: Res<ButtonInput<KeyCode>>,
    mut probe: ResMut<Probe>,
    mut orbits: Query<&mut PanOrbitCamera>,
) {
    let shift = input.pressed(KeyCode::ShiftLeft) || input.pressed(KeyCode::ShiftRight);
    if input.just_pressed(KeyCode::KeyC) {
        let n = CANDIDATES.len();
        probe.candidate = if shift { (probe.candidate + n - 1) % n } else { (probe.candidate + 1) % n };
    }
    if input.just_pressed(KeyCode::KeyG) {
        probe.glow = (probe.glow + 1) % GLOWS.len();
    }
    if input.just_pressed(KeyCode::KeyV) {
        probe.view = (probe.view + 1) % VIEWS.len();
    }
    if input.just_pressed(KeyCode::KeyN) {
        probe.night = !probe.night;
    }
    if input.just_pressed(KeyCode::KeyF) {
        let (linear, to_zero) = match (probe.linear, probe.to_zero) {
            (false, false) => (true, false),
            (true, _) => (false, true),
            _ => (false, false),
        };
        probe.linear = linear;
        probe.to_zero = to_zero;
    }
    if input.just_pressed(KeyCode::KeyP) {
        probe.p256 = !probe.p256;
    }
    if input.just_pressed(KeyCode::KeyR) {
        probe.reach32 = !probe.reach32;
    }
    if input.just_pressed(KeyCode::KeyM) {
        probe.smoothed = !probe.smoothed;
    }
    if input.just_pressed(KeyCode::KeyT) {
        probe.tonemap = (probe.tonemap + 1) % TONEMAPS.len();
    }
    if input.just_pressed(KeyCode::KeyB) {
        probe.bright = !probe.bright;
    }
    if input.just_pressed(KeyCode::KeyH) {
        probe.panel = !probe.panel;
    }
    let digits = [
        KeyCode::Digit1,
        KeyCode::Digit2,
        KeyCode::Digit3,
        KeyCode::Digit4,
        KeyCode::Digit5,
        KeyCode::Digit6,
        KeyCode::Digit7,
    ];
    for (i, key) in digits.iter().enumerate() {
        if input.just_pressed(*key) {
            probe.place = i;
            let (_, eye, target) = PLACES[i];
            for mut orbit in orbits.iter_mut() {
                orbit_at(&mut orbit, eye, target);
            }
        }
    }
}

/// Keeps the world's config, the probe's solve and texture, and day or night
/// in step with the probe's settings.
fn apply(
    mut probe: ResMut<Probe>,
    mut world: ResMut<ColourWorld>,
    mut images: ResMut<Assets<Image>>,
    mut ambient: ResMut<GlobalAmbientLight>,
    mut clear: ResMut<ClearColor>,
    mut sun: Query<&mut DirectionalLight, With<Sun>>,
    mut tonemapping: Query<&mut Tonemapping>,
) {
    for mut t in tonemapping.iter_mut() {
        if *t != TONEMAPS[probe.tonemap].1 {
            *t = TONEMAPS[probe.tonemap].1;
        }
    }
    let key = probe.key();
    let precision = if key.p256 { LightPrecision::Steps256 } else { LightPrecision::Steps16 };
    let curve = if key.linear {
        LevelCurve::linear()
    } else if key.to_zero {
        LevelCurve::from_fn(|l| to_zero_or(false, true, l))
    } else {
        LevelCurve::default()
    };
    let smoothing = if probe.smoothed { LightSmoothing::Smoothed } else { LightSmoothing::Stepped };
    let brightness_of_voxels = if probe.bright { 4000.0 } else { 1000.0 };
    let light = &world.light;
    if light.look.voxel_brightness != brightness_of_voxels {
        world.light.look.voxel_brightness = brightness_of_voxels;
    }
    let light = &world.light;
    if light.precision != precision
        || light.voxel_reach as u32 != key.reach
        || light.look.voxel_falloff != curve
        || light.look.smoothing != smoothing
    {
        let light = &mut world.light;
        light.precision = precision;
        light.voxel_reach = key.reach as u8;
        light.look.voxel_falloff = curve;
        light.look.smoothing = smoothing;
    }
    if probe.solved.as_ref().map(|s| s.key) != Some(key) {
        let solved = solve(&probe.open, key);
        probe.solved = Some(solved);
    }
    if probe.composed != Some((key, probe.candidate)) {
        let started = std::time::Instant::now();
        let candidate = probe.candidate;
        let Probe { solved, data, .. } = &mut *probe;
        compose(solved.as_ref().unwrap(), candidate, data);
        if let Some(mut image) = images.get_mut(&COLOUR_TEXTURE) {
            let bytes: Vec<u8> = probe.data.iter().flat_map(|v| v.to_le_bytes()).collect();
            image.data = Some(bytes);
        }
        probe.composed = Some((key, candidate));
        println!(
            "composed {} in {:.0} ms",
            CANDIDATES[candidate],
            started.elapsed().as_secs_f64() * 1000.0
        );
    }
    let (brightness, illuminance, sky) = if probe.night {
        (0.0, 0.0, Color::srgb(0.02, 0.02, 0.03))
    } else {
        (600.0, light_consts::lux::AMBIENT_DAYLIGHT, Color::srgb(0.58, 0.74, 0.92))
    };
    if ambient.brightness != brightness {
        ambient.brightness = brightness;
    }
    if clear.0 != sky {
        clear.0 = sky;
    }
    for mut s in sun.iter_mut() {
        if s.illuminance != illuminance {
            s.illuminance = illuminance;
        }
    }
}

fn push_params(
    probe: Res<Probe>,
    world: Res<ColourWorld>,
    handle: Option<Res<VoxelWorldMaterialHandle<VoxelWorldMaterial<ColourMaterial>>>>,
    mut materials: ResMut<Assets<VoxelWorldMaterial<ColourMaterial>>>,
) {
    let Some(handle) = handle else { return };
    let look = &world.light.look;
    let params = Params {
        box_min: BOX_MIN.extend(0),
        box_size: BOX_SIZE.extend(0),
        today_min: TODAY_MIN.extend(0),
        today_max: TODAY_MAX.extend(0),
        candidate: probe.candidate as u32,
        glow: probe.glow as u32,
        view: probe.view as u32,
        smoothed: probe.smoothed as u32,
        emissive_glow: look.emissive_glow,
        voxel_brightness: look.voxel_brightness,
        pad0: 0.0,
        pad1: 0.0,
    };
    let Some(material) = materials.get(&handle.handle) else { return };
    let current = &material.base.extension.params;
    if current.candidate != params.candidate
        || current.glow != params.glow
        || current.view != params.view
        || current.smoothed != params.smoothed
        || current.box_size != params.box_size
        || current.emissive_glow != params.emissive_glow
        || current.voxel_brightness != params.voxel_brightness
    {
        if let Some(mut material) = materials.get_mut(&handle.handle) {
            material.base.extension.params = params;
        }
    }
}

fn hud(probe: Res<Probe>, mut text: Query<(&mut Text, &mut Visibility), With<Hud>>) {
    for (mut text, mut visibility) in text.iter_mut() {
        *visibility = if probe.panel { Visibility::Visible } else { Visibility::Hidden };
        let key = probe.key();
        text.0 = format!(
            "Coloured light probe: {} (1-6)\n\
             C  {}\n   {}\n\
             G  glow: {}\n\
             V  view: {}\n\
             N  {}   F  falloff: {}   M  smoothing: {}\n\
             P  {} levels   R  voxel reach {}   B  voxel brightness x{}   H  hide\n\
             T  tonemapping: {}",
            PLACES[probe.place].0,
            CANDIDATES[probe.candidate],
            CANDIDATE_NOTES[probe.candidate],
            GLOWS[probe.glow],
            VIEWS[probe.view],
            if probe.night { "night" } else { "day" },
            if key.linear { "linear" } else if key.to_zero { "0.7^(15 - level), level 1 black" } else { "0.7^(15 - level)" },
            if probe.smoothed { "on" } else { "off" },
            if key.p256 { 256 } else { 16 },
            key.reach,
            if probe.bright { 4 } else { 1 },
            TONEMAPS[probe.tonemap].0,
        );
    }
}

fn take_shots(
    mut commands: Commands,
    mut shots: ResMut<Shots>,
    mut probe: ResMut<Probe>,
    mut orbits: Query<&mut PanOrbitCamera>,
    mut exit: MessageWriter<AppExit>,
) {
    shots.frames += 1;
    while let Some(skip) = SHOTS.get(shots.index) {
        if shots.only.as_ref().is_some_and(|only| !skip.name.starts_with(only.as_str())) {
            shots.index += 1;
        } else {
            break;
        }
    }
    let Some(shot) = SHOTS.get(shots.index) else {
        exit.write(AppExit::Success);
        return;
    };
    if shots.frames == 1 {
        let key_before = probe.key();
        probe.candidate = shot.candidate;
        probe.glow = shot.glow;
        probe.night = shot.night;
        probe.linear = shot.linear;
        probe.to_zero = shot.to_zero;
        probe.p256 = shot.p256;
        probe.reach32 = shot.reach32;
        probe.view = shot.view;
        probe.bright = shot.bright;
        probe.tonemap = shot.tonemap;
        probe.place = shot.place;
        probe.panel = false;
        if probe.key() != key_before {
            shots.wait = shots.wait.max(240);
        }
        let (_, eye, target) = PLACES[shot.place];
        for mut orbit in orbits.iter_mut() {
            orbit_at(&mut orbit, eye, target);
        }
    }
    if !shots.taken && shots.frames >= shots.wait {
        let path = format!("{}/{}.png", shots.out, shot.name);
        println!("shot {path}");
        commands.spawn(Screenshot::primary_window()).observe(save_to_disk(path));
        shots.taken = true;
    }
    if shots.taken && shots.frames >= shots.wait + 15 {
        shots.index += 1;
        shots.frames = 0;
        shots.wait = 30;
        shots.taken = false;
    }
}
