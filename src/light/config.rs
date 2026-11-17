//! The settings of light propagation, which a world returns from
//! [`VoxelWorldConfig::light_propagation`](crate::prelude::VoxelWorldConfig::light_propagation).
//!
//! Each field's documentation says when a change to it takes effect:
//!
//! - **Read once at plugin build:** changing it later logs a warning once and
//!   does nothing.
//! - **Relights on change:** every chunk with a solid near its open voxels
//!   works its light out again. Old light shows until new light lands,
//!   nearest the camera first, and the world settles within a few frames
//!   on native.
//! - **Replaces the texture on change:** as a relight, and the world's light
//!   texture is made again in the new format.
//! - **Shows on the next frame:** only the shader reads it.
//! - **A limit:** it caps what the world may use.
//!
//! Out-of-range values are clamped, with a warning logged once, where the crate
//! reads them. The value the world returns is never rewritten.

use std::fmt::Display;
use std::sync::{Mutex, PoisonError};
use std::time::Duration;

use bevy::color::{Color, LinearRgba};
use bevy::log::{warn, warn_once};
use bevy::reflect::Reflect;

/// Light propagation for one world.
///
/// A world opts in by returning `Some` from
/// [`VoxelWorldConfig::light_propagation`](crate::prelude::VoxelWorldConfig::light_propagation),
/// and returning `None`, the default, costs nothing: no memory, no chunk task
/// time, no mesh attribute, no texture binding and no shader change.
///
/// Light spreads from its sources through the world's open voxels, `Air` and
/// `Unset`, and stops at `Solid` ones. A voxel with no open path to a source is
/// pitch black, however close it is to the surface. There are two kinds of
/// light, each a level from 0 to 15 at every voxel:
///
/// - **Sky light** enters from the open sky and falls straight down, with no
///   loss, to the first solid voxel. Past that it spreads sideways and down,
///   fading with distance.
/// - **Voxel light** comes from emissive voxels, declared by material in
///   [`emissive_materials`](crate::prelude::VoxelWorldConfig::emissive_materials),
///   and spreads the same way.
///
/// Each world has its own light: one world's solids never shadow another
/// world. Up is +Y in every world.
///
/// Every field says when a change to it takes effect, as one of: read once at
/// plugin build, relights on change, replaces the texture on change, shows on
/// the next frame, or a limit.
///
/// # Changing settings while the world runs
///
/// A world keeps its `LightPropagation` in a field of its config and returns
/// a clone of it. A system changes it through `ResMut` of the config, which it
/// can hold together with [`VoxelWorld`](crate::prelude::VoxelWorld). On
/// frames where the config changed, the crate compares the new settings and
/// [`emissive_materials`](crate::prelude::VoxelWorldConfig::emissive_materials)
/// with those it last applied and does the least the difference needs, so a
/// value set equal to the current one costs nothing, and settings at rest
/// cost no work per chunk. Out-of-range values set at runtime are clamped, as
/// at plugin build.
///
/// A change that relights solves again every chunk with an open voxel beside a
/// solid, counting the solids just past its faces, so its cost grows with how
/// many such chunks are loaded. The relights run on the task pool after chunk
/// tasks, so streaming goes on at its usual pace, and their light lands over
/// the next few frames, nearest the camera first, with a brief seam between
/// old and new light.
/// [`VoxelWorld::light_is_settled`](crate::prelude::VoxelWorld::light_is_settled)
/// is false until every relight has landed.
///
/// A change of an emissive material's light colour alone relights only the
/// chunks whose light box holds an emissive voxel of that material: see
/// [`emissive_materials`](crate::prelude::VoxelWorldConfig::emissive_materials).
///
/// A chunk of air with no solid beside it isn't relit, so
/// [`VoxelWorld::light_at`](crate::prelude::VoxelWorld::light_at) there can
/// read light from the old settings. A chunk above LOD level 0 takes the new
/// settings only when its chunk task runs again.
///
/// Whether `light_propagation` returns `None`, and [`shown`](Self::shown), are
/// read once at plugin build: changing either later logs a warning once and
/// changes nothing.
///
/// Light needs the crate's `light_propagation` cargo feature.
///
/// # Voxel edits
///
/// Light follows a voxel set with
/// [`VoxelWorld::set_voxel`](crate::prelude::VoxelWorld::set_voxel) on its
/// own. Every edit set in one frame is worked out together as one change, on
/// the main thread in the frame it is committed. That work grows with the
/// number of voxels set, and with the columns the change opens to the sky,
/// whose falling sky light it follows down.
///
/// The chunks the change's light reaches then relight on the task pool,
/// starting at once rather than queuing behind chunk tasks, so a single voxel
/// edit's light lands within 2 frames while streaming leaves the pool room.
/// Near where chunks meet at an edge or a corner, the relight ripple reaches
/// some of them only through the others, a frame for each step, so an edit
/// there can take a frame longer, or two where its light has several colours.
/// Landing relit light takes at most about 1 ms of the main thread per frame,
/// so an edit whose light reaches many chunks lands over a few more frames.
/// An edit where no light is, such as in a dark cave, relights nothing.
///
/// # Light colour
///
/// An emissive material's light can take a colour, with
/// [`Emission::with_color`]. Where light of several colours reaches a voxel,
/// each of its red, green and blue takes the brightest of them, so a red
/// and a green emissive voxel light the voxels between them yellow, no
/// brighter than either. Sky light has no colour of its own: it takes the
/// colour of Bevy's ambient light, environment map and sun.
///
/// A world whose emissive materials are all white does no colour work: it
/// solves, holds and shows its light as it would if light had no colour. Its
/// chunk materials bind a one-texel placeholder for the colour texture, as
/// they would for any binding the world doesn't use.
///
/// In a world with colours other than white, a chunk whose voxel light is
/// all one colour solves as a white chunk does and keeps one colour for all
/// its voxels. Where light of two or more colours reaches a chunk, it costs
/// more, and each cost grows with:
///
/// - **Solving:** the light colours among the emissive voxels within voxel
///   reach of the chunk, since its voxel light is solved once for each
///   colour. The colours around each chunk count, not how many the world
///   declares. See [`skip_unreached_colors`](Self::skip_unreached_colors).
/// - **Memory:** the chunks that light of two or more colours reaches, each
///   holding a light colour for every voxel, 64 KiB a chunk. A coarse
///   chunk's CPU memory also grows with the coarse chunks holding colour
///   slots, each keeping a copy of its colour slot so that `light_at` reads
///   its colours.
/// - **GPU memory:** the most chunks at once whose faces see more than one
///   light colour, each holding a colour slot: see
///   [`slot_capacity`](Self::slot_capacity).
/// - **Shader work:** the faces of chunks whose light isn't all white, which
///   read a colour beside each level they blend.
///
/// A coarse chunk solves once per light colour among its own emissive
/// voxels, and shows no light colour from past its own voxels.
///
/// # WASM
///
/// Light runs in the browser on WebGL2, with no compute shaders. With no
/// threads to run tasks on, relights run on the main thread within
/// [`wasm_relight_budget`](Self::wasm_relight_budget) each frame, and at most
/// one sky probe starts per frame, so a big edit costs time instead of a
/// hitch. A single voxel edit's light still lands as it does on native: see
/// [Voxel edits](#voxel-edits). A chunk
/// whose light box holds emissive voxels of two or more light colours
/// solves once for each colour, so its relight uses the budget up faster,
/// and a big edit among coloured light lands over more frames.
///
/// A WASM application should choose a [`sky_reach`](Self::sky_reach) of 16,
/// the last before the rows the solver spreads light along go from one
/// 64-bit word to two, which makes every solve of sky light cheaper. WASM
/// doesn't set any default: the defaults are chosen for native. The light
/// texture is clamped to what the device holds, which on WebGL2 is at least
/// 343 slots: see [`slot_capacity`](Self::slot_capacity).
///
/// WebGL2 allows a pipeline 16 textures, and the crate's bindings take three
/// of them in every chunk material with the `sub_grids` feature: the light
/// texture, the tag map and the colour texture. Without it they take two,
/// with no tag map. Each is bound from the start, as a placeholder until
/// the world needs it, whatever the world's light colours. The rest are
/// shared by the material's own textures, six for `StandardMaterial`
/// whether or not they're set, and the camera's, such as shadow maps, the
/// tonemapping lookup table and an environment map. So what's left for a
/// game's textures depends on its material and its camera. A pipeline over
/// the limit fails to build, and Bevy's default render error handler then
/// quits the app.
///
/// Light colour works on WebGL2 too. To keep a world's light white there,
/// return white emissions from
/// [`emissive_materials`](crate::prelude::VoxelWorldConfig::emissive_materials)
/// under `cfg(target_arch = "wasm32")`:
///
/// ```
/// # use bevy::prelude::*;
/// # use bevy_voxel_world::prelude::*;
/// # #[derive(Resource, Clone, Default)]
/// # struct MyWorld;
/// # impl VoxelWorldConfig for MyWorld {
/// #     type MaterialIndex = u8;
/// #     type ChunkUserBundle = ();
/// fn emissive_materials(&self) -> Vec<(u8, Emission)> {
///     let torch = Emission::level(15.0);
///     #[cfg(not(target_arch = "wasm32"))]
///     let torch = torch.with_color(LinearRgba::rgb(1.0, 0.6, 0.2));
///     vec![(3, torch)]
/// }
/// # }
/// ```
///
/// # Showing light
///
/// With [`shown`](Self::shown) on, the crate draws the world's chunks with its
/// material wrapped in
/// [`VoxelWorldBindings`](crate::rendering::VoxelWorldBindings), and:
///
/// - The crate owns the `MeshTag` of every chunk in the world, and overwrites
///   any tag the world's own systems set.
/// - Lit chunk materials are forward only: voxel light is added in the forward
///   pass, so a chunk drawn in a deferred pass shows none.
/// - Exact black is zero light before tonemapping. Where neither kind of light
///   reaches, a face is exactly black, which Bevy's default tonemapper maps to
///   1/255, with dither noise of up to 1/255.
///
/// Propagated light scales Bevy's ambient light down in shade, so a lit world
/// wants a [`GlobalAmbientLight`](bevy::light::GlobalAmbientLight) of about
/// 600, where an unlit one looks right nearer 100. Caves stay black at any
/// value.
///
/// Propagated light shades chunks only. Other entities keep Bevy's lighting,
/// so a character walking into a dark cave stays fully lit unless the game
/// dims it: read the light at its position with
/// [`VoxelWorld::light_at`](crate::prelude::VoxelWorld::light_at) and scale
/// its colour by the look's curves, as [`LevelCurve::at`] describes. The
/// `light_tuning` example does this.
///
/// # Limits
///
/// - **Sky light falls straight down.** It has no direction and doesn't
///   follow the sun. So under [`SkyFill::Paths`], a floating solid shows a
///   straight-down shade on the ground below it as well as its sun shadow.
///   [`SkyFill::Openness`] removes that shade where a place sees most of the
///   sky. Bevy's shadow cascades stop 150 voxels from the camera by default,
///   and past them the sun casts no shadow at all.
/// - **A shadow ends about the sky clearance below what casts it,** in a
///   flat line on a chunk boundary, and a sealed space taller than the
///   clearance lights up on its floor: see [`Sky::Clearance`].
/// - **Lookups must be deterministic.** With sky light on, the world's voxel
///   lookup must return the same voxel for a position every time. After
///   changing what lookups return, call
///   [`VoxelWorld::reset_light`](crate::prelude::VoxelWorld::reset_light).
/// - **A chunk's light can change after it appears.** Its first light reads
///   the chunks around it that aren't loaded yet as solid, so it gets
///   brighter as they arrive; a face on its boundary shows the light in
///   front of it only once the neighbour's light has landed; and a column
///   nothing has proved blocked shows direct sky until a sky probe proves
///   otherwise, so a cave can show sky light for a moment before going dark.
///   Most of these last a frame or two, sky probes longer after a stream, and
///   [`VoxelWorld::light_is_settled`](crate::prelude::VoxelWorld::light_is_settled)
///   says when they're over. Light beside a chunk that isn't loaded can be
///   darker than the truth, never brighter.
/// - **The end of the loaded world shows in a cave.** Past the spawning
///   distance nothing is drawn, so a gap at the end of a dark cave shows
///   whatever the camera draws behind the world, often a bright clear colour.
///   A larger spawning distance, or LOD levels, move that edge further away.
///   Fog, the clear colour and any skybox belong to the application, which
///   can tie them to `light_at` at the camera, at the cost of black sky when
///   looking out of a cave mouth or from shade.
/// - **Sky fill has limits of its own**, such as a cave behind a narrow
///   opening staying black past its pool of light: see [`SkyFill`].
/// - **Slots are capped per world.** A chunk that loses its slot looks flat
///   and its lamps lose their glow: see [`slot_capacity`](Self::slot_capacity).
/// - **`light_at` reads what has landed,** and coarse chunks plausibly rather
///   than exactly: see
///   [`VoxelWorld::light_at`](crate::prelude::VoxelWorld::light_at).
/// - **Light comes from voxels, not meshes,** so a custom mesh must use the
///   chunk's padded local coordinates: see
///   [`VoxelWorldBindings`](crate::rendering::VoxelWorldBindings).
///
/// ```
/// use bevy::prelude::*;
/// use bevy_voxel_world::prelude::*;
///
/// #[derive(Resource, Clone, Default)]
/// struct MyWorld;
///
/// impl VoxelWorldConfig for MyWorld {
///     type MaterialIndex = u8;
///     type ChunkUserBundle = ();
///
///     fn light_propagation(&self) -> Option<LightPropagation> {
///         Some(LightPropagation {
///             voxel_reach: 10,
///             ..default()
///         })
///     }
///
///     fn emissive_materials(&self) -> Vec<(u8, Emission)> {
///         vec![(3, Emission::level(15.0))]
///     }
/// }
/// ```
#[derive(Reflect, Clone, PartialEq, Debug)]
pub struct LightPropagation {
    /// Whether chunks are drawn with propagated light. With `false`, light is
    /// worked out and can be read, but chunks keep Bevy's lighting, and the
    /// world holds no light texture, slots or tags.
    ///
    /// Read once at plugin build. Defaults to `true`.
    pub shown: bool,

    /// Where sky light enters the world.
    ///
    /// Relights on change.
    ///
    /// Two changes need many sky probes, and settle far slower than other
    /// relights. Turning sky light on after [`Sky::Off`] starts with an empty
    /// record, so caves can show sky light until probes prove their columns
    /// blocked. Raising the clearance past what the record has seen sends sky
    /// probes up the chunk positions it adds, except those a fill hint
    /// declares uniform, so a world that hints the air above it settles much
    /// sooner.
    ///
    /// Defaults to [`Sky::Clearance`] of 16 chunks.
    pub sky: Sky,

    /// How many voxels sky light spreads past where it falls straight down,
    /// from 1 to 32.
    ///
    /// Relights on change: every chunk with a solid near its open voxels is
    /// solved again on the task pool. With sky light off it relights nothing.
    ///
    /// Defaults to 30.
    pub sky_reach: u8,

    /// How many voxels voxel light spreads from an emissive voxel at level 15,
    /// from 1 to 32.
    ///
    /// Relights on change: every chunk with a solid near its open voxels is
    /// solved again on the task pool.
    ///
    /// Defaults to 14, which loses one level per voxel.
    pub voxel_reach: u8,

    /// The shape light spreads in, for both kinds of light.
    ///
    /// Relights on change: every chunk with a solid near its open voxels is
    /// solved again on the task pool.
    ///
    /// Defaults to [`LightShape::Round`].
    pub shape: LightShape,

    /// How sky light reaches past where it falls and spreads, such as under a
    /// high floating island or into a cave behind a wide opening. `None`
    /// turns it off, so sky light comes only from direct sky and spreads
    /// `sky_reach` voxels from it: the ground under a high island is then
    /// black in its middle, as under a low slab.
    ///
    /// Sky fill works out a level for every 8-voxel cell from which cells of
    /// the chunk positions the world has seen are open, kept in the sky
    /// record, and seeds those levels into the light of each chunk with an
    /// open voxel beside a solid. It reads only columns proved clear as sky,
    /// and positions it hasn't seen as blocked, so it is exact at rest and
    /// darker than the truth before, never brighter. A change of fill
    /// relights the chunks around it at most every 500 ms. With it off, a
    /// world keeps no fill and runs no fill probes. With
    /// [`Sky::Off`] it does nothing.
    ///
    /// Under paths, keeping the field current costs a little main-thread time
    /// per chunk streamed in, and the field counts against
    /// [`sky_record_cap`](Self::sky_record_cap). See [`SkyFill`] for how sky
    /// openness compares.
    ///
    /// Relights on change.
    ///
    /// Defaults to [`SkyFill::Paths`] with its defaults.
    pub sky_fill: Option<SkyFill>,

    /// How many light levels each voxel holds. Levels read from 0 to 15
    /// either way, in steps of 1/17 at 256 levels.
    ///
    /// At 16 levels plain surfaces show a faint terrace one level high
    /// wherever light loses a level. 256 levels smooths them away and lets
    /// sky fill's fractional levels reach the voxels it lights. It costs 2
    /// bytes per voxel of light on the CPU, where 16 levels cost 1, and twice
    /// the GPU memory per slot: see [`slot_capacity`](Self::slot_capacity).
    ///
    /// Replaces the texture on change: every chunk with a slot drops to one
    /// sky light level and one voxel light level, the old texture is released,
    /// every loaded chunk relights, coarse chunks included, and the new texture
    /// is made when a chunk next needs a slot. Chunks look flat until their
    /// light lands again, which takes longer than other relights.
    ///
    /// Defaults to [`LightPrecision::Steps16`].
    pub precision: LightPrecision,

    /// How many chunks the world's light texture holds at once, at least 1.
    ///
    /// A chunk takes a slot in the texture when its faces see more than one
    /// level of light, or more than one light colour. Each slot costs about
    /// 38 KiB of GPU memory at 16 levels and 77 KiB at 256, so the default of
    /// 1,024 slots takes about 38 MiB or 77 MiB. The texture is made when the
    /// first chunk needs a slot, so a world where every chunk's faces see one
    /// level costs nothing.
    ///
    /// In a world with light colours other than white, a chunk whose faces see
    /// more than one light colour also takes a colour slot, about 77 KiB more at
    /// either precision, in a second texture. That texture is made when the
    /// first chunk needs a colour slot, grows as more chunks need one at once,
    /// up to this count, and never shrinks. So its GPU memory grows with the
    /// most chunks at once whose faces see more than one light colour.
    ///
    /// The count is clamped to what the device's largest 3D texture holds,
    /// with a warning: WebGL2 guarantees 256 texels a side, which is 343
    /// slots.
    ///
    /// When every slot is taken, the chunks farthest from the camera give
    /// theirs to nearer ones, and the world logs a warning once. A chunk
    /// with no slot shows one level of each kind of light on all its faces:
    /// it looks flat, losing its shadows, and its lamps' faces lose their
    /// glow.
    /// It also shows one light colour, the most common one among its voxels
    /// at the level it shows. Colour slots give way the same way.
    /// Raise the count if that warning appears.
    ///
    /// Replaces the texture on change: every chunk with a slot drops to one
    /// sky light level and one voxel light level, the old texture is released,
    /// the settings relight runs, and the new texture is made when a chunk next
    /// needs a slot. Chunks look flat until they take slots again.
    ///
    /// Defaults to 1,024.
    pub slot_capacity: u32,

    /// How light looks on chunk faces.
    ///
    /// Shows on the next frame. See [`LightLook`] for its defaults.
    pub look: LightLook,

    /// The most memory, in bytes, the world keeps in its sky record: which
    /// columns of each chunk position it has seen hold a solid voxel.
    ///
    /// A chunk reads the record to know which of its columns sky light enters
    /// without looking up the voxels above it again, so a chunk spawning
    /// again under positions the record holds gets its direct sky right on
    /// its first publish. Past the cap, the least recently used positions are
    /// forgotten. Forgetting costs only time, never different light: a
    /// forgotten position is looked up again when a chunk needs it.
    ///
    /// A mixed position takes about 192 bytes, and a position that is all air
    /// or has a solid in every column about 64. With sky fill on, each
    /// position takes 16 bytes more for its cells, and the sky fill field
    /// counts against the cap too, at roughly 200 bytes per position it
    /// knows. Values below 3 KiB are raised to it with a warning.
    ///
    /// Applied at once on change.
    ///
    /// Defaults to 16 MiB.
    pub sky_record_cap: usize,

    /// The most main-thread time per frame spent relighting on WASM, from
    /// 100 µs to 100 ms. Ignored on native, where relights run on the task
    /// pool.
    ///
    /// On WASM, tasks get no thread of their own, so relights run on the main
    /// thread from the world's queue, the chunks holding edited voxels first,
    /// until this much of the frame is spent. The last relight to start can
    /// take the frame past it, since one relight can't be split. Each frame
    /// that relights costs up to this much frame time: a bigger budget
    /// settles light sooner after a big edit, a settings change or streaming,
    /// and a smaller one spreads the same work over more frames.
    ///
    /// Defaults to 4 ms.
    pub wasm_relight_budget: Duration,

    /// Whether a chunk with emissive voxels of four or more light colours within
    /// voxel reach first works out which of those colours' light can reach it
    /// through open voxels, and solves only those.
    ///
    /// A chunk's voxel light is solved once for each light colour among the
    /// emissive voxels within voxel reach of it. A colour whose light solids
    /// keep from the chunk costs a whole solve and lights nothing. Working out
    /// which colours reach costs about one more solve, so it saves time where
    /// solids keep colours apart, as in caves and rooms, and costs a little
    /// where every colour reaches, as on open ground.
    ///
    /// It never changes the light, so a change applies to the next solves and
    /// relights nothing. It does nothing in a world whose emissive voxels are
    /// all white.
    ///
    /// Defaults to `true`.
    pub skip_unreached_colors: bool,

    /// What sky a sub-grid's faces show: the world's sky capped by the
    /// sub-grid's enclosure, or the world's sky alone. The enclosure caps
    /// the world's voxel light too. See [`SubGridSky`].
    ///
    /// Relights every sub-grid chunk on change, and no world chunk.
    ///
    /// Defaults to [`SubGridSky::Enclosed`].
    #[cfg(feature = "sub_grids")]
    pub sub_grid_sky: SubGridSky,
}

impl Default for LightPropagation {
    fn default() -> Self {
        Self {
            shown: true,
            sky: Sky::default(),
            sky_reach: 30,
            voxel_reach: 14,
            shape: LightShape::default(),
            sky_fill: Some(SkyFill::default()),
            precision: LightPrecision::default(),
            slot_capacity: 1024,
            look: LightLook::default(),
            sky_record_cap: 16 * 1024 * 1024,
            wasm_relight_budget: Duration::from_millis(4),
            skip_unreached_colors: true,
            #[cfg(feature = "sub_grids")]
            sub_grid_sky: SubGridSky::default(),
        }
    }
}

impl LightPropagation {
    /// This config with every out-of-range value clamped, logging a warning
    /// once for each kind of value clamped.
    pub(crate) fn clamped(&self) -> Self {
        let mut config = self.clone();
        config.sky_reach = clamp_setting(config.sky_reach, 1, 32, "sky_reach");
        config.voxel_reach = clamp_setting(config.voxel_reach, 1, 32, "voxel_reach");
        if config.slot_capacity == 0 {
            warn_once!("light propagation: slot_capacity 0 is below 1, using 1");
            config.slot_capacity = 1;
        }
        if let Sky::Clearance { chunks: 0 } = config.sky {
            warn_once!(
                "light propagation: a sky clearance of 0 chunks is below 1, using 1"
            );
            config.sky = Sky::Clearance { chunks: 1 };
        }
        if config.sky_record_cap < super::record::MIN_CAP {
            warn_once!(
                "light propagation: sky_record_cap {} is below {}, using {}",
                config.sky_record_cap,
                super::record::MIN_CAP,
                super::record::MIN_CAP
            );
            config.sky_record_cap = super::record::MIN_CAP;
        }
        config.wasm_relight_budget = clamp_relight_budget(config.wasm_relight_budget);
        config.sky_fill = config.sky_fill.map(|fill| match fill {
            SkyFill::Paths(paths) => SkyFill::Paths(paths.clamped()),
            SkyFill::Openness(open) => SkyFill::Openness(open.clamped()),
        });
        config.look = config.look.clamped();
        config
    }

    /// Whether a sub-grid's enclosure caps the world's voxel light its faces
    /// show, as it caps the world's sky: under [`SubGridSky::Enclosed`] with
    /// sky light on. With sky light off no enclosure is solved, so there is
    /// nothing to cap with.
    #[cfg(feature = "sub_grids")]
    pub(crate) fn enclosure_caps_voxel_light(&self) -> bool {
        self.sky != Sky::Off && self.sub_grid_sky == SubGridSky::Enclosed
    }
}

/// The shortest and longest `wasm_relight_budget`.
const RELIGHT_BUDGETS: (Duration, Duration) =
    (Duration::from_micros(100), Duration::from_millis(100));

fn clamp_relight_budget(budget: Duration) -> Duration {
    let (min, max) = RELIGHT_BUDGETS;
    if (min..=max).contains(&budget) {
        return budget;
    }
    let clamped = budget.clamp(min, max);
    warn_once!(
        "light propagation: wasm_relight_budget {budget:?} is outside {min:?} to {max:?}, \
         using {clamped:?}"
    );
    clamped
}

/// `value` kept within `min..=max`, logging a warning the first time a value
/// named `name` is out of range. NaN becomes `min`.
fn clamp_setting<T: PartialOrd + Display + Copy>(
    value: T,
    min: T,
    max: T,
    name: &str,
) -> T {
    if value >= min && value <= max {
        return value;
    }
    let clamped = if value > max { max } else { min };
    if first_warning_for(name) {
        warn!("light propagation: {name} {value} is outside {min} to {max}, using {clamped}");
    }
    clamped
}

/// Whether no warning about the setting named `name` was logged yet, noting
/// that one is now.
///
/// `warn_once!` warns once per call site, and one call site clamps many
/// settings, so each setting is noted by name instead.
fn first_warning_for(name: &str) -> bool {
    static WARNED: Mutex<Vec<String>> = Mutex::new(Vec::new());
    let mut warned = WARNED.lock().unwrap_or_else(PoisonError::into_inner);
    if warned.iter().any(|warned| warned == name) {
        return false;
    }
    warned.push(name.to_owned());
    true
}

/// What sky a sub-grid's faces show, from
/// [`LightPropagation::sub_grid_sky`].
///
/// A sub-grid's voxels never shade the world's light, so the world's sky
/// and lamps alone light a closed cabin's inside as brightly as the open air
/// around it.
#[cfg(feature = "sub_grids")]
#[derive(Reflect, Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum SubGridSky {
    /// The world's sky, capped by the sub-grid's enclosure. A voxel of the
    /// sub-grid has direct sky where any of the six lines along the
    /// sub-grid's axes from it leaves the sub-grid without meeting one of its
    /// voxels, and sky spreads from there with the world's `sky_reach`,
    /// `precision` and `shape`. So a closed cabin is dark inside under an
    /// open sky, and its doorway fades inward the way a cave mouth does.
    /// Which way up the sub-grid is changes nothing, so turning it solves
    /// nothing.
    ///
    /// The enclosure caps the world's voxel light the same way, so a world
    /// lamp outside a closed cabin lights its outside and not its inside.
    /// The cap says how open a voxel is to the outside, not which way, so
    /// light from a world lamp beside the cabin still shows under an opening
    /// in its roof. With sky light off no enclosure is solved, and the
    /// world's voxel light shows inside as it does under `World`.
    ///
    /// It costs memory and edit time. Each sub-grid keeps a record of which
    /// of its lines hold a voxel, which grows with its chunks. A voxel edit
    /// also follows the three lines through each edited voxel across the
    /// sub-grid, and relights the chunks around the voxels whose direct sky
    /// it changed, which can lie far from the edit.
    #[default]
    Enclosed,
    /// The world's light alone, wherever the sub-grid is, inside or out. A
    /// cabin's inside is as bright as the air around it, from the world's
    /// sky and its lamps. Nothing is kept or solved for enclosure.
    World,
}

/// Where sky light enters the world.
///
/// With sky light on, the world's voxel lookup must return the same voxel for
/// a position every time it's asked, leaving voxel edits aside, since a column
/// is judged open or blocked from what the lookups return.
#[derive(Reflect, Clone, PartialEq, Debug)]
pub enum Sky {
    /// Sky light enters the top of a chunk in a column when that column holds
    /// no solid voxel in the `chunks` whole chunks straight above it, at least 1.
    ///
    /// A shadow ends about this far below what casts it, and a sealed space
    /// taller than it lights up on its floor.
    Clearance { chunks: u32 },
    /// Sky light enters a column that holds no solid voxel up to `y`, rounded
    /// up to the top of its chunk. Voxel edits above it cast no shadow.
    Ceiling { y: i32 },
    /// No sky light. Voxel light still works.
    Off,
}

impl Default for Sky {
    fn default() -> Self {
        Sky::Clearance { chunks: 16 }
    }
}

/// How sky light is carried past where it falls and spreads, worked out per
/// 8-voxel cell.
///
/// [`Paths`](Self::Paths) is the default. Its light settles far sooner than
/// under [`Openness`](Self::Openness), its field costs far less to keep
/// current while chunks stream in, and it shades the ground under islands and
/// overhangs with smooth gradients, where sky openness draws rings and
/// blotches stepped at 8-voxel cells.
///
/// Each model has limits. Under paths, a cave behind an opening narrower
/// than the cave itself is black past the pool of light around its opening,
/// and a sky full of small floating things gets no fill. Under sky openness,
/// a cave is black past what its opening shows of the sky, every voxel edit
/// casts again the rays of the cells within its reach, and a fresh stretch
/// of world is slow to settle.
#[derive(Reflect, Clone, PartialEq, Debug)]
pub enum SkyFill {
    /// Light passes from open space to open space through the openings between
    /// them, losing more through narrower openings.
    Paths(PathFill),
    /// Each cell is lit by how much of the sky it sees, and a cell seeing
    /// most of it counts as direct sky and gets the sun.
    Openness(OpenFill),
}

impl Default for SkyFill {
    fn default() -> Self {
        SkyFill::Paths(PathFill::default())
    }
}

/// The settings of [`SkyFill::Paths`].
///
/// Each chunk position's open 8-voxel cells split into regions of cells
/// joined by faces, and each region gets a level. A region passes light to a
/// region beside it through the cells they share, losing more the less of the
/// receiving region's width the opening fills, and a region whose top cells
/// see the sky gets light from it.
///
/// A world wanting brighter caves should lower [`opening_power`](Self::opening_power)
/// first: it costs no more. Lowering [`whole_face_loss`](Self::whole_face_loss)
/// lets light go further, but every change of the field then reaches more
/// positions and costs more.
///
/// Two limits come with the default settings. A cave behind an opening
/// narrower than the cave itself is black past the pool of light around its
/// opening. And a sky full of small floating things gets no fill: an 8×8
/// patch of the sky rarely has all its columns clear, so no region sees the
/// sky.
#[derive(Reflect, Clone, PartialEq, Debug)]
pub struct PathFill {
    /// The light level of open space beside the open sky, from 0 to 15.
    /// Defaults to 12.
    ///
    /// Costs nothing per frame. Relights on change.
    pub beside_open_sky: f32,
    /// The levels lost passing through a whole open face of 8-voxel cells, at
    /// least 0. Defaults to 2.
    ///
    /// A lower loss carries light through more positions, and every change of
    /// a position's cells reaches that many more: halving the loss doubles how
    /// far light goes each way, so up to about 8 times as many positions in a
    /// big open world. Relights on change.
    pub whole_face_loss: f32,
    /// How much more a narrower opening loses, at least 0. Passing through an
    /// opening loses `whole_face_loss × (16 / f) ^ opening_power` levels, where
    /// `f`, from 1 to 16, is how many cells of the receiving space's widest
    /// cross-section the opening fills. Defaults to 2.
    ///
    /// Costs nothing per frame, and a lower power costs no more to keep
    /// current. Relights on change.
    pub opening_power: f32,
}

impl Default for PathFill {
    fn default() -> Self {
        Self {
            beside_open_sky: 12.0,
            whole_face_loss: 2.0,
            opening_power: 2.0,
        }
    }
}

impl PathFill {
    fn clamped(self) -> Self {
        Self {
            beside_open_sky: clamp_setting(
                self.beside_open_sky,
                0.0,
                15.0,
                "beside_open_sky",
            ),
            whole_face_loss: clamp_setting(
                self.whole_face_loss,
                0.0,
                f32::MAX,
                "whole_face_loss",
            ),
            opening_power: clamp_setting(
                self.opening_power,
                0.0,
                f32::MAX,
                "opening_power",
            ),
        }
    }
}

/// The settings of [`SkyFill::Openness`].
///
/// Each open 8-voxel cell that isn't direct sky casts rays over the sky above
/// it and is lit by the share that reaches open sky: a cell whose columns are
/// all proved clear, met before a cell holding a solid. Under the default
/// [`sky_falloff`](LightLook::sky_falloff), a cell's brightness is that share
/// × [`gain`](Self::gain). A cell seeing at least [`sun_at`](Self::sun_at) of
/// the sky counts as direct sky: it reads 15 and gets the sun. So the ground
/// under a high floating island loses its straight-down shade and keeps only
/// the sun's shadow, and a cave is black past what its opening shows of the
/// sky.
///
/// Rays cost more than [`SkyFill::Paths`]: every cell that isn't direct sky
/// casts each of its rays again when a voxel within
/// [`reach`](Self::reach) below or beside it changes.
#[derive(Reflect, Clone, PartialEq, Debug)]
pub struct OpenFill {
    /// The share of the sky, from 0 to 1, a cell must see to count as direct
    /// sky and get the sun. A cell seeing none of the sky never does.
    /// Defaults to 0.8.
    ///
    /// Costs nothing per frame. Relights on change.
    pub sun_at: f32,
    /// How bright a cell is for the share of the sky it sees, at least 0:
    /// under the default sky falloff, the cell's brightness is the share ×
    /// `gain`, up to level 14. Defaults to 1.
    ///
    /// Costs nothing per frame. Relights on change.
    pub gain: f32,
    /// How many rays each cell casts toward the sky, at least 1. Defaults to
    /// 32.
    ///
    /// Working out a cell costs about as much as its rays: twice the rays,
    /// twice the time. Relights on change.
    pub rays: u32,
    /// How far, in voxels, a ray goes looking for the sky, from 1 to 512.
    /// Defaults to 128.
    ///
    /// A longer reach lets a ray look further, and makes a voxel edit cast
    /// again the cells of more positions around it: the positions below and
    /// beside an edit within `reach`. Relights on change.
    pub reach: u32,
}

impl Default for OpenFill {
    fn default() -> Self {
        Self {
            sun_at: 0.8,
            gain: 1.0,
            rays: 32,
            reach: 128,
        }
    }
}

impl OpenFill {
    fn clamped(self) -> Self {
        Self {
            sun_at: clamp_setting(self.sun_at, 0.0, 1.0, "sun_at"),
            gain: clamp_setting(self.gain, 0.0, f32::MAX, "gain"),
            rays: clamp_setting(self.rays, 1, u32::MAX, "rays"),
            reach: clamp_setting(self.reach, 1, 512, "reach"),
        }
    }
}

/// The shape light spreads in.
#[derive(Reflect, Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum LightShape {
    /// Pools of light come out nearly circular.
    #[default]
    Round,
    /// Light loses a level per face step, so pools come out diamond shaped.
    Diamond,
}

/// How many light levels each voxel holds. Levels read from 0 to 15 either way.
#[derive(Reflect, Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum LightPrecision {
    /// 16 levels per kind of light, 1 byte per voxel.
    #[default]
    Steps16,
    /// 256 levels per kind of light, 2 bytes per voxel, with smoother sky fill.
    Steps256,
}

/// How light looks on chunk faces.
///
/// Every field shows on the next frame: a change to any of them costs one
/// write of the world's look uniform, with no relight and no work per chunk.
/// The crate notices a change on frames where the world's config resource
/// changed, so a system can tune the look by writing to that resource.
#[derive(Reflect, Clone, PartialEq, Debug)]
pub struct LightLook {
    /// Whether light blends between neighbouring voxels on a face, or shows
    /// each voxel's level as a flat step for a blocky style.
    ///
    /// Shows on the next frame, for one uniform write. Defaults to
    /// [`LightSmoothing::Smoothed`].
    pub smoothing: LightSmoothing,
    /// The brightness of each sky light level, as the share of Bevy's ambient
    /// light and environment map a face gets.
    ///
    /// Shows on the next frame, for one uniform write. Defaults to
    /// `0.7^(15 - level)`, with level 0 exactly black.
    pub sky_falloff: LevelCurve,
    /// The brightness of each voxel light level, as a share of
    /// `voxel_brightness`.
    ///
    /// Where light of several colours meets, the colours mix in the
    /// proportions of the default curve, whatever this curve is, so changing
    /// it never relights. A curve whose level 1 isn't black, such as
    /// [`LevelCurve::linear`], ends each pool of light in a faint step.
    ///
    /// Shows on the next frame, for one uniform write. Defaults to
    /// [`LevelCurve::exponential_to_black(0.7)`](LevelCurve::exponential_to_black),
    /// with levels 0 and 1 exactly black.
    pub voxel_falloff: LevelCurve,
    /// Where Bevy's directional lights shine.
    ///
    /// Shows on the next frame, for one uniform write. Defaults to
    /// [`SunGate::DirectSkyOnly`].
    pub sun: SunGate,
    /// The brightness of voxel light at level 15, multiplied by the camera's
    /// exposure, at least 0.
    ///
    /// Shows on the next frame, for one uniform write. Defaults to 1,000.
    pub voxel_brightness: f32,
    /// How brightly an emissive voxel's own faces glow, as a multiple of voxel
    /// light at level 15, at least 0. A lamp declared below level 15 glows
    /// less, by `voxel_falloff` at its level.
    ///
    /// Shows on the next frame, for one uniform write. Defaults to 6.
    pub emissive_glow: f32,
    /// Whether an emissive voxel's own faces glow in their base colour tinted by
    /// its light colour, rather than in their base colour alone.
    ///
    /// Off, the material decides how an emissive voxel looks, and a white one
    /// glows the same either way. On, the faces of coloured emissive voxels cost
    /// a little more shader work, and no other face does.
    ///
    /// Shows on the next frame, for one uniform write. Defaults to `false`.
    pub tinted_glow: bool,
    /// What the mesh's ambient occlusion darkens.
    ///
    /// Shows on the next frame, for one uniform write. Defaults to
    /// [`LightAo::IndirectLight`].
    pub ao: LightAo,
    /// The least share of Bevy's ambient light any face gets, at least 0, so
    /// caves needn't be pitch black. At 0, a face no light reaches is exactly
    /// black.
    ///
    /// Shows on the next frame, for one uniform write. Defaults to 0.
    pub minimum_brightness: f32,
}

impl Default for LightLook {
    fn default() -> Self {
        Self {
            smoothing: LightSmoothing::default(),
            sky_falloff: LevelCurve::default(),
            voxel_falloff: LevelCurve::exponential_to_black(0.7),
            sun: SunGate::default(),
            voxel_brightness: 1000.0,
            emissive_glow: 6.0,
            tinted_glow: false,
            ao: LightAo::default(),
            minimum_brightness: 0.0,
        }
    }
}

impl LightLook {
    /// This look with every out-of-range value clamped, logging a warning
    /// once for each kind of value clamped.
    pub(crate) fn clamped(self) -> Self {
        Self {
            voxel_brightness: clamp_setting(
                self.voxel_brightness,
                0.0,
                f32::MAX,
                "voxel_brightness",
            ),
            emissive_glow: clamp_setting(
                self.emissive_glow,
                0.0,
                f32::MAX,
                "emissive_glow",
            ),
            minimum_brightness: clamp_setting(
                self.minimum_brightness,
                0.0,
                f32::MAX,
                "minimum_brightness",
            ),
            sky_falloff: self
                .sky_falloff
                .clamped(0.0, f32::MAX, "a sky_falloff entry"),
            voxel_falloff: self.voxel_falloff.clamped(
                0.0,
                f32::MAX,
                "a voxel_falloff entry",
            ),
            sun: match self.sun {
                SunGate::ByLevel(curve) => {
                    SunGate::ByLevel(curve.clamped(0.0, 1.0, "a sun curve entry"))
                }
                direct => direct,
            },
            ..self
        }
    }
}

/// Whether light blends between neighbouring voxels on a face.
#[derive(Reflect, Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum LightSmoothing {
    /// Light blends smoothly across a face.
    #[default]
    Smoothed,
    /// Each voxel's light shows as a flat step.
    Stepped,
}

/// The brightness of each light level from 0 to 15, as 16 entries.
///
/// Level 0 is always black: the entry for it is read as 0 whatever it holds,
/// so a voxel no light reaches stays black under any curve. Brightness is at
/// least 0, and a negative or NaN entry is clamped to 0 with a warning.
#[derive(Reflect, Clone, PartialEq, Debug)]
pub struct LevelCurve {
    pub(crate) brightness: [f32; 16],
}

impl Default for LevelCurve {
    /// `0.7^(15 - level)`, so each level is 70% as bright as the one above it.
    /// It's the default [`sky_falloff`](LightLook::sky_falloff). The default
    /// [`voxel_falloff`](LightLook::voxel_falloff) is
    /// [`exponential_to_black(0.7)`](Self::exponential_to_black).
    fn default() -> Self {
        Self::exponential(0.7)
    }
}

impl LevelCurve {
    /// `base^(15 - level)`: each level is `base` times as bright as the one
    /// above it, and level 15 is 1.
    pub fn exponential(base: f32) -> Self {
        // Repeated multiplication rather than `powi`, which the compiler may
        // work out at compile time to different last bits than at run time:
        // this way `exponential(0.7) == exponential(0.7)` wherever each is
        // built.
        Self::from_fn(|level| (level..15).fold(1.0, |brightness, _| brightness * base))
    }

    /// `(base^(15 - level) - base^14) / (1 - base^14)`: as
    /// [`exponential`](Self::exponential), shifted and scaled so level 1 is black
    /// and level 15 is 1. A pool of light then fades out at its edge instead of
    /// ending in a faint step. At a `base` of 1 it's linear, from black at
    /// level 1 to 1 at level 15.
    pub fn exponential_to_black(base: f32) -> Self {
        let exponential = Self::exponential(base);
        let floor = exponential.brightness[1];
        if 1.0 - floor <= f32::EPSILON {
            return Self::from_fn(|level| level.saturating_sub(1) as f32 / 14.0);
        }
        Self::from_fn(|level| {
            if level == 0 {
                0.0
            } else {
                (exponential.brightness[level as usize] - floor) / (1.0 - floor)
            }
        })
    }

    /// `(level / 15)^gamma`, so level 15 is 1. `gamma(2.2)` spreads the levels
    /// evenly to the eye on a display with that gamma.
    pub fn gamma(gamma: f32) -> Self {
        Self::from_fn(|level| (level as f32 / 15.0).powf(gamma))
    }

    /// `level / 15`, so level 15 is 1.
    pub fn linear() -> Self {
        Self::from_fn(|level| level as f32 / 15.0)
    }

    /// The brightness `brightness(level)` returns for each level from 0 to 15.
    pub fn from_fn(brightness: impl FnMut(u8) -> f32) -> Self {
        let mut brightness = brightness;
        Self {
            brightness: std::array::from_fn(|level| brightness(level as u8)),
        }
    }

    /// The brightness of each level from 0 to 15, in order.
    pub fn from_table(brightness: [f32; 16]) -> Self {
        Self { brightness }
    }

    /// The brightness of `level`, as the shader reads it: 0 at level 0 or
    /// below, the entry at a whole level, between the two entries around a
    /// level between whole levels, and level 15's above 15.
    ///
    /// Gameplay code can dim its own entities the way voxels dim: read the
    /// light at the entity's voxel with
    /// [`VoxelWorld::light_at`](crate::prelude::VoxelWorld::light_at), and
    /// scale the entity's colour by `sky_falloff.at(levels.sky)` for the
    /// ambient light it gets, and by `voxel_falloff.at(levels.voxel)` times
    /// `voxel_brightness` times `levels.voxel_color` for voxel light, using the
    /// world's [`LightLook`].
    pub fn at(&self, level: f32) -> f32 {
        if level.is_nan() || level <= 0.0 {
            return 0.0;
        }
        let level = level.min(15.0);
        let low = level.floor() as usize;
        let high = (low + 1).min(15);
        let entry = |level: usize| {
            if level == 0 {
                0.0
            } else {
                self.brightness[level]
            }
        };
        let t = level - low as f32;
        entry(low) * (1.0 - t) + entry(high) * t
    }

    /// This curve with every entry kept within `min..=max`, logging a warning
    /// once if one wasn't. NaN becomes `min`.
    fn clamped(self, min: f32, max: f32, name: &str) -> Self {
        Self {
            brightness: self
                .brightness
                .map(|entry| clamp_setting(entry, min, max, name)),
        }
    }
}

/// Where Bevy's directional lights shine.
#[derive(Reflect, Clone, PartialEq, Debug, Default)]
pub enum SunGate {
    /// Only where a voxel has direct sky, so the sun never shines through rock
    /// that hasn't spawned.
    ///
    /// Bevy's shadow map holds only the chunks that have spawned, and by
    /// default chunks spawn around the camera and in view, so rock out of view
    /// casts no shadow and the shadow map would let the sun into caves. Direct
    /// sky is worked out from the world's voxels instead. Since sky light falls
    /// straight down, the sun stops at an overhang's drip line, however the
    /// sun is tilted, and ground under the overhang gets only spreading sky
    /// light.
    #[default]
    DirectSkyOnly,
    /// Scaled by the sky light level through this curve, from 0, no sun, to
    /// 1, the whole sun, so the sun follows sky light past direct sky and
    /// fades as it spreads. Entries are clamped to 0 to 1.
    ///
    /// Bevy's shadow map still applies, but rock that hasn't spawned casts no
    /// shadow, so a curve that lets the sun below level 15 can let it into
    /// caves where their sky light reaches.
    ByLevel(LevelCurve),
}

/// What the ambient occlusion of a lit chunk's mesh darkens.
///
/// The ambient occlusion is what the mesher puts in the mesh's vertex colour.
#[derive(Reflect, Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum LightAo {
    /// Bevy's ambient light and voxel light, leaving the base colour and the
    /// sun alone, so rock keeps its shape in dim places and sunlit faces show
    /// no dark corners.
    #[default]
    IndirectLight,
    /// The base colour, so every light darkens in corners, the sun's too, as
    /// without light propagation.
    BaseColor,
    /// Nothing: ambient occlusion is ignored.
    Off,
}

/// The light an emissive material gives off: a level, and a light colour.
///
/// Built with [`Emission::level`], which gives white light, and coloured
/// with [`with_color`](Self::with_color).
#[derive(Reflect, Clone, Copy, PartialEq, Debug)]
pub struct Emission {
    level: f32,
    /// The light colour, scaled so its brightest component is 1, or black.
    color: LinearRgba,
}

impl Emission {
    /// White light at `level`, from 0 to 15. A voxel at level 8 lights about
    /// half as far as one at 15.
    pub fn level(level: f32) -> Self {
        Self {
            level,
            color: LinearRgba::WHITE,
        }
    }

    /// This emission with its light in `color`.
    ///
    /// Only the colour's hue and saturation count. It's scaled so its brightest
    /// component is 1, so `Color::srgb(0.5, 0.0, 0.0)` gives the same light as
    /// `Color::srgb(1.0, 0.0, 0.0)`, and grey gives white. The level alone sets
    /// how bright the light is and how far it reaches. Black gives no light, as
    /// level 0 does. Alpha is ignored, and a negative component is read as 0,
    /// with a warning logged once.
    pub fn with_color(self, color: impl Into<Color>) -> Self {
        let (color, negative) = super::color::scaled(color.into().to_linear());
        if negative {
            warn_once!(
                "light propagation: an emission colour has a negative component, \
                 which is read as 0"
            );
        }
        Self { color, ..self }
    }

    /// The level as a whole level from 0 to 15, and 0 for black light.
    pub(crate) fn whole_level(&self) -> u8 {
        if self.color == LinearRgba::BLACK {
            return 0;
        }
        self.level.round().clamp(0.0, 15.0) as u8
    }

    /// The light colour, with its brightest component 1, or black.
    pub(crate) fn color(&self) -> LinearRgba {
        self.color
    }

    /// This emission with its level clamped to 0 to 15, logging a warning once
    /// if it wasn't.
    pub(crate) fn clamped(self) -> Self {
        Self {
            level: clamp_setting(self.level, 0.0, 15.0, "emission level"),
            ..self
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn each_setting_warns_the_first_time_it_is_out_of_range() {
        assert!(first_warning_for("a test setting"));
        assert!(first_warning_for("another test setting"));
        assert!(!first_warning_for("a test setting"));
    }
}
