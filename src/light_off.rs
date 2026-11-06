//! Stand-ins for light propagation's types in a build without the
//! `light_propagation` feature. Naming any of them fails to compile with an
//! error that says to enable the feature, instead of one that says the type
//! doesn't exist.
//!
//! Overriding the light config method names `LightPropagation`, so it fails the
//! same way:
//!
//! ```compile_fail,E0277
//! use bevy::prelude::*;
//! use bevy_voxel_world::prelude::*;
//!
//! #[derive(Resource, Clone, Default)]
//! struct MyWorld;
//!
//! impl VoxelWorldConfig for MyWorld {
//!     type MaterialIndex = u8;
//!     type ChunkUserBundle = ();
//!
//!     fn light_propagation(&self) -> Option<LightPropagation> {
//!         Some(LightPropagation::default())
//!     }
//! }
//! ```
//!
//! Naming a light type alone fails too:
//!
//! ```compile_fail,E0277
//! use bevy_voxel_world::prelude::*;
//!
//! let _ = Emission::level(15.0);
//! ```

/// Implemented by nothing, so every type below fails to resolve with this
/// message.
#[doc(hidden)]
#[diagnostic::on_unimplemented(
    message = "light propagation needs the `light_propagation` cargo feature of bevy_voxel_world",
    label = "used without the `light_propagation` feature",
    note = "enable it in Cargo.toml: bevy_voxel_world = {{ version = \"..\", features = [\"light_propagation\"] }}"
)]
pub trait NeedsLightPropagationFeature {
    type Type;
}

macro_rules! needs_feature {
    ($($name:ident),*) => {
        $(
            #[doc(hidden)]
            pub type $name = <() as NeedsLightPropagationFeature>::Type;
        )*
    };
}

needs_feature!(
    LightPropagation,
    Sky,
    SkyFill,
    PathFill,
    OpenFill,
    LightShape,
    LightPrecision,
    LightLook,
    LightSmoothing,
    LevelCurve,
    SunGate,
    LightAo,
    Emission
);
