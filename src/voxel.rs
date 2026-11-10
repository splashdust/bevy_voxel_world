use bevy::math::Vec3;
use block_mesh::{MergeVoxel, Voxel, VoxelVisibility};

pub const VOXEL_SIZE: f32 = 1.;

/// One voxel of a world, holding the world's material index `I` when solid.
///
/// The variant order is part of the crate's public API, so that a serialised
/// voxel keeps its layout across releases.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum WorldVoxel<I = u8> {
    #[default]
    Unset,
    Air,
    Solid(I),
}

impl<I: PartialEq> WorldVoxel<I> {
    pub fn is_unset(&self) -> bool {
        *self == WorldVoxel::Unset
    }

    pub fn is_air(&self) -> bool {
        *self == WorldVoxel::Air
    }

    pub fn is_solid(&self) -> bool {
        matches!(self, WorldVoxel::Solid(_))
    }
}

impl<I: PartialEq> Voxel for WorldVoxel<I> {
    fn get_visibility(&self) -> VoxelVisibility {
        if *self == WorldVoxel::Air || *self == WorldVoxel::Unset {
            VoxelVisibility::Empty
        } else {
            VoxelVisibility::Opaque
        }
    }
}

impl<I: PartialEq + Eq + Default + Copy> MergeVoxel for WorldVoxel<I> {
    type MergeValue = I;

    fn merge_value(&self) -> Self::MergeValue {
        match self {
            WorldVoxel::Solid(v) => *v,
            _ => I::default(),
        }
    }
}

#[derive(Eq, PartialEq, Copy, Clone, Debug)]
pub enum VoxelFace {
    None,
    Bottom,
    Top,
    Left,
    Right,
    Back,
    Forward,
}

impl TryFrom<VoxelFace> for Vec3 {
    type Error = ();

    fn try_from(value: VoxelFace) -> Result<Self, Self::Error> {
        match value {
            VoxelFace::None => Err(()),
            VoxelFace::Bottom => Ok(-Vec3::Y),
            VoxelFace::Top => Ok(Vec3::Y),
            VoxelFace::Left => Ok(-Vec3::X),
            VoxelFace::Right => Ok(Vec3::X),
            VoxelFace::Back => Ok(-Vec3::Z),
            VoxelFace::Forward => Ok(Vec3::Z),
        }
    }
}
