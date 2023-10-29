use block_mesh::{MergeVoxel, Voxel, VoxelVisibility};

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum WorldVoxel {
    Unset,
    Air,
    Solid(u8),
}

impl WorldVoxel {
    pub fn is_unset(&self) -> bool {
        *self == WorldVoxel::Unset
    }
}

impl Voxel for WorldVoxel {
    fn get_visibility(&self) -> VoxelVisibility {
        if *self == WorldVoxel::Air || *self == WorldVoxel::Unset {
            VoxelVisibility::Empty
        } else {
            VoxelVisibility::Opaque
        }
    }
}

impl MergeVoxel for WorldVoxel {
    type MergeValue = u8;

    fn merge_value(&self) -> Self::MergeValue {
        match self {
            WorldVoxel::Solid(v) => *v,
            _ => 0,
        }
    }
}
