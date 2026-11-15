//! The pieces `chunk_lifecycle`'s lift burst scenarios lift out of the sine
//! terrain: 100 boxes, each lifted whole with every solid voxel it holds.
//!
//! A piece's chunk count is its sub-grid's: the lift puts the lowest solid
//! voxels at (1, 1, 1), and a chunk joins wherever a solid voxel lies within
//! one voxel of a chunk border, so a piece fits one chunk along an axis only
//! while its solid voxels span at most 30 voxels there. Each box reaches
//! below the surface under every one of its columns, so its solid voxels
//! span its whole width and depth, and from its lowest corner up to the
//! highest surface voxel inside it.
//!
//! 80 pieces of 12 by 12 voxels take one chunk each. 15 take two or four,
//! by being 40 voxels long along one or two axes, or deep. 5 take eight,
//! by being 40 voxels long along all three. The small pieces stand 20
//! voxels apart, so some of them cross chunk corners of the world, and the
//! large ones 48 apart. Every box lies within 7 chunks of the origin, where
//! the bench's camera starts, so its chunks are streamed in.

use bevy::prelude::*;
use bevy_voxel_world::custom_meshing::CHUNK_SIZE_I;

/// A box of the sine terrain to lift, and what its sub-grid should hold.
#[derive(Clone, Copy, Debug)]
pub struct Piece {
    /// The box's lowest corner, inclusive.
    pub min: IVec3,
    /// The box's highest corner, exclusive. Its top is above the terrain.
    pub max: IVec3,
    /// How many level-0 chunks the piece's sub-grid should have, padding
    /// included.
    pub chunks: usize,
}

impl Piece {
    /// Whether the box's solid voxels cross a chunk border of the world
    /// along both x and z, so the lift and the stamp touch the four chunks
    /// around a chunk corner, or eight when the piece also crosses one
    /// along y.
    pub fn crosses_chunk_corner(&self) -> bool {
        let crosses = |min: i32, max: i32| {
            min.div_euclid(CHUNK_SIZE_I) != (max - 1).div_euclid(CHUNK_SIZE_I)
        };
        crosses(self.min.x, self.max.x) && crosses(self.min.z, self.max.z)
    }
}

/// The y every box reaches up to, exclusive: above the highest voxel of
/// the sine terrain.
const TOP: i32 = 20;

/// A piece's footprint along x and z, and how far below the highest surface
/// voxel inside it the box reaches.
struct Shape {
    width: i32,
    length: i32,
    depth: i32,
}

const SMALL: Shape = Shape {
    width: 12,
    length: 12,
    depth: 10,
};

/// The shapes of the 20 larger pieces, in the order they take their sites.
const LARGE: [(Shape, usize); 4] = [
    (
        Shape {
            width: 40,
            length: 40,
            depth: 44,
        },
        5,
    ),
    (
        Shape {
            width: 40,
            length: 40,
            depth: 12,
        },
        5,
    ),
    (
        Shape {
            width: 40,
            length: 12,
            depth: 10,
        },
        5,
    ),
    (
        Shape {
            width: 12,
            length: 12,
            depth: 44,
        },
        5,
    ),
];

/// The pieces, given the sine terrain's `height`: the first voxel of air
/// above its surface at x, z.
pub fn pieces(height: fn(i32, i32) -> i32) -> Vec<Piece> {
    let small = (0..8).flat_map(|row| {
        (0..10)
            .map(move |column| (IVec2::new(-100 + 20 * column, -100 + 20 * row), &SMALL))
    });
    let large_sites = (0..8)
        .map(|k| IVec2::new(-164 + 48 * k, 76))
        .chain((0..8).map(|k| IVec2::new(-164 + 48 * k, 128)))
        .chain((0..4).map(|k| IVec2::new(-68 + 48 * k, -152)));
    let large_shapes = LARGE
        .iter()
        .flat_map(|(shape, count)| std::iter::repeat_n(shape, *count));
    small
        .chain(large_sites.zip(large_shapes))
        .map(|(corner, shape)| piece(height, corner, shape))
        .collect()
}

fn piece(height: fn(i32, i32) -> i32, corner: IVec2, shape: &Shape) -> Piece {
    let heights: Vec<i32> = (corner.x..corner.x + shape.width)
        .flat_map(|x| (corner.y..corner.y + shape.length).map(move |z| height(x, z)))
        .collect();
    let lowest = *heights.iter().min().unwrap();
    let highest = *heights.iter().max().unwrap();
    // Two solid voxels under every column at least, so the solid voxels
    // span the box's whole footprint.
    let bottom = (highest - shape.depth).min(lowest - 2);
    let span = IVec3::new(shape.width, highest - bottom, shape.length);
    // Solid voxels from 1 to `span` on the sub-grid, and padding up to
    // `span + 1`.
    let chunks = ((span + 1) / CHUNK_SIZE_I + 1).element_product() as usize;
    Piece {
        min: IVec3::new(corner.x, bottom, corner.y),
        max: IVec3::new(corner.x + shape.width, TOP, corner.y + shape.length),
        chunks,
    }
}
