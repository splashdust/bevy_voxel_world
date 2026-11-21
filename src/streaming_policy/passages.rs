//! Passages through chunks
//!
//! This module works out the passages of a chunk: which pairs of its six faces
//! its air joins. The chunk task works them out from the voxels of the chunk
//! once they are sampled, and stores them in its `ChunkData`. The streaming
//! policy reads them back through `Occupancy::Mixed`, and lets a spawning ray
//! through a chunk that is solid in places only from a face to a face its air
//! joins.
//!
//! A ray sees through a chunk only along a line of air voxels. Such a line
//! steps from voxel to voxel across their faces, so it stays inside one body
//! of air joined across voxel faces. The ray can therefore only leave by a face
//! that body touches, and the body touches the face the ray came in by too. The
//! rule never stops a ray that would see something, though it lets through
//! many rays that a solid voxel stops further in.
//!
//! A mesh that draws its surface between voxel samples can show air between
//! voxels that share only an edge or a corner. Under
//! `RayPassage::CornerJoined`, air joins across those too.
//!
//! Every solid voxel counts as opaque, as it does for `Occupancy`.
//!
//! # Cost
//!
//! The air is read into rows of one bit per voxel, and each body of air is
//! filled a run of a row at a time. The cost grows with the data voxels of the
//! chunk for reading the rows, and with the runs of air for the fill. Joining
//! across edges and corners visits eight neighbouring rows from each run in
//! place of four, and fills the padding too.

use bevy::math::UVec3;

use super::Face;
use crate::{chunk::ChunkData, configuration::RayPassage, voxel::WorldVoxel};

/// Which pairs of the six faces of a chunk its air joins, one bit per pair.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct Passages(u16);

impl Passages {
    /// Every pair of faces is joined, as in a chunk of air.
    pub const ALL: Self = Self((1 << 15) - 1);

    /// No pair of faces is joined, as in a solid chunk.
    pub const NONE: Self = Self(0);

    /// Whether the air joins faces `a` and `b`. A face counts as joined to
    /// itself.
    pub fn joins(self, a: Face, b: Face) -> bool {
        a == b || self.0 & pair_bit(a, b) != 0
    }

    /// The passages of one body of air that touches `faces`.
    #[cfg(test)]
    pub(crate) fn touching(faces: &[Face]) -> Self {
        Self::of_touched(faces.iter().fold(0, |touched, face| touched | 1 << face.0))
    }

    /// The passages of a body of air touching the faces whose bits are set in
    /// `touched`, bit `f` for face `f`.
    fn of_touched(touched: u8) -> Self {
        let mut pairs = 0;
        for a in 0..6 {
            for b in a + 1..6 {
                if touched >> a & 1 == 1 && touched >> b & 1 == 1 {
                    pairs |= pair_bit(Face(a), Face(b));
                }
            }
        }
        Self(pairs)
    }
}

/// The bit of the pair of faces `a` and `b`, which differ.
///
/// The pairs are numbered in order: faces 0 and 1 are pair 0, faces 0 and 5
/// are pair 4, faces 1 and 2 are pair 5, and faces 4 and 5 are pair 14.
fn pair_bit(a: Face, b: Face) -> u16 {
    let (low, high) = (a.0.min(b.0), a.0.max(b.0));
    let first = [0, 5, 9, 12, 14][low as usize];
    1 << (first + high - low - 1)
}

/// The passages of the chunk with `data`, under `rule`.
///
/// A chunk with no air has no passages, and a chunk with no solid voxels joins
/// every pair of faces. So does any chunk under [`RayPassage::AnyAir`].
pub(crate) fn passages_of<I: PartialEq>(
    data: &ChunkData<I>,
    rule: RayPassage,
) -> Passages {
    if data.is_full {
        return Passages::NONE;
    }
    let Some(voxels) = data.voxels.as_deref().filter(|_| !data.is_empty) else {
        return Passages::ALL;
    };
    let shape = data.data_shape;
    if shape.cmplt(UVec3::splat(3)).any() {
        // No voxels inside the padding, so nothing to block a ray.
        return Passages::ALL;
    }
    match rule {
        RayPassage::AnyAir => Passages::ALL,
        RayPassage::FaceJoined => {
            let air = AirRows::of(voxels, shape, UVec3::ONE, shape - 2);
            let high = UVec3::new(air.cells_x - 1, air.size.y - 1, air.size.z - 1);
            joined(&air, FACE_NEIGHBOURS, UVec3::ZERO, high)
        }
        RayPassage::CornerJoined => {
            // A surface drawn between voxel samples near a face of the chunk
            // comes from the voxels on both sides of the face, so the padding
            // is filled too. The faces of the chunk are still the first and
            // last layers inside the padding.
            let air = AirRows::of(voxels, shape, UVec3::ZERO, shape);
            let low = UVec3::new(air.cell_x(1), 1, 1);
            let high = UVec3::new(air.cell_x(shape.x - 2), shape.y - 2, shape.z - 2);
            joined(&air, CORNER_NEIGHBOURS, low, high)
        }
    }
}

/// The air of a box of voxels, as rows along x of one bit per voxel.
///
/// A row holds 64 bits. In a box wider than that, neighbouring voxels along x
/// share a bit, which holds air when any of them does. Two bodies of air that
/// are apart can then join, but no body is cut in two, so the passages only
/// gain pairs.
struct AirRows {
    /// The box, in voxels.
    size: UVec3,
    /// The bits along x: `size.x`, or 64 in a wider box.
    cells_x: u32,
    /// Row `y + z * size.y` holds the voxels at `y` and `z`, voxel `x` in bit
    /// `cell_x(x)`.
    rows: Vec<u64>,
}

impl AirRows {
    /// The air of the box of `size` voxels at `min` in `voxels`, which have the
    /// padded data `shape`, x fastest.
    fn of<I: PartialEq>(
        voxels: &[WorldVoxel<I>],
        shape: UVec3,
        min: UVec3,
        size: UVec3,
    ) -> Self {
        let mut air = Self {
            size,
            cells_x: size.x.min(64),
            rows: vec![0; (size.y * size.z) as usize],
        };
        let (row_length, slab) = (shape.x as usize, (shape.x * shape.y) as usize);
        for z in 0..size.z {
            for y in 0..size.y {
                let start = (min.z + z) as usize * slab
                    + (min.y + y) as usize * row_length
                    + min.x as usize;
                let row = &voxels[start..start + size.x as usize];
                let mut bits = 0u64;
                if air.cells_x == size.x {
                    for (x, voxel) in row.iter().enumerate() {
                        bits |= u64::from(!voxel.is_solid()) << x;
                    }
                } else {
                    for (x, voxel) in row.iter().enumerate() {
                        bits |= u64::from(!voxel.is_solid()) << air.cell_x(x as u32);
                    }
                }
                air.rows[(y + z * size.y) as usize] = bits;
            }
        }
        air
    }

    /// The bit of voxel `x` in a row.
    fn cell_x(&self, x: u32) -> u32 {
        // Voxels next to each other along x land in the same bit or in bits
        // next to each other, and the last voxel lands in the last bit.
        (x as u64 * self.cells_x as u64 / self.size.x as u64) as u32
    }
}

/// The rows a run of air joins across voxel faces, as offsets in y and z.
const FACE_NEIGHBOURS: &[(i32, i32)] = &[(-1, 0), (1, 0), (0, -1), (0, 1)];

/// The rows a run of air joins across voxel faces, edges and corners, as
/// offsets in y and z.
const CORNER_NEIGHBOURS: &[(i32, i32)] = &[
    (-1, -1),
    (0, -1),
    (1, -1),
    (-1, 0),
    (1, 0),
    (-1, 1),
    (0, 1),
    (1, 1),
];

/// Which pairs of faces the air of `air` joins.
///
/// A run of air joins the runs it overlaps in the rows at `neighbours`. With
/// [`CORNER_NEIGHBOURS`], a run also joins the runs one bit to either side of
/// it in those rows, which share an edge or a corner with it. A body of air
/// touches the low x face where it has air in bit `low.x`, the high x face
/// where it has air in bit `high.x`, and likewise for y and z.
fn joined(air: &AirRows, neighbours: &[(i32, i32)], low: UVec3, high: UVec3) -> Passages {
    let corners = neighbours.len() > FACE_NEIGHBOURS.len();
    let (size_y, size_z) = (air.size.y as i32, air.size.z as i32);
    let in_row = u64::MAX >> (64 - air.cells_x);
    let (low_x, high_x) = (1u64 << low.x, 1u64 << high.x);
    let mut visited = vec![0u64; air.rows.len()];
    // Rows to visit, each with the bits of air to fill the runs from.
    let mut stack: Vec<(usize, u64)> = Vec::new();
    let mut joined = 0u16;
    for start in 0..air.rows.len() {
        loop {
            let unvisited = air.rows[start] & !visited[start];
            if unvisited == 0 {
                break;
            }
            // A new body of air, filled from its first voxel in this row.
            let mut touched = 0u8;
            stack.push((start, unvisited.isolate_lowest_one()));
            while let Some((row, seeds)) = stack.pop() {
                let open = air.rows[row] & !visited[row];
                if open & seeds == 0 {
                    continue;
                }
                let run = runs_through(open, seeds);
                visited[row] |= run;
                let (y, z) = (row as u32 % air.size.y, row as u32 / air.size.y);
                touched |= u8::from(run & low_x != 0)
                    | u8::from(run & high_x != 0) << 1
                    | u8::from(y == low.y) << 2
                    | u8::from(y == high.y) << 3
                    | u8::from(z == low.z) << 4
                    | u8::from(z == high.z) << 5;
                let reach = if corners {
                    (run | run << 1 | run >> 1) & in_row
                } else {
                    run
                };
                for &(dy, dz) in neighbours {
                    let (next_y, next_z) = (y as i32 + dy, z as i32 + dz);
                    if !(0..size_y).contains(&next_y) || !(0..size_z).contains(&next_z) {
                        continue;
                    }
                    let next = (next_y + next_z * size_y) as usize;
                    let seeds = reach & air.rows[next] & !visited[next];
                    if seeds != 0 {
                        stack.push((next, seeds));
                    }
                }
            }
            joined |= Passages::of_touched(touched).0;
            if joined == Passages::ALL.0 {
                return Passages::ALL;
            }
        }
    }
    Passages(joined)
}

/// The runs of set bits in `row` that hold a bit of `seeds`.
///
/// Adding a seed to the row carries through the set bits above it, which
/// flips the run above the seed. The same on the reversed bits flips the run
/// below it.
fn runs_through(row: u64, seeds: u64) -> u64 {
    let seeds = seeds & row;
    let up = row & (row.wrapping_add(seeds) ^ row);
    let (row_down, seeds_down) = (row.reverse_bits(), seeds.reverse_bits());
    let down = (row_down & (row_down.wrapping_add(seeds_down) ^ row_down)).reverse_bits();
    seeds | up | down
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chunk::{FillType, PADDED_CHUNK_SIZE};
    use bevy::math::{IVec3, Vec3};
    use rand::{rngs::StdRng, RngExt, SeedableRng};
    use std::{collections::VecDeque, sync::Arc};

    /// Chunk data of `shape` whose voxel at each padded position is solid
    /// where `solid` says.
    fn chunk(shape: UVec3, solid: impl Fn(UVec3) -> bool) -> ChunkData<u8> {
        let mut voxels = Vec::new();
        for z in 0..shape.z {
            for y in 0..shape.y {
                for x in 0..shape.x {
                    voxels.push(if solid(UVec3::new(x, y, z)) {
                        WorldVoxel::Solid(1)
                    } else {
                        WorldVoxel::Air
                    });
                }
            }
        }
        let solid_count = voxels.iter().filter(|voxel| voxel.is_solid()).count();
        let mut data = ChunkData::new();
        data.data_shape = shape;
        data.is_empty = solid_count == 0;
        data.is_full = solid_count == voxels.len();
        data.fill_type = FillType::Mixed;
        data.voxels = Some(Arc::from(voxels));
        data
    }

    /// The default padded shape.
    const SHAPE: UVec3 = UVec3::splat(PADDED_CHUNK_SIZE);

    /// The faces named by their sides and axes.
    const LOW_X: Face = Face(0);
    const HIGH_X: Face = Face(1);
    const LOW_Y: Face = Face(2);
    const HIGH_Y: Face = Face(3);
    const LOW_Z: Face = Face(4);
    const HIGH_Z: Face = Face(5);

    #[test]
    fn every_pair_of_faces_has_a_bit_of_its_own() {
        let mut bits = 0u16;
        for a in 0..6 {
            for b in a + 1..6 {
                let bit = pair_bit(Face(a), Face(b));
                assert_eq!(bit, pair_bit(Face(b), Face(a)));
                assert_eq!(bits & bit, 0, "faces {a} and {b} share a bit");
                bits |= bit;
            }
        }
        assert_eq!(Passages(bits), Passages::ALL);
    }

    #[test]
    fn runs_through_fills_the_runs_that_hold_a_seed() {
        let row = 0b0111_0110_1110u64 | 1 << 63;
        assert_eq!(runs_through(row, 0b0000_0100_0000), 0b0000_0110_0000);
        assert_eq!(runs_through(row, 0b0100_0000_1000), 0b0111_0000_1110);
        assert_eq!(runs_through(row, 1 << 63), 1 << 63);
        assert_eq!(runs_through(row, 0b1000_0001_0001), 0);
        assert_eq!(runs_through(u64::MAX, 1 << 20), u64::MAX);
    }

    #[test]
    fn a_straight_tunnel_joins_only_the_faces_at_its_ends() {
        // Air along x at y and z 10 to 12, inside the padding.
        let data = chunk(SHAPE, |p| {
            !((10..13).contains(&p.y) && (10..13).contains(&p.z))
        });

        let passages = passages_of(&data, RayPassage::FaceJoined);

        assert_eq!(passages, Passages::touching(&[LOW_X, HIGH_X]));
    }

    #[test]
    fn a_bent_tunnel_joins_the_faces_it_turns_between() {
        // Air from the low x face to the middle, then up to the high y face.
        let data = chunk(SHAPE, |p| {
            let along_x = p.x <= 16 && (15..18).contains(&p.y) && p.z == 16;
            let up = (14..17).contains(&p.x) && p.y >= 15 && p.z == 16;
            !(along_x || up)
        });

        let passages = passages_of(&data, RayPassage::FaceJoined);

        assert_eq!(passages, Passages::touching(&[LOW_X, HIGH_Y]));
    }

    #[test]
    fn two_tunnels_join_only_their_own_faces() {
        let data = chunk(SHAPE, |p| {
            let along_x = p.y == 8 && p.z == 8;
            let along_z = p.x == 24 && p.y == 24;
            !(along_x || along_z)
        });

        let passages = passages_of(&data, RayPassage::FaceJoined);

        assert_eq!(
            passages.0,
            Passages::touching(&[LOW_X, HIGH_X]).0
                | Passages::touching(&[LOW_Z, HIGH_Z]).0
        );
        assert!(!passages.joins(LOW_X, LOW_Z));
    }

    #[test]
    fn air_in_the_padding_alone_joins_no_faces_across_voxel_faces() {
        let data = chunk(SHAPE, |p| (1..33).contains(&p.x) && (1..33).contains(&p.y));

        assert_eq!(passages_of(&data, RayPassage::FaceJoined), Passages::NONE);
    }

    #[test]
    fn air_joined_only_across_an_edge_joins_faces_only_when_corners_count() {
        // Two tunnels, one from the low x face and one from the high y face,
        // that meet only along the edge of two voxels.
        let data = chunk(SHAPE, |p| {
            let from_x = p.x <= 16 && p.y == 16 && p.z == 16;
            let from_y = p.x == 17 && p.y >= 17 && p.z == 16;
            !(from_x || from_y)
        });

        let faces = passages_of(&data, RayPassage::FaceJoined);
        let corners = passages_of(&data, RayPassage::CornerJoined);

        assert!(!faces.joins(LOW_X, HIGH_Y));
        assert!(corners.joins(LOW_X, HIGH_Y));
    }

    #[test]
    fn a_chunk_without_solid_voxels_joins_every_face_and_a_full_one_none() {
        let air = chunk(SHAPE, |_| false);
        let solid = chunk(SHAPE, |_| true);
        let mixed = chunk(SHAPE, |p| p.y < 16);

        for rule in [
            RayPassage::FaceJoined,
            RayPassage::CornerJoined,
            RayPassage::AnyAir,
        ] {
            assert_eq!(passages_of(&air, rule), Passages::ALL);
            assert_eq!(passages_of(&solid, rule), Passages::NONE);
        }
        assert_eq!(passages_of(&mixed, RayPassage::AnyAir), Passages::ALL);
        assert!(!passages_of(&mixed, RayPassage::FaceJoined).joins(LOW_Y, HIGH_Y));
    }

    /// The passages of `data` found one voxel at a time, joining voxels that
    /// share a face, or with `corners` any that touch.
    fn reference(data: &ChunkData<u8>, corners: bool) -> Passages {
        let shape = data.data_shape.as_ivec3();
        let voxels = data.voxels.as_ref().unwrap();
        // The filled box, and the layers that touch each face.
        let (min, max) = if corners {
            (IVec3::ZERO, shape - 1)
        } else {
            (IVec3::ONE, shape - 2)
        };
        let index = |p: IVec3| (p.x + shape.x * (p.y + shape.y * p.z)) as usize;
        let is_air = |p: IVec3| !voxels[index(p)].is_solid();
        let mut seen = vec![false; voxels.len()];
        let mut joined = Passages::NONE;
        for z in min.z..=max.z {
            for y in min.y..=max.y {
                for x in min.x..=max.x {
                    let start = IVec3::new(x, y, z);
                    if !is_air(start) || seen[index(start)] {
                        continue;
                    }
                    seen[index(start)] = true;
                    let mut touched = 0u8;
                    let mut queue = VecDeque::from([start]);
                    while let Some(p) = queue.pop_front() {
                        for axis in 0..3 {
                            touched |= u8::from(p[axis] == 1) << (2 * axis);
                            touched |=
                                u8::from(p[axis] == shape[axis] - 2) << (2 * axis + 1);
                        }
                        for dz in -1..=1 {
                            for dy in -1..=1 {
                                for dx in -1..=1 {
                                    let offset = IVec3::new(dx, dy, dz);
                                    let steps = offset.abs().element_sum();
                                    if steps == 0 || (!corners && steps > 1) {
                                        continue;
                                    }
                                    let next = p + offset;
                                    if next.cmplt(min).any()
                                        || next.cmpgt(max).any()
                                        || !is_air(next)
                                        || seen[index(next)]
                                    {
                                        continue;
                                    }
                                    seen[index(next)] = true;
                                    queue.push_back(next);
                                }
                            }
                        }
                    }
                    joined.0 |= Passages::of_touched(touched).0;
                }
            }
        }
        joined
    }

    /// Chunks of `shape` with random air, from sparse to dense, and with
    /// air in blobs as well as single voxels.
    fn random_chunks(shape: UVec3, seed: u64) -> Vec<ChunkData<u8>> {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut chunks = Vec::new();
        for air in [0.05, 0.2, 0.3, 0.45, 0.7] {
            let solid: Vec<bool> = (0..shape.element_product())
                .map(|_| rng.random::<f32>() >= air)
                .collect();
            chunks.push(chunk(shape, |p| {
                solid[(p.x + shape.x * (p.y + shape.y * p.z)) as usize]
            }));
            // Blobs: air within a random radius of a few random centres.
            let centres: Vec<(Vec3, f32)> = (0..6)
                .map(|_| {
                    let centre = Vec3::new(rng.random(), rng.random(), rng.random())
                        * shape.as_vec3();
                    (centre, rng.random_range(2.0..8.0) * air * 2.0)
                })
                .collect();
            chunks.push(chunk(shape, |p| {
                centres
                    .iter()
                    .all(|(centre, radius)| p.as_vec3().distance(*centre) > *radius)
            }));
        }
        chunks
    }

    #[test]
    fn the_passages_match_a_fill_one_voxel_at_a_time() {
        // The widest shape has 64 voxels in a row with its padding, and so
        // fits a row under either rule.
        for shape in [SHAPE, UVec3::new(20, 34, 9), UVec3::new(64, 5, 6)] {
            for (i, data) in random_chunks(shape, 5).iter().enumerate() {
                for (rule, corners) in [
                    (RayPassage::FaceJoined, false),
                    (RayPassage::CornerJoined, true),
                ] {
                    assert_eq!(
                        passages_of(data, rule),
                        reference(data, corners),
                        "chunk {i} of shape {shape} under {rule:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn a_chunk_wider_than_a_row_joins_at_least_what_its_voxels_join() {
        for shape in [
            UVec3::new(66, 5, 6),
            UVec3::new(100, 12, 10),
            UVec3::new(200, 6, 5),
        ] {
            for (i, data) in random_chunks(shape, 6).iter().enumerate() {
                for (rule, corners) in [
                    (RayPassage::FaceJoined, false),
                    (RayPassage::CornerJoined, true),
                ] {
                    let passages = passages_of(data, rule);
                    let exact = reference(data, corners);
                    assert_eq!(
                        passages.0 & exact.0,
                        exact.0,
                        "chunk {i} of shape {shape} under {rule:?}"
                    );
                }
            }
        }
    }
}
