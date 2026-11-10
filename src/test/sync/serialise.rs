//! Tests of edit deltas and voxels through `serde`, with `postcard` as the
//! format.

use bevy::ecs::system::RunSystemOnce;
use bevy::prelude::*;

use super::{
    chunk_status, full_delta, get_voxel, send, sync_app, update_until, Receiver, Sender,
};
use crate::edits::{Mask, Voxels};
use crate::prelude::*;

fn round_trip(delta: &EditDelta<u8>) -> EditDelta<u8> {
    let bytes = postcard::to_allocvec(delta).unwrap();
    postcard::from_bytes(&bytes).unwrap()
}

#[test]
fn a_sparse_delta_serialises_its_position_as_three_i32() {
    let delta = EditDelta {
        chunk: IVec3::new(1, -2, 3),
        seq: 300,
        full: false,
        voxels: Voxels::Sparse(vec![(5, WorldVoxel::Solid(7)), (1000, WorldVoxel::Air)]),
    };

    // postcard writes an `i32` as a zigzag varint, and a `u64`, a `u16`, a
    // length and an enum variant as a varint.
    #[rustfmt::skip]
    let expected = [
        2, 3, 6,       // chunk: 1, -2, 3
        0xac, 0x02,    // seq: 300
        0,             // full: false
        0,             // Voxels::Sparse
        2,             // two voxels
        5, 2, 7,       // index 5, WorldVoxel::Solid(7)
        0xe8, 0x07, 1, // index 1000, WorldVoxel::Air
    ];
    assert_eq!(postcard::to_allocvec(&delta).unwrap(), expected);
    assert_eq!(round_trip(&delta), delta);
}

#[test]
fn a_dense_delta_serialises_its_mask_as_4096_little_endian_bytes() {
    let mut mask = Mask::default();
    mask.insert(0);
    mask.insert(65);
    let delta = EditDelta {
        chunk: IVec3::ZERO,
        seq: 2,
        full: true,
        voxels: Voxels::Dense {
            mask,
            voxels: vec![WorldVoxel::Solid(1), WorldVoxel::Unset],
        },
    };

    let mut mask_bytes = vec![0u8; 4096];
    // Voxel 0 is bit 0 of word 0; voxel 65 is bit 1 of word 1.
    mask_bytes[0] = 1;
    mask_bytes[8] = 2;
    let mut expected = vec![
        0, 0, 0, // chunk
        2, // seq
        1, // full: true
        1, // Voxels::Dense
        0x80, 0x20, // 4096 mask bytes
    ];
    expected.extend(mask_bytes);
    expected.extend([
        2, // two voxels
        2, 1, // WorldVoxel::Solid(1)
        0, // WorldVoxel::Unset
    ]);
    assert_eq!(postcard::to_allocvec(&delta).unwrap(), expected);
    assert_eq!(round_trip(&delta), delta);
}

#[test]
fn a_mask_of_any_other_length_is_a_deserialise_error() {
    for length in [0, 4095, 4097] {
        // A full dense delta at sequence 1 of the origin chunk, whose mask is
        // `length` zero bytes, and with no voxels.
        let mut bytes = vec![0, 0, 0, 1, 1, 1];
        bytes.extend(postcard::to_allocvec(&vec![0u8; length]).unwrap());
        bytes.push(0);
        assert!(
            postcard::from_bytes::<EditDelta<u8>>(&bytes).is_err(),
            "a mask of {length} bytes deserialised"
        );
    }
}

#[test]
fn sparse_dense_and_empty_full_deltas_survive_postcard() {
    let mut app = sync_app();
    let (sparse, dense, empty) = (IVec3::ZERO, IVec3::new(0, -1, 0), IVec3::new(1, 0, 0));
    send(&mut app, |voxel_world| {
        voxel_world.set_voxel(IVec3::new(3, 4, 5), WorldVoxel::Solid(7));
        voxel_world.set_voxel(IVec3::new(9, 0, 2), WorldVoxel::Air);
        voxel_world.set_voxels_in(IVec3::new(0, -32, 0), IVec3::new(32, 0, 32), |p| {
            Some(WorldVoxel::Solid((p.x % 5) as u8 + 2))
        });
    });
    let deltas: Vec<_> = [sparse, dense, empty]
        .into_iter()
        .map(|chunk| full_delta::<Sender>(&mut app, chunk))
        .collect();
    assert!(matches!(deltas[0].voxels, Voxels::Sparse(_)));
    assert!(matches!(deltas[1].voxels, Voxels::Dense { .. }));
    assert!(deltas[2].is_empty());

    // An edit the receiver sets in the frame it applies the empty full delta
    // is replaced by it.
    let received: Vec<_> = deltas.iter().map(round_trip).collect();
    let applied = app
        .world_mut()
        .run_system_once(move |mut voxel_world: VoxelWorld<Receiver>| {
            voxel_world.set_voxel(IVec3::new(40, 1, 1), WorldVoxel::Solid(9));
            received
                .iter()
                .map(|delta| voxel_world.apply_delta(delta.clone()))
                .collect::<Vec<_>>()
        })
        .unwrap();
    assert_eq!(applied, vec![Ok(()); 3]);
    app.update();

    for position in [
        IVec3::new(3, 4, 5),
        IVec3::new(9, 0, 2),
        IVec3::new(7, -3, 9),
        IVec3::new(31, -32, 31),
    ] {
        assert_eq!(
            get_voxel::<Receiver>(&mut app, position),
            get_voxel::<Sender>(&mut app, position),
            "at {position}"
        );
    }
    // The sender spawns no chunks to read a generated voxel from, so the
    // receiver's chunk is the reference.
    update_until(&mut app, "the receiver's chunk to be ready", |app| {
        chunk_status::<Receiver>(app, empty) == Some(ChunkStatus::Ready)
    });
    assert_eq!(
        get_voxel::<Receiver>(&mut app, IVec3::new(40, 1, 1)),
        WorldVoxel::Air
    );
}

/// A material index with no `serde` impls.
#[derive(Clone, Copy, Debug, Default, Hash, PartialEq, Eq)]
struct PlainIndex(u8);

#[derive(Resource, Clone, Default)]
struct PlainIndexWorld;

impl VoxelWorldConfig for PlainIndexWorld {
    type MaterialIndex = PlainIndex;
    type ChunkUserBundle = ();

    fn spawning_distance(&self) -> u32 {
        2
    }
}

#[test]
fn a_world_whose_material_index_has_no_serde_still_exports_deltas() {
    let mut app = App::new();
    app.add_plugins((
        super::test_plugins(),
        VoxelWorldPlugin::<PlainIndexWorld>::minimal(),
    ));
    app.update();
    app.world_mut()
        .run_system_once(|mut voxel_world: VoxelWorld<PlainIndexWorld>| {
            voxel_world.set_voxel(IVec3::ONE, WorldVoxel::Solid(PlainIndex(3)));
        })
        .unwrap();
    app.update();

    let full = app
        .world_mut()
        .run_system_once(|voxel_world: VoxelWorld<PlainIndexWorld>| {
            voxel_world.full_delta(IVec3::ZERO)
        })
        .unwrap();
    assert_eq!(
        full.iter().collect::<Vec<_>>(),
        vec![(IVec3::ONE, WorldVoxel::Solid(PlainIndex(3)))]
    );
}
