---
title: Syncing and saving edits
feature: serde
module: edits
api:
  - EditsCommitted
  - EditDelta
  - Voxels
  - Mask
  - EditGap
  - SubGridEditsCommitted
---

Every commit of voxel edits writes an `EditsCommitted<C>` message for each level-0 chunk whose edits changed, carrying an `EditDelta`: the chunk's changed voxels and its edit sequence after them. This can be used to move edits to another world, over a network or into a save file.

```rust
// On the server: forward each committed delta to the clients that need it.
fn forward(mut committed: MessageReader<EditsCommitted<MyWorld>>) {
    for message in committed.read() {
        send_to_clients(&message.delta);
    }
}

// On a client: apply each delta that arrives.
fn receive(mut voxel_world: VoxelWorld<MyWorld>) {
    for delta in received_deltas() {
        if let Err(gap) = voxel_world.apply_delta(delta) {
            // A delta for this chunk went missing: ask the server for
            // `full_delta(gap.chunk)` and apply that.
            request_full_delta(gap.chunk);
        }
    }
}
```

A save is the `full_delta` of each chunk in `edited_chunks`, and a load is `forget_edits` followed by `apply_delta` for each saved delta. `clear_voxels_in` and `clear_edits` remove edits and reach other worlds as full deltas, and `regenerate_chunks` makes chunks look up their voxels again after the world's generator changes, for example to a new seed.

The opt-in `serde` feature serialises voxels and deltas in any format. The [`edits` module docs](https://docs.rs/bevy_voxel_world/latest/bevy_voxel_world/edits/index.html) describe the rule a delta is applied by, the channel it assumes, the join, gap and heartbeat flows, and the serialised layout.

The [`multiplayer_loopback` example](https://github.com/splashdust/bevy_voxel_world/blob/main/examples/multiplayer_loopback.rs) runs a server and two clients in one app over a fake network, and the [`save_load` example](https://github.com/splashdust/bevy_voxel_world/blob/main/examples/save_load.rs) saves edits and the terrain seed to a file and loads them back.

```bash
$ cargo run -r --example multiplayer_loopback --features serde,sub_grids
$ cargo run -r --example save_load --features serde,sub_grids
```

Both examples also lift and stamp sub-grids, so they need the `sub_grids` feature too.
