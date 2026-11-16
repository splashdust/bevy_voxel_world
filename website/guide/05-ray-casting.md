---
title: Ray casting
api:
  - VoxelRaycastResult
  - voxel_line_traversal
  - voxel_cartesian_traversal
  - GridRaycastFn
  - GridRaycastHit
---

To find a voxel location in the world from a pixel location on the screen, for example the mouse location, you can ray cast into the voxel world with [`VoxelWorld::raycast`].

```rust
fn do_something_with_mouse_voxel_pos(
    voxel_world: VoxelWorld<MyWorld>,
    camera_info: Query<(&Camera, &GlobalTransform), With<VoxelWorldCamera<MyWorld>>>,
    mut cursor_evr: EventReader<CursorMoved>,
) {
    for ev in cursor_evr.read() {
        // Get a ray from the cursor position into the world
        let (camera, cam_gtf) = camera_info.single();
        let Some(ray) = camera.viewport_to_world(cam_gtf, ev.position) else {
            return;
        };

        if let Some(result) = voxel_world.raycast(ray, &|(_pos, _vox)| true) {
            // result.position will be the world location of the voxel as a Vec3
            // To get the empty location next to the voxel in the direction of the surface where the ray intersected you can use result.normal:
            // let empty_pos = result.position + result.normal;
        }
    }
}
```

```api
VoxelWorld::raycast
VoxelRaycastResult
```

See this [full example of ray casting](https://github.com/splashdust/bevy_voxel_world/blob/main/examples/ray_cast.rs) for more details.
