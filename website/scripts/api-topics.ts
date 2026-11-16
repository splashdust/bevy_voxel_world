// PROTOTYPE: the docs section's topics. Every public item belongs to exactly one,
// and the build stops if one is left out or listed twice.

export type Topic = {
  id: string;
  title: string;
  /** The README section (its `##` heading) that introduces the topic. */
  readme?: string;
  /** A module whose docs belong with the topic. */
  module?: string;
  /** An introduction, in Markdown, for a topic the README has no section for. */
  blurb?: string;
  /** The feature the topic is about. Its page also lists what the feature adds to other topics' items. */
  feature?: string;
  /** Item paths, in reading order. */
  items: string[];
};

export const TOPICS: Topic[] = [
  {
    id: 'setup',
    title: 'Setting up a world',
    readme: 'Basic setup',
    items: [
      'prelude::VoxelWorldPlugin',
      'prelude::VoxelWorldConfig',
      'prelude::DefaultWorld',
      'prelude::VoxelWorldCamera',
      'prelude::VoxelWorldSet',
      'prelude::ChunkSpawnStrategy',
      'prelude::ChunkDespawnStrategy',
    ],
  },
  {
    id: 'voxels',
    title: 'Reading and editing voxels',
    readme: 'Accessing the world',
    items: [
      'prelude::VoxelWorld',
      'prelude::WorldVoxel',
      'prelude::VoxelFace',
      'prelude::VOXEL_SIZE',
      'prelude::get_chunk_voxel_position',
    ],
  },
  {
    id: 'generation',
    title: 'Generating voxels',
    blurb:
      'A world asks its config for a voxel lookup function per chunk, and calls it for each voxel the chunk holds. A fill hint can say what a chunk holds before it is generated.',
    items: ['prelude::VoxelLookupDelegate', 'prelude::VoxelLookupFn', 'prelude::FillHint'],
  },
  {
    id: 'chunks',
    title: 'Chunks, LOD and events',
    readme: 'Level of Detail',
    items: [
      'prelude::Chunk',
      'prelude::ChunkKey',
      'prelude::ChunkStatus',
      'prelude::LodLevel',
      'prelude::MAX_LOD_LEVEL',
      'prelude::ChunkWillSpawn',
      'prelude::ChunkPublished',
      'prelude::ChunkWillUpdate',
      'prelude::ChunkWillRemesh',
      'prelude::ChunkWillDespawn',
    ],
  },
  {
    id: 'raycasting',
    title: 'Ray casting',
    readme: 'Ray casting',
    items: [
      'prelude::VoxelRaycastResult',
      'traversal_alg::voxel_line_traversal',
      'traversal_alg::voxel_cartesian_traversal',
      'prelude::GridRaycastFn',
      'prelude::GridRaycastHit',
    ],
  },
  {
    id: 'materials',
    title: 'Materials and shaders',
    readme: 'Voxel materials',
    module: 'rendering',
    items: [
      'prelude::TextureIndexMapperFn',
      'rendering::VoxelWorldMaterialHandle',
      'rendering::VOXEL_TEXTURE_SHADER_HANDLE',
      'rendering::vertex_layout',
      'rendering::ATTRIBUTE_TEX_INDEX',
      'rendering::VoxelWorldBindings',
      'rendering::VoxelWorldMaterial',
    ],
  },
  {
    id: 'meshing',
    title: 'Custom meshing',
    readme: 'Custom meshing',
    items: [
      'prelude::ChunkMeshingDelegate',
      'prelude::ChunkMeshingFn',
      'prelude::default_chunk_meshing_delegate',
      'custom_meshing::generate_chunk_mesh',
      'custom_meshing::generate_chunk_mesh_for_shape',
      'custom_meshing::mesh_from_quads',
      'custom_meshing::scale_texture_coordinates',
      'prelude::padded_chunk_shape',
      'prelude::padded_chunk_shape_uniform',
      'custom_meshing::PaddedChunkShape',
      'custom_meshing::VoxelArray',
      'custom_meshing::CHUNK_SIZE_U',
      'custom_meshing::CHUNK_SIZE_I',
      'custom_meshing::CHUNK_SIZE_F',
    ],
  },
  {
    id: 'light',
    title: 'Light',
    readme: 'Light propagation',
    feature: 'light_propagation',
    items: [
      'prelude::LightPropagation',
      'prelude::Sky',
      'prelude::SkyFill',
      'prelude::PathFill',
      'prelude::OpenFill',
      'prelude::SunGate',
      'prelude::Emission',
      'prelude::LightShape',
      'prelude::LightPrecision',
      'prelude::LightLook',
      'prelude::LightSmoothing',
      'prelude::LightAo',
      'prelude::LevelCurve',
      'prelude::LightLevels',
      'prelude::SubGridSky',
    ],
  },
  {
    id: 'sync',
    title: 'Syncing and saving edits',
    readme: 'Syncing and saving voxel edits',
    module: 'edits',
    feature: 'serde',
    items: [
      'prelude::EditsCommitted',
      'prelude::EditDelta',
      'edits::Voxels',
      'edits::Mask',
      'prelude::EditGap',
      'prelude::SubGridEditsCommitted',
    ],
  },
  {
    id: 'sub-grids',
    title: 'Sub-grids',
    readme: 'Sub-grids',
    feature: 'sub_grids',
    items: [
      'prelude::SubGrid',
      'prelude::Grid',
      'prelude::GridVoxels',
      'prelude::GridVoxelsMut',
      'prelude::GridRotation',
      'prelude::StampVoxel',
      'prelude::StampPreview',
      'prelude::Stamped',
      'prelude::StampError',
      'prelude::incoming_replaces',
      'prelude::Lifted',
      'prelude::LiftError',
    ],
  },
  {
    id: 'fragments',
    title: 'Dislodged fragments',
    readme: 'Dislodged fragments',
    feature: 'dislodged_fragments',
    items: ['prelude::FragmentDetection', 'prelude::Connectivity', 'prelude::FragmentDislodged'],
  },
  {
    id: 'debug',
    title: 'Debug drawing',
    blurb: 'Gizmos that draw chunk bounds, voxels and rays, for finding out what a world is doing.',
    items: [
      'debug::VoxelWorldDebugDrawPlugin',
      'debug::VoxelWorldDebugDraw',
      'debug::ChunkGizmos',
      'debug::VoxelGizmo',
      'debug::RayGizmo',
      'debug::debug_draw_chunks',
    ],
  },
];
