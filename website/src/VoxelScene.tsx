import { valueNoise } from './noise';

const SIZE = 15;
const WATER_LEVEL = 4;
const CRATER = { x: 8.5, z: 7, radius: 4.3 };

// Screen size of one voxel. Half a tile wide, a quarter tile deep, one tile tall.
const TILE_X = 20;
const TILE_Y = 10;
const TILE_HEIGHT = 20;

type Shades = { top: string; left: string; right: string };

const GRASS: Shades = { top: '#46a53b', left: '#1f6d1c', right: '#2b8127' };
const DIRT: Shades = { top: '#7a5a24', left: '#4a3408', right: '#5d4310' };
const ROCK: Shades = { top: '#8c8c8c', left: '#5e5e5e', right: '#727272' };
const WATER: Shades = { top: '#d5f9ff', left: '#93cfdf', right: '#b3e6f1' };

/** The height of a column before anything digs into it. */
function groundHeight(x: number, z: number): number {
  const hills = valueNoise(x * 0.14 + 0.6, z * 0.14 + 2.3);
  const detail = valueNoise(x * 0.5 + 11.3, z * 0.5 + 2.9);
  return Math.round(1 + hills * 10 + detail * 1.2);
}

/** The height of a column with the crater dug out of it. */
function columnHeight(x: number, z: number): number {
  const ground = Math.max(groundHeight(x, z), WATER_LEVEL);
  const dx = x - CRATER.x;
  const dz = z - CRATER.z;
  const inside = CRATER.radius ** 2 - dx * dx - dz * dz;
  if (inside <= 0) {
    return ground;
  }
  const floor = groundHeight(CRATER.x, CRATER.z) - Math.round(Math.sqrt(inside) * 1.1);
  return Math.max(1, Math.min(ground, floor));
}

function voxelShades(x: number, y: number, z: number): Shades {
  const ground = groundHeight(x, z);
  if (y >= ground) {
    return WATER;
  }
  const depth = ground - 1 - y;
  if (depth === 0) {
    return GRASS;
  }
  return depth <= 3 ? DIRT : ROCK;
}

function project(x: number, y: number, z: number): string {
  return `${(x - z) * TILE_X},${(x + z) * TILE_Y - y * TILE_HEIGHT}`;
}

function polygon(...corners: [number, number, number][]): string {
  return corners.map((corner) => project(...corner)).join(' ');
}

const heightAt = (x: number, z: number) =>
  x < SIZE && z < SIZE ? columnHeight(x, z) : 0;

/** One voxel's visible faces. A side shows only where the next column is lower. */
function Voxel({ x, y, z, top }: { x: number; y: number; z: number; top: boolean }) {
  const shades = voxelShades(x, y, z);
  return (
    <>
      {heightAt(x, z + 1) <= y && (
        <polygon
          fill={shades.left}
          points={polygon(
            [x, y, z + 1],
            [x + 1, y, z + 1],
            [x + 1, y + 1, z + 1],
            [x, y + 1, z + 1],
          )}
        />
      )}
      {heightAt(x + 1, z) <= y && (
        <polygon
          fill={shades.right}
          points={polygon(
            [x + 1, y, z],
            [x + 1, y, z + 1],
            [x + 1, y + 1, z + 1],
            [x + 1, y + 1, z],
          )}
        />
      )}
      {top && (
        <polygon
          fill={shades.top}
          points={polygon(
            [x, y + 1, z],
            [x + 1, y + 1, z],
            [x + 1, y + 1, z + 1],
            [x, y + 1, z + 1],
          )}
        />
      )}
    </>
  );
}

function Column({ x, z }: { x: number; z: number }) {
  const height = columnHeight(x, z);
  const center = (SIZE - 1) / 2;
  const delay = Math.hypot(x - center, z - center) * 45;
  return (
    <g className="voxel-column" style={{ animationDelay: `${delay}ms` }}>
      {Array.from({ length: height }, (_, y) => (
        <Voxel key={y} x={x} y={y} z={z} top={y === height - 1} />
      ))}
    </g>
  );
}

/** An isometric chunk of voxel terrain with a crater in it. */
export function VoxelScene() {
  const coordinates = Array.from({ length: SIZE }, (_, i) => i);
  const top = -13 * TILE_HEIGHT;
  const width = SIZE * TILE_X * 2;
  const height = SIZE * TILE_Y * 2 - top;
  return (
    <svg
      className="voxel-scene"
      viewBox={`${-SIZE * TILE_X} ${top} ${width} ${height}`}
      aria-hidden="true"
    >
      {coordinates.map((x) =>
        coordinates.map((z) => <Column key={`${x},${z}`} x={x} z={z} />),
      )}
    </svg>
  );
}
