import { valueNoise } from './noise';

const WIDTH = 2000;
const CELL = 20;
const MAX_STEPS = 5;

type SteppedEdgeProps = {
  /** Noise offset, so that each edge has its own profile. */
  seed: number;
  /** Colour of the layer below the edge. */
  fill: string;
  /** Colour of the single-voxel layer on top, like grass on dirt. */
  cap?: string;
};

/** The blocky top of a terrain layer, drawn between two sections of the page. */
export function SteppedEdge({ seed, fill, cap }: SteppedEdgeProps) {
  const height = (MAX_STEPS + 2) * CELL;
  const columns = Array.from({ length: WIDTH / CELL }, (_, column) => {
    const noise =
      valueNoise(column * 0.09 + seed, seed) * 0.75 +
      valueNoise(column * 0.37 + seed, seed + 5) * 0.25;
    const steps = Math.round(((noise - 0.2) / 0.6) * MAX_STEPS);
    return { x: column * CELL, y: Math.min(Math.max(steps, 0), MAX_STEPS) * CELL };
  });
  return (
    <svg
      className="stepped-edge"
      viewBox={`0 0 ${WIDTH} ${height}`}
      preserveAspectRatio="xMidYMax slice"
      style={{ height }}
      aria-hidden="true"
    >
      {columns.map(({ x, y }) => (
        <g key={x}>
          <rect x={x} y={y} width={CELL} height={height - y} fill={fill} />
          {cap && <rect x={x} y={y} width={CELL} height={CELL} fill={cap} />}
        </g>
      ))}
    </svg>
  );
}
