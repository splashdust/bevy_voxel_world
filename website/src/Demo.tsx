import { useCallback, useEffect, useRef, useState } from 'react';
import { loadDemo, type VoxelDemo } from './loadDemo';

type Phase =
  | { name: 'waiting' }
  | { name: 'startable' }
  | { name: 'downloading'; loadedBytes: number }
  | { name: 'starting' }
  | { name: 'running' }
  | { name: 'unsupported' }
  | { name: 'failed' };

const PROGRESS_CELLS = 16;

function supportsWebGL2(): boolean {
  return document.createElement('canvas').getContext('webgl2') !== null;
}

/**
 * Whether the demo may start by itself once it's on screen. It's a large download
 * of moving 3D, so phones, data savers and people who prefer reduced motion get a
 * button instead.
 */
function mayStartByItself(): boolean {
  const connection = (navigator as { connection?: { saveData?: boolean } }).connection;
  return (
    !connection?.saveData &&
    !matchMedia('(pointer: coarse)').matches &&
    !matchMedia('(prefers-reduced-motion: reduce)').matches
  );
}

function Progress({ loadedBytes }: { loadedBytes: number }) {
  const fraction = __DEMO_WASM_BYTES__ > 0 ? loadedBytes / __DEMO_WASM_BYTES__ : 0;
  const filled = Math.round(Math.min(fraction, 1) * PROGRESS_CELLS);
  return (
    <div className="progress">
      <div className="progress-cells" aria-hidden="true">
        {Array.from({ length: PROGRESS_CELLS }, (_, cell) => (
          <span key={cell} className={cell < filled ? 'filled' : undefined} />
        ))}
      </div>
      Loading {Math.round(Math.min(fraction, 1) * 100)}%
    </div>
  );
}

/** The live demo: a canvas that Bevy draws to, with its loading states on top. */
export function Demo() {
  const frame = useRef<HTMLDivElement>(null);
  const demo = useRef<VoxelDemo | undefined>(undefined);
  const onScreen = useRef(true);
  const [phase, setPhase] = useState<Phase>({ name: 'waiting' });

  const start = useCallback(() => {
    setPhase({ name: 'downloading', loadedBytes: 0 });
    loadDemo((update) =>
      setPhase(
        update.stage === 'downloading'
          ? { name: 'downloading', loadedBytes: update.loadedBytes }
          : { name: 'starting' },
      ),
    ).then(
      (loaded) => {
        demo.current = loaded;
        loaded.set_canvas_visible(onScreen.current);
        setPhase({ name: 'running' });
      },
      (error) => {
        console.error(error);
        setPhase({ name: 'failed' });
      },
    );
  }, []);

  useEffect(() => {
    if (!supportsWebGL2()) {
      setPhase({ name: 'unsupported' });
      return;
    }
    const startsByItself = mayStartByItself();
    if (!startsByItself) {
      setPhase({ name: 'startable' });
    }

    let started = false;
    const observer = new IntersectionObserver(
      ([entry]) => {
        onScreen.current = entry.isIntersecting;
        demo.current?.set_canvas_visible(entry.isIntersecting);
        if (entry.isIntersecting && startsByItself && !started) {
          started = true;
          start();
        }
      },
      { rootMargin: '200px' },
    );
    observer.observe(frame.current!);
    return () => observer.disconnect();
  }, [start]);

  return (
    <div className="demo-frame" ref={frame}>
      <canvas id="voxel-demo" aria-label="A voxel house on a mountain ledge at dusk, which you can walk through" />
      <div className={`demo-overlay ${phase.name === 'running' ? 'hidden' : ''}`} aria-live="polite">
        {phase.name === 'waiting' && 'Loading'}
        {phase.name === 'startable' && (
          <button className="button" type="button" onClick={start}>
            Start the demo
          </button>
        )}
        {phase.name === 'downloading' && <Progress loadedBytes={phase.loadedBytes} />}
        {phase.name === 'starting' && 'Starting'}
        {phase.name === 'unsupported' && 'This demo needs WebGL2, which your browser doesn’t offer.'}
        {phase.name === 'failed' && 'The demo failed to start. The browser console has the details.'}
      </div>
    </div>
  );
}
