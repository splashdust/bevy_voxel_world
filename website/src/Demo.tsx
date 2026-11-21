import { useCallback, useEffect, useRef, useState } from 'react';
import { loadDemo, type VoxelDemo } from './loadDemo';
import { FIRST_CHOICES, TouchControls } from './TouchControls';

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
  // Touch controls show on a touch screen, and give way to the mouse and
  // keyboard as soon as a mouse is pressed, as on a laptop with a touch screen.
  const [touch, setTouch] = useState(() => matchMedia('(pointer: coarse)').matches);
  const [touchChoices, setTouchChoices] = useState(FIRST_CHOICES);
  const running = phase.name === 'running';

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

  useEffect(() => {
    const element = frame.current!;
    const choose = (event: PointerEvent) => {
      if (event.pointerType === 'touch') {
        setTouch(true);
      } else if (event.pointerType === 'mouse') {
        setTouch(false);
      }
    };
    element.addEventListener('pointerdown', choose, { capture: true });
    return () => element.removeEventListener('pointerdown', choose, { capture: true });
  }, []);

  useEffect(() => {
    if (running) {
      demo.current!.set_touch_controls(touch);
    }
  }, [running, touch]);

  return (
    <div className="demo-frame" ref={frame}>
      <canvas id="voxel-demo" aria-label="A voxel house on a mountain ledge at dusk, which you can walk through" />
      {running && touch && (
        <TouchControls demo={demo.current!} frame={frame} choices={touchChoices} onChoose={setTouchChoices} />
      )}
      <div className={`demo-overlay ${running ? 'hidden' : ''}`} aria-live="polite">
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
