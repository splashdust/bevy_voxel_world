import {
  useEffect,
  useRef,
  useState,
  type CSSProperties,
  type PointerEvent as ReactPointerEvent,
  type ReactNode,
  type RefObject,
} from 'react';
import type { VoxelDemo } from './loadDemo';

/** The radius of the walking stick, in CSS pixels. */
const STICK_RADIUS = 44;
/** How far the stick must move, as a fraction of its radius, before it walks. */
const DEAD_ZONE = 0.2;
/** How far a thumb must drag the stick to run, as a fraction of its radius. */
const RUN_DISTANCE = 1.5;
/** How much of the scene, from its left edge, walks rather than looks. */
const STICK_ZONE = 0.4;

/**
 * The voxels the place button can place, by the names the demo gives their
 * buttons, with the tile of the example's texture that shows each one.
 */
const MATERIALS = [
  { name: 'stone', tile: 3 },
  { name: 'timber', tile: 8 },
  { name: 'lantern', tile: 13 },
  { name: 'crystal', tile: 14 },
] as const;
const TILE_COUNT = 18;
const TEXTURE = `${import.meta.env.BASE_URL}voxel_demo/mountain_house_texture.png`;

/**
 * What the touch controls have chosen, which the demo keeps until they choose
 * again. It outlasts the controls, which a mouse press hides.
 */
export type TouchChoices = { placing: string; carrying: boolean };
export const FIRST_CHOICES: TouchChoices = { placing: MATERIALS[0].name, carrying: false };

type Stick = { pointer: number; x: number; y: number; dx: number; dy: number };
type Drag = { pointer: number; x: number; y: number };

function capture(event: ReactPointerEvent<HTMLElement>) {
  try {
    event.currentTarget.setPointerCapture(event.pointerId);
  } catch {
    // The pointer has already gone, so there is nothing to follow.
  }
}

function fullscreenSupported(): boolean {
  return document.fullscreenEnabled;
}

/**
 * Controls for a touch screen, over the demo's canvas. A touch on the left of
 * the scene walks with a stick that appears under the thumb, and a touch
 * anywhere else looks around. The buttons stand for the keys and mouse buttons
 * of the example.
 */
export function TouchControls({
  demo,
  frame,
  choices,
  onChoose,
}: {
  demo: VoxelDemo;
  frame: RefObject<HTMLDivElement | null>;
  choices: TouchChoices;
  onChoose: (choices: TouchChoices) => void;
}) {
  const stick = useRef<Stick | null>(null);
  const look = useRef<Drag | null>(null);
  const [shownStick, setShownStick] = useState<Stick | null>(null);
  const [fullscreen, setFullscreen] = useState(false);

  useEffect(() => {
    const follow = () => setFullscreen(document.fullscreenElement === frame.current);
    document.addEventListener('fullscreenchange', follow);
    return () => document.removeEventListener('fullscreenchange', follow);
  }, [frame]);

  function walk(next: Stick | null) {
    stick.current = next;
    setShownStick(next);
    const distance = next ? Math.hypot(next.dx, next.dy) : 0;
    if (!next || distance < DEAD_ZONE * STICK_RADIUS) {
      demo.touch_walk(0, 0, false);
      return;
    }
    demo.touch_walk(next.dx / distance, -next.dy / distance, distance >= RUN_DISTANCE * STICK_RADIUS);
  }

  function onPointerDown(event: ReactPointerEvent<HTMLDivElement>) {
    const area = event.currentTarget.getBoundingClientRect();
    const x = event.clientX - area.left;
    const y = event.clientY - area.top;
    if (x < area.width * STICK_ZONE && !stick.current) {
      capture(event);
      walk({ pointer: event.pointerId, x, y, dx: 0, dy: 0 });
    } else if (!look.current) {
      capture(event);
      look.current = { pointer: event.pointerId, x: event.clientX, y: event.clientY };
    }
  }

  function onPointerMove(event: ReactPointerEvent<HTMLDivElement>) {
    const held = stick.current;
    if (held?.pointer === event.pointerId) {
      const area = event.currentTarget.getBoundingClientRect();
      walk({ ...held, dx: event.clientX - area.left - held.x, dy: event.clientY - area.top - held.y });
    } else if (look.current?.pointer === event.pointerId) {
      demo.touch_look(event.clientX - look.current.x, event.clientY - look.current.y);
      look.current = { pointer: event.pointerId, x: event.clientX, y: event.clientY };
    }
  }

  function onPointerEnd(event: ReactPointerEvent<HTMLDivElement>) {
    if (stick.current?.pointer === event.pointerId) {
      walk(null);
    } else if (look.current?.pointer === event.pointerId) {
      look.current = null;
    }
  }

  function toggleFullscreen() {
    if (document.fullscreenElement) {
      void document.exitFullscreen();
    } else {
      void frame.current?.requestFullscreen({ navigationUI: 'hide' });
    }
  }

  return (
    <div
      className="touch-controls"
      onPointerDown={onPointerDown}
      onPointerMove={onPointerMove}
      onPointerUp={onPointerEnd}
      onPointerCancel={onPointerEnd}
      onContextMenu={(event) => event.preventDefault()}
    >
      <TouchStick stick={shownStick} />

      <div className="touch-bar start">
        {MATERIALS.map(({ name, tile }) => (
          <HoldButton
            key={name}
            demo={demo}
            name={name}
            label={`Place ${name}`}
            className="touch-swatch"
            pressed={choices.placing === name}
            onPress={() => onChoose({ ...choices, placing: name })}
            style={{
              backgroundImage: `url(${TEXTURE})`,
              backgroundPosition: `0 ${(tile / (TILE_COUNT - 1)) * 100}%`,
            }}
          />
        ))}
      </div>

      <div className="touch-bar end">
        <div className="touch-sun">
          <HoldButton demo={demo} name="sun_down" label="Lower the sun">
            <ArrowIcon down />
          </HoldButton>
          <SunIcon />
          <HoldButton demo={demo} name="sun_up" label="Raise the sun">
            <ArrowIcon />
          </HoldButton>
        </div>
        <HoldButton
          demo={demo}
          name="carry_lantern"
          label="Carry a lantern"
          pressed={choices.carrying}
          onPress={() => onChoose({ ...choices, carrying: !choices.carrying })}
        >
          <LanternIcon />
        </HoldButton>
        {fullscreenSupported() && (
          <button
            type="button"
            className="touch-button"
            aria-label="Full screen"
            aria-pressed={fullscreen}
            tabIndex={-1}
            onPointerDown={(event) => event.stopPropagation()}
            onClick={toggleFullscreen}
          >
            <FullscreenIcon />
          </button>
        )}
      </div>

      <div className="touch-actions">
        <HoldButton demo={demo} name="remove" label="Remove">
          Remove
        </HoldButton>
        <HoldButton demo={demo} name="place" label="Place">
          Place
        </HoldButton>
        <HoldButton demo={demo} name="jump" label="Jump" className="jump">
          Jump
        </HoldButton>
      </div>
    </div>
  );
}

/** The walking stick under the thumb, or where it waits for one. */
function TouchStick({ stick }: { stick: Stick | null }) {
  const size = STICK_RADIUS * 2;
  const reach = stick ? Math.hypot(stick.dx, stick.dy) : 0;
  const scale = reach > STICK_RADIUS ? STICK_RADIUS / reach : 1;
  const running = reach >= RUN_DISTANCE * STICK_RADIUS;
  return (
    <div
      className={`touch-stick ${stick ? '' : 'resting'} ${running ? 'running' : ''}`}
      aria-hidden="true"
      style={{
        width: size,
        height: size,
        margin: -STICK_RADIUS,
        ...(stick
          ? { left: stick.x, top: stick.y }
          : {
              left: `calc(var(--touch-inset) + ${STICK_RADIUS}px)`,
              top: `calc(100% - var(--touch-inset) - ${STICK_RADIUS}px)`,
            }),
      }}
    >
      <div
        className="touch-knob"
        style={stick ? { transform: `translate(${stick.dx * scale}px, ${stick.dy * scale}px)` } : undefined}
      />
    </div>
  );
}

/**
 * A button that holds a key or mouse button of the example for as long as a
 * finger stays on it. `name` is the demo's name for the button.
 */
function HoldButton({
  demo,
  name,
  label,
  className,
  pressed,
  onPress,
  style,
  children,
}: {
  demo: VoxelDemo;
  name: string;
  label: string;
  className?: string;
  pressed?: boolean;
  onPress?: () => void;
  style?: CSSProperties;
  children?: ReactNode;
}) {
  const held = useRef(false);
  const [down, setDown] = useState(false);

  function release() {
    if (held.current) {
      held.current = false;
      setDown(false);
      demo.touch_button(name, false);
    }
  }

  return (
    <button
      type="button"
      className={`touch-button ${className ?? ''} ${down ? 'down' : ''}`}
      aria-label={label}
      aria-pressed={pressed}
      tabIndex={-1}
      style={style}
      onPointerDown={(event) => {
        event.stopPropagation();
        capture(event);
        held.current = true;
        setDown(true);
        demo.touch_button(name, true);
        onPress?.();
      }}
      onPointerUp={release}
      onPointerCancel={release}
      onLostPointerCapture={release}
    >
      {children}
    </button>
  );
}

/** Pixel art icons, drawn on a 12 by 12 grid. */
function Icon({ children }: { children: ReactNode }) {
  return (
    <svg className="touch-icon" viewBox="0 0 12 12" shapeRendering="crispEdges" aria-hidden="true">
      {children}
    </svg>
  );
}

function ArrowIcon({ down = false }: { down?: boolean }) {
  return (
    <Icon>
      <path d="M5 3h2v1h1v1h1v1h1v1h1v1H1V7h1V6h1V5h1V4h1z" transform={down ? 'rotate(180 6 6)' : undefined} />
    </Icon>
  );
}

function SunIcon() {
  return (
    <span className="touch-sun-icon">
      <Icon>
        <path d="M4 3h4v1h1v4H8v1H4V8H3V4h1zM5 0h2v2H5zM5 10h2v2H5zM0 5h2v2H0zM10 5h2v2h-2zM1 1h2v2H1zM9 1h2v2H9zM1 9h2v2H1zM9 9h2v2H9z" />
      </Icon>
    </span>
  );
}

function LanternIcon() {
  return (
    <Icon>
      <path d="M4 0h4v1H4zM4 1h1v1H4zM7 1h1v1H7zM3 2h6v1H3zM3 3h1v6H3zM8 3h1v6H8zM3 9h6v1H3zM4 10h4v1H4z" />
      <path className="touch-glow" d="M4 3h4v6H4z" />
    </Icon>
  );
}

function FullscreenIcon() {
  return (
    <Icon>
      <path d="M1 1h4v1H2v3H1zM7 1h4v4h-1V2H7zM1 7h1v3h3v1H1zM10 7h1v4H7v-1h3z" />
    </Icon>
  );
}
