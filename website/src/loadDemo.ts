/** The functions the demo's WebAssembly module exports to the page. */
export type VoxelDemo = {
  set_canvas_visible(visible: boolean): void;
};

type DemoModule = VoxelDemo & {
  default(options: { module_or_path: Response }): Promise<unknown>;
};

export type LoadStage =
  | { stage: 'downloading'; loadedBytes: number }
  | { stage: 'starting' };

let loading: Promise<VoxelDemo> | undefined;

/**
 * Downloads the demo and starts it on the page's `#voxel-demo` canvas. Resolves
 * once it has started drawing.
 *
 * The demo can only start once per page, so later calls return the same promise.
 */
export function loadDemo(onStage: (stage: LoadStage) => void): Promise<VoxelDemo> {
  loading ??= load(onStage);
  return loading;
}

async function load(onStage: (stage: LoadStage) => void): Promise<VoxelDemo> {
  const base = `${import.meta.env.BASE_URL}voxel_demo/`;
  const demo: DemoModule = await import(/* @vite-ignore */ `${base}voxel_demo.js`);

  const response = await fetch(`${base}voxel_demo_bg.wasm`);
  if (!response.ok) {
    throw new Error(`Downloading the demo failed with status ${response.status}`);
  }

  await new Promise((resolve, reject) => {
    window.addEventListener('voxel-demo-ready', resolve, { once: true });
    window.addEventListener('error', (event) => {
      if (event.error instanceof WebAssembly.RuntimeError) {
        reject(event.error);
      }
    });
    demo.default({ module_or_path: reportProgress(response, onStage) }).catch(reject);
  });
  return demo;
}

/** Copies a response, reporting the bytes as they arrive. */
function reportProgress(response: Response, onStage: (stage: LoadStage) => void): Response {
  let loadedBytes = 0;
  const counter = new TransformStream<Uint8Array, Uint8Array>({
    transform(chunk, controller) {
      loadedBytes += chunk.byteLength;
      onStage({ stage: 'downloading', loadedBytes });
      controller.enqueue(chunk);
    },
    flush() {
      onStage({ stage: 'starting' });
    },
  });
  return new Response(response.body!.pipeThrough(counter), { headers: response.headers });
}
