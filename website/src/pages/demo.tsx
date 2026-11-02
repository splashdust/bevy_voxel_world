import { Demo } from '../Demo';
import { mountPage, REPOSITORY } from '../Page';
import { SteppedEdge } from '../SteppedEdge';

mountPage(
  'demo',
  <div className="demo-page">
    <div className="hero">
      <SteppedEdge seed={2.4} fill="var(--dirt)" cap="var(--grass)" />
    </div>
    <main className="demo-section">
      <Demo />
      <div className="demo-text">
        <h1>Running in your browser</h1>
        <p>
          This is the crate compiled to WebAssembly. Chunks spawn ahead of the camera as it flies,
          and every crater is a sphere of <code>set_voxel</code> calls.{' '}
          <strong>Click or tap the terrain to dig your own.</strong>
        </p>
        <p className="demo-note">
          A browser gives the demo one thread, so native builds stream much further and faster.{' '}
          <a href={`${REPOSITORY}/blob/main/website/voxel_demo/src/lib.rs`}>
            Read the demo’s source
          </a>
        </p>
      </div>
    </main>
  </div>,
);
