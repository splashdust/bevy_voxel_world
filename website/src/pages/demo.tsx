import { Demo } from "../Demo";
import { mountPage, REPOSITORY } from "../Page";
import { SteppedEdge } from "../SteppedEdge";

mountPage(
  "demo",
  <div className="demo-page">
    <div className="hero">
      <SteppedEdge seed={2.4} fill="var(--dirt)" cap="var(--grass)" />
    </div>
    <main className="demo-section">
      <Demo />
      <div className="demo-text">
        <h1>Running in your browser</h1>
        <p>
          This is the <code>light_mountain_house</code> example compiled to
          WebAssembly, running in your browser. Sky light falls in through the
          door, the windows and the cave mouth. The lanterns, hearth and the
          crystals spread light of their own.
        </p>
        <p>
          <strong>Click the scene to look around.</strong> Walk with WASD and
          jump with Space. A left click places a voxel and a right click removes
          one, and the light follows. Remove a voxel of a trunk to fell a tree.
          1 to 4 choose what to place, [ and ] lower and raise the sun, and Esc
          frees the mouse.
        </p>
        <p>
          <strong>On a touch screen</strong>, drag on the left of the scene to
          walk, and drag past the edge of the stick to run. Drag anywhere else
          to look around. The buttons place and remove voxels and jump, and the
          ones along the top choose what to place, lower and raise the sun,
          carry a lantern and fill the screen.
        </p>
        <p className="demo-note">
          A browser gives the demo a single thread and WebGL2, so a plain sky
          stands in for Bevy’s atmosphere, sky light reaches less far, and the
          world ends a little past the ledge, where a native build streams on to
          the hills across the valley.{" "}
          <a href={`${REPOSITORY}/blob/main/examples/light_mountain_house.rs`}>
            Read the example’s source
          </a>
        </p>
      </div>
    </main>
  </div>,
);
