import readme from "../../../README.md";
import { CopyCommand } from "../CopyCommand";
import { DocMenu } from "../DocMenu";
import { mountPage, pageUrl, REPOSITORY } from "../Page";
import { SteppedEdge } from "../SteppedEdge";
import { VoxelScene } from "../VoxelScene";

mountPage(
  "landing",
  <>
    <header className="hero">
      <div className="hero-content">
        <p className="eyebrow">
          v{__CRATE_VERSION__} · Bevy {__BEVY_VERSION__} · MIT or Apache-2.0
        </p>
        <h1 className="crate-name">
          bevy<span className="underscore">_</span>voxel
          <span className="underscore">_</span>world
        </h1>
        <div className="hero-columns">
          <div className="hero-text">
            <p className="lede">
              Ergonomic voxel engine for{" "}
              <a
                href="https://bevyengine.org/"
                target="_blank"
                rel="noopener noreferrer"
              >
                Bevy
              </a>
              .
            </p>
            <p>
              <strong>bevy_voxel_world</strong> takes care of chunk streaming,
              multithreaded meshing, levels of detail (LOD) and texture mapping.
              Your code supplies the voxels and calls <code>set_voxel</code> to
              change them as the game runs.
            </p>
            <div className="actions">
              <CopyCommand command="cargo add bevy_voxel_world" />
              <a className="button" href={pageUrl("demo")}>
                WASM Demo
              </a>
              <a className="button" href={REPOSITORY}>
                GitHub
              </a>
              <a className="button" href={pageUrl("examples")}>
                Examples
              </a>
            </div>
          </div>
          <VoxelScene />
        </div>
      </div>
      <SteppedEdge seed={2.4} fill="var(--dirt)" cap="var(--grass)" />
    </header>

    <main className="doc-section">
      <div className="doc-with-menu">
        <DocMenu headings={readme.headings} />
        <div
          className="prose"
          dangerouslySetInnerHTML={{ __html: readme.html }}
        />
      </div>
    </main>
  </>,
);
