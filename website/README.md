# Website

The bevy_voxel_world site, built with Vite and React. It has three pages, each with its own HTML file:

- `index.html` renders the repository's `README.md`
- `changelog/index.html` renders its `CHANGELOG.md`
- `demo/index.html` runs a live demo built from `voxel_demo/`, a Bevy app compiled to WebAssembly that runs the `light_mountain_house` example
- `examples/index.html` lists the crate's examples

The examples page is generated from what `cargo metadata` reports. Each example's description is the `//!` comment at the top of its file, and the build stops with an error if an example doesn't have one. A PNG with the same name next to the source file, like `examples/bombs.png`, becomes its screenshot, and an example without one shows no image.

Building the site needs Rust, and the demo also needs the `wasm32-unknown-unknown` target:

```bash
rustup target add wasm32-unknown-unknown
```

Then, from this folder:

```bash
npm install
npm run build:demo
npm run dev
```

`npm run build:demo` puts the demo in `public/voxel_demo`, with the `light_mountain_house` example's voxel texture from `assets/`. The first run takes several minutes. Re-run it after changing `voxel_demo/`, the example or the crate.

## Docs section (prototype)

`docs/index.html` shows the guide in `guide/` with the crate's API, from the rustdoc JSON of a nightly toolchain. `npm run build:api` runs `api_extract/`, which builds that JSON once with every feature that adds API and once without each of them, and writes `api.json`: each public item's signature, docs, members and the features it needs.

Each file in `guide/` is one topic, in the order of the file names. The number at the start of a name only sets that order, so `08-light.md` is the topic `light`. Its front matter gives the title and lists the API items whose reference follows the guide:

```markdown
---
title: Light
feature: light_propagation
api:
  - LightPropagation
  - Sky
---
```

`feature` adds a section listing what the feature adds to other topics' items, and `module` adds a module's docs. In the body, [`LightPropagation`] and [`VoxelWorld::raycast`] link to the API, as in a doc comment. A code block with the language `api` shows the items or members it lists, one per line, at that point in the guide. The build stops if a link or `api` entry names nothing, or if a public item is in no guide's `api` list or in two.

```bash
rustup toolchain install nightly
npm run build:api
npm run dev
```

Set `API_DOCS_TOOLCHAIN` to use a nightly other than `nightly`; it must write the rustdoc JSON format that `api_extract` reads. In `npm run dev`, a bar at the bottom of the docs page flips between three layouts: `?variant=A` (topic pages), `B` (search first) and `C` (explorer columns).

`npm run build` writes the static site to `dist`. Pushing to `main` builds and publishes it to GitHub Pages through `.github/workflows/pages.yml`.
