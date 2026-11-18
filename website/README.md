# Website

The bevy_voxel_world site, built with Vite and React. Each page has its own HTML file:

- `index.html`, Home, renders the repository's `README.md`
- `docs/index.html` is where the documentation starts, and leads to:
  - `docs/guides/index.html`, the guides in `guide/` with the API each one uses
  - `docs/api/index.html`, the whole API on one searchable page
  - `docs/examples/index.html`, the crate's examples
- `demo/index.html` runs a live demo built from `voxel_demo/`, a Bevy app compiled to WebAssembly that runs the `light_mountain_house` example
- `changelog/index.html` renders its `CHANGELOG.md`

`examples/index.html` sends old links on to `docs/examples/`.

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

The guides and API pages show the crate's API from the rustdoc JSON of a nightly toolchain. `npm run build:api` runs `api_extract/`, which builds that JSON once with every feature that adds API and once without each of them, and writes `api.json`: each public item's signature, docs, members and the features it needs.

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

`feature` adds a section listing what the feature adds to other topics' items, and `module` adds a module's docs. In the body, [`LightPropagation`] and [`VoxelWorld::raycast`] link to the API, as in a doc comment. A code block with the language `api` shows the items or members it lists, one per line, at that point in the guide. An image's path is relative to the guide's file, so `![A voxel](img/voxel.png)` or `<img src="img/voxel.png">` shows `guide/img/voxel.png`, both on the site and on GitHub; the build copies it, and stops if it isn't there. The build stops if a link or `api` entry names nothing, or if an item is in two guides' `api` lists. An item in none is fine: the API page lists every public item, and puts those under Other. The build names them in a warning.

```bash
rustup toolchain install nightly
npm run build:api
npm run dev
```

Set `API_DOCS_TOOLCHAIN` to use a nightly other than `nightly`; it must write the rustdoc JSON format that `api_extract` reads. A guide is at `docs/guides/?topic=light`, and the API page takes a search as `docs/api/?q=stamp`.

`npm run build` writes the static site to `dist`. Pushing to `main` builds and publishes it to GitHub Pages through `.github/workflows/pages.yml`.
