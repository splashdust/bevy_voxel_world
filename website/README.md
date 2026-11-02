# Website

The bevy_voxel_world site, built with Vite and React. It has three pages, each with its own HTML file:

- `index.html` renders the repository's `README.md`
- `changelog/index.html` renders its `CHANGELOG.md`
- `demo/index.html` runs a live demo built from `voxel_demo/`, a Bevy app compiled to WebAssembly
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

`npm run build:demo` puts the demo in `public/voxel_demo`. The first run takes several minutes. Re-run it after changing `voxel_demo/` or the crate.

`npm run build` writes the static site to `dist`. Pushing to `main` builds and publishes it to GitHub Pages through `.github/workflows/pages.yml`.
