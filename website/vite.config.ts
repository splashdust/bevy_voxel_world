import { existsSync, readFileSync, statSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import react from '@vitejs/plugin-react';
import { defineConfig } from 'vite';
import { examples } from './scripts/examples.ts';
import { markdown } from './scripts/markdown.ts';

const manifest = readFileSync(new URL('../Cargo.toml', import.meta.url), 'utf8');
const crateVersion = /^version = "(.+)"/m.exec(manifest)?.[1];
const bevyVersion = /^bevy = \{ version = "(.+?)"/m.exec(manifest)?.[1];

const demoWasm = new URL('public/voxel_demo/voxel_demo_bg.wasm', import.meta.url);

const page = (path: string) => fileURLToPath(new URL(path, import.meta.url));

export default defineConfig({
  // GitHub Pages serves a project site from a folder named after the repository.
  base: '/bevy_voxel_world/',
  // Each page is its own HTML file, and Bevy can't be stopped once it's running,
  // so leaving the demo page must be a real page load.
  appType: 'mpa',
  plugins: [markdown(), examples(), react()],
  define: {
    __CRATE_VERSION__: JSON.stringify(crateVersion),
    __BEVY_VERSION__: JSON.stringify(bevyVersion),
    __DEMO_WASM_BYTES__: existsSync(demoWasm) ? statSync(demoWasm).size : 0,
  },
  build: {
    rolldownOptions: {
      input: [
        page('index.html'),
        page('changelog/index.html'),
        page('demo/index.html'),
        page('examples/index.html'),
      ],
    },
  },
  server: {
    // The site renders the README and changelog from the repository root.
    fs: { allow: ['..'] },
  },
});
