// Builds the demo crate for the browser and puts the result in public/voxel_demo,
// where the site loads it from.
import { execFileSync } from 'node:child_process';
import { copyFileSync, existsSync, readFileSync, rmSync } from 'node:fs';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';

const website = fileURLToPath(new URL('..', import.meta.url));
const crate = join(website, 'voxel_demo');
const out = join(website, 'public', 'voxel_demo');
const target = 'wasm32-unknown-unknown';

function run(command, args, cwd = website) {
  console.log(`> ${command} ${args.join(' ')}`);
  execFileSync(command, args, { cwd, stdio: 'inherit' });
}

run('cargo', ['build', '--release', '--target', target], crate);

// The wasm-bindgen CLI must match the library's version exactly, so the version
// comes from the lockfile and whatever is on the PATH is left alone.
const lockfile = readFileSync(join(crate, 'Cargo.lock'), 'utf8');
const version = /name = "wasm-bindgen"\nversion = "(.+)"/.exec(lockfile)[1];
const tools = join(crate, 'target', 'tools', `wasm-bindgen-${version}`);
const wasmBindgen = join(tools, 'bin', 'wasm-bindgen');
if (!existsSync(wasmBindgen)) {
  run('cargo', ['install', 'wasm-bindgen-cli', '--version', version, '--locked', '--root', tools]);
}

rmSync(out, { recursive: true, force: true });
run(wasmBindgen, [
  join(crate, 'target', target, 'release', 'voxel_demo.wasm'),
  '--target', 'web',
  '--no-typescript',
  '--out-dir', out,
  '--out-name', 'voxel_demo',
]);

// The demo loads its voxel texture from beside its WebAssembly.
const texture = 'mountain_house_texture.png';
copyFileSync(join(website, '..', 'assets', texture), join(out, texture));

// These are the WebAssembly features rustc targets by default. wasm-opt's
// --all-features would be shorter, but it also emits encodings browsers reject.
const wasmFeatures = [
  '--enable-bulk-memory',
  '--enable-bulk-memory-opt',
  '--enable-call-indirect-overlong',
  '--enable-multivalue',
  '--enable-mutable-globals',
  '--enable-nontrapping-float-to-int',
  '--enable-reference-types',
  '--enable-sign-ext',
];
const wasm = join(out, 'voxel_demo_bg.wasm');
run(join(website, 'node_modules', '.bin', 'wasm-opt'), [wasm, '-Oz', ...wasmFeatures, '--output', wasm]);

if (!WebAssembly.validate(readFileSync(wasm))) {
  throw new Error(`${wasm} is not valid WebAssembly for this engine`);
}
