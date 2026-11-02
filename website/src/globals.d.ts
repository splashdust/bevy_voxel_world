declare const __CRATE_VERSION__: string;
declare const __BEVY_VERSION__: string;
/** Size of the demo's WebAssembly file, or 0 if it hasn't been built. */
declare const __DEMO_WASM_BYTES__: number;

/** Rendered at build time by scripts/markdown.ts. */
declare module '*.md' {
  const document: {
    html: string;
    headings: import('../scripts/markdown.ts').Heading[];
  };
  export default document;
}

/** Generated at build time by scripts/examples.ts. */
declare module 'virtual:examples' {
  const examples: import('../scripts/examples.ts').Example[];
  export default examples;
}
