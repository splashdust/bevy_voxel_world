import { execFileSync } from 'node:child_process';
import { existsSync, readFileSync } from 'node:fs';
import { relative } from 'node:path';
import { fileURLToPath } from 'node:url';
import { Marked } from 'marked';
import type { Plugin } from 'vite';

export type Example = {
  name: string;
  /** The example's doc comment, as HTML. */
  description: string;
  command: string;
  /** Its source file, relative to the repository. */
  path: string;
  screenshot?: { url: string; width: number; height: number };
};

type CargoTarget = {
  name: string;
  kind: string[];
  src_path: string;
  'required-features'?: string[];
};

const MODULE_ID = 'virtual:examples';
const RESOLVED_MODULE_ID = `\0${MODULE_ID}`;

const repository = fileURLToPath(new URL('../..', import.meta.url));

/** Returns the Markdown in the `//!` comment that a Rust file starts with. */
function docComment(source: string): string {
  const lines = source.split('\n');
  const end = lines.findIndex((line) => !line.startsWith('//!'));
  return lines
    .slice(0, end)
    .map((line) => line.replace(/^\/\/! ?/, ''))
    .join('\n');
}

/** Returns a PNG's size, which is at a fixed position in its header. */
function pngSize(file: string) {
  const png = readFileSync(file);
  return { width: png.readUInt32BE(16), height: png.readUInt32BE(20) };
}

/**
 * Lists the crate's examples as Cargo finds them, which covers every file in `examples/`.
 * Each comes with the PNG next to its source file, if there is one.
 */
function listExamples(): { example: Example; screenshotFile?: string }[] {
  const metadata = execFileSync('cargo', ['metadata', '--no-deps', '--format-version', '1'], {
    cwd: repository,
    encoding: 'utf8',
  });
  const targets: CargoTarget[] = JSON.parse(metadata).packages[0].targets;
  const marked = new Marked();

  return targets
    .filter((target) => target.kind.includes('example'))
    .map((target) => {
      const path = relative(repository, target.src_path);
      const comment = docComment(readFileSync(target.src_path, 'utf8'));
      if (!comment) {
        throw new Error(`${path} needs a //! doc comment at the top, for the site's examples page`);
      }
      const features = target['required-features'] ?? [];
      const screenshotFile = target.src_path.replace(/\.rs$/, '.png');
      return {
        example: {
          name: target.name,
          description: marked.parse(comment, { async: false }),
          command:
            `cargo run -r --example ${target.name}` +
            (features.length > 0 ? ` --features ${features.join(',')}` : ''),
          path,
        },
        screenshotFile: existsSync(screenshotFile) ? screenshotFile : undefined,
      };
    })
    .sort((a, b) => a.example.name.localeCompare(b.example.name));
}

/** Lets the site import `virtual:examples`, the list of the crate's examples. */
export function examples(): Plugin {
  return {
    name: 'examples',
    resolveId(id) {
      return id === MODULE_ID ? RESOLVED_MODULE_ID : null;
    },
    load(id) {
      if (id !== RESOLVED_MODULE_ID) {
        return null;
      }
      // Importing a screenshot makes it part of the build, and gives it its URL.
      const imports: string[] = [];
      const entries = listExamples().map(({ example, screenshotFile }, index) => {
        this.addWatchFile(`${repository}${example.path}`);
        if (!screenshotFile) {
          return JSON.stringify(example);
        }
        const { width, height } = pngSize(screenshotFile);
        imports.push(`import screenshot${index} from ${JSON.stringify(screenshotFile)};`);
        return `{ ...${JSON.stringify(example)}, screenshot: { url: screenshot${index}, width: ${width}, height: ${height} } }`;
      });
      return `${imports.join('\n')}\nexport default [${entries.join(',')}];`;
    },
  };
}
