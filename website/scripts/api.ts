// PROTOTYPE: lets the site import `virtual:api`, the guide in website/guide/
// with the crate's API sorted into its topics and the docs rendered to HTML. It
// reads api.json, which `npm run build:api` writes.
//
// Each guide file is one topic. Its front matter names the topic and lists the
// API items whose reference it holds:
//
//   ---
//   title: Light
//   feature: light_propagation   # the page also lists what the feature adds elsewhere
//   module: rendering            # optional: a module whose docs belong here
//   api:
//     - LightPropagation
//     - Sky
//   ---
//
// The body is Markdown. [`LightPropagation`] and [`VoxelWorld::raycast`] link to
// the API, as in rustdoc, and a code block with the language `api` shows the
// items or members it lists, one per line, where it stands.

import { existsSync, readdirSync, readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { Marked, type Token as MarkdownToken, type TokensList } from 'marked';
import type { Plugin } from 'vite';
import { CODE_COLORS, CODE_THEME, type Heading, highlighter, slugify } from './markdown.ts';

/** One piece of a signature: text, a link (`h`) or a keyword or name (`k`). */
export type Token = { t: string; h?: string; k?: 'kw' | 'name' };

type RawMember = {
  key: string;
  name: string;
  section: string;
  features: string[];
  signature: Token[];
  docs: string;
  links: Record<string, string>;
};

type RawItem = {
  path: string;
  aliases: string[];
  name: string;
  kind: string;
  features: string[];
  signature: Token[];
  docs: string;
  links: Record<string, string>;
  source?: string;
  deprecated?: string;
  members: RawMember[];
  traits: { name: string; href?: string; features: string[] }[];
};

type RawApi = {
  crate: string;
  version: string;
  formatVersion: number;
  features: string[];
  modules: { path: string; docs: string; links: Record<string, string> }[];
  items: RawItem[];
};

export type MemberView = {
  /** The member's anchor: its item's path and its own key, like `prelude::VoxelWorld::fn.light_at`. */
  anchor: string;
  name: string;
  section: string;
  /** The features the member needs beyond its item's. */
  features: string[];
  signature: Token[];
  html: string;
};

export type ItemView = {
  path: string;
  aliases: string[];
  name: string;
  kind: string;
  features: string[];
  signature: Token[];
  html: string;
  /** The first paragraph of the docs, as plain text. */
  summary: string;
  source?: string;
  deprecated?: string;
  members: MemberView[];
  traits: { name: string; href?: string; features: string[] }[];
  /** The guide whose reference holds the item. Not every item is in a guide. */
  topic?: string;
};

/** A run of a guide's HTML, or the API items and members an `api` block shows. */
export type GuideBlock = { html: string } | { embed: string[] };

export type TopicView = {
  id: string;
  title: string;
  /** The guide file, relative to the repository. */
  file: string;
  feature?: string;
  blocks: GuideBlock[];
  /** The guide's `##` and `###` headings. */
  headings: Heading[];
  /** The module docs that belong with the topic, as HTML. */
  moduleDocs?: { path: string; html: string };
  items: string[];
  /** What the topic's feature adds to other topics' items. */
  elsewhere: { item: string; members: string[]; traits: string[] }[];
};

export type FeatureView = {
  name: string;
  /** The comment above the feature in Cargo.toml. */
  doc: string;
  /** The features it turns on. */
  enables: string[];
  /** How many items and members need it. */
  adds: number;
  topic?: string;
};

export type ApiView = {
  version: string;
  features: FeatureView[];
  /** The features that add API, which the site can filter by. */
  apiFeatures: string[];
  topics: TopicView[];
  /** Every public item, in the order the crate exports them. */
  items: Record<string, ItemView>;
  /** The items that no guide's reference holds, which only the API page shows. */
  unguided: string[];
};

const MODULE_ID = 'virtual:api';
const RESOLVED_MODULE_ID = `\0${MODULE_ID}`;

const apiFile = new URL('../api.json', import.meta.url);
const guideDir = new URL('../guide/', import.meta.url);
const manifestFile = new URL('../../Cargo.toml', import.meta.url);

/** Code block attributes that rustdoc reads, and that still mean the code is Rust. */
const RUSTDOC_ATTRIBUTES = /^(compile_fail|ignore|no_run|should_panic|test_harness|standalone_crate|edition\d+|E\d+)$/;

function codeLanguage(info: string | undefined): string {
  const named = (info ?? '')
    .split(',')
    .map((part) => part.trim())
    .find((part) => part && !RUSTDOC_ATTRIBUTES.test(part));
  return named ?? 'rust';
}

/** Leaves out the lines rustdoc hides from a Rust example: those starting with `# `. */
function visibleRust(code: string): string {
  return code
    .split('\n')
    .filter((line) => {
      const trimmed = line.trimStart();
      return trimmed !== '#' && !trimmed.startsWith('# ');
    })
    .map((line) => line.replace(/^(\s*)##/, '$1#'))
    .join('\n');
}

const DOCS_RS = /^https:\/\/docs\.rs\/bevy_voxel_world\/[^/]+\/bevy_voxel_world\/(.+)$/;

/** The key a docs.rs link to this crate points at, so links copied from the README stay on the site. */
function docsRsKey(href: string): string | undefined {
  const match = DOCS_RS.exec(href);
  if (!match) {
    return undefined;
  }
  const parts = match[1].split('/');
  const file = parts.pop()!;
  if (file === 'index.html') {
    return parts.join('::');
  }
  const page = /^(?:struct|enum|trait|fn|type|constant|union|static|macro)\.(\w+)\.html/.exec(file);
  return page ? [...parts, page[1]].join('::') : undefined;
}

const escape = (text: string) =>
  text.replace(/&/g, '&amp;').replace(/"/g, '&quot;').replace(/</g, '&lt;');

function anchor(href: string, inner: string): string {
  const key = href.startsWith('api:') ? href.slice(4) : docsRsKey(href);
  if (key) {
    return `<a href="#${escape(key)}" data-api="${escape(key)}">${inner}</a>`;
  }
  return `<a href="${escape(href)}">${inner}</a>`;
}

/**
 * rustdoc's intra-doc links, like [`Sky`], are reference links without a
 * definition. Appending a definition for each one resolves them.
 */
function withDefinitions(source: string, links: Record<string, string>): string {
  const definitions = Object.entries(links)
    .filter(([text]) => !/[[\]\n]/.test(text))
    .map(([text, href]) => `[${text}]: <${href}>`)
    .join('\n');
  return `${source}\n\n${definitions}\n`;
}

async function markdownTools() {
  const code = await highlighter;

  /** A renderer that moves headings `shift` levels down, and records the ids it gives `##` and `###` headings. */
  const make = (links: Record<string, string>, shift: number, headings?: Heading[]) => {
    const usedIds = new Map<string, number>();
    return new Marked({
      renderer: {
        heading({ tokens, depth }) {
          const level = Math.min(6, depth + shift);
          const inner = this.parser.parseInline(tokens);
          if (!headings || level > 3) {
            return `<h${level}>${inner}</h${level}>\n`;
          }
          const text = this.parser.parseInline(tokens, this.parser.textRenderer);
          const slug = slugify(text);
          const uses = usedIds.get(slug) ?? 0;
          usedIds.set(slug, uses + 1);
          const id = uses === 0 ? slug : `${slug}-${uses}`;
          headings.push({ id, text, depth: level });
          return `<h${level} id="${id}">${inner}</h${level}>\n`;
        },
        code({ text, lang }) {
          const language = codeLanguage(lang);
          const loaded = code.getLoadedLanguages().includes(language);
          return code.codeToHtml(language === 'rust' ? visibleRust(text) : text, {
            lang: loaded ? language : 'text',
            theme: CODE_THEME,
            colorReplacements: CODE_COLORS,
          });
        },
        link({ href, tokens }) {
          return anchor(links[href] ?? href, this.parser.parseInline(tokens));
        },
      },
    });
  };

  /** Renders the Markdown of a doc comment. */
  const docs = (source: string, links: Record<string, string>, shift: number): string =>
    source.trim() ? make(links, shift).parse(withDefinitions(source, links), { async: false }) : '';

  /** Renders a guide, split at its `api` blocks. */
  const guide = (source: string, links: Record<string, string>) => {
    const headings: Heading[] = [];
    const marked = make(links, 0, headings);
    const tokens = marked.lexer(withDefinitions(source, links));
    const blocks: GuideBlock[] = [];
    let start = 0;
    const flush = (end: number) => {
      const run = Object.assign(tokens.slice(start, end), { links: tokens.links }) as TokensList;
      const html = marked.parser(run);
      if (html.trim()) {
        blocks.push({ html });
      }
    };
    tokens.forEach((token: MarkdownToken, i: number) => {
      if (token.type === 'code' && token.lang === 'api') {
        flush(i);
        blocks.push({ embed: token.text.split('\n').map((line: string) => line.trim()).filter(Boolean) });
        start = i + 1;
      }
    });
    flush(tokens.length);
    return { blocks, headings };
  };

  return { docs, guide };
}

/** The first paragraph of rustdoc Markdown, as plain text. */
function summary(docs: string): string {
  const first = docs.split(/\n\s*\n/)[0] ?? '';
  return first
    .replace(/\[([^\]]+)\]\([^)]*\)/g, '$1')
    .replace(/\[([^\]]+)\]/g, '$1')
    .replace(/`/g, '')
    .replace(/\s+/g, ' ')
    .trim();
}

/** Reads a guide's front matter: `key: value` lines, and `key:` followed by `  - item` lines. */
function frontMatter(source: string, file: string) {
  const match = /^---\n([\s\S]*?)\n---\n/.exec(source);
  if (!match) {
    throw new Error(`${file} must start with front matter that gives its title`);
  }
  const data: Record<string, string | string[]> = {};
  let list: string[] | undefined;
  for (const line of match[1].split('\n')) {
    const content = line.replace(/\s+#.*$/, '');
    if (!content.trim()) {
      continue;
    }
    const entry = /^\s+-\s+(.+)$/.exec(content);
    if (entry && list) {
      list.push(entry[1].trim());
      continue;
    }
    const field = /^(\w+):\s*(.*)$/.exec(content);
    if (!field) {
      throw new Error(`${file}: can't read the front matter line "${line}"`);
    }
    if (field[2]) {
      data[field[1]] = field[2].trim();
      list = undefined;
    } else {
      list = [];
      data[field[1]] = list;
    }
  }
  return { data, body: source.slice(match[0].length) };
}

/**
 * Finds what a Rust path names, the way a guide writes it: `LightPropagation`,
 * `rendering::VoxelWorldBindings` or `VoxelWorld::raycast`. Returns its key.
 */
function resolver(items: RawItem[]) {
  const byPath = new Map<string, RawItem>();
  const byName = new Map<string, RawItem[]>();
  for (const item of items) {
    for (const path of [item.path, ...item.aliases]) {
      byPath.set(path, item);
    }
    byName.set(item.name, [...(byName.get(item.name) ?? []), item]);
  }
  const item = (path: string): RawItem | undefined => {
    const found = byPath.get(path) ?? byPath.get(`prelude::${path}`);
    if (found) {
      return found;
    }
    const named = byName.get(path) ?? [];
    if (named.length > 1) {
      throw new Error(`${path} could mean ${named.map((i) => i.path).join(' or ')}`);
    }
    return named[0];
  };
  return (reference: string): string | undefined => {
    const path = reference
      .replace(/^(crate|bevy_voxel_world)::/, '')
      .replace(/\(\)$/, '')
      .replace(/!$/, '');
    const found = item(path);
    if (found) {
      return found.path;
    }
    const split = path.lastIndexOf('::');
    const owner = split > 0 ? item(path.slice(0, split)) : undefined;
    if (!owner) {
      return undefined;
    }
    const name = path.slice(split + 2);
    const members = owner.members.filter((m) => m.name === name || m.key === name);
    if (members.length > 1) {
      throw new Error(`${path} could mean ${members.map((m) => m.key).join(' or ')}; write ${owner.name}::${members[0].key}`);
    }
    return members[0] && `${owner.path}::${members[0].key}`;
  };
}

/** Each feature in Cargo.toml, with the comment above it. */
function cargoFeatures(): { name: string; doc: string; enables: string[] }[] {
  const manifest = readFileSync(manifestFile, 'utf8');
  const section = /\n\[features\]\n([\s\S]*?)\n\[/.exec(manifest)?.[1] ?? '';
  const features = [];
  let comment: string[] = [];
  for (const line of section.split('\n')) {
    if (line.startsWith('#')) {
      comment.push(line.replace(/^#\s?/, ''));
      continue;
    }
    const match = /^(\w+)\s*=\s*\[(.*)\]/.exec(line);
    if (match && match[1] !== 'default') {
      const enables = [...match[2].matchAll(/"([\w]+)"/g)].map((m) => m[1]);
      features.push({ name: match[1], doc: comment.join(' '), enables });
    }
    comment = [];
  }
  return features;
}

async function buildView(): Promise<{ view: ApiView; files: string[] }> {
  if (!existsSync(apiFile)) {
    throw new Error('website/api.json is missing. Run `npm run build:api` first.');
  }
  const api: RawApi = JSON.parse(readFileSync(apiFile, 'utf8'));
  const render = await markdownTools();
  const resolve = resolver(api.items);
  const errors: string[] = [];

  // Each guide file is a topic, in the order of their names.
  const guideFiles = readdirSync(guideDir).filter((f) => f.endsWith('.md')).sort();
  const guides = guideFiles.map((file) => {
    const { data, body } = frontMatter(readFileSync(new URL(file, guideDir), 'utf8'), file);
    if (typeof data.title !== 'string') {
      throw new Error(`${file} needs a title in its front matter`);
    }
    const items = ((data.api ?? []) as string[]).flatMap((reference) => {
      const key = resolve(reference);
      if (!key || !api.items.some((item) => item.path === key)) {
        errors.push(`${file}: api lists ${reference}, which isn't a public item`);
        return [];
      }
      return [key];
    });
    // [`Name`] links to the API, as in a doc comment.
    const links: Record<string, string> = {};
    for (const match of body.matchAll(/\[`([^`\]\n]+)`\](?!\()/g)) {
      const reference = match[1];
      const atLineStart = match.index === 0 || body[match.index - 1] === '\n';
      if (atLineStart && body[match.index + match[0].length] === ':') {
        continue; // A link definition, not a link.
      }
      const key = resolve(reference);
      if (key) {
        links[`\`${reference}\``] = `api:${key}`;
      } else {
        errors.push(`${file}: [\`${reference}\`] names nothing in the API`);
      }
    }
    return {
      file,
      id: file.replace(/^\d+-/, '').replace(/\.md$/, ''),
      title: data.title,
      feature: typeof data.feature === 'string' ? data.feature : undefined,
      module: typeof data.module === 'string' ? data.module : undefined,
      items,
      body,
      links,
    };
  });

  const topicOf = new Map<string, string>();
  for (const guide of guides) {
    for (const path of guide.items) {
      if (topicOf.has(path)) {
        errors.push(`${guide.file}: ${path} is already in ${topicOf.get(path)}`);
      }
      topicOf.set(path, guide.file);
    }
  }
  // An item no guide explains is still in the API page, under Other.
  const unguided = api.items.filter((item) => !topicOf.has(item.path)).map((item) => item.path);
  if (errors.length > 0) {
    throw new Error(`The guide doesn't match the API:\n  ${errors.join('\n  ')}`);
  }

  const items: Record<string, ItemView> = {};
  for (const item of api.items) {
    items[item.path] = {
      path: item.path,
      aliases: item.aliases,
      name: item.name,
      kind: item.kind,
      features: item.features,
      signature: item.signature,
      html: render.docs(item.docs, item.links, 2),
      summary: summary(item.docs),
      source: item.source,
      deprecated: item.deprecated,
      members: item.members.map((member) => ({
        anchor: `${item.path}::${member.key}`,
        name: member.name,
        section: member.section,
        features: member.features.filter((f) => !item.features.includes(f)),
        signature: member.signature,
        html: render.docs(member.docs, member.links, 4),
      })),
      traits: item.traits.map((t) => ({ ...t, features: t.features.filter((f) => !item.features.includes(f)) })),
      topic: guides.find((g) => g.file === topicOf.get(item.path))?.id,
    };
  }

  const topics: TopicView[] = guides.map((guide) => {
    const module = guide.module ? api.modules.find((m) => m.path === guide.module) : undefined;
    const { blocks, headings } = render.guide(guide.body, guide.links);
    for (const block of blocks) {
      if ('embed' in block) {
        block.embed = block.embed.map((reference) => {
          const key = resolve(reference);
          if (!key) {
            throw new Error(`${guide.file}: an api block lists ${reference}, which names nothing in the API`);
          }
          return key;
        });
      }
    }
    const feature = guide.feature;
    const elsewhere = feature
      ? Object.values(items)
          .filter((item) => item.topic !== guide.id && !item.features.includes(feature))
          .map((item) => ({
            item: item.path,
            members: item.members.filter((m) => m.features.includes(feature)).map((m) => m.anchor),
            traits: item.traits.filter((t) => t.features.includes(feature)).map((t) => t.name),
          }))
          .filter((e) => e.members.length > 0 || e.traits.length > 0)
      : [];
    return {
      id: guide.id,
      title: guide.title,
      file: `website/guide/${guide.file}`,
      feature,
      blocks,
      headings,
      moduleDocs: module && { path: module.path, html: render.docs(module.docs, module.links, 2) },
      items: guide.items,
      elsewhere,
    };
  });

  const features: FeatureView[] = cargoFeatures().map((feature) => ({
    ...feature,
    adds: api.items.reduce(
      (count, item) =>
        count +
        (item.features.includes(feature.name) ? 1 : 0) +
        item.members.filter((m) => m.features.includes(feature.name) && !item.features.includes(feature.name)).length,
      0,
    ),
    topic: topics.find((topic) => topic.feature === feature.name)?.id,
  }));

  return {
    view: { version: api.version, features, apiFeatures: api.features, topics, items, unguided },
    files: guideFiles.map((file) => fileURLToPath(new URL(file, guideDir))),
  };
}

/** Lets the site import `virtual:api`, the guide and the crate's API. */
export function api(): Plugin {
  const guidePath = fileURLToPath(guideDir);
  const apiPath = fileURLToPath(apiFile);
  return {
    name: 'api',
    resolveId(id) {
      return id === MODULE_ID ? RESOLVED_MODULE_ID : null;
    },
    async load(id) {
      if (id !== RESOLVED_MODULE_ID) {
        return null;
      }
      const { view, files } = await buildView();
      for (const file of [apiPath, ...files]) {
        this.addWatchFile(file);
      }
      if (view.unguided.length > 0) {
        this.warn(`No guide explains these items, so only the API page shows them:\n  ${view.unguided.join('\n  ')}`);
      }
      return `export default ${JSON.stringify(view)};`;
    },
    // Editing, adding or removing a guide, or rebuilding api.json, reloads the
    // page, since nothing imports those files directly.
    configureServer(server) {
      server.watcher.add([guidePath, apiPath]);
      const reload = (file: string) => {
        if (!file.startsWith(guidePath) && file !== apiPath) {
          return;
        }
        const module = server.moduleGraph.getModuleById(RESOLVED_MODULE_ID);
        if (module) {
          server.moduleGraph.invalidateModule(module);
        }
        server.ws.send({ type: 'full-reload' });
      };
      server.watcher.on('change', reload);
      server.watcher.on('add', reload);
      server.watcher.on('unlink', reload);
    },
  };
}
