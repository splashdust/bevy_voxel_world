// PROTOTYPE: lets the site import `virtual:api`, the crate's API sorted into the
// topics of api-topics.ts, with its docs rendered to HTML. It reads api.json,
// which `npm run build:api` writes.

import { existsSync, readFileSync } from 'node:fs';
import { Marked } from 'marked';
import type { Plugin } from 'vite';
import { CODE_COLORS, CODE_THEME, highlighter } from './markdown.ts';
import { TOPICS } from './api-topics.ts';

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
  topic: string;
};

export type TopicView = {
  id: string;
  title: string;
  feature?: string;
  /** The README section or blurb that introduces the topic, as HTML. */
  intro: string;
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
  items: Record<string, ItemView>;
};

const MODULE_ID = 'virtual:api';
const RESOLVED_MODULE_ID = `\0${MODULE_ID}`;

const apiFile = new URL('../api.json', import.meta.url);
const readmeFile = new URL('../../README.md', import.meta.url);
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

/** The key a docs.rs link to this crate points at, so the README's links stay on the site. */
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

async function docRenderer() {
  const code = await highlighter;
  /** Renders rustdoc Markdown, moving its headings `shift` levels down. */
  return (source: string, links: Record<string, string>, shift: number): string => {
    if (!source.trim()) {
      return '';
    }
    const marked = new Marked({
      renderer: {
        heading({ tokens, depth }) {
          const level = Math.min(6, depth + shift);
          return `<h${level}>${this.parser.parseInline(tokens)}</h${level}>\n`;
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
    // rustdoc's intra-doc links, like [`Sky`], are reference links without a
    // definition. Defining each one resolves them.
    const definitions = Object.entries(links)
      .filter(([text]) => !/[[\]\n]/.test(text))
      .map(([text, href]) => `[${text}]: <${href}>`)
      .join('\n');
    return marked.parse(`${source}\n\n${definitions}\n`, { async: false });
  };
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

/** Returns the body of the README section under `## title`, without its heading. */
function readmeSection(readme: string, title: string): string {
  const lines = readme.split('\n');
  const start = lines.findIndex((line) => line === `## ${title}`);
  if (start < 0) {
    throw new Error(`README.md has no "## ${title}" section, which api-topics.ts names`);
  }
  const end = lines.findIndex((line, i) => i > start && line.startsWith('## '));
  return lines.slice(start + 1, end < 0 ? undefined : end).join('\n');
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

async function buildView(): Promise<ApiView> {
  if (!existsSync(apiFile)) {
    throw new Error('website/api.json is missing. Run `npm run build:api` first.');
  }
  const api: RawApi = JSON.parse(readFileSync(apiFile, 'utf8'));
  const readme = readFileSync(readmeFile, 'utf8');
  const render = await docRenderer();
  const byPath = new Map(api.items.map((item) => [item.path, item]));

  const topicOf = new Map<string, string>();
  for (const topic of TOPICS) {
    for (const path of topic.items) {
      if (!byPath.has(path)) {
        throw new Error(`api-topics.ts lists ${path}, which isn't a public item`);
      }
      if (topicOf.has(path)) {
        throw new Error(`api-topics.ts lists ${path} in both ${topicOf.get(path)} and ${topic.id}`);
      }
      topicOf.set(path, topic.id);
    }
  }
  const missing = api.items.filter((item) => !topicOf.has(item.path)).map((item) => item.path);
  if (missing.length > 0) {
    throw new Error(`These public items are in no topic. Add each to api-topics.ts:\n  ${missing.join('\n  ')}`);
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
      html: render(item.docs, item.links, 2),
      summary: summary(item.docs),
      source: item.source,
      deprecated: item.deprecated,
      members: item.members.map((member) => ({
        anchor: `${item.path}::${member.key}`,
        name: member.name,
        section: member.section,
        features: member.features.filter((f) => !item.features.includes(f)),
        signature: member.signature,
        html: render(member.docs, member.links, 4),
      })),
      traits: item.traits.map((t) => ({ ...t, features: t.features.filter((f) => !item.features.includes(f)) })),
      topic: topicOf.get(item.path)!,
    };
  }

  const topics: TopicView[] = TOPICS.map((topic) => {
    const module = topic.module ? api.modules.find((m) => m.path === topic.module) : undefined;
    const intro = [topic.blurb ?? '', topic.readme ? readmeSection(readme, topic.readme) : ''].join('\n\n');
    const elsewhere = topic.feature
      ? Object.values(items)
          .filter((item) => item.topic !== topic.id && !item.features.includes(topic.feature!))
          .map((item) => ({
            item: item.path,
            members: item.members.filter((m) => m.features.includes(topic.feature!)).map((m) => m.anchor),
            traits: item.traits.filter((t) => t.features.includes(topic.feature!)).map((t) => t.name),
          }))
          .filter((e) => e.members.length > 0 || e.traits.length > 0)
      : [];
    return {
      id: topic.id,
      title: topic.title,
      feature: topic.feature,
      intro: render(intro, {}, 0),
      moduleDocs: module && { path: module.path, html: render(module.docs, module.links, 2) },
      items: topic.items,
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
    topic: TOPICS.find((topic) => topic.feature === feature.name)?.id,
  }));

  return { version: api.version, features, apiFeatures: api.features, topics, items };
}

/** Lets the site import `virtual:api`, the crate's API as the docs section shows it. */
export function api(): Plugin {
  return {
    name: 'api',
    resolveId(id) {
      return id === MODULE_ID ? RESOLVED_MODULE_ID : null;
    },
    async load(id) {
      if (id !== RESOLVED_MODULE_ID) {
        return null;
      }
      this.addWatchFile(apiFile.pathname);
      this.addWatchFile(readmeFile.pathname);
      return `export default ${JSON.stringify(await buildView())};`;
    },
  };
}
