import { Marked, type Token } from 'marked';
import { createHighlighter } from 'shiki';
import type { Plugin } from 'vite';

export type Heading = { id: string; text: string; depth: number };

const CODE_THEME = 'vitesse-dark';
/** Swaps the theme's black background for the site's --dirt-deep. */
const CODE_COLORS = { '#121212': '#241900' };

const highlighter = createHighlighter({
  themes: [CODE_THEME],
  langs: ['rust', 'bash', 'toml'],
});

/** Makes the same anchors as GitHub, so links to a section work on both sites. */
function slugify(text: string): string {
  return text
    .toLowerCase()
    .replace(/[^\w\s-]/g, '')
    .replace(/\s/g, '-');
}

async function renderMarkdown(source: string) {
  const code = await highlighter;
  const headings: Heading[] = [];
  const usedIds = new Map<string, number>();

  const marked = new Marked({
    renderer: {
      heading({ tokens, depth }) {
        const text = this.parser.parseInline(tokens, this.parser.textRenderer);
        const slug = slugify(text);
        const uses = usedIds.get(slug) ?? 0;
        usedIds.set(slug, uses + 1);
        const id = uses === 0 ? slug : `${slug}-${uses}`;
        headings.push({ id, text, depth });
        return `<h${depth} id="${id}">${this.parser.parseInline(tokens)}</h${depth}>\n`;
      },
      code({ text, lang }) {
        const known = lang && code.getLoadedLanguages().includes(lang);
        return code.codeToHtml(text, {
          lang: known ? lang : 'text',
          theme: CODE_THEME,
          colorReplacements: CODE_COLORS,
        });
      },
      html({ text }) {
        // An image without a height takes up no space until it loads, so only the
        // ones with a height can load late without moving the page under the reader.
        return text.replace(/<img (?=[^>]*\bheight=)/g, '<img loading="lazy" ');
      },
    },
  });

  // Each page shows its own title, so the document starts at its first section.
  const tokens = marked.lexer(source);
  const isSection = (token: Token) => token.type === 'heading' && token.depth === 2;
  const body = tokens.slice(tokens.findIndex(isSection));

  // The last section is wrapped so a page can make it tall enough for its heading
  // to scroll up to the top of the screen.
  const lastHeading = body.findLastIndex((token) => token.type === 'heading');
  const html =
    marked.parser(body.slice(0, lastHeading)) +
    `<section class="last-section">${marked.parser(body.slice(lastHeading))}</section>`;

  return { html, headings };
}

/** Lets the site import a Markdown file as its HTML and its list of headings. */
export function markdown(): Plugin {
  return {
    name: 'markdown',
    enforce: 'pre',
    async transform(source, id) {
      if (!id.endsWith('.md')) {
        return null;
      }
      return { code: `export default ${JSON.stringify(await renderMarkdown(source))};`, map: null };
    },
  };
}
