import { StrictMode, type ReactNode } from 'react';
import { createRoot } from 'react-dom/client';
import '@fontsource-variable/martian-mono/wdth.css';
import './styles.css';
import { SteppedEdge } from './SteppedEdge';

export const REPOSITORY = 'https://github.com/splashdust/bevy_voxel_world';

const PAGES = [
  { id: 'landing', label: 'Home', path: '' },
  { id: 'docs', label: 'Docs', path: 'docs/' },
  { id: 'demo', label: 'Demo', path: 'demo/' },
  { id: 'changelog', label: 'Changelog', path: 'changelog/' },
] as const;

/** The pages under Docs, each a tab in the docs header. */
const DOCS_PAGES = [
  { id: 'docs', label: 'Overview', path: 'docs/' },
  { id: 'guides', label: 'Guides', path: 'docs/guides/' },
  { id: 'api', label: 'API', path: 'docs/api/' },
  { id: 'examples', label: 'Examples', path: 'docs/examples/' },
] as const;

type DocsPageId = (typeof DOCS_PAGES)[number]['id'];
type PageId = (typeof PAGES)[number]['id'] | DocsPageId;

export const pageUrl = (id: PageId) =>
  import.meta.env.BASE_URL + [...PAGES, ...DOCS_PAGES].find((page) => page.id === id)!.path;

function SiteNav({ current }: { current: PageId }) {
  // Every page under Docs marks Docs as the current page.
  const section = DOCS_PAGES.some((page) => page.id === current) ? 'docs' : current;
  return (
    <nav className="site-nav" aria-label="Site">
      {PAGES.map((page) => (
        <a
          key={page.id}
          className={`site-nav-page site-nav-${page.id}`}
          href={pageUrl(page.id)}
          aria-current={page.id === section ? 'page' : undefined}
        >
          {page.label}
        </a>
      ))}
      <a className="site-nav-page site-nav-external" href={REPOSITORY}>
        GitHub
      </a>
    </nav>
  );
}

/** The sky at the top of a page, for pages without the landing page's hero. */
export function PageHeader({ title }: { title: string }) {
  return (
    <header className="hero page-header">
      <div className="hero-content">
        <h1>{title}</h1>
      </div>
      <SteppedEdge seed={2.4} fill="var(--dirt)" cap="var(--grass)" />
    </header>
  );
}

/** The header of a page under Docs: tabs for the docs pages, then the page's title. */
export function DocsHeader({ current, title }: { current: DocsPageId; title: string }) {
  return (
    <header className="hero page-header docs-header">
      <div className="hero-content">
        <nav className="docs-tabs" aria-label="Docs">
          {DOCS_PAGES.map((page) => (
            <a key={page.id} href={pageUrl(page.id)} aria-current={page.id === current ? 'page' : undefined}>
              {page.label}
            </a>
          ))}
        </nav>
        <h1>{title}</h1>
      </div>
      <SteppedEdge seed={2.4} fill="var(--dirt)" cap="var(--grass)" />
    </header>
  );
}

function Footer() {
  return (
    <footer>
      <SteppedEdge seed={8.1} fill="var(--rock)" />
      <div className="footer-content">
        <a href={REPOSITORY}>GitHub</a>
        <a href="https://crates.io/crates/bevy_voxel_world">crates.io</a>
        <a href="https://docs.rs/bevy_voxel_world">docs.rs</a>
        <a href="https://bevyengine.org">Bevy</a>
      </div>
    </footer>
  );
}

/** Renders a page's content between the site's navigation and footer. */
export function mountPage(current: PageId, content: ReactNode) {
  createRoot(document.getElementById('root')!).render(
    <StrictMode>
      <SiteNav current={current} />
      {content}
      <Footer />
    </StrictMode>,
  );
}
