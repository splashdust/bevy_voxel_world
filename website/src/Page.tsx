import { StrictMode, type ReactNode } from 'react';
import { createRoot } from 'react-dom/client';
import '@fontsource-variable/martian-mono/wdth.css';
import './styles.css';
import { SteppedEdge } from './SteppedEdge';

export const REPOSITORY = 'https://github.com/splashdust/bevy_voxel_world';

const PAGES = [
  { id: 'landing', label: 'bevy_voxel_world', path: '' },
  { id: 'changelog', label: 'Changelog', path: 'changelog/' },
  { id: 'demo', label: 'Demo', path: 'demo/' },
  { id: 'examples', label: 'Examples', path: 'examples/' },
] as const;

type PageId = (typeof PAGES)[number]['id'];

export const pageUrl = (id: PageId) =>
  import.meta.env.BASE_URL + PAGES.find((page) => page.id === id)!.path;

function SiteNav({ current }: { current: PageId }) {
  return (
    <nav className="site-nav" aria-label="Site">
      {PAGES.map((page) => (
        <a
          key={page.id}
          className={`site-nav-page site-nav-${page.id}`}
          href={pageUrl(page.id)}
          aria-current={page.id === current ? 'page' : undefined}
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
