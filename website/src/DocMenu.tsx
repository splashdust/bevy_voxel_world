import { useEffect, useState } from 'react';
import type { Heading } from '../scripts/markdown.ts';

/**
 * How far below the top of the viewport a heading counts as reached, in pixels.
 * Must stay more than `scroll-padding-top` in styles.css, so that a heading the
 * reader jumped to counts as reached.
 */
const READING_LINE = 100;

/** Returns the id of the heading of the section being read. */
function useCurrentHeading(headings: Heading[]): string {
  const [current, setCurrent] = useState(headings[0].id);

  useEffect(() => {
    let frame = 0;
    const update = () => {
      // Read on every update, since images loading above a heading move it.
      const reached = headings.filter(
        ({ id }) => document.getElementById(id)!.getBoundingClientRect().top <= READING_LINE,
      );
      setCurrent((reached.at(-1) ?? headings[0]).id);
    };
    const schedule = () => {
      cancelAnimationFrame(frame);
      frame = requestAnimationFrame(update);
    };

    update();
    addEventListener('scroll', schedule, { passive: true });
    addEventListener('resize', schedule);
    return () => {
      cancelAnimationFrame(frame);
      removeEventListener('scroll', schedule);
      removeEventListener('resize', schedule);
    };
  }, [headings]);

  return current;
}

/**
 * A menu of a document's headings that follows the reader's position. It's a
 * column beside the document on wide screens, and a collapsed list above it on
 * narrow ones.
 */
export function DocMenu({ headings }: { headings: Heading[] }) {
  const current = useCurrentHeading(headings);

  // The document only exists once React has rendered it, which can be too late
  // for the browser's own jump to the address's #heading.
  useEffect(() => {
    const target = decodeURIComponent(location.hash.slice(1));
    if (target) {
      document.getElementById(target)?.scrollIntoView();
    }
  }, []);

  const links = (
    <ol>
      {headings.map(({ id, text, depth }) => (
        <li key={id} className={`depth-${depth}`}>
          <a href={`#${id}`} aria-current={id === current ? 'location' : undefined}>
            {text}
          </a>
        </li>
      ))}
    </ol>
  );

  return (
    <>
      <details className="doc-menu-compact">
        <summary>On this page</summary>
        {links}
      </details>
      <nav className="doc-menu" aria-label="On this page">
        <p>On this page</p>
        {links}
      </nav>
    </>
  );
}
