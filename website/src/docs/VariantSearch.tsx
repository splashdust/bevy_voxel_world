// PROTOTYPE, variant B: the whole API on one page, search first. Every item is
// one row with its summary, grouped by topic; a row opens in place to its full
// docs. No guide text: the README stays on the landing page.

import { useEffect, useRef, useState } from 'react';
import {
  type ApiView,
  FeatureChips,
  FeatureToggles,
  type ItemView,
  ItemDoc,
  itemOfKey,
  KIND_LABEL,
  scrollToId,
  useApiLinks,
  useFeatureFilter,
} from './shared';

export function SearchFirst({ api }: { api: ApiView }) {
  const filter = useFeatureFilter(api.apiFeatures);
  const [query, setQuery] = useState('');
  const [open, setOpen] = useState<Set<string>>(() => {
    const item = itemOfKey(api, decodeURIComponent(location.hash.slice(1)));
    return new Set(item ? [item.path] : []);
  });
  const search = useRef<HTMLInputElement>(null);

  useEffect(() => {
    const key = decodeURIComponent(location.hash.slice(1));
    if (key) {
      scrollToId(key);
    }
    const onKey = (event: KeyboardEvent) => {
      const typing = (event.target as Element).closest?.('input, textarea, [contenteditable]');
      if (event.key === '/' && !typing) {
        event.preventDefault();
        search.current?.focus();
      }
    };
    addEventListener('keydown', onKey);
    return () => removeEventListener('keydown', onKey);
  }, []);

  // A link opens the item it points at, and clears a search that hides it.
  useApiLinks((key) => {
    const item = itemOfKey(api, key);
    if (!item) {
      return;
    }
    setQuery('');
    setOpen((previous) => new Set(previous).add(item.path));
    history.replaceState(null, '', `#${key}`);
    scrollToId(key === item.path ? `row-${key}` : key);
  });

  const toggle = (path: string) =>
    setOpen((previous) => {
      const next = new Set(previous);
      if (!next.delete(path)) {
        next.add(path);
      }
      return next;
    });

  const needle = query.trim().toLowerCase();
  const match = (item: ItemView) => {
    if (!needle) {
      return { hit: true, members: [] as string[] };
    }
    const own = [item.name, item.path, item.summary].some((text) => text.toLowerCase().includes(needle));
    const members = item.members
      .filter((m) => filter.shows(m.features) && m.name.toLowerCase().includes(needle))
      .map((m) => m.name);
    return { hit: own || members.length > 0, members };
  };

  const groups = api.topics
    .map((topic) => ({
      topic,
      rows: topic.items
        .map((path) => api.items[path])
        .filter((item) => filter.shows(item.features))
        .map((item) => ({ item, match: match(item) }))
        .filter((row) => row.match.hit),
    }))
    .filter((group) => group.rows.length > 0);
  const count = groups.reduce((sum, group) => sum + group.rows.length, 0);

  return (
    <main className="doc-section docs-b">
      <div className="docs-b-inner">
        <div className="docs-b-search">
          <h1>
            API <span className="docs-b-version">v{api.version}</span>
          </h1>
          <input
            ref={search}
            type="search"
            placeholder="Search types, functions and methods    /"
            aria-label="Search the API"
            value={query}
            onChange={(event) => setQuery(event.target.value)}
          />
          <div className="docs-b-filters">
            <FeatureToggles filter={filter} />
            <p className="docs-b-count">{count} items</p>
          </div>
        </div>
        {groups.map(({ topic, rows }) => (
          <section key={topic.id} className="docs-b-group">
            <h2>
              {topic.title}
              {topic.feature && <FeatureChips features={[topic.feature]} />}
            </h2>
            <ul>
              {rows.map(({ item, match }) => {
                const isOpen = open.has(item.path);
                return (
                  <li key={item.path} id={`row-${item.path}`} className={isOpen ? 'open' : undefined}>
                    <button
                      type="button"
                      className="docs-b-row"
                      aria-expanded={isOpen}
                      onClick={() => toggle(item.path)}
                    >
                      <span className="api-kind">{KIND_LABEL[item.kind] ?? item.kind}</span>
                      <span className="docs-b-title">
                        <code className="docs-b-name">{item.name}</code>
                        <FeatureChips features={item.features.filter((f) => f !== topic.feature)} />
                      </span>
                      <span className="docs-b-summary">{item.summary}</span>
                      {match.members.length > 0 && (
                        <span className="docs-b-hits">
                          {match.members.map((name) => (
                            <code key={name}>.{name}</code>
                          ))}
                        </span>
                      )}
                    </button>
                    {isOpen && (
                      <div className="docs-b-detail" id={item.path}>
                        <ItemDoc item={item} filter={filter} headless />
                      </div>
                    )}
                  </li>
                );
              })}
            </ul>
          </section>
        ))}
        {groups.length === 0 && <p className="docs-b-empty">Nothing matches “{query}”.</p>}
      </div>
    </main>
  );
}
