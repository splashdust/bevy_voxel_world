// PROTOTYPE: Docs › API, the whole API on one page, search first. Every item is
// one row with its summary, grouped by the guide that explains it; a row opens
// in place to its full docs. `?q=` starts a search, so any page can link here.

import { useEffect, useRef, useState } from 'react';
import { DocsHeader } from '../Page';
import {
  type ApiView,
  FeatureChips,
  FeatureToggles,
  guideHref,
  type ItemView,
  ItemDoc,
  itemOfKey,
  KIND_LABEL,
  type MemberView,
  scrollToId,
  useApiLinks,
  useFeatureFilter,
} from './shared';

export function ApiSearch({ api }: { api: ApiView }) {
  const filter = useFeatureFilter(api.apiFeatures);
  const [query, setQuery] = useState(() => new URLSearchParams(location.search).get('q') ?? '');
  const [open, setOpen] = useState<Set<string>>(() => {
    const item = itemOfKey(api, decodeURIComponent(location.hash.slice(1)));
    return new Set(item ? [item.path] : []);
  });
  const search = useRef<HTMLInputElement>(null);

  useEffect(() => {
    const key = decodeURIComponent(location.hash.slice(1));
    if (key) {
      scrollToId(key);
    } else {
      search.current?.focus();
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

  // The address keeps the search, so a search can be shared or reloaded.
  useEffect(() => {
    const params = new URLSearchParams(location.search);
    if (query) {
      params.set('q', query);
    } else {
      params.delete('q');
    }
    const search = params.toString();
    history.replaceState(null, '', `${location.pathname}${search ? `?${search}` : ''}${location.hash}`);
  }, [query]);

  // A link opens the item it points at, and clears a search that hides it.
  useApiLinks((key) => {
    const item = itemOfKey(api, key);
    if (!item) {
      return;
    }
    setQuery('');
    setOpen((previous) => new Set(previous).add(item.path));
    history.replaceState(null, '', `${location.pathname}#${key}`);
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
      return { hit: true, members: [] as MemberView[] };
    }
    const own = [item.name, item.path, item.summary].some((text) => text.toLowerCase().includes(needle));
    const members = item.members.filter((m) => filter.shows(m.features) && m.name.toLowerCase().includes(needle));
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
    <>
      <DocsHeader current="api" title="API" />
      <main className="doc-section api-page">
        <div className="api-page-inner">
          <div className="api-page-search">
            <input
              ref={search}
              type="search"
              placeholder="Search types, functions and methods    /"
              aria-label="Search the API"
              value={query}
              onChange={(event) => setQuery(event.target.value)}
            />
            <div className="api-page-filters">
              <FeatureToggles filter={filter} />
              <p className="api-page-count">
                {count} items · v{api.version}
              </p>
            </div>
          </div>
          {groups.map(({ topic, rows }) => (
            <section key={topic.id} className="api-page-group">
              <h2>
                {topic.title}
                {topic.feature && <FeatureChips features={[topic.feature]} />}
                <a className="api-page-guide" href={guideHref(topic.id)}>
                  Guide →
                </a>
              </h2>
              <ul>
                {rows.map(({ item, match }) => {
                  const isOpen = open.has(item.path);
                  return (
                    <li key={item.path} id={`row-${item.path}`} className={isOpen ? 'open' : undefined}>
                      <button
                        type="button"
                        className="api-page-row"
                        aria-expanded={isOpen}
                        onClick={() => {
                          toggle(item.path);
                          // Opening a row a method matched goes straight to that method.
                          if (!isOpen && match.members.length > 0) {
                            scrollToId(match.members[0].anchor);
                          }
                        }}
                      >
                        <span className="api-kind">{KIND_LABEL[item.kind] ?? item.kind}</span>
                        <span className="api-page-title">
                          <code className="api-page-name">{item.name}</code>
                          <FeatureChips features={item.features.filter((f) => f !== topic.feature)} />
                        </span>
                        <span className="api-page-summary">{item.summary}</span>
                        {match.members.length > 0 && (
                          <span className="api-page-hits">
                            {match.members.map((member) => (
                              <code key={member.anchor}>.{member.name}</code>
                            ))}
                          </span>
                        )}
                      </button>
                      {isOpen && (
                        <div className="api-page-detail" id={item.path}>
                          <p className="api-page-explained">
                            Explained in <a href={guideHref(topic.id, item.path)}>{topic.title}</a>
                          </p>
                          <ItemDoc item={item} filter={filter} headless />
                        </div>
                      )}
                    </li>
                  );
                })}
              </ul>
            </section>
          ))}
          {groups.length === 0 && <p className="api-page-empty">Nothing matches “{query}”.</p>}
        </div>
      </main>
    </>
  );
}
