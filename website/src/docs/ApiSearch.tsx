// Docs › API, every public item on one page, search first. Items are
// grouped by the guide that explains them, and those no guide explains come
// last under Other. `?q=` starts a search and `?features=` picks features, so
// any page can link to either.

import { useEffect, useRef, useState } from 'react';
import { DocsHeader } from '../Page';
import { type Hit, ItemList, type ItemRow, useOpenItems } from './ItemList';
import { type ApiView, FeatureChips, guideHref, type ItemView, scrollToId, useApiLinks } from './shared';

/** Cargo's name for the features a crate builds with when none are asked for. */
const DEFAULT = 'default';

/**
 * Which features to show the API of. With none picked, the page shows
 * everything; with some, it shows only what needs one of them, and `default`
 * stands for what needs no feature at all.
 */
function useFeaturePicks(features: string[]) {
  const all = [DEFAULT, ...features];
  const [picked, setPicked] = useState<Set<string>>(() => {
    const listed = new URLSearchParams(location.search).get('features')?.split(',') ?? [];
    return new Set(listed.filter((f) => all.includes(f)));
  });
  return {
    all,
    picked,
    shows: (needs: string[]) =>
      picked.size === 0 || (needs.length === 0 ? picked.has(DEFAULT) : needs.some((f) => picked.has(f))),
    toggle: (feature: string) =>
      setPicked((previous) => {
        const next = new Set(previous);
        if (!next.delete(feature)) {
          next.add(feature);
        }
        return next;
      }),
    clear: () => setPicked(new Set()),
  };
}

export function ApiSearch({ api }: { api: ApiView }) {
  const picks = useFeaturePicks(api.apiFeatures);
  const [query, setQuery] = useState(() => new URLSearchParams(location.search).get('q') ?? '');
  const items = useOpenItems(api);
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

  // The address keeps the search and the picked features, so either can be shared.
  useEffect(() => {
    const params = new URLSearchParams(location.search);
    const set = (name: string, value: string) => (value ? params.set(name, value) : params.delete(name));
    set('q', query);
    set('features', [...picks.picked].join(','));
    const search = params.toString();
    history.replaceState(null, '', `${location.pathname}${search ? `?${search}` : ''}${location.hash}`);
  }, [query, picks.picked]);

  // A link opens the item it points at, and clears a search or pick that hides it.
  useApiLinks((key) => {
    setQuery('');
    picks.clear();
    history.replaceState(null, '', `${location.pathname}#${key}`);
    items.reveal(key);
  });

  const needle = query.trim().toLowerCase();

  /** The item's row if the picked features and the search both let it through. */
  const row = (item: ItemView): ItemRow | undefined => {
    const shown = picks.shows(item.features);
    // An item a pick hides still shows when it has members or trait impls the pick covers.
    const pickedHits: Hit[] =
      picks.picked.size > 0 && !shown
        ? [
            ...item.members
              .filter((m) => picks.shows([...item.features, ...m.features]))
              .map((m) => ({ label: `.${m.name}`, anchor: m.anchor })),
            ...item.traits
              .filter((t) => picks.shows([...item.features, ...t.features]))
              .map((t) => ({ label: `impl ${t.name}` })),
          ]
        : [];
    if (!shown && pickedHits.length === 0) {
      return undefined;
    }
    if (!needle) {
      return { item, hits: pickedHits };
    }
    const own = [item.name, item.path, item.summary].some((text) => text.toLowerCase().includes(needle));
    const searchHits = item.members
      .filter((m) => picks.shows([...item.features, ...m.features]) && m.name.toLowerCase().includes(needle))
      .map((m) => ({ label: `.${m.name}`, anchor: m.anchor }));
    if (!own && searchHits.length === 0) {
      return undefined;
    }
    return { item, hits: searchHits.length > 0 ? searchHits : pickedHits };
  };

  const rowsOf = (paths: string[]) =>
    paths.map((path) => row(api.items[path])).filter((r): r is ItemRow => r !== undefined);
  const groups = [
    // An item that several guides hold shows once, under the first of them.
    ...api.topics.map((topic) => ({
      id: topic.id,
      title: topic.title,
      topic,
      rows: rowsOf(topic.items.filter((path) => api.items[path].topics[0] === topic.id)),
    })),
    { id: 'other', title: 'Other', topic: undefined, rows: rowsOf(api.unguided) },
  ].filter((group) => group.rows.length > 0);
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
              <div className="feature-picks" role="group" aria-label="Show only these features">
                <p>{picks.picked.size === 0 ? 'Every feature. Show only:' : 'Showing only:'}</p>
                {picks.all.map((feature) => (
                  <button
                    key={feature}
                    type="button"
                    className={`chip chip-${feature} pick`}
                    aria-pressed={picks.picked.has(feature)}
                    onClick={() => picks.toggle(feature)}
                  >
                    {feature}
                  </button>
                ))}
              </div>
              <p className="api-page-count">
                {count} items · v{api.version}
              </p>
            </div>
          </div>
          {groups.map((group) => (
            <section key={group.id} className="api-page-group">
              <h2>
                {group.title}
                {group.topic?.feature && <FeatureChips features={[group.topic.feature]} />}
                {group.topic && (
                  <a className="api-page-guide" href={guideHref(group.topic.id)}>
                    Guide →
                  </a>
                )}
              </h2>
              <ItemList
                api={api}
                rows={group.rows}
                open={items.open}
                onToggle={items.toggle}
                filter={picks}
                hideFeature={group.topic?.feature}
                guideLinks
              />
            </section>
          ))}
          {groups.length === 0 && <p className="api-page-empty">Nothing matches.</p>}
        </div>
      </main>
    </>
  );
}
