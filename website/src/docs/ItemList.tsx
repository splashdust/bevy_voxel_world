// The list that the API page and each guide's reference share. One
// row per item with its summary, which opens in place to the item's full docs.

import { useState } from 'react';
import {
  type ApiView,
  FeatureChips,
  guideHref,
  type ItemView,
  ItemDoc,
  itemOfKey,
  KIND_LABEL,
  scrollToId,
  type Shows,
} from './shared';

/** A member or trait impl named under an item's summary: what a search or filter matched. */
export type Hit = { label: string; anchor?: string };

export type ItemRow = { item: ItemView; hits?: Hit[] };

/** Which items' rows are open, starting with the one the address points at. */
export function useOpenItems(api: ApiView) {
  const [open, setOpen] = useState<Set<string>>(() => {
    const item = itemOfKey(api, decodeURIComponent(location.hash.slice(1)));
    return new Set(item ? [item.path] : []);
  });
  return {
    open,
    toggle: (path: string) =>
      setOpen((previous) => {
        const next = new Set(previous);
        if (!next.delete(path)) {
          next.add(path);
        }
        return next;
      }),
    /**
     * Opens the row of the item a key names, and scrolls to the key: the row,
     * or a member inside it. Returns false if the key names no item.
     */
    reveal: (key: string) => {
      const item = itemOfKey(api, key);
      if (!item) {
        return false;
      }
      setOpen((previous) => new Set(previous).add(item.path));
      scrollToId(key);
      return true;
    },
  };
}

export function ItemList({
  api,
  rows,
  open,
  onToggle,
  filter,
  hideFeature,
  guideLinks,
}: {
  api: ApiView;
  rows: ItemRow[];
  open: Set<string>;
  onToggle: (path: string) => void;
  /** Which members and traits an open row shows. */
  filter: Shows;
  /** A feature the whole list needs, so its chip would say nothing. */
  hideFeature?: string;
  /** Whether an open row links to the guide that explains the item. */
  guideLinks?: boolean;
}) {
  return (
    <ul className="item-list">
      {rows.map(({ item, hits = [] }) => {
        const isOpen = open.has(item.path);
        const guide = guideLinks && item.topic ? api.topics.find((t) => t.id === item.topic) : undefined;
        return (
          <li key={item.path} id={item.path} className={isOpen ? 'open' : undefined}>
            <button
              type="button"
              className="item-row"
              aria-expanded={isOpen}
              onClick={() => {
                onToggle(item.path);
                // Opening a row a member matched goes straight to that member.
                const first = hits.find((hit) => hit.anchor);
                if (!isOpen && first?.anchor) {
                  scrollToId(first.anchor);
                }
              }}
            >
              <span className="api-kind">{KIND_LABEL[item.kind] ?? item.kind}</span>
              <span className="item-row-title">
                <code className="item-row-name">{item.name}</code>
                <FeatureChips features={item.features.filter((f) => f !== hideFeature)} />
              </span>
              <span className="item-row-summary">{item.summary}</span>
              {hits.length > 0 && (
                <span className="item-row-hits">
                  {hits.map((hit) => (
                    <code key={hit.label}>{hit.label}</code>
                  ))}
                </span>
              )}
            </button>
            {isOpen && (
              <div className="item-detail">
                {guide && (
                  <p className="item-detail-guide">
                    Explained in <a href={guideHref(guide.id, item.path)}>{guide.title}</a>
                  </p>
                )}
                <ItemDoc item={item} filter={filter} headless />
              </div>
            )}
          </li>
        );
      })}
    </ul>
  );
}
