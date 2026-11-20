// Docs › Guides, one page per topic, like a book. Topics down the
// left, the topic's guide from website/guide/, then its reference as the same
// list the API page shows, and a menu on the right that follows the reader.

import { useEffect, useMemo } from 'react';
import { DocMenu } from '../DocMenu';
import { DocsHeader, pageUrl } from '../Page';
import { ItemList, useOpenItems } from './ItemList';
import {
  ApiSearchBox,
  type ApiView,
  Elsewhere,
  FeatureChips,
  guideHref,
  SHOW_ALL,
  TopicGuide,
  topicOfKey,
  useApiLinks,
} from './shared';

export function Guides({ api }: { api: ApiView }) {
  const requested = new URLSearchParams(location.search).get('topic');
  const topic = api.topics.find((t) => t.id === requested) ?? api.topics[0];
  const reference = useOpenItems(api);

  useEffect(() => {
    document.title = `${topic.title} · Guides · bevy_voxel_world`;
  }, [topic]);

  // A link into this topic opens its row; one into another topic loads that
  // topic's page, and one to an item no guide explains opens the API page.
  useApiLinks((key) => {
    const target = topicOfKey(api, key);
    if (target === topic.id) {
      history.replaceState(null, '', `#${key}`);
      if (!reference.reveal(key)) {
        document.getElementById(key)?.scrollIntoView();
      }
    } else if (target) {
      location.assign(guideHref(target, key));
    } else {
      location.assign(`${pageUrl('api')}#${key}`);
    }
  });

  const items = useMemo(() => topic.items.map((path) => api.items[path]), [api, topic]);
  const headings = useMemo(
    () => [
      ...topic.headings,
      ...(items.length > 0 ? [{ id: 'reference', text: 'Reference', depth: 2 }] : []),
      ...items.map((item) => ({ id: item.path, text: item.name, depth: 3 })),
      ...(topic.elsewhere.length > 0 ? [{ id: 'elsewhere', text: 'Elsewhere', depth: 2 }] : []),
    ],
    [topic, items],
  );

  return (
    <>
      <DocsHeader current="guides" title={topic.title} />
      <main className="doc-section">
        <div className="guides">
          <nav className="guides-sidebar" aria-label="Guides">
            <ApiSearchBox />
            <ol>
              {api.topics.map((t) => (
                <li key={t.id}>
                  <a href={guideHref(t.id)} aria-current={t.id === topic.id ? 'page' : undefined}>
                    {t.title}
                    {t.feature && <FeatureChips features={[t.feature]} />}
                  </a>
                </li>
              ))}
            </ol>
          </nav>
          <div className="guides-content">
            <TopicGuide api={api} topic={topic} filter={SHOW_ALL} />
            {items.length > 0 && (
              <>
                <h2 id="reference" className="api-reference-heading">
                  Reference
                </h2>
                <ItemList
                  api={api}
                  rows={items.map((item) => ({ item }))}
                  open={reference.open}
                  onToggle={reference.toggle}
                  filter={SHOW_ALL}
                  hideFeature={topic.feature}
                />
              </>
            )}
            <Elsewhere api={api} topic={topic} filter={SHOW_ALL} />
          </div>
          <DocMenu headings={headings} />
        </div>
      </main>
    </>
  );
}
