// PROTOTYPE: Docs › Guides, one page per topic, like a book. Topics down the
// left, the topic's guide from website/guide/, then each item of its reference
// in full, and a menu on the right that follows the reader.

import { useEffect, useMemo } from 'react';
import { DocMenu } from '../DocMenu';
import { DocsHeader } from '../Page';
import {
  ApiSearchBox,
  type ApiView,
  Elsewhere,
  FeatureChips,
  FeatureToggles,
  guideHref,
  ItemDoc,
  TopicGuide,
  topicOfKey,
  useApiLinks,
  useFeatureFilter,
} from './shared';

export function Guides({ api }: { api: ApiView }) {
  const requested = new URLSearchParams(location.search).get('topic');
  const topic = api.topics.find((t) => t.id === requested) ?? api.topics[0];
  const filter = useFeatureFilter(api.apiFeatures);

  useEffect(() => {
    document.title = `${topic.title} · Guides · bevy_voxel_world`;
  }, [topic]);

  // A link into another topic loads that topic's page.
  useApiLinks((key) => {
    const target = topicOfKey(api, key);
    if (!target) {
      return;
    }
    if (target === topic.id) {
      history.replaceState(null, '', `#${key}`);
      document.getElementById(key)?.scrollIntoView();
    } else {
      location.assign(guideHref(target, key));
    }
  });

  const items = topic.items.map((path) => api.items[path]).filter((item) => filter.shows(item.features));
  const headings = useMemo(
    () => [
      ...topic.headings,
      { id: 'reference', text: 'Reference', depth: 2 },
      ...items.map((item) => ({ id: item.path, text: item.name, depth: 3 })),
      ...(topic.elsewhere.length > 0 ? [{ id: 'elsewhere', text: 'Elsewhere', depth: 2 }] : []),
    ],
    // The items change only with the topic and the filter.
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [topic, filter.enabled],
  );

  return (
    <>
      <DocsHeader current="guides" title={topic.title} />
      <main className="doc-section">
        <div className="guides">
          <nav className="guides-sidebar" aria-label="Guides">
            <ApiSearchBox />
            <FeatureToggles filter={filter} />
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
            <TopicGuide api={api} topic={topic} filter={filter} />
            <h2 id="reference" className="api-reference-heading">
              Reference
            </h2>
            {items.map((item) => (
              <ItemDoc key={item.path} item={item} filter={filter} />
            ))}
            <Elsewhere api={api} topic={topic} filter={filter} />
          </div>
          <DocMenu headings={headings} />
        </div>
      </main>
    </>
  );
}
