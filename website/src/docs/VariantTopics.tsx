// PROTOTYPE, variant A: one page per topic, like a book. Topics down the left,
// the topic's guide from website/guide/, then each item of its reference in
// full, and a menu on the right that follows the reader.

import { useMemo } from 'react';
import { DocMenu } from '../DocMenu';
import { PageHeader } from '../Page';
import {
  type ApiView,
  Elsewhere,
  FeatureChips,
  FeaturesOverview,
  FeatureToggles,
  ItemDoc,
  TopicGuide,
  topicOfKey,
  useApiLinks,
  useFeatureFilter,
} from './shared';

export function TopicPages({ api }: { api: ApiView }) {
  const params = new URLSearchParams(location.search);
  const topicId = params.get('topic') ?? 'overview';
  const topic = api.topics.find((t) => t.id === topicId);
  const filter = useFeatureFilter(api.apiFeatures);

  const topicHref = (id: string) => {
    const next = new URLSearchParams(params);
    next.set('topic', id);
    return `?${next}`;
  };

  // A link into another topic loads that topic's page.
  useApiLinks((key) => {
    const target = topicOfKey(api, key);
    if (!target) {
      return;
    }
    if (target === topicId) {
      history.replaceState(null, '', `#${key}`);
      document.getElementById(key)?.scrollIntoView();
    } else {
      location.assign(`${topicHref(target)}#${key}`);
    }
  });

  const items = (topic?.items ?? []).map((path) => api.items[path]).filter((item) => filter.shows(item.features));
  const headings = useMemo(
    () =>
      topic
        ? [
            ...topic.headings,
            { id: 'reference', text: 'Reference', depth: 2 },
            ...items.map((item) => ({ id: item.path, text: item.name, depth: 3 })),
            ...(topic.elsewhere.length > 0 ? [{ id: 'elsewhere', text: 'Elsewhere', depth: 2 }] : []),
          ]
        : [
            { id: 'features', text: 'Cargo features', depth: 2 },
            { id: 'topics', text: 'Topics', depth: 2 },
          ],
    // The items change only with the topic and the filter.
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [topicId, filter.enabled],
  );

  return (
    <>
      <PageHeader title={topic?.title ?? 'Docs'} />
      <main className="doc-section">
        <div className="docs-a">
          <nav className="docs-a-sidebar" aria-label="Topics">
            <FeatureToggles filter={filter} />
            <ol>
              <li>
                <a href={topicHref('overview')} aria-current={topic ? undefined : 'page'}>
                  Overview
                </a>
              </li>
              {api.topics.map((t) => (
                <li key={t.id}>
                  <a href={topicHref(t.id)} aria-current={t.id === topicId ? 'page' : undefined}>
                    {t.title}
                    {t.feature && <FeatureChips features={[t.feature]} />}
                  </a>
                </li>
              ))}
            </ol>
          </nav>
          <div className="docs-a-content">
            {topic ? (
              <>
                <TopicGuide api={api} topic={topic} filter={filter} />
                <h2 id="reference" className="api-reference-heading">
                  Reference
                </h2>
                {items.map((item) => (
                  <ItemDoc key={item.path} item={item} filter={filter} />
                ))}
                <Elsewhere api={api} topic={topic} filter={filter} />
              </>
            ) : (
              <FeaturesOverview api={api} topicHref={topicHref} />
            )}
          </div>
          {headings.length > 0 && <DocMenu headings={headings} />}
        </div>
      </main>
    </>
  );
}
