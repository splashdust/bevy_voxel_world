// PROTOTYPE, variant C: an explorer in three columns, like a file browser.
// Topics, then the topic's items with their summaries, then one item's full
// docs. Each column scrolls on its own; on a phone, one column at a time.

import { useEffect, useRef, useState } from 'react';
import {
  type ApiView,
  Elsewhere,
  FeatureChips,
  FeatureToggles,
  ItemDoc,
  itemOfKey,
  KIND_LABEL,
  TopicIntro,
  topicOfKey,
  useApiLinks,
  useFeatureFilter,
} from './shared';

type Pane = 'topics' | 'items' | 'detail';

export function Explorer({ api }: { api: ApiView }) {
  const filter = useFeatureFilter(api.apiFeatures);
  const [start] = useState(() => {
    const key = decodeURIComponent(location.hash.slice(1));
    const item = key ? itemOfKey(api, key) : undefined;
    const topic = item?.topic ?? new URLSearchParams(location.search).get('topic') ?? api.topics[0].id;
    return { key, item: item?.path, topic };
  });
  const [topicId, setTopicId] = useState(start.topic);
  const [itemPath, setItemPath] = useState(start.item);
  const [pane, setPane] = useState<Pane>(start.item ? 'detail' : 'items');
  const detail = useRef<HTMLDivElement>(null);

  const remember = (topic: string, key?: string) => {
    const params = new URLSearchParams(location.search);
    params.set('topic', topic);
    history.replaceState(null, '', `?${params}${key ? `#${key}` : ''}`);
  };

  const show = (key: string) => {
    const item = itemOfKey(api, key);
    const topic = item?.topic ?? topicOfKey(api, key);
    if (!topic) {
      return;
    }
    setTopicId(topic);
    setItemPath(item?.path);
    setPane('detail');
    remember(topic, key);
    requestAnimationFrame(() => {
      if (item && key !== item.path) {
        document.getElementById(key)?.scrollIntoView({ block: 'start' });
      } else {
        detail.current?.scrollTo(0, 0);
      }
    });
  };

  useApiLinks(show);
  useEffect(() => {
    if (start.key && start.key !== start.item) {
      requestAnimationFrame(() => document.getElementById(start.key)?.scrollIntoView({ block: 'start' }));
    }
  }, [start]);

  const topic = api.topics.find((t) => t.id === topicId) ?? api.topics[0];
  const items = topic.items.map((path) => api.items[path]).filter((item) => filter.shows(item.features));
  const item = itemPath ? api.items[itemPath] : undefined;

  return (
    <main className={`docs-c pane-${pane}`}>
      <nav className="docs-c-col docs-c-topics" aria-label="Topics">
        <FeatureToggles filter={filter} />
        <ol>
          {api.topics.map((t) => (
            <li key={t.id}>
              <button
                type="button"
                aria-current={t.id === topic.id}
                onClick={() => {
                  setTopicId(t.id);
                  setItemPath(undefined);
                  setPane('items');
                  remember(t.id);
                  detail.current?.scrollTo(0, 0);
                }}
              >
                <span className="docs-c-item-name">{t.title}</span>
                {t.feature && <FeatureChips features={[t.feature]} />}
              </button>
            </li>
          ))}
        </ol>
      </nav>
      <nav className="docs-c-col docs-c-items" aria-label={topic.title}>
        <button type="button" className="docs-c-back" onClick={() => setPane('topics')}>
          ← Topics
        </button>
        <h2>{topic.title}</h2>
        <ol>
          <li>
            <button
              type="button"
              aria-current={!item}
              onClick={() => {
                setItemPath(undefined);
                setPane('detail');
                remember(topic.id);
                detail.current?.scrollTo(0, 0);
              }}
            >
              <span className="docs-c-item-name">About {topic.title.toLowerCase()}</span>
            </button>
          </li>
          {items.map((i) => (
            <li key={i.path}>
              <button type="button" aria-current={i.path === itemPath} onClick={() => show(i.path)}>
                <span className="api-kind">{KIND_LABEL[i.kind] ?? i.kind}</span>
                <span className="docs-c-item-name">{i.name}</span>
                <FeatureChips features={i.features.filter((f) => f !== topic.feature)} />
                <span className="docs-c-summary">{i.summary}</span>
              </button>
            </li>
          ))}
        </ol>
      </nav>
      <div className="docs-c-col docs-c-detail" ref={detail}>
        <button type="button" className="docs-c-back" onClick={() => setPane('items')}>
          ← {topic.title}
        </button>
        {item ? (
          <ItemDoc item={item} filter={filter} />
        ) : (
          <>
            <h1 className="docs-c-title">{topic.title}</h1>
            <TopicIntro topic={topic} />
            <Elsewhere api={api} topic={topic} filter={filter} />
          </>
        )}
      </div>
    </main>
  );
}
