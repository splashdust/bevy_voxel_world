// What the docs pages share: signatures, feature chips, the API search box and
// one item's full docs.

import { Fragment, useEffect } from 'react';
import type { ApiView, ItemView, MemberView, Token, TopicView } from '../../scripts/api.ts';
import { pageUrl, REPOSITORY } from '../Page';

export type { ApiView, ItemView, MemberView, Token, TopicView };

export const KIND_LABEL: Record<string, string> = {
  struct: 'struct',
  enum: 'enum',
  union: 'union',
  trait: 'trait',
  fn: 'function',
  type: 'type alias',
  const: 'constant',
  static: 'static',
  macro: 'macro',
};

/** A link to a public item or member, which the page's own navigation follows. */
export function ApiLink({ to, children }: { to: string; children: React.ReactNode }) {
  return (
    <a href={`#${to}`} data-api={to}>
      {children}
    </a>
  );
}

export function Signature({ tokens, small }: { tokens: Token[]; small?: boolean }) {
  return (
    <pre className={small ? 'signature small' : 'signature'}>
      <code>
        {tokens.map((token, i) => {
          if (token.h?.startsWith('api:')) {
            return (
              <ApiLink key={i} to={token.h.slice(4)}>
                {token.t}
              </ApiLink>
            );
          }
          if (token.h) {
            return (
              <a key={i} href={token.h}>
                {token.t}
              </a>
            );
          }
          if (token.k) {
            return (
              <span key={i} className={`sig-${token.k}`}>
                {token.t}
              </span>
            );
          }
          return <Fragment key={i}>{token.t}</Fragment>;
        })}
      </code>
    </pre>
  );
}

export function FeatureChips({ features }: { features: string[] }) {
  if (features.length === 0) {
    return null;
  }
  return (
    <span className="chips">
      {features.map((feature) => (
        <span key={feature} className={`chip chip-${feature}`} title={`Needs the ${feature} feature`}>
          {feature}
        </span>
      ))}
    </span>
  );
}

/** The Guides page of a topic, at a key's anchor if one is given. */
export const guideHref = (topic: string, key?: string) =>
  `${pageUrl('guides')}?topic=${topic}${key ? `#${key}` : ''}`;

/** A search field that opens the API page with its results. */
export function ApiSearchBox({ placeholder = 'Search the API' }: { placeholder?: string }) {
  return (
    <form className="api-search-box" action={pageUrl('api')} method="get" role="search">
      <input type="search" name="q" placeholder={placeholder} aria-label="Search the API" />
    </form>
  );
}

/** Says whether to show something that needs these features. */
export type Shows = { shows: (features: string[]) => boolean };

/** Shows everything, whatever features it needs. */
export const SHOW_ALL: Shows = { shows: () => true };

/** Calls `follow` with the key of each docs link the reader clicks. */
export function useApiLinks(follow: (key: string) => void) {
  useEffect(() => {
    const onClick = (event: MouseEvent) => {
      if (event.button !== 0 || event.metaKey || event.ctrlKey || event.shiftKey) {
        return;
      }
      const link = (event.target as Element).closest?.('a[data-api]');
      if (link) {
        event.preventDefault();
        follow(link.getAttribute('data-api')!);
      }
    };
    document.addEventListener('click', onClick);
    return () => document.removeEventListener('click', onClick);
  });
}

/** The item a key names: the item itself, or the item a member's key starts with. */
export function itemOfKey(api: ApiView, key: string): ItemView | undefined {
  const parts = key.split('::');
  for (let n = parts.length; n > 0; n--) {
    const item = api.items[parts.slice(0, n).join('::')];
    if (item) {
      return item;
    }
  }
  return undefined;
}

/** The topics a key belongs to: its item's, or that of the module it names. */
export function topicsOfKey(api: ApiView, key: string): string[] {
  const topics = itemOfKey(api, key)?.topics ?? [];
  if (topics.length > 0) {
    return topics;
  }
  const module = api.topics.find((t) => t.moduleDocs?.path === key);
  return module ? [module.id] : [];
}

/** Scrolls to the element with this id once React has drawn it. */
export function scrollToId(id: string) {
  requestAnimationFrame(() => document.getElementById(id)?.scrollIntoView({ block: 'start' }));
}

/**
 * One member's docs. `owner` names the item it belongs to, for a member shown
 * away from its item; `anchored: false` leaves off the id that links land on.
 */
export function MemberDoc({ member, owner, anchored = true }: { member: MemberView; owner?: ItemView; anchored?: boolean }) {
  return (
    <div className="api-member" id={anchored ? member.anchor : undefined}>
      <div className="api-member-head">
        <Signature tokens={member.signature} small />
        <FeatureChips features={member.features} />
      </div>
      {owner && (
        <p className="api-owner">
          on <ApiLink to={owner.path}>{owner.name}</ApiLink>
        </p>
      )}
      {member.html && <div className="prose api-docs" dangerouslySetInnerHTML={{ __html: member.html }} />}
    </div>
  );
}

/** One item's full docs: signature, docs, traits and members. */
export function ItemDoc({ item, filter, headless }: { item: ItemView; filter: Shows; headless?: boolean }) {
  const members = item.members.filter((m) => filter.shows([...item.features, ...m.features]));
  const sections: [string, MemberView[]][] = [];
  for (const member of members) {
    const last = sections.at(-1);
    if (last && last[0] === member.section) {
      last[1].push(member);
    } else {
      sections.push([member.section, [member]]);
    }
  }
  const traits = item.traits.filter((t) => filter.shows([...item.features, ...t.features]));

  return (
    <article className="api-item" id={headless ? undefined : item.path}>
      <header className="api-item-header">
        {!headless && (
          <>
            <span className="api-kind">{KIND_LABEL[item.kind] ?? item.kind}</span>
            <h2 className="api-name">
              <ApiLink to={item.path}>{item.name}</ApiLink>
            </h2>
            <FeatureChips features={item.features} />
          </>
        )}
        {item.source && (
          <a className="api-source" href={`${REPOSITORY}/blob/main/${item.source}`}>
            Source
          </a>
        )}
      </header>
      <p className="api-import">
        <code>use bevy_voxel_world::{item.path};</code>
      </p>
      <Signature tokens={item.signature} />
      {item.deprecated !== undefined && <p className="api-deprecated">Deprecated. {item.deprecated}</p>}
      {item.html && <div className="prose api-docs" dangerouslySetInnerHTML={{ __html: item.html }} />}
      {traits.length > 0 && (
        <p className="api-traits">
          <span className="api-traits-label">Implements</span>
          {traits.map((t) => (
            <span key={t.name} className="api-trait">
              {t.href ? <a href={t.href}>{t.name}</a> : t.name}
              <FeatureChips features={t.features} />
            </span>
          ))}
        </p>
      )}
      {sections.map(([section, sectionMembers]) => (
        <section key={section} className="api-members">
          <h3 className="api-section">{section}</h3>
          {sectionMembers.map((member) => (
            <MemberDoc key={member.anchor} member={member} />
          ))}
        </section>
      ))}
    </article>
  );
}

/**
 * What an `api` block in a guide shows: an item's signature and summary, with
 * a link to its full reference, or a member's full docs.
 */
export function ApiEmbed({ api, keys, filter }: { api: ApiView; keys: string[]; filter: Shows }) {
  return (
    <div className="api-embed">
      {keys.map((key) => {
        const item = itemOfKey(api, key)!;
        if (!filter.shows(item.features)) {
          return null;
        }
        if (key === item.path) {
          return (
            <div key={key} className="api-embed-item">
              <p className="api-embed-head">
                <span className="api-kind">{KIND_LABEL[item.kind] ?? item.kind}</span>
                <ApiLink to={item.path}>{item.name}</ApiLink>
                <FeatureChips features={item.features} />
              </p>
              <Signature tokens={item.signature} small />
              <p className="api-embed-summary">{item.summary}</p>
            </div>
          );
        }
        const member = item.members.find((m) => m.anchor === key)!;
        return filter.shows([...item.features, ...member.features]) ? (
          <MemberDoc key={key} member={member} owner={item} anchored={false} />
        ) : null;
      })}
    </div>
  );
}

/** A topic's guide, with its `api` blocks, and the module docs that belong with it. */
export function TopicGuide({ api, topic, filter }: { api: ApiView; topic: TopicView; filter: Shows }) {
  return (
    <>
      {topic.blocks.map((block, i) =>
        'html' in block ? (
          <div key={i} className="prose" dangerouslySetInnerHTML={{ __html: block.html }} />
        ) : (
          <ApiEmbed key={i} api={api} keys={block.embed} filter={filter} />
        ),
      )}
      <p className="guide-edit">
        <a href={`${REPOSITORY}/blob/main/${topic.file}`}>Edit this page</a>
      </p>
      {topic.moduleDocs && (
        <details className="module-docs" id={topic.moduleDocs.path}>
          <summary>
            More in the <code>{topic.moduleDocs.path}</code> module's docs
          </summary>
          <div className="prose" dangerouslySetInnerHTML={{ __html: topic.moduleDocs.html }} />
        </details>
      )}
    </>
  );
}

/** What a topic's feature adds to items that belong to other topics. */
export function Elsewhere({ api, topic, filter }: { api: ApiView; topic: TopicView; filter: Shows }) {
  const entries = topic.elsewhere.filter((e) => filter.shows(api.items[e.item].features));
  if (!topic.feature || entries.length === 0) {
    return null;
  }
  return (
    <section className="api-elsewhere" id="elsewhere">
      <h2>
        What <code>{topic.feature}</code> adds to other types
      </h2>
      {entries.map((entry) => {
        const item = api.items[entry.item];
        const members = item.members.filter((m) => entry.members.includes(m.anchor));
        return (
          <div key={entry.item} className="api-elsewhere-item">
            {entry.traits.length > 0 && (
              <p className="api-traits">
                <ApiLink to={item.path}>{item.name}</ApiLink>
                <span className="api-traits-label">implements</span>
                {entry.traits.map((name) => (
                  <span key={name} className="api-trait">
                    {name}
                  </span>
                ))}
              </p>
            )}
            {members.map((member) => (
              <MemberDoc key={member.anchor} member={{ ...member, features: [] }} owner={item} />
            ))}
          </div>
        );
      })}
    </section>
  );
}

/** The crate's features and what each adds. */
export function FeaturesOverview({ api, topicHref }: { api: ApiView; topicHref: (id: string) => string }) {
  return (
    <div className="prose features-overview">
      <h2 id="features">Cargo features</h2>
      <p>
        All the features that can be enabled for <strong>bevy_voxel_world</strong>
      </p>
      <table>
        <tbody>
          {api.features.map((feature) => (
            <tr key={feature.name} id={`feature-${feature.name}`}>
              <td>
                <FeatureChips features={[feature.name]} />
              </td>
              <td>
                {feature.doc}
                {feature.enables.length > 0 && ` Turns on ${feature.enables.join(', ')}.`}
              </td>
              <td>
                {feature.adds === 0
                  ? 'No API'
                  : feature.topic
                    ? <a href={topicHref(feature.topic)}>{feature.adds} items</a>
                    : `${feature.adds} items`}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
