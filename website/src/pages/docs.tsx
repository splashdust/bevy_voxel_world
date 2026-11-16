// PROTOTYPE: three layouts of the docs section, switched by `?variant=`.
//   A  Topic pages       ?variant=A&topic=light
//   B  Search first      ?variant=B
//   C  Explorer columns  ?variant=C&topic=light

import api from 'virtual:api';
import '../docs/docs.css';
import { mountPage } from '../Page';
import { PrototypeSwitcher } from '../docs/PrototypeSwitcher';
import { Explorer } from '../docs/VariantExplorer';
import { SearchFirst } from '../docs/VariantSearch';
import { TopicPages } from '../docs/VariantTopics';

const VARIANTS = [
  { key: 'A', name: 'Topic pages', Page: TopicPages },
  { key: 'B', name: 'Search first', Page: SearchFirst },
  { key: 'C', name: 'Explorer columns', Page: Explorer },
];

const requested = new URLSearchParams(location.search).get('variant');
const variant = VARIANTS.find((v) => v.key === requested) ?? VARIANTS[0];

mountPage(
  'docs',
  <>
    <variant.Page api={api} />
    {import.meta.env.DEV && <PrototypeSwitcher variants={VARIANTS} current={variant.key} />}
  </>,
);
