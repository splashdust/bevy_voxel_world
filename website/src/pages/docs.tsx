// The Docs overview, which leads to the guides, the API and the
// examples, and lists the crate's features.

import api from 'virtual:api';
import examples from 'virtual:examples';
import '../docs/docs.css';
import { ApiSearchBox, FeatureChips, FeaturesOverview, guideHref } from '../docs/shared';
import { DocsHeader, mountPage, pageUrl } from '../Page';

const itemCount = Object.keys(api.items).length;

mountPage(
  'docs',
  <>
    <DocsHeader current="docs" title="Docs" />
    <main className="doc-section">
      <div className="docs-home">
        <div className="docs-cards">
          <section className="docs-card">
            <h2>
              <a href={pageUrl('guides')}>Guides</a>
            </h2>
            <p>A tutorial from basic to advanced usage of BVW. Work in progress!</p>
            <ol className="docs-card-topics">
              {api.topics.map((topic) => (
                <li key={topic.id}>
                  <a href={guideHref(topic.id)}>{topic.title}</a>
                  {topic.feature && <FeatureChips features={[topic.feature]} />}
                </li>
              ))}
            </ol>
          </section>
          <section className="docs-card">
            <h2>
              <a href={pageUrl('api')}>API</a>
            </h2>
            <p>
              All {itemCount} public types, functions and constants, with their methods, on one
              searchable page.
            </p>
            <ApiSearchBox placeholder="Search types and methods" />
          </section>
          <section className="docs-card">
            <h2>
              <a href={pageUrl('examples')}>Examples</a>
            </h2>
            <p>{examples.length} runnable examples, each with what it shows and the command to run it.</p>
          </section>
        </div>
        <FeaturesOverview api={api} topicHref={(id) => guideHref(id)} />
      </div>
    </main>
  </>,
);
