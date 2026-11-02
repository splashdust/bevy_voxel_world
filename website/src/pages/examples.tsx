import examples from 'virtual:examples';
import { CopyCommand } from '../CopyCommand';
import { mountPage, PageHeader, REPOSITORY } from '../Page';

mountPage(
  'examples',
  <div className="narrow-page">
    <PageHeader title="Examples" />
    <main className="doc-section">
      <div className="example-list">
        <p>
          To run these, clone the <a href={REPOSITORY}>repository</a> and use each command from its
          root.
        </p>
        {examples.map((example) => (
          <article key={example.name} id={example.name} className="example">
            <h2>{example.name}</h2>
            {example.screenshot && (
              <img
                className="example-screenshot"
                src={example.screenshot.url}
                width={example.screenshot.width}
                height={example.screenshot.height}
                loading="lazy"
                alt={`The ${example.name} example running`}
              />
            )}
            <div className="prose" dangerouslySetInnerHTML={{ __html: example.description }} />
            <div className="example-actions">
              <CopyCommand command={example.command} />
              <a href={`${REPOSITORY}/blob/main/${example.path}`}>Source</a>
            </div>
          </article>
        ))}
      </div>
    </main>
  </div>,
);
