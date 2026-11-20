import changelog from '../../../CHANGELOG.md';
import { DocMenu } from '../DocMenu';
import { mountPage, PageHeader } from '../Page';

// The menu lists the releases, not the sections inside each one.
const releases = changelog.headings.filter((heading) => heading.depth === 2);

mountPage(
  'changelog',
  <>
    <PageHeader title="Changelog" />
    <main className="doc-section">
      <div className="doc-with-menu">
        <DocMenu headings={releases} />
        <div className="prose" dangerouslySetInnerHTML={{ __html: changelog.html }} />
      </div>
    </main>
  </>,
);
