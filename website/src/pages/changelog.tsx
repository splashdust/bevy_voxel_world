import changelog from '../../../CHANGELOG.md';
import { mountPage, PageHeader } from '../Page';

mountPage(
  'changelog',
  <div className="narrow-page">
    <PageHeader title="Changelog" />
    <main className="doc-section">
      <div className="prose" dangerouslySetInnerHTML={{ __html: changelog.html }} />
    </main>
  </div>,
);
