// PROTOTYPE: Docs › Guides, one topic at a time: ?topic=light.

import api from 'virtual:api';
import '../docs/docs.css';
import { Guides } from '../docs/Guides';
import { mountPage } from '../Page';

mountPage('guides', <Guides api={api} />);
