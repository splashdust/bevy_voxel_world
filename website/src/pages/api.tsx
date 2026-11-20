// Docs › API, the whole API on one searchable page: ?q=stamp.

import api from 'virtual:api';
import '../docs/docs.css';
import { ApiSearch } from '../docs/ApiSearch';
import { mountPage } from '../Page';

mountPage('api', <ApiSearch api={api} />);
