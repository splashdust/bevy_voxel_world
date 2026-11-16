// PROTOTYPE: a floating bar for flipping between the docs page's variants.
// Only `npm run dev` shows it.

import { useEffect } from 'react';

export type Variant = { key: string; name: string };

export function PrototypeSwitcher({ variants, current }: { variants: Variant[]; current: string }) {
  const index = variants.findIndex((v) => v.key === current);

  const go = (step: number) => {
    const next = variants[(index + step + variants.length) % variants.length];
    const params = new URLSearchParams(location.search);
    params.set('variant', next.key);
    location.assign(`?${params}${location.hash}`);
  };

  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if ((event.target as Element).closest?.('input, textarea, [contenteditable]')) {
        return;
      }
      if (event.key === 'ArrowLeft') {
        go(-1);
      } else if (event.key === 'ArrowRight') {
        go(1);
      }
    };
    addEventListener('keydown', onKey);
    return () => removeEventListener('keydown', onKey);
  });

  return (
    <div className="prototype-switcher" role="group" aria-label="Prototype variants">
      <button type="button" onClick={() => go(-1)} aria-label="Previous variant">
        ←
      </button>
      <span>
        {current} · {variants[index]?.name}
      </span>
      <button type="button" onClick={() => go(1)} aria-label="Next variant">
        →
      </button>
    </div>
  );
}
