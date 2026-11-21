import { useCallback, useEffect, useState, type RefObject } from 'react';

/** Marks the history entry that full screen adds, as this page load's own. */
const ENTRY = { demoFullScreen: Math.random() };

function onOwnEntry(): boolean {
  return history.state?.demoFullScreen === ENTRY.demoFullScreen;
}

/**
 * Whether the demo fills the screen, and a function that changes it.
 *
 * Full screen makes the frame fill the window through CSS, which works in
 * every browser. An iPhone can't put an element in full screen, so on an
 * iPhone that is all full screen does. Where the browser can, the frame also
 * goes full screen for real, which hides the bars of the browser.
 *
 * Full screen adds an entry to the history, so that going back leaves full
 * screen rather than the page. On an iPhone a swipe from the left edge goes
 * back, and the walking stick sits at that edge.
 */
export function useFullScreen(frame: RefObject<HTMLElement | null>): [boolean, (on: boolean) => void] {
  const [fullScreen, setFullScreen] = useState(false);

  useEffect(() => {
    const leave = () => {
      setFullScreen(false);
      if (document.fullscreenElement) {
        void document.exitFullscreen();
      }
    };
    // The browser leaves its own full screen on Esc, or on the back button of a
    // phone, and the frame follows.
    const follow = () => {
      if (!document.fullscreenElement && onOwnEntry()) {
        history.back();
      }
    };
    addEventListener('popstate', leave);
    document.addEventListener('fullscreenchange', follow);
    return () => {
      removeEventListener('popstate', leave);
      document.removeEventListener('fullscreenchange', follow);
    };
  }, []);

  const change = useCallback(
    (on: boolean) => {
      if (on) {
        history.pushState(ENTRY, '');
        setFullScreen(true);
        if (document.fullscreenEnabled) {
          frame.current?.requestFullscreen({ navigationUI: 'hide' }).catch(() => {
            // The frame still fills the window.
          });
        }
      } else if (onOwnEntry()) {
        // Going back leaves full screen, through the popstate listener.
        history.back();
      } else {
        setFullScreen(false);
      }
    },
    [frame],
  );

  return [fullScreen, change];
}
