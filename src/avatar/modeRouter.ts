/**
 * Mode router for the dual-modality app.
 *
 * EchoLink (default) mounts on the main window; AvatarLink mounts when the
 * window is loaded with the `#/avatar` hash (the Electron companion window
 * loads `index.html#/avatar`). Pure presentation routing — no coupling to
 * EchoLink internals.
 */

export type AppMode = 'echolink' | 'avatar';

/** Resolve the mode from the current URL hash (and ?mode= fallback). */
export function getModeFromLocation(): AppMode {
  try {
    const hash = window.location.hash || '';
    if (hash.replace(/^#\/?/, '').toLowerCase().startsWith('avatar')) {
      return 'avatar';
    }
    const params = new URLSearchParams(window.location.search);
    if ((params.get('mode') || '').toLowerCase() === 'avatar') {
      return 'avatar';
    }
  } catch {
    // window may be unavailable in non-browser contexts
  }
  return 'echolink';
}

/** Minimal shape of the preload-exposed avatar API (see electron/preload.cjs). */
export interface AvatarAPI {
  switchMode: (mode: AppMode) => Promise<void>;
  closeAvatar: () => Promise<void>;
  getMode: () => Promise<AppMode>;
  onModeChanged: (cb: (mode: AppMode) => void) => void;
}

/** Safe accessor for the avatar API (undefined when not running in Electron). */
export function getAvatarAPI(): AvatarAPI | undefined {
  return (window as unknown as { avatarAPI?: AvatarAPI }).avatarAPI;
}
