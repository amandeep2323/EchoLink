/**
 * SpoutBridge — interface for the later-phase GPU texture broadcast to OBS via
 * Spout2 (Windows/DirectX). Phase 1 ships a no-op default so the in-app widget
 * works and can be captured by OBS *window capture*; the native node addon is
 * wired in a later phase (see backend/avatar3d/native/README.md).
 */

export interface SpoutBridge {
  /** True only when the native Spout2 addon + runtime are available. */
  isAvailable(): boolean;
  /** Begin publishing under a named Spout sender (e.g. "AvatarLink"). */
  start(senderName: string): boolean;
  /** Push a shared GPU texture handle (from Electron offscreen paint). */
  sendSharedTexture(handle: ArrayBufferLike, width: number, height: number): void;
  /** Stop publishing and release the shared texture. */
  stop(): void;
}

/** Default no-op bridge used until the native addon lands (graceful degradation). */
export class NoopSpoutBridge implements SpoutBridge {
  isAvailable(): boolean {
    return false;
  }
  start(): boolean {
    console.info('[Spout] Not available — use OBS window capture for now.');
    return false;
  }
  sendSharedTexture(): void {
    /* no-op */
  }
  stop(): void {
    /* no-op */
  }
}

/**
 * Resolve the active Spout bridge. In a later phase this will probe for the
 * native addon (e.g. exposed on `window.spoutAPI` via preload) and return it
 * when present; today it always returns the no-op implementation.
 */
export function getSpoutBridge(): SpoutBridge {
  const native = (window as unknown as { spoutAPI?: SpoutBridge }).spoutAPI;
  if (native && native.isAvailable()) return native;
  return new NoopSpoutBridge();
}
