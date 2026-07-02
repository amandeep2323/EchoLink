/**
 * ModeMenu — turns the "EchoLink" header branding into a dropdown that switches
 * between EchoLink and AvatarLink (presentation-only; delegates to avatarAPI).
 *
 * The dropdown renders via a React portal into document.body so it is never
 * clipped by any ancestor's overflow:hidden or stacking-context rules.
 */

import { useEffect, useRef, useState, useLayoutEffect } from 'react';
import { createPortal } from 'react-dom';
import { cn } from '@/utils/cn';
import { getAvatarAPI, type AppMode } from './modeRouter';

export function ModeMenu() {
  const [open, setOpen] = useState(false);
  const [mode, setMode] = useState<AppMode>('echolink');
  const triggerRef = useRef<HTMLDivElement | null>(null);
  const [dropdownPos, setDropdownPos] = useState({ top: 0, left: 0 });

  useEffect(() => {
    const api = getAvatarAPI();
    api?.getMode().then(setMode).catch(() => {});
    api?.onModeChanged(setMode);
  }, []);

  // Close on outside click
  useEffect(() => {
    const onDoc = (e: MouseEvent) => {
      if (triggerRef.current && !triggerRef.current.contains(e.target as Node)) {
        // Also allow clicks inside the portal dropdown
        const portal = document.getElementById('mode-menu-portal');
        if (!portal?.contains(e.target as Node)) setOpen(false);
      }
    };
    document.addEventListener('mousedown', onDoc);
    return () => document.removeEventListener('mousedown', onDoc);
  }, []);

  // Recalculate drop position whenever we open
  useLayoutEffect(() => {
    if (open && triggerRef.current) {
      const r = triggerRef.current.getBoundingClientRect();
      setDropdownPos({ top: r.bottom + 4, left: r.left });
    }
  }, [open]);

  const select = (m: AppMode) => {
    setOpen(false);
    const api = getAvatarAPI();
    if (api) {
      // Electron context — use IPC
      void api.switchMode(m);
    } else {
      // Dev browser fallback — navigate via hash so the scene mounts
      if (m === 'avatar') {
        window.location.hash = '/avatar';
        window.location.reload();
      } else {
        window.location.hash = '';
        window.location.reload();
      }
    }
  };

  const dropdown = open
    ? createPortal(
        <div
          id="mode-menu-portal"
          style={{ position: 'fixed', top: dropdownPos.top, left: dropdownPos.left, zIndex: 99999 }}
          className="w-44 rounded-lg border border-slate-700/60 bg-slate-900/98 backdrop-blur-md shadow-2xl py-1"
        >
          <MenuItem label="EchoLink" hint="ASL recognition" active={mode === 'echolink'} onClick={() => select('echolink')} />
          <MenuItem label="AvatarLink" hint="Text → 3D sign" active={mode === 'avatar'} onClick={() => select('avatar')} />
        </div>,
        document.body
      )
    : null;

  return (
    <>
      <div className="relative" ref={triggerRef}>
        <button
          onClick={() => setOpen((v) => !v)}
          className="flex items-center gap-1 group"
          title="Switch mode"
        >
          <h1 className="text-lg font-bold bg-gradient-to-r from-violet-400 to-indigo-400 bg-clip-text text-transparent">
            EchoLink
          </h1>
          <svg
            className={cn('w-3.5 h-3.5 text-slate-500 transition-transform', open && 'rotate-180')}
            fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}
          >
            <path strokeLinecap="round" strokeLinejoin="round" d="M19 9l-7 7-7-7" />
          </svg>
        </button>
      </div>
      {dropdown}
    </>
  );
}

function MenuItem({ label, hint, active, onClick }: { label: string; hint: string; active: boolean; onClick: () => void }) {
  return (
    <button
      onClick={onClick}
      className={cn(
        'w-full text-left px-3 py-2 hover:bg-slate-800/70 transition-colors flex items-center justify-between',
        active && 'bg-slate-800/40'
      )}
    >
      <span className="flex flex-col">
        <span className="text-sm text-slate-100">{label}</span>
        <span className="text-[10px] text-slate-500">{hint}</span>
      </span>
      {active && <span className="w-1.5 h-1.5 rounded-full bg-emerald-400" />}
    </button>
  );
}
