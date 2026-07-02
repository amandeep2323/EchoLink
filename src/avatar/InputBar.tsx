/** Input bar + playback controls for AvatarLink (bottom layer). */

import { useState } from 'react';

interface InputBarProps {
  onSubmit: (text: string) => void;
  onReplay: () => void;
  onStop: () => void;
  disabled?: boolean;
  status?: string;
}

const noDrag = { WebkitAppRegion: 'no-drag' } as React.CSSProperties;

export function InputBar({ onSubmit, onReplay, onStop, disabled, status }: InputBarProps) {
  const [text, setText] = useState('');

  const submit = () => {
    const trimmed = text.trim();
    if (!trimmed) return; // empty input is a no-op
    onSubmit(trimmed);
  };

  return (
    <div className="flex flex-col gap-1.5 px-3 pb-3" style={noDrag}>
      {status && <div className="text-[10px] text-slate-400 text-center">{status}</div>}
      <div className="flex items-center gap-2 rounded-xl bg-slate-900/60 backdrop-blur-md border border-slate-700/50 px-2 py-1.5">
        <input
          value={text}
          onChange={(e) => setText(e.target.value)}
          onKeyDown={(e) => { if (e.key === 'Enter') submit(); }}
          placeholder="Type a phrase to sign…"
          className="flex-1 bg-transparent text-sm text-slate-100 placeholder:text-slate-500 outline-none"
          style={noDrag}
        />
        <button
          onClick={submit}
          disabled={disabled}
          className="rounded-lg bg-violet-600 hover:bg-violet-500 disabled:opacity-40 px-3 py-1 text-xs font-medium text-white"
        >
          Sign
        </button>
        <button
          onClick={onReplay}
          disabled={disabled}
          title="Replay last phrase"
          className="rounded-lg bg-slate-700/60 hover:bg-slate-600 disabled:opacity-40 px-2 py-1 text-xs text-slate-200"
        >
          ↻
        </button>
        <button
          onClick={onStop}
          title="Stop"
          className="rounded-lg bg-slate-700/60 hover:bg-slate-600 px-2 py-1 text-xs text-slate-200"
        >
          ■
        </button>
      </div>
    </div>
  );
}
