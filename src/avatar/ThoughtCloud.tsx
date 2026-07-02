/** Thought Cloud overlay — shows the tokens being processed as bubbles. */

import { cn } from '@/utils/cn';

export interface ThoughtBubble {
  label: string;
}

interface ThoughtCloudProps {
  bubbles: ThoughtBubble[];
  activeIndex: number;
  visible: boolean;
}

export function ThoughtCloud({ bubbles, activeIndex, visible }: ThoughtCloudProps) {
  if (!visible || bubbles.length === 0) return null;
  return (
    <div
      className="pointer-events-none flex flex-wrap items-center justify-center gap-1.5 px-3 transition-opacity duration-300"
      style={{ WebkitAppRegion: 'no-drag' } as React.CSSProperties}
    >
      {bubbles.map((b, i) => (
        <span
          key={`${b.label}-${i}`}
          className={cn(
            'rounded-full px-2.5 py-1 text-xs font-medium backdrop-blur-md transition-all',
            i === activeIndex
              ? 'bg-violet-500/80 text-white scale-105'
              : i < activeIndex
                ? 'bg-slate-700/40 text-slate-400'
                : 'bg-slate-800/50 text-slate-300'
          )}
        >
          {b.label}
        </span>
      ))}
    </div>
  );
}
