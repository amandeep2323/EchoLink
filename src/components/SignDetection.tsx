import { useState, useEffect, useRef } from 'react';
import type { SignDetectionData } from '@/types';
import { cn } from '@/utils/cn';

interface SignDetectionProps {
  data: SignDetectionData | null;
}

function getConfidenceColor(confidence: number): string {
  if (confidence >= 0.8) return 'from-emerald-500 to-emerald-400';
  if (confidence >= 0.6) return 'from-violet-500 to-indigo-500';
  if (confidence >= 0.4) return 'from-amber-500 to-yellow-500';
  return 'from-red-500 to-orange-500';
}

function getConfidenceTextColor(confidence: number): string {
  if (confidence >= 0.8) return 'text-emerald-400';
  if (confidence >= 0.6) return 'text-violet-400';
  if (confidence >= 0.4) return 'text-amber-400';
  return 'text-red-400';
}

export function SignDetection({ data }: SignDetectionProps) {
  const [flash, setFlash] = useState(false);
  const [lastAccepted, setLastAccepted] = useState<string>('');
  const flashTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  // Flash green when a letter is accepted
  useEffect(() => {
    if (data?.letter_added) {
      setFlash(true);
      setLastAccepted(data.sign);

      if (flashTimeoutRef.current) clearTimeout(flashTimeoutRef.current);
      flashTimeoutRef.current = setTimeout(() => setFlash(false), 400);
    }

    return () => {
      if (flashTimeoutRef.current) clearTimeout(flashTimeoutRef.current);
    };
  }, [data?.letter_added, data?.sign]);

  const smoothedConf = data?.smoothed_confidence ?? data?.confidence ?? 0;

  return (
    <div
      className={cn(
        'bg-slate-900/80 rounded-xl border p-5 shrink-0 transition-all duration-200',
        flash
          ? 'border-emerald-500/60 shadow-lg shadow-emerald-500/10'
          : 'border-slate-800/60'
      )}
    >
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-xs font-semibold text-slate-500 uppercase tracking-wider">
          Sign Detection
        </h2>
        <div className="flex items-center gap-2">
          {data?.letter_added && (
            <div className="px-2 py-0.5 rounded-full text-[10px] font-bold bg-emerald-500/15 text-emerald-400 animate-scale-in">
              ✓ ACCEPTED
            </div>
          )}
          {data && !data.letter_added && (
            <div className={cn(
              'px-2 py-0.5 rounded-full text-[10px] font-medium',
              smoothedConf >= 0.6
                ? 'bg-violet-500/10 text-violet-400'
                : 'bg-amber-500/10 text-amber-400'
            )}>
              {smoothedConf >= 0.6 ? 'DETECTING' : 'LOW'}
            </div>
          )}
        </div>
      </div>

      {data ? (
        <div className="animate-fade-in">
          {/* Current Sign */}
          <div className="text-center mb-4">
            <div className={cn(
              'text-4xl font-bold mb-1 tracking-tight transition-colors duration-200',
              flash ? 'text-emerald-400' : 'text-white'
            )}>
              {data.sign}
            </div>
            <div className="flex items-center justify-center gap-3">
              <div className={cn('text-lg font-semibold tabular-nums', getConfidenceTextColor(data.confidence))}>
                {(data.confidence * 100).toFixed(1)}%
              </div>
              {data.smoothed_confidence > 0 && data.smoothed_confidence !== data.confidence && (
                <div className="text-xs text-slate-500 tabular-nums">
                  avg: {(data.smoothed_confidence * 100).toFixed(1)}%
                </div>
              )}
            </div>
          </div>

          {/* Confidence bars */}
          <div className="space-y-1.5 mb-4">
            {/* Raw confidence */}
            <div className="flex items-center gap-2">
              <span className="text-[10px] text-slate-600 w-8">Raw</span>
              <div className="flex-1 bg-slate-800/80 rounded-full h-1.5 overflow-hidden">
                <div
                  className={cn(
                    'h-full rounded-full bg-gradient-to-r transition-all duration-300 ease-out',
                    getConfidenceColor(data.confidence)
                  )}
                  style={{ width: `${Math.min(data.confidence * 100, 100)}%` }}
                />
              </div>
            </div>
            {/* Smoothed confidence */}
            <div className="flex items-center gap-2">
              <span className="text-[10px] text-slate-600 w-8">Avg</span>
              <div className="flex-1 bg-slate-800/80 rounded-full h-1.5 overflow-hidden">
                <div
                  className={cn(
                    'h-full rounded-full transition-all duration-500 ease-out',
                    flash ? 'bg-emerald-500' : 'bg-slate-400/60'
                  )}
                  style={{ width: `${Math.min(smoothedConf * 100, 100)}%` }}
                />
              </div>
            </div>
          </div>

          {/* Last accepted letter */}
          {lastAccepted && (
            <div className="flex items-center gap-2 mb-4 px-3 py-1.5 bg-emerald-500/5 rounded-lg border border-emerald-500/10">
              <span className="text-[10px] text-emerald-500/60 uppercase font-semibold">Last Accepted</span>
              <span className="text-sm font-bold text-emerald-400">{lastAccepted}</span>
            </div>
          )}

          {/* Top 3 Predictions */}
          {data.top_3 && data.top_3.length > 0 && (
            <div className="space-y-2.5">
              <h3 className="text-[10px] font-semibold text-slate-600 uppercase tracking-wider">
                Top Predictions
              </h3>
              {data.top_3.map((pred, i) => (
                <div key={`${pred.sign}-${i}`} className="flex items-center gap-2.5">
                  <span className="text-xs text-slate-600 w-4 font-mono">{i + 1}</span>
                  <span className={cn(
                    'text-sm flex-1 font-medium truncate',
                    pred.sign === data.sign ? 'text-white' : 'text-slate-400'
                  )}>
                    {pred.sign}
                  </span>
                  <div className="w-16 bg-slate-800/60 rounded-full h-1.5 overflow-hidden">
                    <div
                      className="bg-slate-500/60 h-full rounded-full transition-all duration-300"
                      style={{ width: `${Math.min(pred.confidence * 100, 100)}%` }}
                    />
                  </div>
                  <span className="text-xs text-slate-500 w-12 text-right tabular-nums font-mono">
                    {(pred.confidence * 100).toFixed(1)}%
                  </span>
                </div>
              ))}
            </div>
          )}
        </div>
      ) : (
        <div className="flex flex-col items-center justify-center py-8 text-slate-600">
          <span className="text-4xl mb-3 opacity-40">✋</span>
          <p className="text-sm font-medium text-slate-500">No Signs Detected</p>
          <p className="text-xs text-slate-600 mt-1">
            Start the pipeline to begin recognition
          </p>
        </div>
      )}
    </div>
  );
}
