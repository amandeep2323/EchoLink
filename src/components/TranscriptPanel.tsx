import { useRef, useEffect, useState } from 'react';
import type { TranscriptData } from '@/types';
import { cn } from '@/utils/cn';

interface TranscriptPanelProps {
  transcript: TranscriptData;
  onClear?: () => void;
}

export function TranscriptPanel({ transcript, onClear }: TranscriptPanelProps) {
  const scrollRef = useRef<HTMLDivElement>(null);
  const [copied, setCopied] = useState(false);

  // Auto-scroll to bottom
  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [transcript.full_text, transcript.latest_word]);

  const getFullText = () => {
    const parts: string[] = [];
    if (transcript.full_text) parts.push(transcript.full_text);
    if (transcript.latest_word) parts.push(transcript.latest_word);
    return parts.join(' ').trim();
  };

  const handleCopy = async () => {
    const text = getFullText();
    if (!text) return;
    try {
      await navigator.clipboard.writeText(text);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch { /* noop */ }
  };

  const handleExport = () => {
    const text = getFullText();
    if (!text) return;

    const timestamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19);
    const content = [
      `SignSpeak Transcript`,
      `Date: ${new Date().toLocaleString()}`,
      `${'─'.repeat(40)}`,
      '',
      text,
      '',
      `${'─'.repeat(40)}`,
      `Words: ${text.split(/\s+/).filter(Boolean).length}`,
    ].join('\n');

    const blob = new Blob([content], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `signspeak-transcript-${timestamp}.txt`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  const fullText = getFullText();
  const wordCount = fullText ? fullText.split(/\s+/).filter(Boolean).length : 0;
  const hasContent = !!transcript.full_text || !!transcript.latest_word;

  return (
    <div className="flex-1 flex flex-col bg-slate-900/80 rounded-xl border border-slate-800/60 min-h-0">
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-2.5 border-b border-slate-800/50 shrink-0 gap-2">
        <div className="flex items-center gap-2 min-w-0">
          <h2 className="text-xs font-semibold text-slate-500 uppercase tracking-wider shrink-0">
            Transcript
          </h2>
          {wordCount > 0 && (
            <span className="text-[10px] text-slate-600 bg-slate-800/60 px-1.5 py-0.5 rounded-full tabular-nums shrink-0">
              {wordCount}
            </span>
          )}
          {transcript.latest_word && (
            <span className="flex items-center gap-1 text-[10px] text-violet-400/70 bg-violet-500/8 px-1.5 py-0.5 rounded-full font-medium shrink-0">
              <span className="inline-block w-1 h-1 rounded-full bg-violet-400 animate-pulse" />
              spelling
            </span>
          )}
        </div>
        <div className="flex items-center shrink-0">
          <button
            onClick={handleExport}
            disabled={!hasContent}
            className={cn(
              'p-1.5 rounded-md transition-all',
              hasContent
                ? 'text-slate-500 hover:text-blue-400 hover:bg-blue-500/10'
                : 'opacity-30 cursor-not-allowed text-slate-600'
            )}
            title="Export transcript (Ctrl+E)"
          >
            <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M3 16.5v2.25A2.25 2.25 0 0 0 5.25 21h13.5A2.25 2.25 0 0 0 21 18.75V16.5M16.5 12 12 16.5m0 0L7.5 12m4.5 4.5V3" />
            </svg>
          </button>
          {onClear && (
            <button
              onClick={onClear}
              disabled={!hasContent}
              className={cn(
                'p-1.5 rounded-md transition-all',
                hasContent
                  ? 'text-slate-500 hover:text-red-400 hover:bg-red-500/10'
                  : 'opacity-30 cursor-not-allowed text-slate-600'
              )}
              title="Clear transcript"
            >
              <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                <path strokeLinecap="round" strokeLinejoin="round" d="m14.74 9-.346 9m-4.788 0L9.26 9m9.968-3.21c.342.052.682.107 1.022.166m-1.022-.165L18.16 19.673a2.25 2.25 0 0 1-2.244 2.077H8.084a2.25 2.25 0 0 1-2.244-2.077L4.772 5.79m14.456 0a48.108 48.108 0 0 0-3.478-.397m-12 .562c.34-.059.68-.114 1.022-.165m0 0a48.11 48.11 0 0 1 3.478-.397m7.5 0v-.916c0-1.18-.91-2.164-2.09-2.201a51.964 51.964 0 0 0-3.32 0c-1.18.037-2.09 1.022-2.09 2.201v.916m7.5 0a48.667 48.667 0 0 0-7.5 0" />
              </svg>
            </button>
          )}
          <button
            onClick={handleCopy}
            disabled={!hasContent}
            className={cn(
              'p-1.5 rounded-md transition-all',
              copied
                ? 'text-emerald-400 bg-emerald-500/10'
                : 'text-slate-500 hover:text-slate-300 hover:bg-slate-800/60',
              !hasContent && 'opacity-30 cursor-not-allowed'
            )}
            title={copied ? 'Copied!' : 'Copy transcript'}
          >
            <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
              {copied ? (
                <path strokeLinecap="round" strokeLinejoin="round" d="m4.5 12.75 6 6 9-13.5" />
              ) : (
                <path strokeLinecap="round" strokeLinejoin="round" d="M15.666 3.888A2.25 2.25 0 0 0 13.5 2.25h-3c-1.03 0-1.9.693-2.166 1.638m7.332 0c.055.194.084.4.084.612v0a.75.75 0 0 1-.75.75H9.75a.75.75 0 0 1-.75-.75v0c0-.212.03-.418.084-.612m7.332 0c.646.049 1.288.11 1.927.184 1.1.128 1.907 1.077 1.907 2.185V19.5a2.25 2.25 0 0 1-2.25 2.25H6.75A2.25 2.25 0 0 1 4.5 19.5V6.257c0-1.108.806-2.057 1.907-2.185a48.208 48.208 0 0 1 1.927-.184" />
              )}
            </svg>
          </button>
        </div>
      </div>

      {/* Transcript Content */}
      <div ref={scrollRef} className="flex-1 overflow-y-auto p-5 min-h-0">
        {hasContent ? (
          <div className="space-y-3">
            <div className="text-sm leading-relaxed whitespace-pre-wrap">
              {/* ── Completed words (white, final, spell-corrected) ── */}
              {transcript.full_text && (
                <span className="text-slate-200">{transcript.full_text}</span>
              )}

              {/* ── Current word being spelled (violet + blinking cursor) ── */}
              {transcript.latest_word && (
                <>
                  {transcript.full_text ? ' ' : ''}
                  <span className="relative inline-flex items-baseline">
                    <span className="px-1 py-0.5 rounded bg-violet-500/10 text-violet-300 font-semibold tracking-wide border border-violet-500/20">
                      {transcript.latest_word}
                    </span>
                    <span className="inline-block w-0.5 h-4 bg-violet-400 ml-0.5 animate-cursor-blink rounded-full" />
                  </span>
                </>
              )}
            </div>

            {/* Sentence complete indicator */}
            {transcript.is_sentence_complete && (
              <div className="flex items-center gap-2 mt-2 px-3 py-2 rounded-lg bg-emerald-500/5 border border-emerald-500/10">
                <svg className="w-3.5 h-3.5 text-emerald-400" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="m4.5 12.75 6 6 9-13.5" />
                </svg>
                <span className="text-[11px] text-emerald-400/80 font-medium">
                  Sentence complete — spoken via TTS
                </span>
              </div>
            )}
          </div>
        ) : (
          <div className="flex flex-col items-center justify-center h-full text-slate-600">
            <span className="text-3xl mb-3 opacity-30">📝</span>
            <p className="text-sm font-medium text-slate-500">No Transcript Yet</p>
            <p className="text-xs text-slate-600 mt-1.5 text-center leading-relaxed">
              Sign ASL letters to build words.
              <br />
              Remove hand to complete a word.
            </p>
          </div>
        )}
      </div>
    </div>
  );
}
