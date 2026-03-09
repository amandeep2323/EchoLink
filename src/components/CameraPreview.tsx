interface CameraPreviewProps {
  frame: string | null;
  isLoading?: boolean;
  pipelineRunning?: boolean;
}

export function CameraPreview({ frame, isLoading, pipelineRunning }: CameraPreviewProps) {
  return (
    <div className="relative flex-1 bg-slate-900/80 rounded-xl overflow-hidden border border-slate-800/60 min-h-0">
      {frame ? (
        <img
          src={`data:image/jpeg;base64,${frame}`}
          alt="Camera feed with sign language detection"
          className="w-full h-full object-contain"
          draggable={false}
        />
      ) : (
        <div className="absolute inset-0 flex flex-col items-center justify-center gap-4">
          {isLoading ? (
            /* ── Loading State: Pipeline is starting ── */
            <>
              <div className="relative">
                {/* Spinner ring */}
                <div className="w-16 h-16 rounded-full border-2 border-slate-800 border-t-violet-500 animate-spin-slow" />
                {/* Center icon */}
                <div className="absolute inset-0 flex items-center justify-center">
                  <svg className="w-6 h-6 text-violet-400" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
                    <path strokeLinecap="round" strokeLinejoin="round" d="m15.75 10.5 4.72-4.72a.75.75 0 0 1 1.28.53v11.38a.75.75 0 0 1-1.28.53l-4.72-4.72M4.5 18.75h9a2.25 2.25 0 0 0 2.25-2.25v-9a2.25 2.25 0 0 0-2.25-2.25h-9A2.25 2.25 0 0 0 2.25 7.5v9a2.25 2.25 0 0 0 2.25 2.25Z" />
                  </svg>
                </div>
              </div>
              <div className="text-center space-y-1.5">
                <p className="text-sm font-medium text-violet-400">Starting Pipeline...</p>
                <p className="text-xs text-slate-500">Loading camera, model, and MediaPipe</p>
              </div>
              {/* Progress dots */}
              <div className="flex items-center gap-1.5">
                <div className="w-1.5 h-1.5 rounded-full bg-violet-500 animate-pulse" style={{ animationDelay: '0ms' }} />
                <div className="w-1.5 h-1.5 rounded-full bg-violet-500 animate-pulse" style={{ animationDelay: '200ms' }} />
                <div className="w-1.5 h-1.5 rounded-full bg-violet-500 animate-pulse" style={{ animationDelay: '400ms' }} />
              </div>
            </>
          ) : (
            /* ── Idle State: Not started ── */
            <>
              <div className="relative">
                <div className="w-20 h-20 rounded-2xl bg-slate-800/60 border border-slate-700/40 flex items-center justify-center">
                  <svg className="w-10 h-10 text-slate-600" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
                    <path strokeLinecap="round" strokeLinejoin="round" d="m15.75 10.5 4.72-4.72a.75.75 0 0 1 1.28.53v11.38a.75.75 0 0 1-1.28.53l-4.72-4.72M4.5 18.75h9a2.25 2.25 0 0 0 2.25-2.25v-9a2.25 2.25 0 0 0-2.25-2.25h-9A2.25 2.25 0 0 0 2.25 7.5v9a2.25 2.25 0 0 0 2.25 2.25Z" />
                  </svg>
                </div>
                <div className="absolute inset-0 rounded-2xl border-2 border-slate-700/30 animate-pulse" />
              </div>
              <div className="text-center space-y-1">
                <p className="text-sm font-medium text-slate-500">No Camera Feed</p>
                <p className="text-xs text-slate-600">
                  {pipelineRunning ? 'Waiting for frames...' : 'Click Start or press Ctrl+Enter'}
                </p>
              </div>
              {!pipelineRunning && (
                <div className="flex items-center gap-2 text-xs text-slate-600">
                  <kbd className="px-1.5 py-0.5 bg-slate-800 rounded border border-slate-700 text-slate-500 font-mono text-[10px]">
                    Ctrl
                  </kbd>
                  <span>+</span>
                  <kbd className="px-1.5 py-0.5 bg-slate-800 rounded border border-slate-700 text-slate-500 font-mono text-[10px]">
                    Enter
                  </kbd>
                  <span className="text-slate-600">to start</span>
                </div>
              )}
            </>
          )}
        </div>
      )}

      {/* Subtle gradient overlay at bottom */}
      {frame && (
        <div className="absolute bottom-0 left-0 right-0 h-12 bg-gradient-to-t from-slate-950/40 to-transparent pointer-events-none" />
      )}
    </div>
  );
}
