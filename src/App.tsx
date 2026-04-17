import { useState, useEffect, useCallback, useRef } from 'react';
import { useWebSocket } from '@/hooks/useWebSocket';
import { useSettings } from '@/hooks/useSettings';
import { CameraPreview } from '@/components/CameraPreview';
import { SignDetection } from '@/components/SignDetection';
import { TranscriptPanel } from '@/components/TranscriptPanel';
import { ControlBar } from '@/components/ControlBar';
import { Settings } from '@/components/Settings';
import { StatusIndicator } from '@/components/StatusIndicator';
import { ToastContainer, useToast } from '@/components/Toast';
import type { PipelineSettings, ModelInfo } from '@/types';

function formatDuration(seconds: number): string {
  const h = Math.floor(seconds / 3600);
  const m = Math.floor((seconds % 3600) / 60);
  const s = Math.floor(seconds % 60);
  if (h > 0) return `${h}h ${m}m ${s}s`;
  if (m > 0) return `${m}m ${s}s`;
  return `${s}s`;
}

export function App() {
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [pipelineLoading, setPipelineLoading] = useState(false);
  const [sessionStart, setSessionStart] = useState<number | null>(null);
  const [sessionDuration, setSessionDuration] = useState(0);
  const { settings, updateSettings: updateLocalSettings, resetSettings } = useSettings();
  const { toasts, addToast, removeToast } = useToast();

  const {
    connectionStatus,
    backendOnline,
    frame,
    signDetection,
    transcript,
    status,
    devices,
    errors,
    dismissError,
    startPipeline,
    stopPipeline,
    updateSettings: updateRemoteSettings,
    requestDevices,
    clearTranscript,
    models,
    activeModelId,
    switchModel,
  } = useWebSocket();

  // ── Track previous status for toast triggers ──
  const prevStatusRef = useRef(status);
  const prevBackendRef = useRef(backendOnline);

  useEffect(() => {
    const prev = prevStatusRef.current;

    // Pipeline started
    if (status.pipeline_running && !prev.pipeline_running) {
      setPipelineLoading(false);
      setSessionStart(Date.now());
      addToast({ message: 'Pipeline started', type: 'success', icon: '▶' });
    }

    // Pipeline stopped
    if (!status.pipeline_running && prev.pipeline_running) {
      setSessionStart(null);
      addToast({ message: 'Pipeline stopped', type: 'info', icon: '■' });
    }

    // Model loaded
    if (status.model_loaded && !prev.model_loaded) {
      addToast({ message: 'Sign language model loaded', type: 'success', icon: '🧠' });
    }

    // VCam toggled
    if (status.vcam_active && !prev.vcam_active) {
      addToast({ message: 'Virtual camera active', type: 'success', icon: '📷' });
    }
    if (!status.vcam_active && prev.vcam_active) {
      addToast({ message: 'Virtual camera stopped', type: 'info', icon: '📷' });
    }

    // VMic toggled
    if (status.vmic_active && !prev.vmic_active) {
      addToast({ message: 'Virtual microphone active', type: 'success', icon: '🎤' });
    }
    if (!status.vmic_active && prev.vmic_active) {
      addToast({ message: 'Virtual microphone stopped', type: 'info', icon: '🎤' });
    }

    // Hands detected / lost
    if (status.hands_detected && !prev.hands_detected) {
      addToast({ message: 'Hand detected — start signing!', type: 'info', icon: '✋', duration: 2500 });
    }

    prevStatusRef.current = status;
  }, [status, addToast]);

  // Backend online/offline toasts
  useEffect(() => {
    if (backendOnline && !prevBackendRef.current) {
      addToast({ message: 'Backend connected', type: 'success', icon: '🟢', duration: 3000 });
    }
    if (!backendOnline && prevBackendRef.current) {
      addToast({ message: 'Backend disconnected', type: 'warning', icon: '🔴', duration: 5000 });
    }
    prevBackendRef.current = backendOnline;
  }, [backendOnline, addToast]);

  // ── Session duration timer ──
  useEffect(() => {
    if (!sessionStart) {
      setSessionDuration(0);
      return;
    }
    const interval = setInterval(() => {
      setSessionDuration(Math.floor((Date.now() - sessionStart) / 1000));
    }, 1000);
    return () => clearInterval(interval);
  }, [sessionStart]);

  // ── Handlers ──
  const handleStart = useCallback(() => {
    setPipelineLoading(true);
    startPipeline(settings);
    setTimeout(() => setPipelineLoading(false), 15000);
  }, [startPipeline, settings]);

  const handleStop = useCallback(() => {
    setPipelineLoading(false);
    stopPipeline();
  }, [stopPipeline]);

  const handleSettingChange = useCallback(
    (updates: Partial<PipelineSettings>) => {
      updateLocalSettings(updates);
      if (status.pipeline_running) {
        updateRemoteSettings(updates);
      }
    },
    [updateLocalSettings, updateRemoteSettings, status.pipeline_running]
  );

  const handleSwitchModel = useCallback((modelId: string) => {
    switchModel(modelId);
    const model = models.find(m => m.id === modelId);
    if (model) {
      addToast({ message: `Switched to ${model.name}`, type: 'success', icon: '🧠' });
    }
  }, [switchModel, models, addToast]);

  const handleOpenSettings = useCallback(() => {
    requestDevices();
    setSettingsOpen(true);
  }, [requestDevices]);

  const handleExportTranscript = useCallback(() => {
    const parts: string[] = [];
    if (transcript.full_text) parts.push(transcript.full_text);
    if (transcript.latest_word) parts.push(transcript.latest_word);
    const text = parts.join(' ').trim();
    if (!text) return;

    const timestamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19);
    const content = [
      'SignSpeak Transcript',
      `Date: ${new Date().toLocaleString()}`,
      `Duration: ${formatDuration(sessionDuration)}`,
      '─'.repeat(40),
      '',
      text,
      '',
      '─'.repeat(40),
      `Words: ${text.split(/\s+/).filter(Boolean).length}`,
      `Frames: ${status.frames_processed.toLocaleString()}`,
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

    addToast({ message: 'Transcript exported', type: 'success', icon: '📄', duration: 2000 });
  }, [transcript, sessionDuration, status.frames_processed, addToast]);

  const handleClearTranscript = useCallback(() => {
    clearTranscript();
    addToast({ message: 'Transcript cleared', type: 'info', icon: '🗑', duration: 2000 });
  }, [clearTranscript, addToast]);

  // ── Keyboard Shortcuts ──
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // Ctrl+Enter — Start/Stop pipeline
      if (e.ctrlKey && e.key === 'Enter') {
        e.preventDefault();
        if (status.pipeline_running) handleStop();
        else handleStart();
      }
      // Ctrl+, — Toggle Settings
      else if (e.ctrlKey && e.key === ',') {
        e.preventDefault();
        if (settingsOpen) setSettingsOpen(false);
        else handleOpenSettings();
      }
      // Ctrl+L — Toggle Landmarks
      else if (e.ctrlKey && e.key === 'l') {
        e.preventDefault();
        handleSettingChange({ show_landmarks: !settings.show_landmarks });
      }
      // Ctrl+T — Toggle TTS
      else if (e.ctrlKey && e.key === 't') {
        e.preventDefault();
        handleSettingChange({ tts_enabled: !settings.tts_enabled });
        addToast({
          message: settings.tts_enabled ? 'TTS disabled' : 'TTS enabled',
          type: 'info',
          icon: '🔊',
          duration: 2000,
        });
      }
      // Ctrl+M — Toggle VMic
      else if (e.ctrlKey && e.key === 'm') {
        e.preventDefault();
        handleSettingChange({ vmic_enabled: !settings.vmic_enabled });
        addToast({
          message: settings.vmic_enabled ? 'VMic disabled' : 'VMic enabled',
          type: 'info',
          icon: '🎤',
          duration: 2000,
        });
      }
      // Ctrl+E — Export transcript
      else if (e.ctrlKey && e.key === 'e') {
        e.preventDefault();
        handleExportTranscript();
      }
      // Ctrl+Shift+C — Clear transcript
      else if (e.ctrlKey && e.shiftKey && e.key === 'C') {
        e.preventDefault();
        handleClearTranscript();
      }
      // Escape — Close settings
      else if (e.key === 'Escape' && settingsOpen) {
        e.preventDefault();
        setSettingsOpen(false);
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [
    status.pipeline_running, settings.show_landmarks, settings.tts_enabled,
    settings.vmic_enabled, settingsOpen, handleStart, handleStop,
    handleSettingChange, handleClearTranscript, handleExportTranscript,
    handleOpenSettings, addToast,
  ]);

  // Determine loading state: pipeline was requested but no frames yet
  const isLoading = pipelineLoading && !frame;

  // Word count from transcript
  const fullText = [transcript.full_text, transcript.latest_word].filter(Boolean).join(' ').trim();
  const wordCount = fullText ? fullText.split(/\s+/).filter(Boolean).length : 0;

  return (
    <div className="h-screen bg-slate-950 text-white flex flex-col overflow-hidden">
      {/* ─── Header ─── */}
      <header className="h-14 flex items-center justify-between px-5 border-b border-slate-800/40 bg-slate-900/40 backdrop-blur-md shrink-0">
        <div className="flex items-center gap-3">
          <div className="flex items-center justify-center w-8 h-8 rounded-lg bg-gradient-to-br from-violet-600 to-indigo-600 shadow-lg shadow-violet-500/20">
            <span className="text-lg leading-none">🤟</span>
          </div>
          <div className="flex items-center gap-2">
            <h1 className="text-lg font-bold bg-gradient-to-r from-violet-400 to-indigo-400 bg-clip-text text-transparent">
              SignSpeak
            </h1>
            <span className="text-[10px] text-slate-600 font-medium bg-slate-800/60 px-1.5 py-0.5 rounded-full">
              v1.0
            </span>
          </div>
        </div>

        <StatusIndicator
          connectionStatus={connectionStatus}
          backendOnline={backendOnline}
          status={status}
        />

        <button
          onClick={handleOpenSettings}
          className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-sm text-slate-400 hover:text-white hover:bg-slate-800/60 transition-colors"
          title="Settings (Ctrl+,)"
        >
          <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
            <path strokeLinecap="round" strokeLinejoin="round" d="M9.594 3.94c.09-.542.56-.94 1.11-.94h2.593c.55 0 1.02.398 1.11.94l.213 1.281c.063.374.313.686.645.87.074.04.147.083.22.127.325.196.72.257 1.075.124l1.217-.456a1.125 1.125 0 0 1 1.37.49l1.296 2.247a1.125 1.125 0 0 1-.26 1.431l-1.003.827c-.293.241-.438.613-.43.992a7.723 7.723 0 0 1 0 .255c-.008.378.137.75.43.991l1.004.827c.424.35.534.955.26 1.43l-1.298 2.247a1.125 1.125 0 0 1-1.369.491l-1.217-.456c-.355-.133-.75-.072-1.076.124a6.47 6.47 0 0 1-.22.128c-.331.183-.581.495-.644.869l-.213 1.281c-.09.543-.56.94-1.11.94h-2.594c-.55 0-1.019-.398-1.11-.94l-.213-1.281c-.062-.374-.312-.686-.644-.87a6.52 6.52 0 0 1-.22-.127c-.325-.196-.72-.257-1.076-.124l-1.217.456a1.125 1.125 0 0 1-1.369-.49l-1.297-2.247a1.125 1.125 0 0 1 .26-1.431l1.004-.827c.292-.24.437-.613.43-.991a6.932 6.932 0 0 1 0-.255c.007-.38-.138-.751-.43-.992l-1.004-.827a1.125 1.125 0 0 1-.26-1.43l1.297-2.247a1.125 1.125 0 0 1 1.37-.491l1.216.456c.356.133.751.072 1.076-.124.072-.044.146-.086.22-.128.332-.183.582-.495.644-.869l.214-1.28Z" />
            <path strokeLinecap="round" strokeLinejoin="round" d="M15 12a3 3 0 1 1-6 0 3 3 0 0 1 6 0Z" />
          </svg>
          <span className="hidden sm:inline">Settings</span>
        </button>
      </header>

      {/* ─── Error Banners ─── */}
      {errors.length > 0 && (
        <div className="shrink-0 space-y-1 px-4 pt-3">
          {errors.map(error => (
            <div
              key={error.id}
              className="flex items-center justify-between px-4 py-2.5 bg-red-500/8 border border-red-500/15 rounded-lg animate-slide-down"
            >
              <div className="flex items-center gap-2.5">
                <svg className="w-4 h-4 text-red-400 shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M12 9v3.75m9-.75a9 9 0 1 1-18 0 9 9 0 0 1 18 0Zm-9 3.75h.008v.008H12v-.008Z" />
                </svg>
                <span className="text-sm text-red-300">{error.message}</span>
              </div>
              <button
                onClick={() => dismissError(error.id)}
                className="text-red-400/50 hover:text-red-400 transition-colors ml-4 shrink-0"
              >
                <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M6 18 18 6M6 6l12 12" />
                </svg>
              </button>
            </div>
          ))}
        </div>
      )}

      {/* ─── Main Content ─── */}
      <main className="flex-1 flex gap-4 p-4 min-h-0">
        {/* Left Column: Camera + Controls */}
        <div className="flex-1 flex flex-col gap-4 min-w-0">
          <CameraPreview
            frame={frame}
            isLoading={isLoading}
            pipelineRunning={status.pipeline_running}
          />
          <ControlBar
            settings={settings}
            status={status}
            devices={devices}
            connectionStatus={connectionStatus}
            backendOnline={backendOnline}
            onStart={handleStart}
            onStop={handleStop}
            onSettingChange={handleSettingChange}
          />
        </div>

        {/* Right Column: Detection + Transcript */}
        <div className="w-[380px] flex flex-col gap-4 shrink-0">
          <SignDetection data={signDetection} />
          <TranscriptPanel transcript={transcript} onClear={handleClearTranscript} />
        </div>
      </main>

      {/* ─── Footer ─── */}
      <footer className="shrink-0 px-5 py-2 border-t border-slate-800/30 bg-slate-900/20">
        <div className="flex items-center justify-between text-[10px] text-slate-600">
          <div className="flex items-center gap-4">
            <span>
              <kbd className="px-1 py-0.5 bg-slate-800/60 rounded border border-slate-700/40 font-mono mr-1">Ctrl+Enter</kbd>
              Start/Stop
            </span>
            <span>
              <kbd className="px-1 py-0.5 bg-slate-800/60 rounded border border-slate-700/40 font-mono mr-1">Ctrl+,</kbd>
              Settings
            </span>
            <span>
              <kbd className="px-1 py-0.5 bg-slate-800/60 rounded border border-slate-700/40 font-mono mr-1">Ctrl+L</kbd>
              Landmarks
            </span>
            <span>
              <kbd className="px-1 py-0.5 bg-slate-800/60 rounded border border-slate-700/40 font-mono mr-1">Ctrl+T</kbd>
              TTS
            </span>
            <span>
              <kbd className="px-1 py-0.5 bg-slate-800/60 rounded border border-slate-700/40 font-mono mr-1">Ctrl+M</kbd>
              VMic
            </span>
            <span>
              <kbd className="px-1 py-0.5 bg-slate-800/60 rounded border border-slate-700/40 font-mono mr-1">Ctrl+E</kbd>
              Export
            </span>
            <span>
              <kbd className="px-1 py-0.5 bg-slate-800/60 rounded border border-slate-700/40 font-mono mr-1">Esc</kbd>
              Close
            </span>
          </div>
          <div className="flex items-center gap-3">
            {status.pipeline_running && (
              <>
                <span className="text-slate-500 tabular-nums">
                  ⏱ {formatDuration(sessionDuration)}
                </span>
                <span className="text-slate-600">·</span>
                <span className="text-slate-500 tabular-nums">
                  {wordCount} {wordCount === 1 ? 'word' : 'words'}
                </span>
                <span className="text-slate-600">·</span>
                <span className="text-slate-500 tabular-nums">
                  {status.frames_processed.toLocaleString()} frames
                </span>
              </>
            )}
            <span className="text-slate-700">SignSpeak v1.0</span>
          </div>
        </div>
      </footer>

      {/* ─── Settings Modal ─── */}
      {settingsOpen && (
        <Settings
          settings={settings}
          devices={devices}
          models={models}
          activeModelId={activeModelId}
          onSwitchModel={handleSwitchModel}
          onUpdate={handleSettingChange}
          onReset={resetSettings}
          onClose={() => setSettingsOpen(false)}
        />
      )}

      {/* ─── Toasts ─── */}
      <ToastContainer toasts={toasts} onRemove={removeToast} />
    </div>
  );
}
