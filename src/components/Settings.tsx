import { useEffect, useRef } from 'react';
import type { PipelineSettings, DeviceList, ModelInfo } from '@/types';
import { cn } from '@/utils/cn';

interface SettingsProps {
  settings: PipelineSettings;
  devices: DeviceList;
  models: ModelInfo[];
  activeModelId: string;
  onSwitchModel: (modelId: string) => void;
  onUpdate: (updates: Partial<PipelineSettings>) => void;
  onReset: () => void;
  onClose: () => void;
}

// ── Reusable Toggle Switch ──
function Toggle({
  enabled,
  onChange,
  color = 'violet',
}: {
  enabled: boolean;
  onChange: () => void;
  color?: 'violet' | 'blue' | 'amber' | 'emerald';
}) {
  const colorMap = {
    violet: 'bg-violet-600 shadow-violet-500/30',
    blue: 'bg-blue-600 shadow-blue-500/30',
    amber: 'bg-amber-600 shadow-amber-500/30',
    emerald: 'bg-emerald-600 shadow-emerald-500/30',
  };

  return (
    <button
      type="button"
      onClick={onChange}
      className={cn(
        'relative w-11 h-6 rounded-full transition-all duration-200 shrink-0 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-offset-slate-900',
        enabled
          ? `${colorMap[color]} shadow-sm focus:ring-${color}-500`
          : 'bg-slate-700 hover:bg-slate-600 focus:ring-slate-500'
      )}
    >
      <span
        className={cn(
          'absolute top-0.5 left-0.5 w-5 h-5 rounded-full bg-white shadow-sm transition-transform duration-200',
          enabled && 'translate-x-5'
        )}
      />
    </button>
  );
}

// ── Setting Row ──
function SettingRow({
  label,
  description,
  children,
  indent,
}: {
  label: string;
  description?: string;
  children: React.ReactNode;
  indent?: boolean;
}) {
  return (
    <div
      className={cn(
        'flex items-center justify-between gap-4 py-3',
        indent && 'ml-4 pl-4 border-l-2 border-slate-700/40'
      )}
    >
      <div className="min-w-0">
        <span className="text-sm font-medium text-slate-300">{label}</span>
        {description && (
          <p className="text-xs text-slate-500 mt-0.5 leading-relaxed">{description}</p>
        )}
      </div>
      <div className="shrink-0">{children}</div>
    </div>
  );
}

// ── Section Card ──
function Section({
  icon,
  title,
  children,
}: {
  icon: string;
  title: string;
  children: React.ReactNode;
}) {
  return (
    <section className="rounded-xl bg-slate-800/30 border border-slate-800/50 overflow-hidden">
      <div className="flex items-center gap-2.5 px-5 py-3 border-b border-slate-800/50 bg-slate-800/20">
        <span className="text-base">{icon}</span>
        <h3 className="text-xs font-semibold text-slate-400 uppercase tracking-wider">{title}</h3>
      </div>
      <div className="px-5 divide-y divide-slate-800/40">{children}</div>
    </section>
  );
}

// ── Styled Select ──
function Select({
  value,
  onChange,
  children,
  className,
}: {
  value: string | number;
  onChange: (e: React.ChangeEvent<HTMLSelectElement>) => void;
  children: React.ReactNode;
  className?: string;
}) {
  return (
    <select
      value={value}
      onChange={onChange}
      className={cn(
        'bg-slate-800/80 border border-slate-700/50 rounded-lg px-3 py-2 text-sm text-slate-300',
        'hover:border-slate-600 focus:border-violet-500 transition-colors',
        className
      )}
    >
      {children}
    </select>
  );
}

export function Settings({ settings, devices, models, activeModelId, onSwitchModel, onUpdate, onReset, onClose }: SettingsProps) {
  const modalRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    modalRef.current?.focus();
  }, []);

  useEffect(() => {
    const handleKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        e.preventDefault();
        onClose();
      }
    };
    window.addEventListener('keydown', handleKey);
    return () => window.removeEventListener('keydown', handleKey);
  }, [onClose]);

  const handleResolutionChange = (value: string) => {
    const [w, h] = value.split('x').map(Number);
    onUpdate({ resolution: [w, h] as [number, number] });
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center animate-fade-in">
      {/* Backdrop */}
      <div className="absolute inset-0 bg-black/60 backdrop-blur-sm" onClick={onClose} />

      {/* Modal */}
      <div
        ref={modalRef}
        tabIndex={-1}
        className="relative bg-slate-900 border border-slate-700/50 rounded-2xl shadow-2xl shadow-black/50 w-full max-w-lg max-h-[85vh] overflow-hidden flex flex-col animate-scale-in"
      >
        {/* ── Header ── */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-slate-800/60 shrink-0 bg-slate-900/80">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-slate-700 to-slate-800 flex items-center justify-center border border-slate-600/50">
              <svg className="w-4 h-4 text-slate-300" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M9.594 3.94c.09-.542.56-.94 1.11-.94h2.593c.55 0 1.02.398 1.11.94l.213 1.281c.063.374.313.686.645.87.074.04.147.083.22.127.325.196.72.257 1.075.124l1.217-.456a1.125 1.125 0 0 1 1.37.49l1.296 2.247a1.125 1.125 0 0 1-.26 1.431l-1.003.827c-.293.241-.438.613-.43.992a7.723 7.723 0 0 1 0 .255c-.008.378.137.75.43.991l1.004.827c.424.35.534.955.26 1.43l-1.298 2.247a1.125 1.125 0 0 1-1.369.491l-1.217-.456c-.355-.133-.75-.072-1.076.124a6.47 6.47 0 0 1-.22.128c-.331.183-.581.495-.644.869l-.213 1.281c-.09.543-.56.94-1.11.94h-2.594c-.55 0-1.019-.398-1.11-.94l-.213-1.281c-.062-.374-.312-.686-.644-.87a6.52 6.52 0 0 1-.22-.127c-.325-.196-.72-.257-1.076-.124l-1.217.456a1.125 1.125 0 0 1-1.369-.49l-1.297-2.247a1.125 1.125 0 0 1 .26-1.431l1.004-.827c.292-.24.437-.613.43-.991a6.932 6.932 0 0 1 0-.255c.007-.38-.138-.751-.43-.992l-1.004-.827a1.125 1.125 0 0 1-.26-1.43l1.297-2.247a1.125 1.125 0 0 1 1.37-.491l1.216.456c.356.133.751.072 1.076-.124.072-.044.146-.086.22-.128.332-.183.582-.495.644-.869l.214-1.28Z" />
                <path strokeLinecap="round" strokeLinejoin="round" d="M15 12a3 3 0 1 1-6 0 3 3 0 0 1 6 0Z" />
              </svg>
            </div>
            <div>
              <h2 className="text-base font-semibold text-white">Settings</h2>
              <p className="text-[11px] text-slate-500">Configure pipeline and devices</p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="w-8 h-8 flex items-center justify-center rounded-lg text-slate-500 hover:text-white hover:bg-slate-800 transition-all"
            title="Close (Escape)"
          >
            <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M6 18 18 6M6 6l12 12" />
            </svg>
          </button>
        </div>

        {/* ── Content ── */}
        <div className="flex-1 overflow-y-auto px-6 py-5 space-y-4">
          {/* Models */}
          <Section icon="🧠" title="Models">
            {models.length === 0 ? (
              <div className="py-6 text-center">
                <p className="text-sm text-slate-500">No models found</p>
                <p className="text-xs text-slate-600 mt-1">Place model folders in <code className="text-slate-400 bg-slate-800/60 px-1.5 py-0.5 rounded">python-backend/models/sign/</code></p>
              </div>
            ) : (
              <div className="py-3 space-y-2">
                {models.map(model => (
                  <div
                    key={model.id}
                    className={cn(
                      'flex items-center justify-between gap-3 px-4 py-3 rounded-lg border transition-all',
                      model.id === activeModelId
                        ? 'bg-emerald-500/5 border-emerald-500/20'
                        : 'bg-slate-800/30 border-slate-700/30 hover:border-slate-600/50'
                    )}
                  >
                    <div className="min-w-0 flex-1">
                      <div className="flex items-center gap-2 flex-wrap">
                        <span className="text-sm font-medium text-slate-200">{model.name}</span>
                        <span className={cn(
                          'text-[10px] font-medium px-1.5 py-0.5 rounded-full',
                          model.type === 'fingerspelling' ? 'bg-blue-500/15 text-blue-400' :
                          model.type === 'word' ? 'bg-amber-500/15 text-amber-400' :
                          model.type === 'sentence' ? 'bg-purple-500/15 text-purple-400' :
                          'bg-slate-700/50 text-slate-400'
                        )}>
                          {model.type}
                        </span>
                        <span className="text-[10px] text-slate-500 bg-slate-800/60 px-1.5 py-0.5 rounded-full">
                          {model.labels_count} labels
                        </span>
                      </div>
                      {model.description && (
                        <p className="text-xs text-slate-500 mt-1 truncate">{model.description}</p>
                      )}
                    </div>
                    {model.id === activeModelId ? (
                      <span className="flex items-center gap-1.5 text-xs text-emerald-400 shrink-0">
                        <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse" />
                        Active
                      </span>
                    ) : (
                      <button
                        onClick={() => onSwitchModel(model.id)}
                        className="px-3 py-1.5 text-xs font-medium text-slate-400 hover:text-white bg-slate-700/50 hover:bg-violet-600 rounded-lg transition-all shrink-0"
                      >
                        Activate
                      </button>
                    )}
                  </div>
                ))}
              </div>
            )}
          </Section>

          {/* Camera */}
          <Section icon="📷" title="Camera">
            <div className="py-3">
              <label className="block text-xs font-medium text-slate-400 mb-2">Camera Device</label>
              <Select
                value={settings.camera_index}
                onChange={(e) => onUpdate({ camera_index: Number(e.target.value) })}
                className="w-full"
              >
                {devices.cameras.length > 0 ? (
                  devices.cameras.map(cam => (
                    <option key={cam.index} value={cam.index}>{cam.name}</option>
                  ))
                ) : (
                  <option value={0}>Default Camera</option>
                )}
              </Select>
            </div>
            <div className="py-3">
              <div className="grid grid-cols-2 gap-3">
                <div>
                  <label className="block text-xs font-medium text-slate-400 mb-2">Resolution</label>
                  <Select
                    value={`${settings.resolution[0]}x${settings.resolution[1]}`}
                    onChange={(e) => handleResolutionChange(e.target.value)}
                    className="w-full"
                  >
                    <option value="320x240">320 × 240</option>
                    <option value="640x480">640 × 480</option>
                    <option value="1280x720">1280 × 720 HD</option>
                    <option value="1920x1080">1920 × 1080 FHD</option>
                  </Select>
                </div>
                <div>
                  <label className="block text-xs font-medium text-slate-400 mb-2">Frame Rate</label>
                  <Select
                    value={settings.fps}
                    onChange={(e) => onUpdate({ fps: Number(e.target.value) })}
                    className="w-full"
                  >
                    <option value={15}>15 FPS</option>
                    <option value={24}>24 FPS</option>
                    <option value={30}>30 FPS</option>
                    <option value={60}>60 FPS</option>
                  </Select>
                </div>
              </div>
            </div>
          </Section>

          {/* Recognition */}
          <Section icon="🧠" title="Recognition">
            <div className="py-3">
              <div className="flex items-center justify-between mb-2">
                <label className="text-xs font-medium text-slate-400">Confidence Threshold</label>
                <span className="text-xs font-mono text-violet-400 bg-violet-500/10 px-2 py-0.5 rounded-md">
                  {(settings.confidence_threshold * 100).toFixed(0)}%
                </span>
              </div>
              <input
                type="range"
                min="0"
                max="100"
                value={settings.confidence_threshold * 100}
                onChange={(e) => onUpdate({ confidence_threshold: Number(e.target.value) / 100 })}
                className="w-full"
              />
              <p className="text-[11px] text-slate-600 mt-2">
                Higher values = more accurate but slower detection
              </p>
            </div>
          </Section>

          {/* Display */}
          <Section icon="👁" title="Display">
            <SettingRow label="Show Landmarks" description="Hand/pose wireframe on camera feed">
              <Toggle enabled={settings.show_landmarks} onChange={() => onUpdate({ show_landmarks: !settings.show_landmarks })} />
            </SettingRow>
            <SettingRow label="Transcript Overlay" description="Show transcript bar on video frame">
              <Toggle enabled={settings.show_overlay} onChange={() => onUpdate({ show_overlay: !settings.show_overlay })} />
            </SettingRow>
          </Section>

          {/* Text-to-Speech */}
          <Section icon="🔊" title="Text-to-Speech">
            <SettingRow label="Enable TTS" description="Speak completed words aloud">
              <Toggle enabled={settings.tts_enabled} onChange={() => onUpdate({ tts_enabled: !settings.tts_enabled })} color="amber" />
            </SettingRow>
            <div className="py-3">
              <label className="block text-xs font-medium text-slate-400 mb-2">Voice Model</label>
              <input
                type="text"
                value={settings.tts_voice}
                onChange={(e) => onUpdate({ tts_voice: e.target.value })}
                placeholder="en_US-lessac-medium"
                className="w-full bg-slate-800/80 border border-slate-700/50 rounded-lg px-3 py-2 text-sm text-slate-300 placeholder-slate-600 hover:border-slate-600 focus:border-violet-500 transition-colors"
              />
              <p className="text-[11px] text-slate-600 mt-1.5">
                Piper voice model name • uses Windows SAPI as fallback
              </p>
            </div>
            <div className="py-3">
              <label className="block text-xs font-medium text-slate-400 mb-2">Local Audio Output</label>
              <Select
                value={settings.audio_output_device}
                onChange={(e) => onUpdate({ audio_output_device: e.target.value })}
                className="w-full"
              >
                <option value="">Default Speakers</option>
                {devices.audio_output_devices.map(dev => (
                  <option key={dev.index} value={dev.name}>{dev.name}</option>
                ))}
              </Select>
              <p className="text-[11px] text-slate-600 mt-1.5">
                Where you hear TTS locally (your speakers/headphones)
              </p>
            </div>
          </Section>

          {/* Virtual Devices */}
          <Section icon="🔌" title="Virtual Devices">
            <SettingRow label="Virtual Camera" description="Output to OBS Virtual Camera for Meet/Zoom">
              <Toggle enabled={settings.vcam_enabled} onChange={() => onUpdate({ vcam_enabled: !settings.vcam_enabled })} color="blue" />
            </SettingRow>
            {settings.vcam_enabled && (
              <SettingRow label="Mirror VCam" description="Flip output so participants see correct orientation" indent>
                <Toggle enabled={settings.vcam_mirror} onChange={() => onUpdate({ vcam_mirror: !settings.vcam_mirror })} color="blue" />
              </SettingRow>
            )}
            <SettingRow label="Virtual Microphone" description="Route TTS audio to VB-Audio Virtual Cable">
              <Toggle enabled={settings.vmic_enabled} onChange={() => onUpdate({ vmic_enabled: !settings.vmic_enabled })} color="blue" />
            </SettingRow>
            {settings.vmic_enabled && (
              <div className="py-3 ml-4 pl-4 border-l-2 border-slate-700/40">
                <label className="block text-xs font-medium text-slate-400 mb-2">VMic Output Device</label>
                <Select
                  value={settings.vmic_device}
                  onChange={(e) => onUpdate({ vmic_device: e.target.value })}
                  className="w-full"
                >
                  <option value="">Auto-detect VB-Cable</option>
                  {devices.audio_output_devices.map(dev => (
                    <option key={dev.index} value={dev.name}>{dev.name}</option>
                  ))}
                </Select>
                <div className="mt-2 px-3 py-2 rounded-lg bg-blue-500/5 border border-blue-500/10">
                  <p className="text-[11px] text-blue-400/70 leading-relaxed">
                    <strong>Setup:</strong> Select "CABLE Input" here → In Meet/Zoom, set mic to "CABLE Output"
                  </p>
                </div>
              </div>
            )}
          </Section>
        </div>

        {/* ── Footer ── */}
        <div className="flex items-center justify-between px-6 py-4 border-t border-slate-800/60 shrink-0 bg-slate-900/80">
          <button
            onClick={onReset}
            className="flex items-center gap-2 px-4 py-2 text-sm text-slate-500 hover:text-amber-400 hover:bg-amber-500/5 rounded-lg transition-all border border-transparent hover:border-amber-500/20"
          >
            <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M16.023 9.348h4.992v-.001M2.985 19.644v-4.992m0 0h4.992m-4.993 0 3.181 3.183a8.25 8.25 0 0 0 13.803-3.7M4.031 9.865a8.25 8.25 0 0 1 13.803-3.7l3.181 3.182" />
            </svg>
            Reset Defaults
          </button>
          <div className="flex items-center gap-3">
            <span className="text-[10px] text-slate-600">
              <kbd className="px-1 py-0.5 bg-slate-800 rounded border border-slate-700 font-mono">Esc</kbd> to close
            </span>
            <button
              onClick={onClose}
              className="px-6 py-2 bg-violet-600 hover:bg-violet-500 text-white rounded-lg text-sm font-semibold transition-all shadow-sm shadow-violet-500/20 hover:shadow-md hover:shadow-violet-500/30 active:scale-[0.98]"
            >
              Done
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
