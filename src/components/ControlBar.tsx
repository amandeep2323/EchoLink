import type { PipelineSettings, StatusData, DeviceList, ConnectionStatus } from '@/types';
import { cn } from '@/utils/cn';

interface ControlBarProps {
  settings: PipelineSettings;
  status: StatusData;
  devices: DeviceList;
  connectionStatus: ConnectionStatus;
  backendOnline: boolean;
  onStart: () => void;
  onStop: () => void;
  onSettingChange: (updates: Partial<PipelineSettings>) => void;
}

interface ToggleBtnProps {
  active: boolean;
  icon: string;
  label: string;
  onClick: () => void;
  variant?: 'violet' | 'blue' | 'amber';
  suffix?: string;
}

function ToggleBtn({ active, icon, label, onClick, variant = 'violet', suffix }: ToggleBtnProps) {
  const colorMap = {
    violet: {
      active: 'bg-violet-500/15 text-violet-300 border-violet-500/30 shadow-violet-500/5',
      hover: 'hover:bg-violet-500/10',
    },
    blue: {
      active: 'bg-blue-500/15 text-blue-300 border-blue-500/30 shadow-blue-500/5',
      hover: 'hover:bg-blue-500/10',
    },
    amber: {
      active: 'bg-amber-500/15 text-amber-300 border-amber-500/30 shadow-amber-500/5',
      hover: 'hover:bg-amber-500/10',
    },
  };

  const colors = colorMap[variant];

  return (
    <button
      onClick={onClick}
      className={cn(
        'flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium transition-all border shadow-sm',
        active
          ? `${colors.active} shadow-sm`
          : `bg-slate-800/40 text-slate-500 border-slate-700/40 hover:bg-slate-800 hover:text-slate-400 ${colors.hover}`
      )}
      title={`Toggle ${label}`}
    >
      <span className="text-sm">{icon}</span>
      <span>{label}</span>
      {suffix && (
        <span className="text-[9px] opacity-60">{suffix}</span>
      )}
    </button>
  );
}

export function ControlBar({
  settings,
  status,
  devices,
  connectionStatus,
  backendOnline,
  onStart,
  onStop,
  onSettingChange,
}: ControlBarProps) {
  // connectionStatus and backendOnline available for future use
  void connectionStatus;
  void backendOnline;
  const isRunning = status.pipeline_running;

  return (
    <div className="bg-slate-900/80 rounded-xl border border-slate-800/60 p-4 shrink-0">
      <div className="flex flex-wrap items-center gap-3">
        {/* Start / Stop Button — always clickable */}
        <button
          onClick={() => {
            if (isRunning) {
              onStop();
            } else {
              onStart();
            }
          }}
          className={cn(
            'flex items-center gap-2 px-5 py-2 rounded-lg text-sm font-semibold transition-all border shadow-sm',
            isRunning
              ? 'bg-red-500/15 text-red-400 border-red-500/30 hover:bg-red-500/25 shadow-red-500/5'
              : 'bg-emerald-500/15 text-emerald-400 border-emerald-500/30 hover:bg-emerald-500/25 shadow-emerald-500/5'
          )}
          title={isRunning ? 'Stop Pipeline (Ctrl+Enter)' : 'Start Pipeline (Ctrl+Enter)'}
        >
          <span className="text-base">{isRunning ? '■' : '▶'}</span>
          <span>{isRunning ? 'Stop' : 'Start'}</span>
        </button>

        {/* Separator */}
        <div className="w-px h-7 bg-slate-800" />

        {/* Feature Toggles */}
        <ToggleBtn
          active={settings.show_landmarks}
          icon="🦴"
          label="Landmarks"
          onClick={() => onSettingChange({ show_landmarks: !settings.show_landmarks })}
          variant="violet"
        />
        <ToggleBtn
          active={settings.show_overlay}
          icon="📝"
          label="Overlay"
          onClick={() => onSettingChange({ show_overlay: !settings.show_overlay })}
          variant="violet"
        />

        {/* Separator */}
        <div className="w-px h-7 bg-slate-800" />

        {/* Output Device Toggles */}
        <ToggleBtn
          active={settings.tts_enabled}
          icon="🔊"
          label="TTS"
          onClick={() => onSettingChange({ tts_enabled: !settings.tts_enabled })}
          variant="amber"
        />
        <ToggleBtn
          active={settings.vcam_enabled}
          icon="📷"
          label="VCam"
          onClick={() => onSettingChange({ vcam_enabled: !settings.vcam_enabled })}
          variant="blue"
          suffix={settings.vcam_enabled && settings.vcam_mirror ? '🔄' : undefined}
        />
        <ToggleBtn
          active={settings.vmic_enabled}
          icon="🎤"
          label="VMic"
          onClick={() => onSettingChange({ vmic_enabled: !settings.vmic_enabled })}
          variant="blue"
        />

        {/* Separator */}
        <div className="w-px h-7 bg-slate-800" />

        {/* Confidence Threshold */}
        <div className="flex items-center gap-2.5">
          <label className="text-[10px] font-medium text-slate-500 uppercase tracking-wider">
            Threshold
          </label>
          <input
            type="range"
            min="0"
            max="100"
            value={settings.confidence_threshold * 100}
            onChange={(e) =>
              onSettingChange({ confidence_threshold: Number(e.target.value) / 100 })
            }
            className="w-20"
            title={`Confidence threshold: ${(settings.confidence_threshold * 100).toFixed(0)}%`}
          />
          <span className="text-xs text-slate-400 tabular-nums font-mono w-8">
            {(settings.confidence_threshold * 100).toFixed(0)}%
          </span>
        </div>

        {/* Spacer */}
        <div className="flex-1" />

        {/* Camera Select */}
        <div className="flex items-center gap-2">
          <label className="text-[10px] font-medium text-slate-500 uppercase tracking-wider">
            Camera
          </label>
          <select
            value={settings.camera_index}
            onChange={(e) => onSettingChange({ camera_index: Number(e.target.value) })}
            className="bg-slate-800/60 border border-slate-700/40 rounded-lg px-2.5 py-1.5 text-xs text-slate-300 max-w-[180px] truncate"
          >
            {devices.cameras.length > 0 ? (
              devices.cameras.map(cam => (
                <option key={cam.index} value={cam.index}>
                  {cam.name}
                </option>
              ))
            ) : (
              <option value={0}>Default Camera</option>
            )}
          </select>
        </div>

        {/* FPS Counter */}
        {isRunning && (
          <div className="flex items-center gap-1.5 px-2.5 py-1 bg-slate-800/40 rounded-lg border border-slate-700/30">
            <div className="w-1.5 h-1.5 rounded-full bg-emerald-500 animate-pulse" />
            <span className="text-xs text-slate-400 tabular-nums font-mono">
              {status.fps.toFixed(1)} FPS
            </span>
          </div>
        )}
      </div>
    </div>
  );
}
