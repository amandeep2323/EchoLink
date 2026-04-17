import type { StatusData, ConnectionStatus } from '@/types';
import { cn } from '@/utils/cn';

interface StatusIndicatorProps {
  connectionStatus: ConnectionStatus;
  backendOnline: boolean;
  status: StatusData;
}

interface StatusDot {
  label: string;
  color: string;
  pulse: boolean;
  title: string;
}

export function StatusIndicator({ connectionStatus, backendOnline, status }: StatusIndicatorProps) {
  const getBackendStatus = (): { color: string; pulse: boolean; title: string; label: string } => {
    if (connectionStatus === 'connected') {
      return { color: 'bg-emerald-500', pulse: false, title: 'Backend Connected', label: 'Backend' };
    }
    if (connectionStatus === 'connecting') {
      return { color: 'bg-amber-500', pulse: true, title: 'Connecting...', label: 'Backend' };
    }
    if (backendOnline) {
      return { color: 'bg-amber-500', pulse: true, title: 'Backend Online — Connecting...', label: 'Backend' };
    }
    return { color: 'bg-slate-600', pulse: false, title: 'Backend Offline', label: 'Backend' };
  };

  const backend = getBackendStatus();

  const dots: StatusDot[] = [
    {
      label: backend.label,
      color: backend.color,
      pulse: backend.pulse,
      title: backend.title,
    },
    {
      label: 'Pipeline',
      color: status.pipeline_running ? 'bg-emerald-500' : 'bg-slate-600',
      pulse: status.pipeline_running,
      title: status.pipeline_running ? 'Pipeline Running' : 'Pipeline Stopped',
    },
    {
      label: 'Model',
      color: status.model_loaded ? 'bg-emerald-500' : 'bg-slate-600',
      pulse: false,
      title: status.model_loaded ? 'Model Loaded' : 'Model Not Loaded',
    },
    {
      label: 'Hands',
      color: status.hands_detected ? 'bg-violet-500' : 'bg-slate-600',
      pulse: status.hands_detected,
      title: status.hands_detected ? 'Hands Detected' : 'No Hands Detected',
    },
    {
      label: 'VCam',
      color: status.vcam_active ? 'bg-blue-500' : 'bg-slate-600',
      pulse: false,
      title: status.vcam_active ? 'Virtual Camera Active' : 'Virtual Camera Inactive',
    },
    {
      label: 'VMic',
      color: status.vmic_active ? 'bg-blue-500' : 'bg-slate-600',
      pulse: false,
      title: status.vmic_active ? 'Virtual Mic Active' : 'Virtual Mic Inactive',
    },
  ];

  return (
    <div className="hidden md:flex items-center gap-4">
      {dots.map(dot => (
        <div
          key={dot.label}
          className="flex items-center gap-1.5 cursor-default"
          title={dot.title}
        >
          <div className="relative">
            <div
              className={cn(
                'w-2 h-2 rounded-full transition-colors duration-300',
                dot.color
              )}
            />
            {dot.pulse && (
              <div
                className={cn(
                  'absolute inset-0 w-2 h-2 rounded-full animate-ping opacity-40',
                  dot.color
                )}
              />
            )}
          </div>
          <span className="text-[10px] font-medium text-slate-500 uppercase tracking-wider">
            {dot.label}
          </span>
        </div>
      ))}
    </div>
  );
}
