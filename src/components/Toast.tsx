import { useState, useEffect, useCallback, useRef } from 'react';
import type { Toast as ToastType } from '@/types';
import { cn } from '@/utils/cn';

// ── Toast Hook ──
export function useToast() {
  const [toasts, setToasts] = useState<ToastType[]>([]);

  const addToast = useCallback((toast: Omit<ToastType, 'id'>) => {
    const id = `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
    setToasts(prev => {
      // Deduplicate: don't stack identical messages
      if (prev.some(t => t.message === toast.message)) return prev;
      // Max 5 toasts visible
      const next = [...prev, { ...toast, id }];
      return next.length > 5 ? next.slice(-5) : next;
    });
    return id;
  }, []);

  const removeToast = useCallback((id: string) => {
    setToasts(prev => prev.filter(t => t.id !== id));
  }, []);

  return { toasts, addToast, removeToast };
}

// ── Single Toast Item ──
function ToastItem({
  toast,
  onRemove,
}: {
  toast: ToastType;
  onRemove: (id: string) => void;
}) {
  const [exiting, setExiting] = useState(false);
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const dismiss = useCallback(() => {
    setExiting(true);
    setTimeout(() => onRemove(toast.id), 300);
  }, [onRemove, toast.id]);

  useEffect(() => {
    const dur = toast.duration ?? 4000;
    timerRef.current = setTimeout(dismiss, dur);
    return () => {
      if (timerRef.current) clearTimeout(timerRef.current);
    };
  }, [dismiss, toast.duration]);

  const styleMap = {
    success: {
      bg: 'bg-emerald-500/10 border-emerald-500/25',
      icon: toast.icon ?? '✓',
      iconColor: 'text-emerald-400 bg-emerald-500/20',
      text: 'text-emerald-200',
    },
    info: {
      bg: 'bg-blue-500/10 border-blue-500/25',
      icon: toast.icon ?? 'ℹ',
      iconColor: 'text-blue-400 bg-blue-500/20',
      text: 'text-blue-200',
    },
    warning: {
      bg: 'bg-amber-500/10 border-amber-500/25',
      icon: toast.icon ?? '⚠',
      iconColor: 'text-amber-400 bg-amber-500/20',
      text: 'text-amber-200',
    },
    error: {
      bg: 'bg-red-500/10 border-red-500/25',
      icon: toast.icon ?? '✕',
      iconColor: 'text-red-400 bg-red-500/20',
      text: 'text-red-200',
    },
  };

  const s = styleMap[toast.type];

  return (
    <div
      className={cn(
        'flex items-center gap-3 px-4 py-3 rounded-xl border backdrop-blur-md shadow-lg transition-all duration-300',
        s.bg,
        exiting ? 'opacity-0 translate-x-8 scale-95' : 'opacity-100 translate-x-0 scale-100',
        'animate-toast-in'
      )}
    >
      <div className={cn('w-7 h-7 rounded-lg flex items-center justify-center text-sm shrink-0', s.iconColor)}>
        {s.icon}
      </div>
      <span className={cn('text-sm font-medium flex-1', s.text)}>
        {toast.message}
      </span>
      <button
        onClick={dismiss}
        className="text-slate-500 hover:text-slate-300 transition-colors shrink-0 p-0.5"
      >
        <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
          <path strokeLinecap="round" strokeLinejoin="round" d="M6 18 18 6M6 6l12 12" />
        </svg>
      </button>
    </div>
  );
}

// ── Toast Container ──
export function ToastContainer({
  toasts,
  onRemove,
}: {
  toasts: ToastType[];
  onRemove: (id: string) => void;
}) {
  if (toasts.length === 0) return null;

  return (
    <div className="fixed bottom-6 right-6 z-[100] flex flex-col gap-2 max-w-sm w-full pointer-events-none">
      {toasts.map(toast => (
        <div key={toast.id} className="pointer-events-auto">
          <ToastItem toast={toast} onRemove={onRemove} />
        </div>
      ))}
    </div>
  );
}
