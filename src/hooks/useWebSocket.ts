import { useState, useEffect, useRef, useCallback } from 'react';
import type {
  SignDetectionData,
  TranscriptData,
  StatusData,
  DeviceList,
  ErrorInfo,
  ConnectionStatus,
  PipelineSettings,
  ModelInfo,
} from '@/types';

const WS_URL = 'ws://127.0.0.1:8765/ws';
const HEALTH_URL = 'http://127.0.0.1:8765/health';
const HEALTH_POLL_INTERVAL = 3000;
const STALE_TIMEOUT = 30000;

export function useWebSocket() {
  const [connectionStatus, setConnectionStatus] = useState<ConnectionStatus>('disconnected');
  const [frame, setFrame] = useState<string | null>(null);
  const [signDetection, setSignDetection] = useState<SignDetectionData | null>(null);
  const [transcript, setTranscript] = useState<TranscriptData>({
    full_text: '',
    latest_word: '',
    is_sentence_complete: false,
  });
  const [status, setStatus] = useState<StatusData>({
    pipeline_running: false,
    model_loaded: false,
    hands_detected: false,
    vcam_active: false,
    vmic_active: false,
    fps: 0,
    frames_processed: 0,
  });
  const [devices, setDevices] = useState<DeviceList>({
    cameras: [],
    audio_output_devices: [],
  });
  const [errors, setErrors] = useState<ErrorInfo[]>([]);
  const [backendOnline, setBackendOnline] = useState(false);
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [activeModelId, setActiveModelId] = useState<string>('');

  const wsRef = useRef<WebSocket | null>(null);
  const healthPollRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const staleTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const mountedRef = useRef(true);

  // ── Error Management ──
  const addError = useCallback((message: string) => {
    setErrors(prev => {
      if (prev.some(e => e.message === message)) return prev;
      const newError: ErrorInfo = {
        id: `${Date.now()}-${Math.random().toString(36).substring(2, 11)}`,
        message,
        timestamp: Date.now(),
      };
      return [...prev, newError];
    });
  }, []);

  const dismissError = useCallback((id: string) => {
    setErrors(prev => prev.filter(e => e.id !== id));
  }, []);

  // ── Stale Connection Timer ──
  const resetStaleTimer = useCallback(() => {
    if (staleTimeoutRef.current) clearTimeout(staleTimeoutRef.current);
    staleTimeoutRef.current = setTimeout(() => {
      addError('Connection appears stale — no messages received for 30 seconds');
    }, STALE_TIMEOUT);
  }, [addError]);

  // ── Send Message (direct, no hook dependency) ──
  const sendRaw = (type: string, data: unknown = null): boolean => {
    const ws = wsRef.current;
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ type, data }));
      return true;
    }
    return false;
  };

  const send = useCallback((type: string, data: unknown = null) => {
    return sendRaw(type, data);
  }, []);

  // ── Health Check ──
  const checkHealth = useCallback(async (): Promise<boolean> => {
    try {
      const res = await fetch(HEALTH_URL, { signal: AbortSignal.timeout(2000) });
      return res.ok;
    } catch {
      return false;
    }
  }, []);

  // ── Connect WebSocket ──
  const connectWs = useCallback((): Promise<boolean> => {
    return new Promise((resolve) => {
      if (wsRef.current?.readyState === WebSocket.OPEN) {
        resolve(true);
        return;
      }

      if (wsRef.current?.readyState === WebSocket.CONNECTING) {
        // Wait for existing connection attempt
        const ws = wsRef.current;
        const onOpen = () => { ws.removeEventListener('open', onOpen); resolve(true); };
        ws.addEventListener('open', onOpen);
        setTimeout(() => { ws.removeEventListener('open', onOpen); resolve(false); }, 3000);
        return;
      }

      setConnectionStatus('connecting');

      try {
        const ws = new WebSocket(WS_URL);
        wsRef.current = ws;

        const timeout = setTimeout(() => {
          ws.close();
          resolve(false);
        }, 3000); // 3s timeout (was 5s)

        ws.onopen = () => {
          clearTimeout(timeout);
          if (!mountedRef.current) { ws.close(); return; }

          setConnectionStatus('connected');
          setBackendOnline(true);
          resetStaleTimer();

          // Stop health polling
          if (healthPollRef.current) {
            clearInterval(healthPollRef.current);
            healthPollRef.current = null;
          }

          // Request devices and models
          ws.send(JSON.stringify({ type: 'get_devices', data: null }));
          ws.send(JSON.stringify({ type: 'get_models', data: null }));

          resolve(true);
        };

        ws.onmessage = (event: MessageEvent) => {
          resetStaleTimer();
          try {
            const msg = JSON.parse(event.data);
            switch (msg.type) {
              case 'preview_frame':
                setFrame(msg.data.frame);
                break;
              case 'sign_detected':
                setSignDetection(msg.data as SignDetectionData);
                break;
              case 'transcript_update':
                setTranscript(msg.data as TranscriptData);
                break;
              case 'status_update':
                setStatus(msg.data as StatusData);
                break;
              case 'device_list':
                setDevices(msg.data as DeviceList);
                break;
              case 'error':
                addError(msg.data.message);
                break;
              case 'model_list': {
                const modelList = (msg.data.models || []) as ModelInfo[];
                setModels(modelList);
                setActiveModelId(msg.data.active_model_id || '');
                break;
              }
              case 'model_switched':
                setActiveModelId(msg.data.model_id || '');
                setModels(prev => prev.map(m => ({
                  ...m,
                  active: m.id === msg.data.model_id,
                })));
                break;
            }
          } catch {
            // Ignore parse errors
          }
        };

        ws.onclose = () => {
          clearTimeout(timeout);
          if (!mountedRef.current) return;

          setConnectionStatus('disconnected');
          // Don't set backendOnline=false here — health poll will determine that
          if (staleTimeoutRef.current) clearTimeout(staleTimeoutRef.current);

          // Start health polling to detect backend coming back
          if (!healthPollRef.current && mountedRef.current) {
            healthPollRef.current = setInterval(async () => {
              if (!mountedRef.current) return;
              const online = await checkHealth();
              setBackendOnline(online);
              if (online && wsRef.current?.readyState !== WebSocket.OPEN) {
                connectWs();
              }
            }, HEALTH_POLL_INTERVAL);
          }
        };

        ws.onerror = () => {
          // onclose fires after this
        };
      } catch {
        setConnectionStatus('disconnected');
        resolve(false);
      }
    });
  }, [addError, resetStaleTimer, checkHealth]);

  // ── Pipeline Controls ──

  const startPipeline = useCallback(
    async (settings: PipelineSettings) => {
      // Fast path: already connected
      if (wsRef.current?.readyState === WebSocket.OPEN) {
        sendRaw('start_pipeline', settings);
        return;
      }

      // Not connected — connect first, then send
      const connected = await connectWs();
      if (connected) {
        // Small delay to ensure WS is fully ready
        setTimeout(() => {
          sendRaw('start_pipeline', settings);
        }, 50);
      } else {
        addError('Cannot connect to backend — make sure the Python server is running');
      }
    },
    [connectWs, addError]
  );

  const stopPipeline = useCallback(() => {
    send('stop_pipeline', null);
  }, [send]);

  const updateSettings = useCallback(
    (settings: Partial<PipelineSettings>) => {
      send('update_settings', settings);
    },
    [send]
  );

  const requestDevices = useCallback(() => {
    send('get_devices', null);
  }, [send]);

  const clearTranscript = useCallback(() => {
    send('clear_transcript', null);
    setTranscript({ full_text: '', latest_word: '', is_sentence_complete: false });
    setSignDetection(null);
  }, [send]);

  const switchModel = useCallback((modelId: string) => {
    send('switch_model', { model_id: modelId });
  }, [send]);

  const getModels = useCallback(() => {
    send('get_models', null);
  }, [send]);

  // ── Mount: initial health check → connect or poll ──
  useEffect(() => {
    mountedRef.current = true;

    (async () => {
      const online = await checkHealth();
      if (!mountedRef.current) return;

      setBackendOnline(online);

      if (online) {
        connectWs();
      } else {
        // Start silent background polling
        healthPollRef.current = setInterval(async () => {
          if (!mountedRef.current) return;
          const isOnline = await checkHealth();
          setBackendOnline(isOnline);
          if (isOnline && wsRef.current?.readyState !== WebSocket.OPEN) {
            connectWs();
          }
        }, HEALTH_POLL_INTERVAL);
      }
    })();

    return () => {
      mountedRef.current = false;
      if (healthPollRef.current) {
        clearInterval(healthPollRef.current);
        healthPollRef.current = null;
      }
      if (staleTimeoutRef.current) clearTimeout(staleTimeoutRef.current);
      wsRef.current?.close();
    };
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  // Auto-dismiss errors after 8 seconds
  useEffect(() => {
    if (errors.length === 0) return;
    const timers = errors.map(error =>
      setTimeout(() => dismissError(error.id), 8000)
    );
    return () => timers.forEach(t => clearTimeout(t));
  }, [errors, dismissError]);

  return {
    connectionStatus,
    backendOnline,
    frame,
    signDetection,
    transcript,
    status,
    devices,
    errors,
    models,
    activeModelId,
    dismissError,
    startPipeline,
    stopPipeline,
    updateSettings,
    requestDevices,
    clearTranscript,
    switchModel,
    getModels,
  };
}
