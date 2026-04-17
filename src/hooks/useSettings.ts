import { useState, useCallback, useEffect } from 'react';
import type { PipelineSettings } from '@/types';

const STORAGE_KEY = 'signspeak-settings';

const DEFAULT_SETTINGS: PipelineSettings = {
  camera_index: 0,
  resolution: [640, 480],
  fps: 30,
  show_landmarks: true,
  show_overlay: true,
  tts_enabled: false,
  vcam_enabled: false,
  vcam_mirror: false,
  vmic_enabled: false,
  vmic_device: '',
  confidence_threshold: 0.6,
  tts_voice: 'en_US-lessac-medium',
  audio_output_device: '',
};

export function useSettings() {
  const [settings, setSettings] = useState<PipelineSettings>(() => {
    try {
      const stored = localStorage.getItem(STORAGE_KEY);
      if (stored) {
        return { ...DEFAULT_SETTINGS, ...JSON.parse(stored) };
      }
    } catch {
      // Ignore parse errors, use defaults
    }
    return DEFAULT_SETTINGS;
  });

  // Persist to localStorage on change
  useEffect(() => {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(settings));
    } catch {
      // Ignore storage errors
    }
  }, [settings]);

  const updateSettings = useCallback((updates: Partial<PipelineSettings>) => {
    setSettings(prev => ({ ...prev, ...updates }));
  }, []);

  const resetSettings = useCallback(() => {
    setSettings(DEFAULT_SETTINGS);
    try {
      localStorage.removeItem(STORAGE_KEY);
    } catch {
      // Ignore
    }
  }, []);

  return { settings, updateSettings, resetSettings, DEFAULT_SETTINGS };
}
