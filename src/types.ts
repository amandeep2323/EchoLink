export interface PipelineSettings {
  camera_index: number;
  resolution: [number, number];
  fps: number;
  show_landmarks: boolean;
  show_overlay: boolean;
  tts_enabled: boolean;
  vcam_enabled: boolean;
  vcam_mirror: boolean;
  vmic_enabled: boolean;
  vmic_device: string;
  confidence_threshold: number;
  tts_voice: string;
  audio_output_device: string;
}

export interface SignDetectionData {
  sign: string;
  confidence: number;
  smoothed_confidence: number;
  letter_added: boolean;
  top_3: Array<{ sign: string; confidence: number }>;
}

export interface TranscriptData {
  full_text: string;
  latest_word: string;
  is_sentence_complete: boolean;
}

export interface StatusData {
  pipeline_running: boolean;
  model_loaded: boolean;
  hands_detected: boolean;
  vcam_active: boolean;
  vmic_active: boolean;
  fps: number;
  frames_processed: number;
}

export interface DeviceList {
  cameras: Array<{ index: number; name: string }>;
  audio_output_devices: Array<{ index: number; name: string }>;
}

export interface ErrorInfo {
  id: string;
  message: string;
  timestamp: number;
}

export interface ModelInfo {
  id: string;
  name: string;
  description: string;
  type: string;
  active: boolean;
  model_type: string;
  labels_count: number;
}

export type ConnectionStatus = 'disconnected' | 'connecting' | 'connected' | 'reconnecting';

export interface Toast {
  id: string;
  message: string;
  type: 'success' | 'info' | 'warning' | 'error';
  icon?: string;
  duration?: number;
}
