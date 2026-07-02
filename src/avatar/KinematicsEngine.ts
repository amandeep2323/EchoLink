/**
 * KinematicsEngine — drives avatar bones from a queue of resolved pose frames
 * using quaternion Slerp. State machine per frame: BLEND_IN -> HOLD -> next.
 * At the end of a sequence the avatar blends back to the neutral rest pose.
 *
 * Whole-word GLB clips are played via AnimationMixer (cross-faded); pose-JSON
 * is the MVP path. Missing poses/clips are skipped gracefully.
 */

import { Quaternion, AnimationMixer, type AnimationClip } from 'three';
import type { BoneResolution } from './BoneMap';
import type { Pose, PoseFrame } from './poseStore';

export interface EngineConfig {
  blendMs: number;
  holdMs: number;
}

export interface EngineEvents {
  onTokenStart?: (index: number, label: string) => void;
  onComplete?: () => void;
}

interface QueueItem {
  label: string;
  frames: PoseFrame[];
  clip?: AnimationClip;
}

type Phase = 'idle' | 'blend' | 'hold';

const DEFAULT_CONFIG: EngineConfig = { blendMs: 180, holdMs: 350 };

export class KinematicsEngine {
  private bones: BoneResolution;
  private restPose: Map<string, Quaternion>;
  private cfg: EngineConfig;
  private events: EngineEvents;
  private mixer?: AnimationMixer;

  private queue: QueueItem[] = [];
  private itemIndex = -1;
  private frameIndex = 0;
  private phase: Phase = 'idle';
  private elapsed = 0;
  private fromQuats = new Map<string, Quaternion>();
  private targetQuats = new Map<string, Quaternion>();

  constructor(
    bones: BoneResolution,
    restPose: Map<string, Quaternion>,
    events: EngineEvents = {},
    cfg: Partial<EngineConfig> = {}
  ) {
    this.bones = bones;
    this.restPose = restPose;
    this.events = events;
    this.cfg = { ...DEFAULT_CONFIG, ...cfg };
  }

  get isPlaying(): boolean {
    return this.phase !== 'idle';
  }

  /** Replace the queue and start playing immediately. */
  play(items: QueueItem[]): void {
    this.queue = items.filter((it) => (it.frames && it.frames.length) || it.clip);
    this.itemIndex = -1;
    this.advanceItem();
  }

  /** Stop and return to rest. */
  stop(): void {
    this.queue = [];
    this.beginBlendTo(this.restPose, 'rest');
  }

  private advanceItem(): void {
    this.itemIndex += 1;
    this.frameIndex = 0;
    if (this.itemIndex >= this.queue.length) {
      // Sequence finished — blend back to rest, then idle.
      this.beginBlendTo(this.restPose, 'rest');
      this.queue = [];
      this.events.onComplete?.();
      return;
    }
    const item = this.queue[this.itemIndex];
    this.events.onTokenStart?.(this.itemIndex, item.label);
    if (item.clip && this.mixer) {
      const action = this.mixer.clipAction(item.clip);
      action.reset().play();
    }
    this.beginFrame(item.frames[0]);
  }

  private beginFrame(frame?: PoseFrame): void {
    if (!frame) {
      this.advanceItem();
      return;
    }
    this.beginBlendTo(frame.quats, 'frame', frame.holdMs);
  }

  private pendingHoldMs = 350;

  private beginBlendTo(targets: Map<string, Quaternion>, _kind: string, holdMs?: number): void {
    this.fromQuats.clear();
    this.targetQuats.clear();
    this.bones.bones.forEach((bone, logical) => {
      const target = targets.get(logical) ?? this.restPose.get(logical);
      if (target) {
        this.fromQuats.set(logical, bone.quaternion.clone());
        this.targetQuats.set(logical, target.clone());
      }
    });
    this.pendingHoldMs = holdMs ?? this.cfg.holdMs;
    this.elapsed = 0;
    this.phase = 'blend';
  }

  /** Call every render tick with delta seconds. */
  update(deltaSeconds: number): void {
    if (this.mixer) this.mixer.update(deltaSeconds);
    if (this.phase === 'idle') return;

    this.elapsed += deltaSeconds * 1000;

    if (this.phase === 'blend') {
      const t = Math.min(1, this.elapsed / Math.max(1, this.cfg.blendMs));
      this.fromQuats.forEach((from, logical) => {
        const to = this.targetQuats.get(logical)!;
        const bone = this.bones.bones.get(logical)!;
        bone.quaternion.copy(from).slerp(to, t);
      });
      if (t >= 1) {
        this.phase = 'hold';
        this.elapsed = 0;
      }
      return;
    }

    if (this.phase === 'hold') {
      if (this.elapsed >= this.pendingHoldMs) {
        // Next frame in current item, or next item.
        const item = this.queue[this.itemIndex];
        if (item && this.frameIndex + 1 < item.frames.length) {
          this.frameIndex += 1;
          this.beginFrame(item.frames[this.frameIndex]);
        } else if (this.queue.length > 0) {
          this.advanceItem();
        } else {
          this.phase = 'idle';
        }
      }
    }
  }

  attachMixer(mixer: AnimationMixer): void {
    this.mixer = mixer;
  }
}

export type { QueueItem };
export type ResolvedPoseItem = { pose: Pose; label: string };
