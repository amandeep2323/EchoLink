/**
 * Pose store — loads the Fingerspelling_Set and whole-word clip manifest from
 * `public/poses/`, converting authored Euler degrees into quaternions.
 */

import { Euler, Quaternion, MathUtils } from 'three';

export type PoseSide = 'left' | 'right' | 'both';

/** Raw authored pose JSON (Euler degrees per bone). */
export interface RawPose {
  id: string;
  side?: PoseSide;
  description?: string;
  rotations?: Record<string, [number, number, number]>;
  frames?: Array<{ holdMs?: number; rotations: Record<string, [number, number, number]> }>;
}

/** A single resolved keyframe: local quaternion per logical bone name. */
export interface PoseFrame {
  holdMs: number;
  quats: Map<string, Quaternion>;
}

/** A resolved pose/clip ready for the kinematics engine. */
export interface Pose {
  id: string;
  side: PoseSide;
  frames: PoseFrame[];
}

export interface ClipManifestEntry {
  clipId: string;
  kind: 'pose' | 'gltf';
  file: string;
}

export interface ClipManifest {
  words: Record<string, ClipManifestEntry>;
}

const POSES_BASE = 'poses';

function degToQuat(deg: [number, number, number]): Quaternion {
  const e = new Euler(
    MathUtils.degToRad(deg[0]),
    MathUtils.degToRad(deg[1]),
    MathUtils.degToRad(deg[2]),
    'XYZ'
  );
  return new Quaternion().setFromEuler(e);
}

function toFrames(raw: RawPose): PoseFrame[] {
  const rawFrames = raw.frames ?? (raw.rotations ? [{ holdMs: 350, rotations: raw.rotations }] : []);
  return rawFrames.map((f) => {
    const quats = new Map<string, Quaternion>();
    for (const [bone, deg] of Object.entries(f.rotations ?? {})) {
      quats.set(bone, degToQuat(deg));
    }
    return { holdMs: f.holdMs ?? 350, quats };
  });
}

async function fetchJson<T>(path: string): Promise<T | null> {
  try {
    const res = await fetch(path, { cache: 'no-cache' });
    if (!res.ok) return null;
    return (await res.json()) as T;
  } catch {
    return null;
  }
}

/** Load one pose JSON by file name (e.g. "letter_A.json"). */
export async function loadPoseFile(file: string): Promise<Pose | null> {
  const raw = await fetchJson<RawPose>(`${POSES_BASE}/${file}`);
  if (!raw) return null;
  return { id: raw.id, side: raw.side ?? 'right', frames: toFrames(raw) };
}

/** Load the fingerspelling set A–Z (+ digits) that are present. */
export async function loadFingerspellingSet(): Promise<Map<string, Pose>> {
  const map = new Map<string, Pose>();
  const letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'.split('');
  const digits = '0123456789'.split('');
  const targets: Array<[string, string]> = [
    ...letters.map((c) => [c, `letter_${c}.json`] as [string, string]),
    ...digits.map((d) => [d, `digit_${d}.json`] as [string, string]),
  ];
  await Promise.all(
    targets.map(async ([char, file]) => {
      const pose = await loadPoseFile(file);
      if (pose) map.set(char, pose);
    })
  );
  return map;
}

/** Load the clip manifest (whole-word availability). */
export async function loadClipManifest(): Promise<ClipManifest> {
  const m = await fetchJson<ClipManifest>(`${POSES_BASE}/clip_manifest.json`);
  return m ?? { words: {} };
}
