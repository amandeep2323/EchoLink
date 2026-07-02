/**
 * Bone-map resolver for Mixamo / Ready Player Me humanoid skeletons.
 *
 * RPM GLB skeletons use Mixamo-style joint names WITHOUT the `mixamorig:`
 * prefix; Mixamo FBX/GLB exports use the `mixamorig:` prefix. We normalize by
 * stripping a leading `mixamorig:` / `mixamorig` before comparison so the same
 * canonical map works for both.
 */

import type { Object3D, Bone } from 'three';

const FINGERS = ['Thumb', 'Index', 'Middle', 'Ring', 'Pinky'] as const;

/** Build the canonical logical bone-name list for one side. */
function sideBones(side: 'Left' | 'Right'): string[] {
  const names: string[] = [
    `${side}Shoulder`,
    `${side}Arm`,
    `${side}ForeArm`,
    `${side}Hand`,
  ];
  for (const finger of FINGERS) {
    for (let seg = 1; seg <= 3; seg++) {
      names.push(`${side}Hand${finger}${seg}`);
    }
  }
  return names;
}

/** All canonical bone names targeted by the kinematics engine (both arms). */
export const CANONICAL_BONES: string[] = [
  ...sideBones('Left'),
  ...sideBones('Right'),
];

/** Normalize a skeleton bone name for matching (strip mixamorig prefix). */
export function normalizeBoneName(name: string): string {
  return name.replace(/^mixamorig[:_]?/i, '');
}

export interface BoneResolution {
  bones: Map<string, Bone>;
  missing: string[];
}

/**
 * Walk the loaded model's hierarchy and resolve each canonical bone name to a
 * THREE.Bone. Tries the exact name, then a `mixamorig:`-prefixed variant.
 * Missing bones are reported (not fatal).
 */
export function resolveBoneMap(root: Object3D): BoneResolution {
  // Index every node by its normalized name for O(1) lookup.
  const byNormalized = new Map<string, Bone>();
  root.traverse((obj) => {
    const asBone = obj as Bone;
    if (obj.name) {
      const norm = normalizeBoneName(obj.name);
      if (!byNormalized.has(norm)) {
        byNormalized.set(norm, asBone);
      }
    }
  });

  const bones = new Map<string, Bone>();
  const missing: string[] = [];

  for (const logical of CANONICAL_BONES) {
    const bone = byNormalized.get(logical);
    if (bone) {
      bones.set(logical, bone);
    } else {
      missing.push(logical);
    }
  }

  if (missing.length > 0) {
    console.warn(
      `[BoneMap] ${missing.length} bone(s) not found on the loaded model:`,
      missing.join(', ')
    );
  }

  return { bones, missing };
}
