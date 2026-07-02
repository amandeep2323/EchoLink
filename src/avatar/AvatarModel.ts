/**
 * AvatarModel — loads the Ready Player Me GLB humanoid using the native
 * Three.js GLTFLoader, resolves its bone map, and captures the neutral rest
 * pose so the kinematics engine can return to it.
 */

import { Group, Object3D, Quaternion, type AnimationClip } from 'three';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';
import { resolveBoneMap, type BoneResolution } from './BoneMap';

export interface LoadedAvatar {
  root: Group;
  bones: BoneResolution;
  /** Rest-pose local quaternion per logical bone name. */
  restPose: Map<string, Quaternion>;
  animations: AnimationClip[];
}

const DEFAULT_GLB = 'avatar/avatar.glb';

export async function loadAvatar(url: string = DEFAULT_GLB): Promise<LoadedAvatar> {
  const loader = new GLTFLoader();
  const gltf = await loader.loadAsync(url);
  const root = gltf.scene as Group;

  const bones = resolveBoneMap(root);

  // Capture neutral rest pose from the freshly loaded skeleton.
  const restPose = new Map<string, Quaternion>();
  bones.bones.forEach((bone, logical) => {
    restPose.set(logical, bone.quaternion.clone());
  });

  return {
    root,
    bones,
    restPose,
    animations: gltf.animations as AnimationClip[],
  };
}

/** Center/scale helper so the avatar frames nicely in the widget. */
export function frameAvatar(root: Object3D): void {
  root.position.set(0, 0, 0);
}
