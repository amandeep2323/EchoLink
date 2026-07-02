import { describe, it, expect } from 'vitest';
import { Bone, Quaternion, Euler, MathUtils } from 'three';
import { KinematicsEngine, type QueueItem } from '../KinematicsEngine';
import type { BoneResolution } from '../BoneMap';
import type { PoseFrame } from '../poseStore';

function setup() {
  const bone = new Bone();
  bone.name = 'RightHandIndex1';
  const bones: BoneResolution = { bones: new Map([['RightHandIndex1', bone]]), missing: [] };
  const restPose = new Map([['RightHandIndex1', new Quaternion()]]); // identity rest
  return { bone, bones, restPose };
}

function targetQuat(deg: number): Quaternion {
  return new Quaternion().setFromEuler(new Euler(MathUtils.degToRad(deg), 0, 0, 'XYZ'));
}

function frame(deg: number, holdMs = 50): PoseFrame {
  return { holdMs, quats: new Map([['RightHandIndex1', targetQuat(deg)]]) };
}

/** Advance the engine by total ms in small steps. */
function run(engine: KinematicsEngine, ms: number, step = 16) {
  for (let t = 0; t < ms; t += step) engine.update(step / 1000);
}

describe('KinematicsEngine (Properties 5 & 6)', () => {
  it('Slerps a bone to the target pose within tolerance', () => {
    const { bone, bones, restPose } = setup();
    const engine = new KinematicsEngine(bones, restPose, {}, { blendMs: 100, holdMs: 50 });
    const item: QueueItem = { label: 'A', frames: [frame(85)] };
    engine.play([item]);
    run(engine, 120); // exceed blend
    const target = targetQuat(85);
    expect(bone.quaternion.angleTo(target)).toBeLessThan(0.02);
  });

  it('returns to neutral rest pose after the sequence and goes idle', () => {
    const { bone, bones, restPose } = setup();
    let completed = false;
    const engine = new KinematicsEngine(bones, restPose, { onComplete: () => { completed = true; } }, { blendMs: 50, holdMs: 30 });
    engine.play([{ label: 'A', frames: [frame(85, 30)] }]);
    run(engine, 600);
    expect(completed).toBe(true);
    expect(bone.quaternion.angleTo(new Quaternion())).toBeLessThan(0.02); // back to identity rest
    expect(engine.isPlaying).toBe(false);
  });

  it('skips items with no frames/clip gracefully (Property 6)', () => {
    const { bones, restPose } = setup();
    const engine = new KinematicsEngine(bones, restPose, {}, { blendMs: 30, holdMs: 20 });
    // empty-frame items are filtered out by play(); a valid one still runs
    engine.play([
      { label: 'empty', frames: [] },
      { label: 'A', frames: [frame(45, 20)] },
    ]);
    expect(() => run(engine, 400)).not.toThrow();
    expect(engine.isPlaying).toBe(false);
  });
});
