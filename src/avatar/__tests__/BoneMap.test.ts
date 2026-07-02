import { describe, it, expect, vi } from 'vitest';
import { Object3D, Bone } from 'three';
import { resolveBoneMap, normalizeBoneName, CANONICAL_BONES } from '../BoneMap';

function makeSkeleton(names: string[]): Object3D {
  const root = new Object3D();
  for (const n of names) {
    const b = new Bone();
    b.name = n;
    root.add(b);
  }
  return root;
}

describe('BoneMap resolver (Property 4: prefix-invariance)', () => {
  it('resolves canonical RPM names without prefix', () => {
    const root = makeSkeleton(CANONICAL_BONES);
    const { bones, missing } = resolveBoneMap(root);
    expect(missing).toHaveLength(0);
    expect(bones.get('RightHandIndex1')).toBeInstanceOf(Bone);
  });

  it('resolves mixamorig:-prefixed names', () => {
    const root = makeSkeleton(CANONICAL_BONES.map((n) => `mixamorig:${n}`));
    const { bones, missing } = resolveBoneMap(root);
    expect(missing).toHaveLength(0);
    expect(bones.get('LeftHandThumb3')).toBeInstanceOf(Bone);
  });

  it('reports missing bones without throwing', () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const partial = CANONICAL_BONES.slice(0, 5);
    const { bones, missing } = resolveBoneMap(makeSkeleton(partial));
    expect(bones.size).toBe(5);
    expect(missing.length).toBe(CANONICAL_BONES.length - 5);
    expect(warn).toHaveBeenCalled();
    warn.mockRestore();
  });

  it('normalizes prefixes', () => {
    expect(normalizeBoneName('mixamorig:RightHand')).toBe('RightHand');
    expect(normalizeBoneName('mixamorigRightHand')).toBe('RightHand');
    expect(normalizeBoneName('RightHand')).toBe('RightHand');
  });
});
