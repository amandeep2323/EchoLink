# Fingerspelling & clip pose format

This folder holds the **Fingerspelling_Set** (one JSON per letter A–Z and digits)
and the **clip manifest** for whole-word animations.

## File naming

- Letters: `letter_A.json` … `letter_Z.json`
- Digits (optional): `digit_0.json` … `digit_9.json`
- Whole-word pose clips (optional): `word_HELLO.json` (multi-frame, see below)
- `clip_manifest.json` — lists which whole words have authored clips

Use `letter_A_template.json` as the starting point for each letter.

## Pose JSON schema (single handshape)

```json
{
  "id": "letter_A",
  "side": "right",                 // "right" | "left" | "both"
  "description": "free text",
  "rotations": {
    "<BoneName>": [x, y, z]         // LOCAL rotation in EULER DEGREES (XYZ order)
  }
}
```

- Rotations are **local** bone rotations in **degrees**, applied in XYZ Euler order.
  The engine converts them to quaternions at load and Slerps toward them.
- Only include bones you want to drive; omitted bones keep their rest rotation.
- Bone names follow Mixamo/Ready Player Me conventions (no `mixamorig:` prefix
  needed — it is stripped automatically). Per hand:
  - Wrist: `RightHand` / `LeftHand`
  - Thumb: `…HandThumb1/2/3`
  - Index: `…HandIndex1/2/3`
  - Middle: `…HandMiddle1/2/3`
  - Ring: `…HandRing1/2/3`
  - Pinky: `…HandPinky1/2/3`
  - Arm: `…Arm`, `…ForeArm`, `…Shoulder`

## Multi-frame clip schema (whole word, optional)

```json
{
  "id": "word_HELLO",
  "side": "right",
  "frames": [
    { "holdMs": 250, "rotations": { "RightHandIndex1": [80, 0, 0] } },
    { "holdMs": 250, "rotations": { "RightHandIndex1": [10, 0, 0] } }
  ]
}
```

Then register it in `clip_manifest.json`:

```json
{ "words": { "HELLO": { "clipId": "word_HELLO", "kind": "pose", "file": "word_HELLO.json" } } }
```

For GLB animation clips instead of pose JSON, use `"kind": "gltf"` and point
`file` at a `.glb` containing the named animation.
