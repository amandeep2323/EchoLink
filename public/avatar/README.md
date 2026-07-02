# Avatar GLB assets

Drop the Ready Player Me **GLB** humanoid model here as `avatar.glb`.

Requirements for the model:
- Exported as **GLB** (binary glTF).
- Includes a **full humanoid skeleton with finger bones** (RPM "fullbody"/"halfbody"
  with hands). The kinematics engine targets Mixamo/RPM bone names such as
  `LeftHand`, `LeftHandIndex1/2/3`, `LeftHandThumb1/2/3`, `RightForeArm`, etc.
- A `mixamorig:` name prefix is fine — the bone resolver strips it automatically.

The frontend loads `public/avatar/avatar.glb` via the native Three.js `GLTFLoader`.
If you use a different filename, update the loader path in `src/avatar/AvatarModel.ts`.
