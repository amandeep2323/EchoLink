# Model3 Input Format Issue

## Resolution Update (2026-05-31)

This document is now partially superseded.

Confirmed behavior from source code and ONNX inspection:
- Model input is `[..., 543, 3]` where `543` = holistic landmarks per frame.
- The first axis is temporal length (`T`), so runtime input should be `[T, 543, 3]`.

The earlier interpretation ("543 frames with only 3 aggregate features") was incorrect.

## Problem
Model3 expects input shape `[batch, 543, 3]` but the application is providing `[batch, 55, 1086]`.

## Root Cause
The model from jamesjbustos/sign-language-recognition was trained on a **completely different input format** than what we assumed.

### What We Assumed:
- Input: 55 keypoints × 543 frames × 2 coords = `[1, 55, 1086]`
- Format: Nodes-first (keypoints across time)

### What the Model Actually Expects:
- Input: 543 frames × 3 features = `[1, 543, 3]`
- Format: Frames-first with only 3 features per frame

## What Are These 3 Features?

Looking at the jamesjbustos model, it likely uses:
1. **Hand position X** (average or dominant hand)
2. **Hand position Y** (average or dominant hand)  
3. **Hand movement/velocity** or **Z-depth**

This is a **much simpler** feature representation than full 55-keypoint pose data.

## Why This Happened

The model was trained on a simplified feature extraction:
- Instead of using all 55 keypoints
- It extracts just 3 aggregate features per frame
- This makes the model smaller and faster
- But requires different preprocessing

## Solutions

### Option 1: Use Model3 As-Is (Requires Custom Preprocessing)

**What's Needed**:
1. Extract only 3 features per frame instead of 55 keypoints
2. Possible features:
   - Dominant hand centroid (x, y)
   - Hand movement velocity
   - Or hand bounding box center + size

**Pros**:
- Model3 will work correctly
- Simpler input = faster processing

**Cons**:
- Need to implement custom feature extraction
- Less information than full pose
- May be less accurate

### Option 2: Find/Use a Different Model3

**What's Needed**:
- Find an ASL model that uses MediaPipe Holistic keypoints
- Or use a model with similar input format to Model1/Model2

**Pros**:
- Consistent with existing infrastructure
- No custom preprocessing needed

**Cons**:
- Hard to find suitable models
- May not have 250-sign vocabulary

### Option 3: Disable Model3

**What's Needed**:
- Just use Model1 (fingerspelling) for now
- Wait until we find a compatible model

**Pros**:
- Model1 works perfectly
- No wasted effort on incompatible model

**Cons**:
- No word-level recognition
- Only fingerspelling available

## Recommendation

**Disable Model3 for now** and focus on getting Model2 working with OpenPose.

### Rationale:
1. Model3's input format is incompatible with our MediaPipe-based system
2. Implementing custom 3-feature extraction is non-trivial
3. Model2 with OpenPose would give us 2000 words (much better than Model3's 250)
4. Model1 (fingerspelling) already works perfectly

## Next Steps

1. **Disable Model3** - Mark as incompatible
2. **Implement OpenPose for Model2** - This will give us:
   - Model1: Fingerspelling (24 letters) ✅
   - Model2: Word recognition (2000 signs) with OpenPose
3. **Document the issue** - Explain why Model3 doesn't work

---

**Date**: 2026-05-27
**Status**: Incompatible Input Format
**Resolution**: Disable Model3, implement OpenPose for Model2 instead
