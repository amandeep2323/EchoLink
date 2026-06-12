# Intel OPEA TTS Setup Instructions

## Overview

The auto-download feature has been removed from the OPEA TTS module. Users must now run a setup script to download models before using OPEA TTS.

## Changes Made

### 1. Removed Auto-Download from Code

**Files Modified:**
- `python-backend/src/speech/opea_tts/synthesizer.py`
  - Removed `download_models_if_needed()` call from `load()` method
  - Added `_verify_models_cached()` method to check if models exist
  - Now shows clear error message if models are missing

**Before:**
```python
# Step 1: Ensure models are downloaded
success, message = download_models_if_needed(self.model_dir)
if not success:
    print(f"[OPEA TTS] ✗ Model download failed: {message}")
    return False
```

**After:**
```python
# Step 1: Verify models exist (no auto-download)
if not self._verify_models_cached():
    print(f"[OPEA TTS] ✗ Models not found in: {self.model_dir}")
    print("[OPEA TTS] Run the setup script first:")
    print("[OPEA TTS]   python-backend/scripts/setup_opea_tts_models.bat")
    return False
```

### 2. Created Setup Script

**New File:**
`python-backend/scripts/setup_opea_tts_models.bat`

**What it does:**
1. Checks Python installation
2. Verifies directory structure
3. Creates model cache directory
4. Installs required dependencies (transformers, torch, sentencepiece)
5. Downloads models from HuggingFace:
   - `microsoft/speecht5_tts` (Main TTS model, ~585MB)
   - `microsoft/speecht5_hifigan` (Vocoder, ~50MB)
6. Downloads speaker embeddings from Intel repo
7. Verifies installation

**Total download size:** ~650MB (larger than initially estimated due to full PyTorch model)

## How to Use

### For Users:

1. **Run the setup script:**
   ```batch
   cd "asl-speech-desktop-app v6 complete"
   python-backend\scripts\setup_opea_tts_models.bat
   ```

2. **Wait for download to complete:**
   - The script will download ~650MB from HuggingFace
   - Progress bars will show download status
   - This may take 5-15 minutes depending on internet speed

3. **Verify installation:**
   - Script will automatically verify all models downloaded
   - Look for "SUCCESS — OPEA TTS Models Installed" message

4. **Start using OPEA TTS:**
   - Models are now cached at: `python-backend/models/tts/opea_speecht5/`
   - EchoLink will automatically use OPEA TTS if OpenVINO is detected
   - No further downloads needed

### For Developers:

**Testing if models are installed:**
```python
from src.speech.opea_tts import OpeaTtsSynthesizer

synth = OpeaTtsSynthesizer(model_dir="models/tts/opea_speecht5")
success = synth.load()

if success:
    print("✓ OPEA TTS ready!")
else:
    print("✗ Models not found — run setup script")
```

**Error message when models missing:**
```
[OPEA TTS] ✗ Models not found in: models/tts/opea_speecht5
[OPEA TTS] Run the setup script first:
[OPEA TTS]   python-backend/scripts/setup_opea_tts_models.bat
```

## Benefits of Manual Setup

### 1. **User Control**
- Users decide when to download (not during first app launch)
- Can run setup during off-peak hours
- Can pause/resume if needed (via script re-run)

### 2. **Better Error Handling**
- Clear progress indication during download
- Detailed error messages if download fails
- Easy to retry if something goes wrong

### 3. **Offline Readiness**
- Once downloaded, works completely offline
- No surprise downloads during app usage
- Models cached permanently

### 4. **Disk Space Management**
- Users know exactly how much space is needed upfront
- Can check disk space before running
- Clear location of cached files

### 5. **Distribution**
- Can pre-download and distribute models with app
- No dependency on HuggingFace availability at runtime
- Faster app startup

## Troubleshooting

### "sentencepiece not found" Error

**Solution:**
```batch
pip install sentencepiece
```

Then re-run the setup script.

### "Insufficient disk space" Error

**Required:** ~1GB free space (650MB models + cache overhead)

**Solution:**
1. Check available disk space
2. Free up space if needed
3. Re-run setup script

### "HuggingFace servers down" Error

**Solution:**
1. Check internet connection
2. Try again later
3. Visit https://status.huggingface.co/ for server status

### Script Fails at Verification

**Solution:**
1. Delete partial downloads:
   ```batch
   rmdir /s /q "python-backend\models\tts\opea_speecht5"
   ```
2. Re-run setup script to download fresh

## Model Cache Structure

After successful setup:

```
python-backend/models/tts/opea_speecht5/
├── models--microsoft--speecht5_tts/
│   └── snapshots/<hash>/
│       ├── config.json
│       ├── pytorch_model.bin (~585MB)
│       ├── tokenizer_config.json
│       ├── spm_char.model
│       ├── added_tokens.json
│       └── special_tokens_map.json
├── models--microsoft--speecht5_hifigan/
│   └── snapshots/<hash>/
│       ├── config.json
│       └── pytorch_model.bin (~50MB)
├── spk_embed_default.pt
└── spk_embed_male.pt
```

## Integration with TTSEngine

**Backend Priority (when OpenVINO detected):**
1. Intel OPEA TTS (if models installed) ← **NEW**
2. Piper (if model files present)
3. pyttsx3 (system voices, always available)

**Backend Priority (no OpenVINO):**
1. Existing SpeechT5 (legacy backend)
2. Piper
3. pyttsx3

**No manual configuration needed** — TTSEngine automatically detects and uses OPEA TTS when:
- Models are installed via setup script
- OpenVINO is available (optional, for acceleration)

## Optional: OpenVINO Acceleration

For 30-40% faster synthesis, install OpenVINO:

```batch
pip install openvino
```

OPEA TTS will automatically use OpenVINO if available, otherwise falls back to PyTorch CPU.

## File Locations

| File | Purpose |
|------|---------|
| `python-backend/scripts/setup_opea_tts_models.bat` | Setup script (run once) |
| `python-backend/models/tts/opea_speecht5/` | Model cache directory |
| `python-backend/src/speech/opea_tts/synthesizer.py` | Modified to require pre-downloaded models |
| `python-backend/src/speech/opea_tts/model_downloader.py` | Still used by setup script |

## Download Status

As of last run, the setup script was downloading models successfully:
- ✅ SpeechT5 processor downloaded
- ✅ SpeechT5 tokenizer files downloaded  
- 🔄 SpeechT5 model downloading (45% complete at timeout)
- ⏳ HiFi-GAN vocoder (pending)
- ⏳ Speaker embeddings (pending)

**Note:** The download may take 10-15 minutes total. The script will continue until complete.
