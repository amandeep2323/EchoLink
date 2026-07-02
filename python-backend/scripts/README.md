# EchoLink Setup Scripts

This folder contains setup scripts for EchoLink components that require one-time configuration.

## Available Scripts

### `setup_opea_tts_models.bat`

**Purpose:** Downloads Intel OPEA SpeechT5 TTS models for offline speech synthesis.

**When to run:** Before using OPEA TTS backend for the first time.

**What it does:**
- Downloads microsoft/speecht5_tts model (~585MB)
- Downloads microsoft/speecht5_hifigan vocoder (~50MB)
- Downloads Intel speaker embeddings (<1MB)
- Verifies installation

**How to run:**
```batch
cd "asl-speech-desktop-app v6 complete"
python-backend\scripts\setup_opea_tts_models.bat
```

**Requirements:**
- Python 3.9+
- Internet connection (one-time)
- ~1GB free disk space
- Dependencies: transformers, torch, sentencepiece

**Duration:** 5-15 minutes (depending on internet speed)

**Output:** Models cached at `python-backend/models/tts/opea_speecht5/`

## Troubleshooting

### Script won't run

**Check:**
1. Run from project root directory
2. Python is in PATH: `python --version`
3. You have administrator rights (if needed)

### Download fails

**Common causes:**
- No internet connection
- HuggingFace servers down
- Insufficient disk space
- Firewall blocking downloads

**Solution:**
- Check internet
- Check disk space: need ~1GB free
- Try again later
- Check https://status.huggingface.co/

### Missing dependencies

**Error:** `ModuleNotFoundError: No module named 'sentencepiece'`

**Solution:**
```batch
pip install sentencepiece transformers torch
```

Then re-run the script.

## Script Details

### setup_opea_tts_models.bat

**Steps:**
1. ✓ Check Python installation
2. ✓ Verify directory structure
3. ✓ Create model cache directory
4. ✓ Install/verify dependencies
5. ✓ Download models from HuggingFace
6. ✓ Download speaker embeddings
7. ✓ Verify installation

**Success indicators:**
```
================================================================
 SUCCESS — OPEA TTS Models Installed
================================================================

Models are ready at:
  python-backend\models\tts\opea_speecht5\

You can now use OPEA TTS in EchoLink!
```

## After Running Scripts

Once setup is complete:
- Models are cached locally
- No internet connection needed for TTS
- EchoLink automatically detects and uses OPEA TTS
- Faster startup (no runtime downloads)

## Re-running Scripts

It's safe to re-run scripts:
- Existing files are detected and skipped
- Only missing files are downloaded
- Use this to fix incomplete installations

## Model Updates

If Intel releases updated models:
1. Delete old cache: `rmdir /s /q python-backend\models\tts\opea_speecht5`
2. Re-run setup script

## Adding New Scripts

When adding new setup scripts to this folder:
1. Follow same naming pattern: `setup_<component>_<purpose>.bat`
2. Add entry to this README
3. Include progress indicators
4. Verify installation at end
5. Provide clear error messages
