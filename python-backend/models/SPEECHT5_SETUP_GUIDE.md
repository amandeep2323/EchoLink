# Intel OPEA SpeechT5 TTS - Quick Setup Guide

## What's New?
EchoLink now uses **Intel OPEA SpeechT5** as the primary Text-to-Speech engine, providing natural, human-like voices with offline capability.

## One-Time Setup (5 minutes)

### Step 1: Install Dependencies
```bash
cd python-backend
pip install -r requirements.txt

```

This installs:
- Intel OpenVINO Runtime
- HuggingFace Transformers
- PyTorch (CPU version)
- SentencePiece tokenizer
- Other required packages

### Step 2: First Run (Automatic Model Download)
```bash
python main.py
```

**What happens automatically**:
1. ✅ Detects missing SpeechT5 models
2. ✅ Downloads from HuggingFace Hub (~600MB)
3. ✅ Saves to `models/tts/speecht5/`
4. ✅ Tests synthesis
5. ✅ Starts TTS engine

**Console Output**:
```
[TTS] Testing SpeechT5 (Intel OPEA + OpenVINO)...
[TTS] SpeechT5: Loading...
[TTS] SpeechT5: Models not found — downloading...
[TTS] SpeechT5: Starting model download...
[TTS] SpeechT5: Downloading main model (microsoft/speecht5_tts)...
[TTS] SpeechT5: Downloading vocoder (microsoft/speecht5_hifigan)...
[TTS] SpeechT5: Downloading speaker embeddings...
[TTS] SpeechT5: ✓ Model download completed in 120.5s
[TTS] SpeechT5: ✓ Loaded with PyTorch
[TTS] ✓ Backend: SpeechT5 (Intel OPEA + OpenVINO) — rate: 16000Hz
```

**Download Time**: 2-5 minutes (depends on internet speed)

### Step 3: Done!
On subsequent runs, models load instantly from cache. No internet required.

## Verification

### Check Backend Selection
Open the web UI and check console logs:
```
[TTS] ✓ Backend: SpeechT5 (Intel OPEA + OpenVINO) — rate: 16000Hz
```

### Test TTS
1. Enable TTS in the web UI
2. Sign a word (e.g., "HELLO")
3. Listen for natural speech output

## Backend Selection

### Automatic (Default)
The system tries backends in order:
1. **SpeechT5** (best quality, offline)
2. **Piper** (good quality, fast)
3. **pyttsx3** (basic quality, fallback)

### Manual Selection
Force a specific backend by editing `pipeline_manager.py`:
```python
tts.load(backend="speecht5")  # Force SpeechT5
tts.load(backend="piper")     # Force Piper
tts.load(backend="pyttsx3")   # Force pyttsx3
```

## Troubleshooting

### Problem: Models Won't Download
**Error**: "[TTS] SpeechT5: Model download failed"

**Solutions**:
1. Check internet connection
2. Check firewall (allow HuggingFace Hub access)
3. Try manual download:
   ```bash
   pip install huggingface_hub
   huggingface-cli login  # Optional
   huggingface-cli download microsoft/speecht5_tts
   ```

### Problem: Import Errors
**Error**: "ModuleNotFoundError: No module named 'transformers'"

**Solution**:
```bash
pip install transformers sentencepiece torch datasets openvino
```

### Problem: Out of Memory
**Error**: "MemoryError" or slow system

**Solution**: Use Piper instead (lower memory)
```python
tts.load(backend="piper")
```

### Problem: Slow Synthesis
**Symptom**: >1 second per word

**Solutions**:
1. Wait for OpenVINO optimization (coming soon)
2. Use Piper for faster synthesis
3. Check CPU usage (close other apps)

### Problem: No Audio Output
**Check**:
1. Speakers connected and unmuted
2. VirtualMic installed (OBS Studio)
3. TTS enabled in web UI
4. Console shows synthesis messages

## System Requirements

### Minimum
- **OS**: Windows 10/11
- **CPU**: Intel Core i5 or equivalent
- **RAM**: 8GB
- **Disk**: 2GB free space
- **Internet**: Required for first-time setup only

### Recommended
- **CPU**: Intel Core i7 or newer
- **RAM**: 16GB
- **SSD**: For faster model loading

## Performance

### Current (PyTorch CPU)
- **Synthesis**: ~100-200ms per word
- **Quality**: ★★★★★ (Natural, human-like)
- **Memory**: ~500MB
- **Offline**: Yes

### Coming Soon (OpenVINO)
- **Synthesis**: ~50-100ms per word (2-3x faster)
- **Memory**: ~300MB (lower)
- **Optimized**: Intel CPU acceleration

## Disk Usage

```
models/tts/speecht5/          (~1GB total)
├── pytorch_model.bin          ~400MB
├── vocoder_pytorch_model.bin  ~150MB
├── config files               ~50MB
└── HuggingFace cache          ~400MB
```

**Cleanup** (if needed):
```bash
# Remove HuggingFace cache (can re-download)
rm -rf models/tts/speecht5/models--*
```

## Advanced Configuration

### Change Voice Quality
Edit `speecht5_backend.py`:
```python
DEFAULT_SPEAKER_ID = 7645  # Current voice
DEFAULT_SPEAKER_ID = 1234  # Try different speaker
```

Speakers available: 0-9999 (from CMU Arctic dataset)

### Adjust Sample Rate
Currently fixed at 16000 Hz (optimal for VirtualMic)

## Next Steps

1. ✅ Install dependencies
2. ✅ Run first-time setup
3. ✅ Test TTS with recognition
4. ✅ Verify VirtualMic output
5. ⏳ Wait for OpenVINO optimization update

## Need Help?

Check detailed documentation:
- `SPEECHT5_TTS_IMPLEMENTATION.md` - Technical details
- `IMPLEMENTATION_SUMMARY.md` - Project overview

## Changelog

### v1.0 - Initial Release
- ✅ Intel OPEA SpeechT5 integration
- ✅ Automatic model download
- ✅ PyTorch CPU backend
- ✅ Backward compatibility
- ✅ VirtualMic integration

### v1.1 - Coming Soon
- ⏳ OpenVINO acceleration
- ⏳ 2-3x faster synthesis
- ⏳ Lower memory usage
- ⏳ Model caching

Enjoy natural, high-quality text-to-speech with Intel OPEA SpeechT5! 🎉
