#!/usr/bin/env python3
"""
SignSpeak — Environment Diagnostic Script
==========================================
Run this to check if all dependencies are installed correctly.

Usage:
    python diagnose.py
"""

import os
import sys

PASS = "✓"
FAIL = "✗"
WARN = "⚠"


def check(name, test_fn, fix_hint=""):
    """Run a diagnostic check and print the result."""
    try:
        result = test_fn()
        if result:
            print(f"  {PASS} {name}: {result}")
            return True
        else:
            print(f"  {FAIL} {name}: returned falsy")
            if fix_hint:
                print(f"    Fix: {fix_hint}")
            return False
    except Exception as e:
        print(f"  {FAIL} {name}: {e}")
        if fix_hint:
            print(f"    Fix: {fix_hint}")
        return False


def main():
    print()
    print("=" * 60)
    print("  SignSpeak — Environment Diagnostic")
    print("=" * 60)
    print(f"  Python: {sys.version}")
    print(f"  Executable: {sys.executable}")
    print(f"  CWD: {os.getcwd()}")
    print()

    # ── Check for shadowing ──
    print("── Shadowing Check ──")
    shadow_found = False
    for p in [os.getcwd()] + sys.path[:5]:
        for name in ["mediapipe.py", "cv2.py", "numpy.py", "onnxruntime.py"]:
            shadow = os.path.join(p, name)
            if os.path.isfile(shadow):
                print(f"  {FAIL} SHADOW DETECTED: {shadow}")
                print(f"    This file is hiding the real '{name[:-3]}' package!")
                print(f"    Delete or rename it.")
                shadow_found = True
    if not shadow_found:
        print(f"  {PASS} No shadowing issues found")
    print()

    # ── Core Dependencies ──
    print("── Core Dependencies ──")

    check("numpy", lambda: __import__("numpy").__version__,
          "pip install numpy")

    check("opencv", lambda: __import__("cv2").__version__,
          "pip install opencv-python-headless")

    check("fastapi", lambda: __import__("fastapi").__version__,
          "pip install fastapi")

    check("uvicorn", lambda: __import__("uvicorn").__version__,
          "pip install uvicorn[standard]")
    print()

    # ── MediaPipe (the tricky one) ──
    print("── MediaPipe ──")
    mp_ok = False

    def check_mediapipe():
        import mediapipe as mp
        ver = getattr(mp, '__version__', 'unknown')
        loc = getattr(mp, '__file__', 'unknown')
        print(f"    Version: {ver}")
        print(f"    Location: {loc}")
        attrs = [x for x in dir(mp) if not x.startswith('_')]
        print(f"    Attributes: {attrs[:15]}{'...' if len(attrs) > 15 else ''}")
        return ver

    mp_ok = check("mediapipe import", check_mediapipe,
                   "pip install mediapipe==0.10.14")

    if mp_ok:
        # Test mp.solutions.hands
        def check_solutions():
            import mediapipe as mp
            hands = mp.solutions.hands
            return f"mp.solutions.hands found — HAND_CONNECTIONS: {len(hands.HAND_CONNECTIONS)} edges"

        sol_ok = check("mp.solutions.hands", check_solutions)

        if not sol_ok:
            # Try direct import
            def check_direct():
                from mediapipe.python.solutions import hands
                return f"Direct import works — HAND_CONNECTIONS: {len(hands.HAND_CONNECTIONS)} edges"

            check("mediapipe.python.solutions.hands", check_direct)

        # Test actual hand detection
        def check_hands_init():
            import mediapipe as mp
            try:
                hands = mp.solutions.hands
            except AttributeError:
                from mediapipe.python.solutions import hands
            h = hands.Hands(
                static_image_mode=True,
                max_num_hands=1,
                min_detection_confidence=0.5,
            )
            # Test with a blank image
            import numpy as np
            blank = np.zeros((480, 640, 3), dtype=np.uint8)
            import cv2
            rgb = cv2.cvtColor(blank, cv2.COLOR_BGR2RGB)
            result = h.process(rgb)
            h.close()
            return f"Hands() initialized and processed test frame"

        check("MediaPipe Hands inference", check_hands_init)
    print()

    # ── ONNX Runtime ──
    print("── ONNX Runtime ──")

    def check_onnx():
        import onnxruntime as ort
        providers = ort.get_available_providers()
        return f"{ort.__version__} — providers: {providers}"

    check("onnxruntime", check_onnx,
          "pip install onnxruntime")
    print()

    # ── Model Files ──
    print("── Model Files ──")
    backend_root = os.path.dirname(os.path.abspath(__file__))
    sign_dir = os.path.join(backend_root, "models", "sign")
    tts_dir = os.path.join(backend_root, "models", "tts")

    print(f"  Sign model dir: {sign_dir}")
    if os.path.isdir(sign_dir):
        files = os.listdir(sign_dir)
        model_files = [f for f in files if f.endswith(('.onnx', '.h5', '.keras', '.tflite'))]
        label_files = [f for f in files if f.endswith(('.json', '.txt'))]
        if model_files:
            for f in model_files:
                size = os.path.getsize(os.path.join(sign_dir, f)) / (1024 * 1024)
                print(f"    {PASS} {f} ({size:.1f} MB)")
        else:
            print(f"    {FAIL} No model files found (.onnx, .h5, .keras, .tflite)")
            print(f"      Place your model.onnx here")

        if label_files:
            for f in label_files:
                print(f"    {PASS} {f}")
        else:
            print(f"    {WARN} No label file found (will use default ABCDEFGHIKLMNOPQRSTUVWXY)")
    else:
        print(f"    {FAIL} Directory does not exist")
        print(f"      Create it: mkdir -p {sign_dir}")

    print()
    print(f"  TTS model dir: {tts_dir}")
    if os.path.isdir(tts_dir):
        files = os.listdir(tts_dir)
        voice_files = [f for f in files if f.endswith('.onnx') and not f.endswith('.onnx.json')]
        config_files = [f for f in files if f.endswith('.onnx.json')]
        if voice_files:
            for f in voice_files:
                size = os.path.getsize(os.path.join(tts_dir, f)) / (1024 * 1024)
                print(f"    {PASS} {f} ({size:.1f} MB)")
        else:
            print(f"    {WARN} No TTS voice files (TTS won't work without these)")
            print(f"      Download from: https://github.com/rhasspy/piper/releases")
        if config_files:
            for f in config_files:
                print(f"    {PASS} {f}")
    else:
        print(f"    {WARN} Directory does not exist (TTS is optional)")

    print()

    # ── Optional Dependencies ──
    print("── Optional Dependencies ──")

    check("pyvirtualcam", lambda: __import__("pyvirtualcam").__version__,
          "pip install pyvirtualcam (requires OBS Studio)")

    check("sounddevice", lambda: __import__("sounddevice").__version__,
          "pip install sounddevice")

    def check_piper():
        from piper import PiperVoice
        return "PiperVoice importable"
    check("piper-tts", check_piper,
          "pip install piper-tts")

    check("pyttsx3", lambda: __import__("pyttsx3").__version__,
          "pip install pyttsx3 (fallback TTS using Windows SAPI)")
    print()

    # ── TTS Synthesis Test ──
    print("── TTS Synthesis Test ──")

    # Test piper_phonemize
    def check_phonemize():
        import piper_phonemize
        result = piper_phonemize.phonemize_espeak("hello", "en-us")
        if result:
            return f"phonemize('hello') → {result[0][:60]}"
        else:
            return None
    check("piper_phonemize (espeak-ng)", check_phonemize,
          "pip install piper-phonemize  (espeak-ng bundled)")

    # Test Piper synthesis
    if os.path.isdir(tts_dir):
        voice_files = [f for f in os.listdir(tts_dir)
                       if f.endswith('.onnx') and not f.endswith('.onnx.json')]
        if voice_files:
            def check_piper_synth():
                from piper import PiperVoice
                import wave as wave_mod
                import tempfile
                model_path = os.path.join(tts_dir, voice_files[0])
                voice = PiperVoice.load(model_path)

                # Method 1: temp file
                fd, tmp = tempfile.mkstemp(suffix=".wav")
                os.close(fd)
                try:
                    with wave_mod.open(tmp, "wb") as wf:
                        wf.setnchannels(1)
                        wf.setsampwidth(2)
                        wf.setframerate(22050)
                        voice.synthesize("hello world", wf)
                    with wave_mod.open(tmp, "rb") as wr:
                        n = wr.getnframes()
                        sr = wr.getframerate()
                        return f"{n} samples at {sr}Hz ({n/sr:.2f}s) — Piper synthesis works!"
                finally:
                    os.unlink(tmp)

            piper_synth_ok = check("Piper synthesis (temp file)", check_piper_synth)

            if not piper_synth_ok:
                # Show more details
                print("    Detailed Piper debug:")
                try:
                    from piper import PiperVoice
                    model_path = os.path.join(tts_dir, voice_files[0])
                    voice = PiperVoice.load(model_path)

                    # Check synthesize_stream_raw
                    chunks = list(voice.synthesize_stream_raw("hello"))
                    total_bytes = sum(len(c) for c in chunks)
                    print(f"      synthesize_stream_raw: {len(chunks)} chunks, {total_bytes} bytes")

                    if total_bytes == 0:
                        print(f"      {FAIL} 0 bytes — phonemization or inference is failing silently")
                        print(f"      Try: pip install piper-phonemize --force-reinstall")
                except Exception as e:
                    print(f"      Error: {e}")

        else:
            print(f"  {WARN} No TTS voice files — skipping synthesis test")
    else:
        print(f"  {WARN} No TTS dir — skipping synthesis test")

    # Test pyttsx3 synthesis
    def check_pyttsx3_synth():
        import pyttsx3
        import tempfile
        import wave as wave_mod
        engine = pyttsx3.init()
        fd, tmp = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        try:
            engine.save_to_file("hello world", tmp)
            engine.runAndWait()
            if os.path.exists(tmp) and os.path.getsize(tmp) > 100:
                with wave_mod.open(tmp, "rb") as wr:
                    n = wr.getnframes()
                    sr = wr.getframerate()
                    return f"{n} samples at {sr}Hz ({n/sr:.2f}s) — pyttsx3 synthesis works!"
            else:
                return None
        finally:
            engine.stop()
            if os.path.exists(tmp):
                os.unlink(tmp)

    check("pyttsx3 synthesis", check_pyttsx3_synth,
          "pip install pyttsx3")
    print()

    # ── ONNX Model Test ──
    print("── ONNX Model Loading Test ──")
    onnx_files = []
    if os.path.isdir(sign_dir):
        onnx_files = [f for f in os.listdir(sign_dir) if f.endswith('.onnx')]

    if onnx_files:
        def test_onnx_load():
            import onnxruntime as ort
            model_path = os.path.join(sign_dir, onnx_files[0])
            sess = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
            inp = sess.get_inputs()[0]
            out = sess.get_outputs()[0]
            result = (
                f"Input: {inp.name} {inp.shape} | "
                f"Output: {out.name} {out.shape}"
            )

            # Try inference
            import numpy as np
            dummy = np.random.rand(1, 21, 3).astype(np.float32)
            prediction = sess.run([out.name], {inp.name: dummy})[0]
            best = int(np.argmax(prediction[0]))
            conf = float(prediction[0][best])
            letters = "ABCDEFGHIKLMNOPQRSTUVWXY"
            letter = letters[best] if best < len(letters) else f"idx_{best}"
            result += f" | Test predict: {letter} ({conf:.2%})"
            return result

        check(f"Load & run {onnx_files[0]}", test_onnx_load)
    else:
        print(f"  {WARN} No .onnx model found — skipping load test")
    print()

    # ── Summary ──
    print("=" * 60)
    print("  QUICK FIX COMMANDS (copy-paste):")
    print("=" * 60)
    print()
    print("  If mediapipe fails:")
    print("    pip uninstall mediapipe -y")
    print("    pip install mediapipe==0.10.14 --force-reinstall --no-cache-dir")
    print()
    print("  If numpy issues:")
    print("    pip install \"numpy>=1.24,<2.0\"")
    print()
    print("  If onnxruntime fails:")
    print("    pip install onnxruntime --force-reinstall")
    print()
    print("  Full reinstall:")
    print("    pip install -r requirements.txt --force-reinstall")
    print()


if __name__ == "__main__":
    main()
