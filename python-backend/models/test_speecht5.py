"""
SpeechT5 TTS Test Script
========================
Tests Intel OPEA SpeechT5 TTS backend installation and functionality.

Usage:
    python test_speecht5.py
"""

import os
import sys
import time
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))


def test_imports():
    """Test that all required packages are installed."""
    print("\n" + "=" * 60)
    print("  TEST 1: Package Imports")
    print("=" * 60)
    
    required_packages = [
        ("transformers", "HuggingFace Transformers"),
        ("sentencepiece", "SentencePiece"),
        ("torch", "PyTorch"),
        ("datasets", "HuggingFace Datasets"),
        ("openvino", "Intel OpenVINO"),
        ("numpy", "NumPy"),
    ]
    
    missing = []
    for package, name in required_packages:
        try:
            __import__(package)
            print(f"  ✓ {name} ({package})")
        except ImportError:
            print(f"  ✗ {name} ({package}) - MISSING")
            missing.append(package)
    
    if missing:
        print(f"\n  Missing packages: {', '.join(missing)}")
        print(f"  Install: pip install {' '.join(missing)}")
        return False
    
    print("\n  ✓ All packages installed")
    return True


def test_speecht5_backend():
    """Test SpeechT5Synthesizer directly."""
    print("\n" + "=" * 60)
    print("  TEST 2: SpeechT5 Backend")
    print("=" * 60)
    
    try:
        from speech.speecht5_backend import SpeechT5Synthesizer
        
        # Create synthesizer
        model_dir = os.path.join(os.path.dirname(__file__), "models", "tts", "speecht5")
        print(f"  Model directory: {model_dir}")
        
        synth = SpeechT5Synthesizer(model_dir=model_dir)
        
        # Load (will download if needed)
        print("\n  Loading SpeechT5...")
        start_time = time.time()
        
        if synth.load():
            load_time = time.time() - start_time
            print(f"  ✓ SpeechT5 loaded in {load_time:.1f}s")
            print(f"  Sample rate: {synth.sample_rate} Hz")
            
            # Test synthesis
            print("\n  Testing synthesis...")
            test_text = "Hello, this is a test."
            
            synth_start = time.time()
            audio = synth.synthesize(test_text, return_numpy=True)
            synth_time = time.time() - synth_start
            
            if audio is not None and len(audio) > 0:
                duration = len(audio) / synth.sample_rate
                print(f"  ✓ Synthesis successful")
                print(f"    Text: '{test_text}'")
                print(f"    Audio: {len(audio)} samples")
                print(f"    Duration: {duration:.2f}s")
                print(f"    Synthesis time: {synth_time:.2f}s")
                print(f"    RTF: {synth_time/duration:.2f}x (lower is better)")
                
                # Check audio format
                if audio.dtype == np.int16:
                    print(f"  ✓ Audio format: int16 (compatible with VirtualMic)")
                else:
                    print(f"  ✗ Audio format: {audio.dtype} (expected int16)")
                
                return True
            else:
                print("  ✗ Synthesis failed - no audio generated")
                return False
        else:
            print("  ✗ SpeechT5 load failed")
            return False
            
    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_tts_engine():
    """Test TTSEngine with SpeechT5 backend."""
    print("\n" + "=" * 60)
    print("  TEST 3: TTSEngine Integration")
    print("=" * 60)
    
    try:
        from speech import TTSEngine
        
        model_dir = os.path.join(os.path.dirname(__file__), "models", "tts")
        print(f"  Model directory: {model_dir}")
        
        # Create engine
        tts = TTSEngine(model_dir=model_dir, backend="speecht5")
        
        # Load
        print("\n  Loading TTSEngine with SpeechT5...")
        start_time = time.time()
        tts.load()
        load_time = time.time() - start_time
        
        if tts.is_loaded:
            print(f"  ✓ TTSEngine loaded in {load_time:.1f}s")
            print(f"  Backend: {tts.backend_name}")
            print(f"  Voice: {tts.voice_name}")
            print(f"  Sample rate: {tts.sample_rate} Hz")
            
            if tts.backend_name == "speecht5":
                print("  ✓ SpeechT5 is active backend")
                return True
            else:
                print(f"  ⚠ Backend is '{tts.backend_name}', not 'speecht5'")
                print(f"    SpeechT5 may not be available - check previous tests")
                return False
        else:
            print("  ✗ TTSEngine load failed")
            return False
            
    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_backend_fallback():
    """Test backend fallback mechanism."""
    print("\n" + "=" * 60)
    print("  TEST 4: Backend Fallback")
    print("=" * 60)
    
    try:
        from speech import TTSEngine
        
        model_dir = os.path.join(os.path.dirname(__file__), "models", "tts")
        
        # Test auto mode
        print("\n  Testing auto mode (SpeechT5 → Piper → pyttsx3)...")
        tts = TTSEngine(model_dir=model_dir, backend="auto")
        tts.load()
        
        if tts.is_loaded:
            print(f"  ✓ Auto mode selected: {tts.backend_name}")
        else:
            print("  ✗ Auto mode failed to load any backend")
            return False
        
        # Test explicit backends
        backends = ["speecht5", "piper", "pyttsx3"]
        results = {}
        
        for backend in backends:
            print(f"\n  Testing {backend}...")
            tts = TTSEngine(model_dir=model_dir, backend=backend)
            tts.load()
            
            results[backend] = tts.is_loaded
            if tts.is_loaded:
                print(f"    ✓ {backend} available")
            else:
                print(f"    ✗ {backend} not available")
        
        print("\n  Backend Availability:")
        for backend, available in results.items():
            status = "✓" if available else "✗"
            print(f"    {status} {backend}")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_audio_compatibility():
    """Test audio output compatibility with VirtualMic."""
    print("\n" + "=" * 60)
    print("  TEST 5: VirtualMic Compatibility")
    print("=" * 60)
    
    try:
        from speech.speecht5_backend import SpeechT5Synthesizer
        
        model_dir = os.path.join(os.path.dirname(__file__), "models", "tts", "speecht5")
        synth = SpeechT5Synthesizer(model_dir=model_dir)
        
        if synth.load():
            audio = synth.synthesize("test", return_numpy=True)
            
            if audio is not None:
                # Check requirements
                checks = []
                
                # 1. Data type
                if audio.dtype == np.int16:
                    checks.append(("✓", "Data type: int16"))
                else:
                    checks.append(("✗", f"Data type: {audio.dtype} (expected int16)"))
                
                # 2. Sample rate
                if synth.sample_rate in [16000, 22050]:
                    checks.append(("✓", f"Sample rate: {synth.sample_rate} Hz"))
                else:
                    checks.append(("⚠", f"Sample rate: {synth.sample_rate} Hz (unusual)"))
                
                # 3. Shape
                if audio.ndim == 1:
                    checks.append(("✓", f"Shape: {audio.shape} (mono)"))
                else:
                    checks.append(("✗", f"Shape: {audio.shape} (expected 1D)"))
                
                # 4. Value range
                if np.min(audio) >= -32768 and np.max(audio) <= 32767:
                    checks.append(("✓", f"Value range: [{np.min(audio)}, {np.max(audio)}]"))
                else:
                    checks.append(("✗", f"Value range: [{np.min(audio)}, {np.max(audio)}] (out of int16 range)"))
                
                print()
                for status, msg in checks:
                    print(f"  {status} {msg}")
                
                all_pass = all(status == "✓" for status, _ in checks)
                if all_pass:
                    print("\n  ✓ Audio is compatible with VirtualMic")
                    return True
                else:
                    print("\n  ⚠ Audio may have compatibility issues")
                    return False
            else:
                print("  ✗ Synthesis failed")
                return False
        else:
            print("  ✗ SpeechT5 load failed")
            return False
            
    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("  Intel OPEA SpeechT5 TTS - Test Suite")
    print("=" * 60)
    
    results = []
    
    # Run tests
    results.append(("Package Imports", test_imports()))
    
    if results[-1][1]:  # Only continue if imports succeeded
        results.append(("SpeechT5 Backend", test_speecht5_backend()))
        results.append(("TTSEngine Integration", test_tts_engine()))
        results.append(("Backend Fallback", test_backend_fallback()))
        results.append(("VirtualMic Compatibility", test_audio_compatibility()))
    
    # Print summary
    print("\n" + "=" * 60)
    print("  TEST SUMMARY")
    print("=" * 60)
    
    for test_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}: {test_name}")
    
    all_passed = all(passed for _, passed in results)
    
    print("\n" + "=" * 60)
    if all_passed:
        print("  ✓ ALL TESTS PASSED")
        print("=" * 60)
        print("\n  SpeechT5 TTS is ready to use!")
        print("  Run: python main.py")
        return 0
    else:
        print("  ✗ SOME TESTS FAILED")
        print("=" * 60)
        print("\n  Check error messages above for details.")
        print("  Install missing packages: pip install -r requirements.txt")
        return 1


if __name__ == "__main__":
    sys.exit(main())
