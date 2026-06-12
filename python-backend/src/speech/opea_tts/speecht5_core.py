# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# Adapted from opea-project/GenAIComps/comps/third_parties/speecht5/src/speecht5_model.py
# Original: https://github.com/opea-project/GenAIComps/blob/main/comps/third_parties/speecht5/src/speecht5_model.py
#
# Modifications for desktop integration:
#   - Removed Docker-specific paths and dependencies
#   - Removed Gaudi2/HPU warmup code  
#   - Added configurable model directory
#   - Integrated runtime backend detector for OpenVINO/PyTorch selection
#   - Enhanced error handling for desktop environment
#   - Removed FastAPI server dependencies

"""
SpeechT5 Model Core

Core SpeechT5 text-to-speech synthesis logic adapted from Intel OPEA project.
Provides model loading, speaker embedding management, and text-to-speech synthesis
with batch processing support for long text inputs.

This module handles:
- Loading SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan models
- Downloading and caching speaker embeddings
- Text splitting for batch processing
- Audio generation from text input
"""

import os
import subprocess
import traceback
from typing import Optional

import numpy as np
import torch
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan


class SpeechT5ModelCore:
    """
    Core SpeechT5 model for text-to-speech synthesis.
    
    Adapted from OPEA's speecht5_model.py to function as a standalone desktop library.
    Supports both PyTorch CPU inference and OpenVINO acceleration.
    
    Attributes:
        model_dir: Directory for model cache storage
        backend: Inference backend ("cpu", "openvino", or "hpu")
        model_name: HuggingFace model identifier for main TTS model
        vocoder_name: HuggingFace model identifier for vocoder
    """
    
    def __init__(self, model_dir: str, backend: str = "cpu"):
        """
        Initialize SpeechT5 model core.
        
        Args:
            model_dir: Path to directory for model cache and speaker embeddings
            backend: Inference backend - "cpu" (PyTorch CPU), "openvino" (Intel OpenVINO),
                     or "hpu" (Gaudi2 - not recommended for desktop)
        """
        self.model_dir = model_dir
        self.backend = backend
        
        # Model identifiers
        self.model_name = "microsoft/speecht5_tts"
        self.vocoder_name = "microsoft/speecht5_hifigan"
        
        # Model components (loaded during initialize())
        self.processor: Optional[SpeechT5Processor] = None
        self.model: Optional[SpeechT5ForTextToSpeech] = None
        self.vocoder: Optional[SpeechT5HifiGan] = None
        self.speaker_embeddings: dict[str, torch.Tensor] = {}
        
        # OpenVINO IR model (loaded when backend == "openvino" and IR is available)
        # The IR is produced by convert_tts_model.py and stored alongside the
        # HuggingFace cache, e.g. models/tts/speecht5_openvino/.
        self.ov_model = None
        self.use_openvino_ir: bool = False
        self.ir_dir: str = os.path.join(
            os.path.dirname(os.path.normpath(model_dir)), "speecht5_openvino"
        )
        
        # Device for inference
        self.device = self._determine_device()
    
    def _determine_device(self) -> str:
        """
        Determine the inference device based on backend setting.
        
        Returns:
            Device string for torch.to() call: "cpu", "cuda", or "hpu"
        """
        if self.backend == "hpu":
            # Gaudi2 HPU (not typical for desktop)
            return "hpu"
        elif self.backend == "openvino":
            # OpenVINO uses CPU but may optimize differently
            # For now, we use CPU and let OpenVINO runtime handle optimization
            return "cpu"
        else:
            # Standard PyTorch CPU
            return "cpu"
    
    def initialize(self) -> bool:
        """
        Load models and speaker embeddings.
        
        Downloads models from HuggingFace on first run and caches them locally.
        Loads speaker embeddings from Intel's speaker embedding repository.
        
        When the OpenVINO IR is available, the heavy PyTorch TTS model and
        vocoder are NOT loaded (the IR handles inference), which dramatically
        cuts initialization time.
        
        Returns:
            True if initialization successful, False otherwise
        """
        try:
            print(f"[OPEA TTS Core] Initializing with backend: {self.backend}")
            
            # Create model directory
            os.makedirs(self.model_dir, exist_ok=True)
            
            # Configure backend-specific optimizations
            if self.backend == "hpu":
                print("[OPEA TTS Core] Warning: HPU/Gaudi2 backend selected but not recommended for desktop")
                # Skip Gaudi2 warmup code (removed for desktop)
            
            # Processor is always needed (tokenization) — lightweight.
            print(f"[OPEA TTS Core] Loading processor from {self.model_name}...")
            self.processor = SpeechT5Processor.from_pretrained(
                self.model_name,
                cache_dir=self.model_dir
            )
            
            # Try the OpenVINO IR first. If it loads, we skip the heavy PyTorch
            # TTS model + vocoder entirely (they would never be used).
            if self.backend == "openvino":
                self._try_load_openvino_ir()
            
            if not self.use_openvino_ir:
                # PyTorch fallback path — load the full model + vocoder.
                print(f"[OPEA TTS Core] Loading TTS model from {self.model_name}...")
                self.model = SpeechT5ForTextToSpeech.from_pretrained(
                    self.model_name,
                    cache_dir=self.model_dir
                )
                self.model = self.model.to(self.device)
                self.model.eval()
                
                print(f"[OPEA TTS Core] Loading vocoder from {self.vocoder_name}...")
                self.vocoder = SpeechT5HifiGan.from_pretrained(
                    self.vocoder_name,
                    cache_dir=self.model_dir
                )
                self.vocoder = self.vocoder.to(self.device)
                self.vocoder.eval()
            
            # Load speaker embeddings (lightweight, needed by both paths)
            print("[OPEA TTS Core] Loading speaker embeddings...")
            self._load_speaker_embeddings()
            
            mode = "OpenVINO IR" if self.use_openvino_ir else "PyTorch"
            print(f"[OPEA TTS Core] ✓ Initialization complete (device: {self.device}, mode: {mode})")
            return True
            
        except Exception as e:
            print(f"[OPEA TTS Core] ✗ Initialization failed: {e}")
            traceback.print_exc()
            return False
    
    def _try_load_openvino_ir(self) -> None:
        """
        Attempt to load the exported OpenVINO IR model for accelerated inference.
        
        The IR is produced by convert_tts_model.py and stored at self.ir_dir
        (e.g. models/tts/speecht5_openvino/). If the directory or the
        optimum-intel runtime is missing, this silently leaves the PyTorch
        pipeline in place so synthesis still works.
        
        Sets:
            self.ov_model and self.use_openvino_ir = True on success.
        """
        ir_xml = os.path.join(self.ir_dir, "openvino_encoder_model.xml")
        if not os.path.exists(ir_xml):
            print(f"[OPEA TTS Core] OpenVINO IR not found at {self.ir_dir} — using PyTorch pipeline")
            print(f"[OPEA TTS Core]   (run convert_tts_model.py to generate the IR)")
            return
        
        try:
            from optimum.intel import OVModelForTextToSpeechSeq2Seq
        except ImportError:
            print("[OPEA TTS Core] optimum-intel not installed — using PyTorch pipeline")
            print("[OPEA TTS Core]   Install with: pip install \"optimum-intel[openvino]\"")
            return
        
        try:
            print(f"[OPEA TTS Core] Loading OpenVINO IR from {self.ir_dir}...")
            # Persistent compile cache: optimum-intel creates its own internal
            # Core() that does NOT inherit CACHE_DIR from the workspace's loader.
            # Without this, every startup recompiles ~580MB of IR (encoder +
            # decoder + postnet + vocoder), which is the dominant cost.
            cache_dir = os.path.join("cache", "openvino_tts")
            os.makedirs(cache_dir, exist_ok=True)
            ov_config = {
                "CACHE_DIR": os.path.abspath(cache_dir),
                "PERFORMANCE_HINT": "LATENCY",
            }
            self.ov_model = OVModelForTextToSpeechSeq2Seq.from_pretrained(
                self.ir_dir, ov_config=ov_config
            )
            self.use_openvino_ir = True
            print(f"[OPEA TTS Core] ✓ OpenVINO IR loaded — using accelerated inference")
            print(f"[OPEA TTS Core]   Compile cache: {cache_dir}")
        except Exception as e:
            print(f"[OPEA TTS Core] ⚠ OpenVINO IR load failed ({e}) — using PyTorch pipeline")
            self.ov_model = None
            self.use_openvino_ir = False
    
    def _load_speaker_embeddings(self) -> None:
        """
        Download and cache speaker embeddings from Intel repository.
        
        Downloads spk_embed_default.pt and spk_embed_male.pt from Intel Extension
        for Transformers repository. If download fails, creates zero embeddings
        as fallback.
        
        Side Effects:
            - Downloads .pt files to model_dir using curl subprocess
            - Updates self.speaker_embeddings dict with loaded tensors
            - Prints download status messages
        """
        # Speaker embedding files from Intel's neural_chat assets
        embeddings = ["spk_embed_default.pt", "spk_embed_male.pt"]
        base_url = "https://raw.githubusercontent.com/intel/intel-extension-for-transformers/main/intel_extension_for_transformers/neural_chat/assets/speaker_embeddings/"
        
        for embed_file in embeddings:
            local_path = os.path.join(self.model_dir, embed_file)
            
            # Download if not cached
            if not os.path.exists(local_path):
                print(f"[OPEA TTS Core] Downloading speaker embedding: {embed_file}")
                try:
                    # Use curl to download (matches OPEA original implementation)
                    subprocess.run(
                        ["curl", "-o", local_path, base_url + embed_file],
                        check=True,
                        capture_output=True
                    )
                    print(f"[OPEA TTS Core] ✓ Downloaded {embed_file}")
                except subprocess.CalledProcessError as e:
                    print(f"[OPEA TTS Core] Warning: Failed to download {embed_file}: {e}")
                    print("[OPEA TTS Core] Creating zero embedding as fallback")
                    # Create zero embedding fallback (512-dim for SpeechT5)
                    torch.save(torch.zeros((1, 512)), local_path)
                except FileNotFoundError:
                    print("[OPEA TTS Core] Warning: curl not found, using alternative download method")
                    # Try with urllib as fallback
                    try:
                        import urllib.request
                        urllib.request.urlretrieve(base_url + embed_file, local_path)
                        print(f"[OPEA TTS Core] ✓ Downloaded {embed_file} (via urllib)")
                    except Exception as e2:
                        print(f"[OPEA TTS Core] Warning: Alternative download failed: {e2}")
                        torch.save(torch.zeros((1, 512)), local_path)
            
            # Load embedding into memory
            try:
                embedding_name = embed_file.replace("spk_embed_", "").replace(".pt", "")
                self.speaker_embeddings[embedding_name] = torch.load(
                    local_path,
                    map_location=self.device
                )
                print(f"[OPEA TTS Core] ✓ Loaded speaker embedding: {embedding_name}")
            except Exception as e:
                print(f"[OPEA TTS Core] Warning: Failed to load {embed_file}: {e}")
                # Create zero embedding fallback
                embedding_name = embed_file.replace("spk_embed_", "").replace(".pt", "")
                self.speaker_embeddings[embedding_name] = torch.zeros(
                    (1, 512),
                    device=self.device
                )
    
    def text_to_speech(self, text: str, voice: str = "default") -> np.ndarray:
        """
        Synthesize speech from text with batch processing support.
        
        Args:
            text: Input text to synthesize (supports long text via batching)
            voice: Speaker voice selection - "default" or "male"
            
        Returns:
            Audio waveform as numpy array (float32, shape: (samples,), 16kHz sample rate)
            
        Raises:
            ValueError: If voice is not "default" or "male"
            RuntimeError: If speaker embedding not loaded
            
        Note:
            Long text (>100 characters) is automatically split into batches
            and synthesized sequentially. Output is concatenated.
        """
        # Validate voice parameter
        if voice not in ["default", "male"]:
            raise ValueError(
                f"Unsupported voice: {voice}. Use 'default' or 'male'."
            )
        
        # Get speaker embedding
        speaker_embedding = self.speaker_embeddings.get(voice)
        if speaker_embedding is None:
            raise RuntimeError(
                f"Speaker embedding '{voice}' not loaded. "
                f"Available: {list(self.speaker_embeddings.keys())}"
            )
        
        # Split long text into batches (from OPEA original implementation)
        text_batches = self._split_long_text(text, batch_length=100)
        
        backend_label = "OpenVINO IR" if self.use_openvino_ir else "PyTorch"
        print(f"[OPEA TTS Core] Synthesizing {len(text_batches)} text batch(es) [{backend_label}]")
        
        # Process text through processor
        inputs = self.processor(
            text=text_batches,
            padding=True,
            max_length=600,  # SpeechT5 max sequence length
            return_tensors="pt"
        )
        
        # Move inputs to device
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)
        speaker_emb = speaker_embedding.to(self.device)
        
        if self.use_openvino_ir and self.ov_model is not None:
            # ── OpenVINO IR inference path ──
            # The OV model drives the generation loop internally and applies the
            # postnet + vocoder, returning a waveform per batch row.
            all_audio = np.array([])
            for row in range(input_ids.size(0)):
                row_ids = input_ids[row : row + 1]
                waveform = self.ov_model.generate(
                    input_ids=row_ids,
                    speaker_embeddings=speaker_emb,
                )
                batch_audio = np.asarray(waveform).reshape(-1)
                all_audio = np.concatenate([all_audio, batch_audio])
            
            print(f"[OPEA TTS Core] ✓ Synthesized {len(all_audio)} audio samples [OpenVINO IR]")
            return all_audio
        
        # ── PyTorch inference path (fallback) ──
        # Generate speech
        with torch.no_grad():
            waveforms, waveform_lengths = self.model.generate_speech(
                input_ids,
                speaker_embeddings=speaker_emb,
                attention_mask=attention_mask,
                vocoder=self.vocoder,
                return_output_lengths=True
            )
        
        # Concatenate batches into single audio output
        all_audio = np.array([])
        for i in range(waveforms.size(0)):
            # Extract valid audio (up to actual length, excluding padding)
            batch_audio = waveforms[i][:waveform_lengths[i]].cpu().numpy()
            all_audio = np.concatenate([all_audio, batch_audio])
        
        print(f"[OPEA TTS Core] ✓ Synthesized {len(all_audio)} audio samples")
        return all_audio
    
    def _split_long_text(self, text: str, batch_length: int = 100) -> list[str]:
        """
        Split long text into batches of shorter sentences.
        
        Adapted from OPEA's original implementation. Splits text at sentence
        boundaries (punctuation marks) to maintain natural prosody across batches.
        
        Args:
            text: Input text to split
            batch_length: Maximum character length per batch
            
        Returns:
            List of text chunks, each ending with punctuation
            
        Algorithm:
            1. Scan through text tracking sentence boundaries
            2. When batch_length reached, split at last sentence boundary
            3. Add period to chunks without punctuation to avoid unexpected EOS
        """
        result = []
        sentence_ends = [",", ".", "?", "!", "。", ";", " "]
        idx = 0
        cur_start = 0
        cur_end = -1
        
        # Scan through text
        while idx < len(text):
            # Check if we've exceeded batch length
            if idx - cur_start > batch_length:
                if cur_end != -1 and cur_end > cur_start:
                    # Split at last sentence boundary
                    result.append(text[cur_start:cur_end + 1])
                else:
                    # No sentence boundary found, force split
                    cur_end = cur_start + batch_length - 1
                    result.append(text[cur_start:cur_end + 1])
                
                # Move to next batch
                idx = cur_end
                cur_start = cur_end + 1
            
            # Track sentence boundaries
            if text[idx] in sentence_ends:
                cur_end = idx
            
            idx += 1
        
        # Handle last chunk
        if cur_start < len(text):
            last_chunk = text[cur_start:]
            # Find last punctuation in final chunk
            last_punc_idx = max(
                [last_chunk.rfind(punc) for punc in sentence_ends[:-1]]
            )
            if last_punc_idx != -1:
                # Keep up to last punctuation
                result.append(last_chunk[:last_punc_idx + 1])
            else:
                # No punctuation, use entire chunk
                result.append(last_chunk)
        
        # Add period to chunks without proper punctuation to avoid unexpected EOS behavior
        result = [
            s if s.rstrip()[-1:] in [".", "!", "?", "。"] else s + "."
            for s in result
        ]
        
        return result
