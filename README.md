# Audio Module - Real-time Audio-Visual Speech System

This module implements the **Audio-only** processing pipeline, acting as a fallback and complement to the LipNet visual module.

## Architecture

The pipeline runs on CPU (Intel i3 / Ryzen 5) and consists of three stages:

1.  **Audio Capture (`audio_capture.py`)**:
    - Captures audio from microphone at 16kHz.
    - Uses **frame-based Energy VAD** (Voice Activity Detection) to detect speech.
    - Buffers speech frames and releases them as a segment after 500ms of silence.

2.  **ASR (`whisper_asr.py`)**:
    - Uses **Whisper (Tiny/Base)** for speech-to-text.
    - **Optimization**: Uses `faster-whisper` (CTranslate2 backend) with `int8` quantization for < 1.5s latency on CPU.
    - Outputs raw text and confidence scores.

3.  **Refinement (`text_refinement_t5.py`)**:
    - Uses **T5-Small** to clean and reconstruct text.
    - **Optimization**: Uses PyTorch **dynamic quantization** to reduce model size and inference int8 ops.

## Requirements

Install dependencies:

```bash
pip install -r requirements.txt
```

**Note**: You need `PyAudio`. On Windows, if `pip install pyaudio` fails, stick to providing the prebuilt `.whl` or use `conda`.

## Usage

Run the pipeline:

```bash
# For Intel i3 (Tiny model)
python audio_pipeline.py --model_size tiny

# For Ryzen 5 (Base model)
python audio_pipeline.py --model_size base
```

## Output Format

The script prints JSON objects to stdout for easy parsing by the fusion module:

```json
{
  "modality": "audio",
  "raw_text": "hello world",
  "refined_text": "Hello world.",
  "confidence": 0.85,
  "low_confidence": false
}
```

## Optimizations for CPU Real-Time

1.  **VAD Pre-filtering**: We only run heavy ASR inference when speech is detected, saving CPU cycles during silence.
2.  **Faster-Whisper (CTranslate2)**: 4x faster than standard OpenAI Whisper on CPU.
3.  **Int8 Quantization**: Reduces memory bandwidth and uses vectorized CPU instructions (AVX2).
4.  **Greedy Decoding**: `beam_size=1` in Whisper prevents slow extensive search.
