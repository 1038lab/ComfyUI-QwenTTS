# ComfyUI-QwenTTS

**ComfyUI custom nodes for Qwen3‑TTS (12Hz)**: **CustomVoice**, **VoiceDesign**, and **VoiceClone** — with practical defaults for stability and speed on **CUDA / Apple Silicon (MPS) / CPU**.

> If this repo saves you time, please ⭐ it — it helps more ComfyUI users discover a working Qwen3‑TTS setup.

![ComfyUI-QwenTTS_V1.1.0](example_workflows/QwenTTS_Nodes.jpg)

---

## What’s New (v1.1.0)

- **Voice Clone** supports reusable **`VOICE`** inputs from the Voices Library.
- New **Tools**: **Create Voice**, **Load Voice**, **Whisper STT**, and **Voice Instruct** presets (EN + CN).
![create_your_voice](example_workflows/Create_your_voice.jpg)
- Advanced nodes expose attention selection: `auto / sage_attn / flash_attn / sdpa / eager`.
- README includes `extra_model_paths.yaml` guidance for custom model locations.
- **Audio Duration** node rewritten: cleaner logic, seconds-based outputs, optional frame calculation.

[More updated Details](Update.md)

---

## Quickstart (3 minutes)

### 1) Install

**Option A — ComfyUI‑Manager (recommended)**
- Open ComfyUI‑Manager → search **ComfyUI‑QwenTTS** → Install.

**Option B — Git clone**
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/1038lab/ComfyUI-QwenTTS.git
```

### 2) Install requirements (important)
Use ComfyUI’s embedded python if you’re on Portable:

**Windows Portable**
```bat
cd <ComfyUI_root>
python_embeded\python.exe -m pip install -r ComfyUI\custom_nodes\ComfyUI-QwenTTS\requirements.txt --no-cache-dir
```

**macOS/Linux (typical)**
```bash
python3 -m pip install -r ComfyUI/custom_nodes/ComfyUI-QwenTTS/requirements.txt --no-cache-dir
```

### 3) Import workflow
- Import: `example_workflows/QwenTTS_sample_workflow.json`
- Run it once (first run is slower due to model download + warmup)

---

## Features

- **Custom Voice (preset speakers)**: easy, high-quality TTS with 9 timbres.
- **Voice Design**: create voices using a natural-language description.
- **Voice Clone**: clone from reference audio + transcript, or reuse a saved **`VOICE`**.
- **Multi‑Device**: auto select CUDA → MPS → CPU.
- **Local‑First models**: prefer `ComfyUI/models/TTS/Qwen3-TTS/`.
- **Tools bundle**: Create/Load Voice, Whisper STT, Voice Instruct presets, Text Token Count.
- **Advanced control nodes**: sampling, max_new_tokens, attention backend, unload.

---
## Model Overview (Qwen3-TTS)

- **Languages**: Chinese, English, Japanese, Korean, German, French, Russian, Portuguese, Spanish, Italian.  
- **Instruction control**: Supports voice style control via natural-language instructions.  
- **Tokenizer**: Uses Qwen3-TTS-Tokenizer-12Hz for speech encoding/decoding.  

## Model Matrix (12Hz)

| Model | Size | Features | Streaming | Instruction |
|---|---|---|---|---|
| CustomVoice | 1.7B | 9 premium timbres, style control | ✅ | ✅ |
| VoiceDesign | 1.7B | Voice design from descriptions | ✅ | ✅ |
| Base | 1.7B | 3s rapid voice clone, FT base | ✅ | - |
| CustomVoice | 0.6B | 9 premium timbres | ✅ | - |
| Base | 0.6B | 3s rapid voice clone | ✅ | - |
| Tokenizer | 12Hz | Speech encode/decode | - | - |

## Models Download

Models can be auto-downloaded to:
```
ComfyUI/models/TTS/Qwen3-TTS/<MODEL_NAME>/
```

Supported model IDs (Hugging Face):
- [Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice)
- [Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice)
- [Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign)
- [Qwen/Qwen3-TTS-12Hz-1.7B-Base](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base)
- [Qwen/Qwen3-TTS-12Hz-0.6B-Base](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-Base)
- [Qwen/Qwen3-TTS-Tokenizer-12Hz](https://huggingface.co/Qwen/Qwen3-TTS-Tokenizer-12Hz)

If a model is missing locally, it will be downloaded automatically on first use.

### Model Folder Policy

All Qwen3-TTS assets are stored in one consistent location:
```
ComfyUI/models/TTS/Qwen3-TTS/<MODEL_NAME>/
```
This node will not download or create model folders elsewhere.

### Extra Model Paths (Optional)

If you store models outside the default ComfyUI path, configure ComfyUI’s
`extra_model_paths.yaml` in the ComfyUI root. This node relies on ComfyUI’s
standard model path system.

Supported location:
- `ComfyUI/extra_model_paths.yaml`

Example (ComfyUI format):
```yaml
comfyui:
  base_path: D:/AI/ComfyUI-Models
  tts: models/TTS/  # use lowercase `tts`
```
If your ComfyUI build does not expose a `TTS` key, keep the default layout
`ComfyUI/models/TTS/Qwen3-TTS/` and skip this section.

How to place Qwen3-TTS models in a custom location:
1) Set `base_path` to your shared models root.
2) Put Qwen3‑TTS models under:
   - `<base_path>/TTS/Qwen3-TTS/<MODEL_NAME>/`
3) Add that root to `extra_model_paths.yaml` (under `tts` as shown above).
4) Restart ComfyUI.

### Why So Many Files?

Qwen3-TTS follows the standard Hugging Face model layout (config, tokenizer, weights, etc.).
Multiple JSON/config files are required by Transformers at runtime, so they cannot be safely
collapsed into a single file without breaking loading.

### Manual Download (Recommended for Slow/Blocked Networks)

You can download models manually and place them into:
```
ComfyUI/models/TTS/Qwen3-TTS/<MODEL_NAME>/
```

Hugging Face CLI example:
```bash
pip install -U "huggingface_hub[cli]"

huggingface-cli download Qwen/Qwen3-TTS-Tokenizer-12Hz --local-dir ./Qwen3-TTS-Tokenizer-12Hz
huggingface-cli download Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice --local-dir ./Qwen3-TTS-12Hz-1.7B-CustomVoice
huggingface-cli download Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign --local-dir ./Qwen3-TTS-12Hz-1.7B-VoiceDesign
huggingface-cli download Qwen/Qwen3-TTS-12Hz-1.7B-Base --local-dir ./Qwen3-TTS-12Hz-1.7B-Base
huggingface-cli download Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice --local-dir ./Qwen3-TTS-12Hz-0.6B-CustomVoice
huggingface-cli download Qwen/Qwen3-TTS-12Hz-0.6B-Base --local-dir ./Qwen3-TTS-12Hz-0.6B-Base
```

Then move each downloaded folder into:
```
ComfyUI/models/TTS/Qwen3-TTS/
```

### Manual Download via ModelScope (Mainland China)
```bash
pip install -U modelscope
modelscope download --model Qwen/Qwen3-TTS-Tokenizer-12Hz --local_dir ./Qwen3-TTS-Tokenizer-12Hz
modelscope download --model Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice --local_dir ./Qwen3-TTS-12Hz-1.7B-CustomVoice
modelscope download --model Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign --local_dir ./Qwen3-TTS-12Hz-1.7B-VoiceDesign
modelscope download --model Qwen/Qwen3-TTS-12Hz-1.7B-Base --local_dir ./Qwen3-TTS-12Hz-1.7B-Base
modelscope download --model Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice --local_dir ./Qwen3-TTS-12Hz-0.6B-CustomVoice
modelscope download --model Qwen/Qwen3-TTS-12Hz-0.6B-Base --local_dir ./Qwen3-TTS-12Hz-0.6B-Base
```

This node auto-downloads missing models to:
```
ComfyUI/models/TTS/Qwen3-TTS/<MODEL_NAME>/
```

---

## Usage overview

### Basic nodes (fast defaults)
- Faster defaults (typically `do_sample=False`)
- Minimal inputs

### Advanced nodes (full control)
- Expose `max_new_tokens`, sampling knobs, **attention backend selection** (`auto/sage_attn/flash_attn/sdpa/eager`), unload_models, seed.

### Tips that fix 80% of “quality/length” issues
- **Set a sensible `max_new_tokens`** (too high can cause long humming / trailing noise).
- Prefer **do_sample=False** for stability.
- Use the speaker’s **native language** for best results.

---

## Optional speedups (CUDA)

### FlashAttention 2
```bash
pip install flash-attn --no-build-isolation
```

### SageAttention (experimental)
```bash
pip install sageattention
```

---

## Troubleshooting (common)

### 1) `'Qwen3TTSTalkerConfig' object has no attribute 'pad_token_id'`
This is usually an incompatible `transformers` build (often 5.x dev/nightly).

Fix (recommended):
```bash
pip install -U "transformers==4.57.3" "tokenizers<0.20" --no-cache-dir
```
Then restart ComfyUI.

### 2) Output always very long / humming
Lower `max_new_tokens` (try 512–1024 for short text), and set `do_sample=False`.
Tip: use **Text Token Count (QwenTTS)** to pick a safe `max_new_tokens` and reduce long trailing noise.

### 3) CUDA OOM
Split long scripts into chunks, lower `max_new_tokens`, and use `precision=bf16`.

---

## License
GPL‑3.0 (see LICENSE).

## Credits
- Qwen3‑TTS by Alibaba Qwen Team
- ComfyUI community
