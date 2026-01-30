# Update v1.1.0

## Highlights
- Voice Clone nodes now support `VOICE` inputs from the Voices Library for stable reuse.
- New Tools bundle: Create Voice, Load Voice, Whisper STT, and Voice Instruct presets (EN + CN).
- Advanced nodes now expose attention selection (auto / sage_attn / flash_attn / sdpa / eager).
- README expanded with ComfyUI `extra_model_paths.yaml` guidance for custom model locations.
- Audio Duration node rewritten: cleaner logic, seconds-based outputs, optional frame calculation.

## New / Updated Nodes
- Create Voice (QwenTTS)
  - Build and save voice prompts to `.pt` in `ComfyUI/output/qwen3-tts_voices` by default.
- Load Voice (QwenTTS)
  - Load saved voices or use a custom path and output `VOICE`.
- Whisper STT (QwenTTS)
  - Transcribe `AUDIO` to text with multiple model sizes.
- Voice Clone (QwenTTS) / Voice Clone (QwenTTS) Advanced
  - Added optional `voice` input; `reference_audio` is only required if no voice is provided.
- Voice Instruct (QwenTTS)
  - English preset builder from `voice_instruct.json`.
- 声音风格指引 (QwenTTS)
  - 中文预设 builder from `voice_instruct_zh.json`.
- Text Token Count (QwenTTS)
  - Counts text tokens and outputs a rounded value for `max_new_tokens`.

## Audio Duration
- Outputs: `duration_int` (seconds), `duration_float` (seconds), `frames`, `audio_path`.
- Optional `fps` input enables frame calculation.
