import json
import math
import os
import tempfile

import torch
import torchaudio

import folder_paths
import AILab_QwenTTS as core

try:
    import whisper
except Exception:
    whisper = None


_WHISPER_CACHE = {}
_TEXT_TOKENIZER_CACHE = {}
_TEXT_TOKENIZER_MODEL_ID = core.MODEL_ID_MAP.get(("CustomVoice", "1.7B")) or core.MODEL_ID_MAP.get(("Base", "1.7B"))

_VOICE_INSTRUCT_CACHE = {"mtime": None, "characters": None, "styles": None}
_VOICE_INSTRUCT_ZH_CACHE = {"mtime": None, "characters": None, "styles": None}


def _voice_instruct_path():
    return os.path.join(os.path.dirname(__file__), "voice_instruct.json")


def _voice_instruct_zh_path():
    return os.path.join(os.path.dirname(__file__), "voice_instruct_zh.json")


def _load_voice_instruct_presets():
    path = _voice_instruct_path()
    try:
        mtime = os.path.getmtime(path)
    except Exception:
        mtime = None

    cached = _VOICE_INSTRUCT_CACHE
    if cached["characters"] is not None and cached["styles"] is not None and cached["mtime"] == mtime:
        return cached["characters"], cached["styles"]

    characters = []
    styles = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        characters = data.get("characters") or []
        styles = data.get("styles") or []
    except Exception:
        characters = []
        styles = []

    cached["mtime"] = mtime
    cached["characters"] = characters
    cached["styles"] = styles
    return characters, styles


def _build_voice_instruct_options():
    characters, styles = _load_voice_instruct_presets()
    if not characters:
        characters = [{"name": "Auto", "text": ""}]
    if not styles:
        styles = [{"name": "Auto", "text": ""}]

    def to_character_label(item):
        return (item.get("name") or "").strip() or "Auto"

    def to_style_label(item):
        return (item.get("name") or "").strip() or "Auto"

    character_labels = [to_character_label(item) for item in characters]
    style_labels = [to_style_label(item) for item in styles]
    character_map = {to_character_label(item): (item.get("text") or "").strip() for item in characters}
    style_map = {to_style_label(item): (item.get("text") or "").strip() for item in styles}
    return character_labels, style_labels, character_map, style_map


def _load_voice_instruct_presets_zh():
    path = _voice_instruct_zh_path()
    try:
        mtime = os.path.getmtime(path)
    except Exception:
        mtime = None

    cached = _VOICE_INSTRUCT_ZH_CACHE
    if cached["characters"] is not None and cached["styles"] is not None and cached["mtime"] == mtime:
        return cached["characters"], cached["styles"]

    characters = []
    styles = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        characters = data.get("characters") or []
        styles = data.get("styles") or []
    except Exception:
        characters = []
        styles = []

    cached["mtime"] = mtime
    cached["characters"] = characters
    cached["styles"] = styles
    return characters, styles


def _build_voice_instruct_options_zh():
    characters, styles = _load_voice_instruct_presets_zh()
    if not characters:
        characters = [{"name": "自动", "text": ""}]
    if not styles:
        styles = [{"name": "自动", "text": ""}]

    def to_character_label(item):
        return (item.get("name") or "").strip() or "自动"

    def to_style_label(item):
        return (item.get("name") or "").strip() or "自动"

    character_labels = [to_character_label(item) for item in characters]
    style_labels = [to_style_label(item) for item in styles]
    character_map = {to_character_label(item): (item.get("text") or "").strip() for item in characters}
    style_map = {to_style_label(item): (item.get("text") or "").strip() for item in styles}
    return character_labels, style_labels, character_map, style_map


def _sanitize_name(name: str) -> str:
    safe = "".join(c for c in name if c.isalnum() or c in ("-", "_")).strip()
    return safe or "voice"


def _default_voice_dir():
    try:
        base = folder_paths.get_output_directory()
    except Exception:
        base = None
    if not base:
        base = os.path.join(os.path.dirname(__file__), "voice_library")
    path = os.path.join(base, "qwen3-tts_voices")
    os.makedirs(path, exist_ok=True)
    return path


def _build_prompt(reference_audio, reference_text, model_size, device, precision, x_vector_only):
    model = core._load_model("Base", model_size, device, precision)
    audio_tuple = core._audio_to_tuple(reference_audio)
    with core._maybe_autocast(core._resolve_device(device), precision):
        items = model.create_voice_clone_prompt(
            ref_audio=audio_tuple,
            ref_text=reference_text.strip() if reference_text and reference_text.strip() else None,
            x_vector_only_mode=bool(x_vector_only),
        )
    return items


def _save_voice(prompt_items, save_path: str | None, name: str | None):
    payload = _serialize_prompt_items(prompt_items)
    if save_path and save_path.strip():
        base_dir = save_path.strip()
        os.makedirs(base_dir, exist_ok=True)
    else:
        base_dir = _default_voice_dir()
    filename = f"{_sanitize_name(name or 'voice')}.pt"
    path = os.path.join(base_dir, filename)

    payload = {"version": 2, "prompt": payload}
    torch.save(payload, path)
    return path


def _load_voice_from_file(path: str):
    if not path:
        return None
    try:
        data = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        data = torch.load(path, map_location="cpu")
    if isinstance(data, dict) and "prompt" in data:
        payload = data["prompt"]
    else:
        payload = data
    if isinstance(payload, list) and payload and isinstance(payload[0], dict):
        return _deserialize_prompt_items(payload)
    return payload


def _get_prompt_item_class():
    try:
        core._load_qwen3_model()
        module = __import__("qwen_tts.inference.qwen3_tts_model", fromlist=["VoiceClonePromptItem"])
        return getattr(module, "VoiceClonePromptItem", None)
    except Exception:
        return None


def _serialize_prompt_item(item):
    return {
        "ref_code": item.ref_code,
        "ref_spk_embedding": item.ref_spk_embedding,
        "x_vector_only_mode": bool(item.x_vector_only_mode),
        "icl_mode": bool(item.icl_mode),
        "ref_text": item.ref_text,
    }


def _deserialize_prompt_item(data):
    cls = _get_prompt_item_class()
    if cls is None:
        return data
    return cls(
        ref_code=data.get("ref_code"),
        ref_spk_embedding=data.get("ref_spk_embedding"),
        x_vector_only_mode=bool(data.get("x_vector_only_mode")),
        icl_mode=bool(data.get("icl_mode")),
        ref_text=data.get("ref_text"),
    )


def _serialize_prompt_items(items):
    return [_serialize_prompt_item(item) for item in items]


def _deserialize_prompt_items(items):
    return [_deserialize_prompt_item(item) for item in items]


def _resolve_device():
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _get_model(model_size: str, device: str):
    key = (model_size, device)
    model = _WHISPER_CACHE.get(key)
    if model is not None:
        return model
    if whisper is None:
        raise RuntimeError("whisper is not installed. Please install it in your ComfyUI environment.")
    model = whisper.load_model(model_size, device=device)
    _WHISPER_CACHE[key] = model
    return model


def _get_text_tokenizer(model_id: str):
    tok = _TEXT_TOKENIZER_CACHE.get(model_id)
    if tok is not None:
        return tok
    try:
        from transformers import AutoTokenizer
    except Exception as e:
        raise RuntimeError(f"transformers is required for token counting: {e}")
    local = None
    try:
        local = core._find_local_model(model_id)
    except Exception:
        local = None
    source = local or model_id
    tok = AutoTokenizer.from_pretrained(source, use_fast=True, trust_remote_code=True)
    _TEXT_TOKENIZER_CACHE[model_id] = tok
    return tok


class Qwen3TTSVoicesLibrary:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "reference_audio": ("AUDIO", {"tooltip": "Reference audio used to build the speaker prompt (required)."}),
                "reference_text": ("STRING", {"multiline": True, "default": "", "tooltip": "Transcript of the reference audio. Required unless x_vector_only is enabled."}),
                "model_size": (["0.6B", "1.7B"], {"default": "1.7B", "tooltip": "Base model size used to extract speaker features."}),
                "device": (core._available_devices(), {"default": "auto", "tooltip": "Compute device for prompt building (auto/cpu/cuda/mps)."}),
                "precision": (["bf16", "fp16", "fp32"], {"default": "bf16", "tooltip": "Compute precision for prompt building. bf16 recommended on modern GPUs."}),
                "x_vector_only": ("BOOLEAN", {"default": False, "tooltip": "Use speaker embedding only. If enabled, reference_text can be empty."}),
                "voice_name": ("STRING", {"default": "voice_1", "multiline": False, "tooltip": "Saved voice name (file name without extension)."}),
            },
            "optional": {
                "save_path": ("STRING", {"default": "", "multiline": False, "tooltip": "Custom folder to save voices. Leave empty to use ComfyUI output/qwen3-tts_voices."}),
                "unload_models": ("BOOLEAN", {"default": True, "tooltip": "Unload cached models after generation"}),
            },
        }

    RETURN_TYPES = ("VOICE",)
    RETURN_NAMES = ("VOICE",)
    FUNCTION = "generate"
    CATEGORY = "🧪AILab/🎙️QwenTTS"

    def generate(
        self,
        reference_audio,
        reference_text,
        model_size,
        device,
        precision,
        x_vector_only=False,
        voice_name="voice_1",
        save_path="",
        unload_models=True,
    ):
        if (not reference_text or not reference_text.strip()) and not x_vector_only:
            raise ValueError("reference_text is required unless x_vector_only is enabled")
        prompt_items = _build_prompt(reference_audio, reference_text, model_size, device, precision, x_vector_only)
        name = voice_name.strip() if voice_name else ""
        path = _save_voice(prompt_items, save_path, name or None)
        if unload_models:
            try:
                core._MODEL_CACHE.clear()
                core.model_management.soft_empty_cache()
                import gc
                gc.collect()
                gc.collect()
            except Exception:
                pass
        return (path,)


class Qwen3TTSLoadVoice:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "voice_name": (cls._list_voices(), {"default": "", "tooltip": "Pick a saved voice from the default voice library folder."}),
            },
            "optional": {
                "custom_path": ("STRING", {"default": "", "multiline": False, "tooltip": "Load a voice .pt from a custom full path. Overrides voice_name if set."}),
            },
        }

    RETURN_TYPES = ("VOICE",)
    RETURN_NAMES = ("VOICE",)
    FUNCTION = "load"
    CATEGORY = "🧪AILab/🎙️QwenTTS"

    @staticmethod
    def _list_voices():
        root = _default_voice_dir()
        if not os.path.isdir(root):
            return [""]
        names = []
        for fname in os.listdir(root):
            if fname.lower().endswith(".pt"):
                names.append(fname)
        return names or [""]

    def load(self, voice_name, custom_path=""):
        if custom_path and custom_path.strip():
            return (custom_path.strip(),)
        if voice_name:
            return (os.path.join(_default_voice_dir(), voice_name),)
        return ("",)


class Qwen3TTSWhisperSTT:
    WHISPER_MODELS = [
        "tiny", "base", "small", "medium", "large", "large-v2", "large-v3", "large-v3-turbo",]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO", {"tooltip": "Audio input to transcribe."}),
                "model_size": (cls.WHISPER_MODELS, {"default": "small", "tooltip": "Whisper model size. Larger models are slower but more accurate."}),
                "language": (["auto", "en", "zh", "ja", "ko", "de", "fr", "es", "it", "pt", "ru"], {"default": "auto", "tooltip": "Force language or use auto detection."}),
            },
            "optional": {
                "unload_models": ("BOOLEAN", {"default": True, "tooltip": "Unload cached models after transcription"}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "transcribe"
    CATEGORY = "🧪AILab/🎙️QwenTTS"

    def transcribe(self, audio, model_size="large-v3-turbo", language="auto", unload_models=True):
        device = _resolve_device()
        print(f"[Whisper STT] Using device: {device}")
        model = _get_model(model_size, device)

        waveform = audio["waveform"]
        sample_rate = int(audio["sample_rate"])
        if waveform.dim() == 3:
            waveform = waveform.squeeze(0)

        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                tmp_path = tmp.name
            torchaudio.save(tmp_path, waveform, sample_rate)
            result = model.transcribe(tmp_path, language=None if language == "auto" else language)
            if language == "auto":
                detected = result.get("language", "unknown")
                conf = float(result.get("language_probability", 0.0))
                print(f"[Whisper STT] Detected: {detected} (conf: {conf:.2f})")
            return (result.get("text", "").strip(),)
        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass
            if unload_models:
                try:
                    _WHISPER_CACHE.clear()
                    core.model_management.soft_empty_cache()
                    import gc
                    gc.collect()
                    gc.collect()
                except Exception:
                    pass


class Qwen3TTSTextTokenCount:
    _RATIO = 8.0
    _MIN_LIMIT = 384
    _MAX_LIMIT = 4096
    _ROUND_TO = 64

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"multiline": True, "default": "", "tooltip": "Text to count tokens for."}),
            }
        }

    RETURN_TYPES = ("STRING", "INT")
    RETURN_NAMES = ("text", "max_new_tokens")
    FUNCTION = "count"
    CATEGORY = "🧪AILab/🎙️QwenTTS"

    def count(self, text):
        if not _TEXT_TOKENIZER_MODEL_ID:
            raise RuntimeError("No suitable Qwen3-TTS model id found for text tokenization.")
        tokenizer = _get_text_tokenizer(_TEXT_TOKENIZER_MODEL_ID)
        ids = tokenizer(text or "", add_special_tokens=False).get("input_ids", [])
        count = int(len(ids))
        est = int(math.ceil(count * float(self._RATIO)))
        min_cap = int(self._MIN_LIMIT) if int(self._MIN_LIMIT) > 0 else 0
        max_cap = int(self._MAX_LIMIT) if int(self._MAX_LIMIT) > 0 else 0
        if min_cap > 0:
            est = max(est, min_cap)
        if max_cap > 0:
            est = min(est, max_cap)
        divisor = int(self._ROUND_TO) if int(self._ROUND_TO) > 0 else 1
        rounded = ((est + divisor - 1) // divisor) * divisor
        if max_cap > 0 and rounded > max_cap:
            rounded = max_cap
        if min_cap > 0 and rounded < min_cap:
            rounded = min_cap
        return (text or "", rounded)


class Qwen3TTSVoiceInstructZH:
    @classmethod
    def INPUT_TYPES(cls):
        character_labels, style_labels, _, _ = _build_voice_instruct_options_zh()
        return {
            "required": {
                "角色": (character_labels, {"default": character_labels[0] if character_labels else "自动", "tooltip": "中文角色预设（voice_instruct_zh.json）。"}),
                "风格": (style_labels, {"default": style_labels[0] if style_labels else "自动", "tooltip": "中文语气预设（voice_instruct_zh.json）。"}),
            },
            "optional": {
                "自定义风格指引": ("STRING", {"default": "", "multiline": True, "tooltip": "自定义中文指令，优先级最高。"}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("VOICE_INSTRUCT",)
    FUNCTION = "build"
    CATEGORY = "🧪AILab/🎙️QwenTTS"

    def build(self, 角色="自动", 风格="自动", 自定义风格指引=""):
        text = (自定义风格指引 or "").strip()
        if text:
            return (text,)

        _, _, character_map, style_map = _build_voice_instruct_options_zh()
        char_text = (character_map.get(角色) or "").strip()
        style_text = (style_map.get(风格) or "").strip()

        parts = []
        if char_text:
            parts.append(char_text)
        if style_text:
            parts.append(style_text)

        return ("\n".join(parts).strip(),)


NODE_CLASS_MAPPINGS = {
    "AILab_Qwen3TTSVoicesLibrary": Qwen3TTSVoicesLibrary,
    "AILab_Qwen3TTSLoadVoice": Qwen3TTSLoadVoice,
    "AILab_Qwen3TTSWhisperSTT": Qwen3TTSWhisperSTT,
    "AILab_Qwen3TTSVoiceInstruct": Qwen3TTSVoiceInstruct,
    "AILab_Qwen3TTSVoiceInstructZH": Qwen3TTSVoiceInstructZH,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AILab_Qwen3TTSVoicesLibrary": "Create Voice (QwenTTS)",
    "AILab_Qwen3TTSLoadVoice": "Load Voice (QwenTTS)",
    "AILab_Qwen3TTSWhisperSTT": "Whisper STT (QwenTTS)",
    "AILab_Qwen3TTSVoiceInstruct": "Voice Instruct (QwenTTS)",
    "AILab_Qwen3TTSVoiceInstructZH": "声音风格指引 (QwenTTS)",
}
