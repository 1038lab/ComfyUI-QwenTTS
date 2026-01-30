# ComfyUI-QwenTTS
# A clean, efficient ComfyUI custom node for Qwen3-TTS (Text-to-Speech) functionality.
# This implementation provides high-quality speech generation and voice cloning
# capabilities using the Qwen3-TTS model family.
#
# Models License Notice:
# - Qwen3-TTS: Apache-2.0 License (https://huggingface.co/collections/Qwen/qwen3-tts)
#
# This integration script follows GPL-3.0 License.
# When using or modifying this code, please respect both the original model licenses
# and this integration's license terms.
#
# Source: https://github.com/1038lab/ComfyUI-QwenTTS

import os
import sys
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
from contextlib import nullcontext
from collections import OrderedDict

import numpy as np
import torch
import folder_paths
import comfy.model_management as model_management

try:
    from huggingface_hub import snapshot_download
except Exception:
    snapshot_download = None

current_dir = Path(__file__).parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

qwen_pkg_dir = current_dir / "qwen_tts"
if qwen_pkg_dir.exists() and str(qwen_pkg_dir) not in sys.path:
    sys.path.insert(0, str(qwen_pkg_dir))

_IMPORT_ERROR = None
Qwen3TTSModel = None

def _ensure_qwen_package():
    if not qwen_pkg_dir.exists():
        return
    pkg_name = "qwen_tts"
    if pkg_name not in sys.modules:
        try:
            import types
            module = types.ModuleType(pkg_name)
            module.__path__ = [str(qwen_pkg_dir)]
            sys.modules[pkg_name] = module
        except Exception:
            return
    core_dir = qwen_pkg_dir / "core"
    if core_dir.exists() and "qwen_tts.core" not in sys.modules:
        try:
            import types
            core_module = types.ModuleType("qwen_tts.core")
            core_module.__path__ = [str(core_dir)]
            sys.modules["qwen_tts.core"] = core_module

            tokenizer_12hz_dir = core_dir / "tokenizer_12hz"
            if tokenizer_12hz_dir.exists():
                token_pkg = types.ModuleType("qwen_tts.core.tokenizer_12hz")
                token_pkg.__path__ = [str(tokenizer_12hz_dir)]
                sys.modules["qwen_tts.core.tokenizer_12hz"] = token_pkg

                import importlib.util
                cfg_path = tokenizer_12hz_dir / "configuration_qwen3_tts_tokenizer_v2.py"
                mdl_path = tokenizer_12hz_dir / "modeling_qwen3_tts_tokenizer_v2.py"
                if cfg_path.exists():
                    spec = importlib.util.spec_from_file_location(
                        "qwen_tts.core.tokenizer_12hz.configuration_qwen3_tts_tokenizer_v2",
                        str(cfg_path),
                    )
                    if spec and spec.loader:
                        mod = importlib.util.module_from_spec(spec)
                        sys.modules[spec.name] = mod
                        spec.loader.exec_module(mod)
                        core_module.Qwen3TTSTokenizerV2Config = getattr(mod, "Qwen3TTSTokenizerV2Config")
                if mdl_path.exists():
                    spec = importlib.util.spec_from_file_location(
                        "qwen_tts.core.tokenizer_12hz.modeling_qwen3_tts_tokenizer_v2",
                        str(mdl_path),
                    )
                    if spec and spec.loader:
                        mod = importlib.util.module_from_spec(spec)
                        sys.modules[spec.name] = mod
                        spec.loader.exec_module(mod)
                        core_module.Qwen3TTSTokenizerV2Model = getattr(mod, "Qwen3TTSTokenizerV2Model")

                try:
                    from transformers import PretrainedConfig, PreTrainedModel

                    class Qwen3TTSTokenizerV1Config(PretrainedConfig):
                        model_type = "qwen3_tts_tokenizer_25hz"

                    class Qwen3TTSTokenizerV1Model(PreTrainedModel):
                        config_class = Qwen3TTSTokenizerV1Config

                        def __init__(self, config):
                            super().__init__(config)

                        def forward(self, *args, **kwargs):
                            raise RuntimeError("Tokenizer 25Hz is not supported in this node.")

                    core_module.Qwen3TTSTokenizerV1Config = Qwen3TTSTokenizerV1Config
                    core_module.Qwen3TTSTokenizerV1Model = Qwen3TTSTokenizerV1Model
                except Exception:
                    pass
        except Exception:
            return

def _load_qwen3_model():
    global Qwen3TTSModel, _IMPORT_ERROR
    if Qwen3TTSModel is not None:
        return Qwen3TTSModel
    _ensure_qwen_package()
    try:
        import importlib.util
        model_path = qwen_pkg_dir / "inference" / "qwen3_tts_model.py"
        if not model_path.exists():
            raise FileNotFoundError(f"Missing qwen3_tts_model.py at {model_path}")
        spec = importlib.util.spec_from_file_location(
            "qwen_tts.inference.qwen3_tts_model",
            str(model_path),
        )
        if not spec or not spec.loader:
            raise ImportError("Failed to create spec for qwen3_tts_model")
        module = importlib.util.module_from_spec(spec)
        sys.modules["qwen_tts.inference.qwen3_tts_model"] = module
        spec.loader.exec_module(module)
        Qwen3TTSModel = getattr(module, "Qwen3TTSModel")
        return Qwen3TTSModel
    except Exception as e:
        _IMPORT_ERROR = str(e)
        print(f"[Qwen3-TTS] Failed to import qwen3_tts_model: {e}")
        return None

LANGUAGE_CHOICES = [
    "Auto",
    "Chinese",
    "English",
    "Japanese",
    "Korean",
    "French",
    "German",
    "Spanish",
    "Portuguese",
    "Russian",
    "Italian",
]
LANGUAGE_MAP = {
    "Auto": "auto",
    "Chinese": "chinese",
    "English": "english",
    "Japanese": "japanese",
    "Korean": "korean",
    "French": "french",
    "German": "german",
    "Spanish": "spanish",
    "Portuguese": "portuguese",
    "Russian": "russian",
    "Italian": "italian",
}

SPEAKER_CHOICES = [
    "Aiden",
    "Dylan",
    "Eric",
    "Ono_Anna",
    "Ryan",
    "Serena",
    "Sohee",
    "Uncle_Fu",
    "Vivian",
]

MODEL_ID_MAP = {
    ("Base", "0.6B"): "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
    ("Base", "1.7B"): "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    ("CustomVoice", "0.6B"): "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
    ("CustomVoice", "1.7B"): "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
    ("VoiceDesign", "1.7B"): "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
}
TOKENIZER_ID = "Qwen/Qwen3-TTS-Tokenizer-12Hz"
ATTENTION_OPTIONS = ["auto", "sage_attn", "flash_attn", "sdpa", "eager"]
_MODEL_CACHE: Dict[Tuple[str, str, str, str, str], Any] = {}


def _available_devices():
    devices = ["auto"]
    if torch.cuda.is_available():
        devices.append("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        devices.append("mps")
    devices.append("cpu")
    return devices


def _flash_attn_available():
    try:
        import flash_attn  # noqa: F401
        return True
    except Exception:
        return False


def _sage_attn_available():
    try:
        import sageattention  # noqa: F401
        return True
    except Exception:
        return False


def _resolve_attention_impl(attention: str, device: str):
    if not device.startswith("cuda"):
        return None
    if attention == "auto":
        if _flash_attn_available():
            return "flash_attention_2"
        if _sage_attn_available():
            return "sage_attn"
        return "sdpa"
    if attention == "sage_attn":
        return "sage_attn"
    if attention == "flash_attn":
        if _flash_attn_available():
            return "flash_attention_2"
        return None
    if attention == "sdpa":
        return "sdpa"
    if attention == "eager":
        return "eager"
    return None


def _patch_sage_attention(model):
    try:
        from sageattention import sageattn
    except Exception:
        raise RuntimeError("sageattention is not installed. Please install it to use sage_attn.")

    patched = 0
    for name, module in model.model.named_modules():
        if getattr(module, "_qwen_sage_patched", False):
            continue
        if "attn" not in name.lower() and "Attention" not in type(module).__name__:
            continue
        if not hasattr(module, "forward"):
            continue

        original_forward = module.forward

        def make_sage_forward(orig_forward):
            def sage_forward(*args, **kwargs):
                if len(args) >= 3:
                    q, k, v = args[0], args[1], args[2]
                    if hasattr(q, "shape") and hasattr(k, "shape") and hasattr(v, "shape"):
                        attn_mask = kwargs.get("attention_mask", None)
                        try:
                            return sageattn(q, k, v, is_causal=False, attn_mask=attn_mask)
                        except Exception:
                            return orig_forward(*args, **kwargs)
                return orig_forward(*args, **kwargs)

            return sage_forward

        module.forward = make_sage_forward(original_forward)
        module._qwen_sage_patched = True
        patched += 1

    print(f"[Qwen3-TTS] sage_attn patched modules: {patched}")


def _resolve_device(device_choice: str):
    if device_choice == "auto":
        if torch.cuda.is_available():
            return str(model_management.get_torch_device())
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    if device_choice == "cuda" and torch.cuda.is_available():
        return str(model_management.get_torch_device())
    if device_choice == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _resolve_dtype(precision: str, device: str):
    if device == "mps":
        if precision in ("bf16", "fp16"):
            return torch.float16
        return torch.float32
    if precision == "bf16":
        return torch.bfloat16
    if precision == "fp16":
        return torch.float16
    return torch.float32


def _maybe_autocast(device: str, precision: str):
    if not device.startswith("cuda"):
        return nullcontext()
    if precision == "bf16":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    if precision == "fp16":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def _model_store_root():
    tts_paths = _get_tts_paths()
    if tts_paths:
        return os.path.join(tts_paths[0], "Qwen3-TTS")
    base = folder_paths.models_dir
    return os.path.join(base, "TTS", "Qwen3-TTS")


def _get_tts_paths():
    folders = getattr(folder_paths, "folder_names_and_paths", {}) or {}
    for key in ("tts", "TTS"):
        if key in folders:
            return folder_paths.get_folder_paths(key) or []
    return []


def _find_local_model(model_id: str) -> Optional[str]:
    model_name = model_id.split("/")[-1]
    candidates = []
    default_root = _model_store_root()
    if os.path.isdir(default_root):
        candidates.append(os.path.join(default_root, model_name))
    try:
        tts_roots = _get_tts_paths()
        for root in tts_roots:
            candidates.append(os.path.join(root, "Qwen3-TTS", model_name))
    except Exception:
        pass

    for path in candidates:
        if os.path.isdir(path) and os.listdir(path):
            return path
    return None


def _download_model(model_id: str) -> Optional[str]:
    if snapshot_download is None:
        return None
    target_root = _model_store_root()
    os.makedirs(target_root, exist_ok=True)
    target_dir = os.path.join(target_root, model_id.split("/")[-1])
    if os.path.isdir(target_dir) and os.listdir(target_dir):
        return target_dir
    try:
        snapshot_download(repo_id=model_id, local_dir=target_dir, local_dir_use_symlinks=False)
        return target_dir
    except Exception as e:
        print(f"[Qwen3-TTS] Download failed for {model_id}: {e}")
        return None


def _resolve_model_source(model_id: str) -> str:
    local_path = _find_local_model(model_id)
    if local_path:
        return local_path
    dl_path = _download_model(model_id)
    if dl_path:
        return dl_path
    return model_id


def _load_model(model_type: str, model_size: str, device_choice: str, precision: str, attention: str = "auto"):
    if model_type == "VoiceDesign" and model_size != "1.7B":
        raise ValueError("VoiceDesign only supports 1.7B models.")
    if Qwen3TTSModel is None:
        _load_qwen3_model()
    if Qwen3TTSModel is None:
        hint = _IMPORT_ERROR or "unknown import error"
        raise RuntimeError(
            "qwen_tts is not available. Please install dependencies in your ComfyUI environment "
            f"and check the package path. Import error: {hint}"
        )

    model_id = MODEL_ID_MAP.get((model_type, model_size))
    if model_id is None:
        raise ValueError(f"Unsupported model type/size: {model_type}/{model_size}")

    device = _resolve_device(device_choice)
    dtype = _resolve_dtype(precision, device)

    attn_impl = _resolve_attention_impl(attention, device)
    cache_key = (model_type, model_size, device, precision, attn_impl or "default")
    if cache_key in _MODEL_CACHE:
        print(f"[Qwen3-TTS] Using cached model: {model_type} {model_size} on {device} ({precision})")
        return _MODEL_CACHE[cache_key]

    source = _resolve_model_source(model_id)
    if device.startswith("cuda"):
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass
    try:
        if attn_impl == "sage_attn":
            if not _sage_attn_available():
                print("[Qwen3-TTS] sage_attn not available. Install 'sageattention' or choose another attention.")
                attn_impl = None
            else:
                model = Qwen3TTSModel.from_pretrained(source, device_map=device, dtype=dtype)
                _patch_sage_attention(model)
                print("[Qwen3-TTS] attention: sage_attn")
        if attn_impl != "sage_attn":
            kwargs = {"device_map": device, "dtype": dtype}
            if attn_impl:
                kwargs["attn_implementation"] = attn_impl
            model = Qwen3TTSModel.from_pretrained(source, **kwargs)
            if device.startswith("cuda") and attn_impl:
                print(f"[Qwen3-TTS] attention: {attn_impl}")
    except Exception as e:
        if "attn_implementation" in str(e) or "flash" in str(e).lower():
            print("[Qwen3-TTS] attention unavailable, fallback to default attention")
            model = Qwen3TTSModel.from_pretrained(source, device_map=device, dtype=dtype)
        else:
            raise e

    _MODEL_CACHE[cache_key] = model
    return model


def _set_seed(seed: int):
    if seed is None or seed < 0:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed % (2 ** 32))


def _get_prompt_item_class():
    try:
        _load_qwen3_model()
        module = __import__("qwen_tts.inference.qwen3_tts_model", fromlist=["VoiceClonePromptItem"])
        return getattr(module, "VoiceClonePromptItem", None)
    except Exception:
        return None


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


def _deserialize_prompt_items(items):
    return [_deserialize_prompt_item(item) for item in items]


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


def _audio_to_tuple(audio: Any) -> Tuple[np.ndarray, int]:
    waveform = None
    sr = None

    if isinstance(audio, dict):
        if "waveform" in audio:
            waveform = audio.get("waveform")
            sr = audio.get("sample_rate") or audio.get("sr") or audio.get("sampling_rate")
        elif "data" in audio and "sampling_rate" in audio:
            waveform = audio.get("data")
            sr = audio.get("sampling_rate")
        elif "audio" in audio and isinstance(audio["audio"], (tuple, list)):
            a0, a1 = audio["audio"]
            if isinstance(a0, (int, float)):
                sr, waveform = int(a0), a1
            else:
                waveform, sr = a0, int(a1)
    elif isinstance(audio, tuple) and len(audio) == 2:
        a0, a1 = audio
        if isinstance(a0, (int, float)):
            sr, waveform = int(a0), a1
        else:
            waveform, sr = a0, int(a1)
    elif isinstance(audio, list) and len(audio) == 2:
        waveform, sr = audio[0], int(audio[1])

    if sr is None or waveform is None:
        raise ValueError("Invalid AUDIO input")

    if isinstance(waveform, torch.Tensor):
        wav = waveform.detach()
        if wav.dim() > 1:
            wav = wav.squeeze()
            if wav.dim() > 1:
                wav = wav.mean(dim=0)
        wav = wav.cpu().numpy()
    else:
        wav = np.asarray(waveform)

    if wav.ndim > 1:
        wav = np.mean(wav, axis=0)

    wav = wav.astype(np.float32)
    if wav.size < 1024:
        pad = 1024 - wav.size
        wav = np.concatenate([wav, np.zeros(pad, dtype=np.float32)])

    return wav, int(sr)


def _to_comfy_audio(wavs, sr: int):
    if isinstance(wavs, list) and len(wavs) > 0:
        wav = wavs[0]
    else:
        wav = wavs
    if isinstance(wav, np.ndarray):
        tensor = torch.from_numpy(wav).float()
    else:
        tensor = wav.float()
    if tensor.dim() > 1:
        tensor = tensor.squeeze()
    tensor = tensor.unsqueeze(0).unsqueeze(0)
    return {"waveform": tensor, "sample_rate": int(sr)}


def _custom_voice_generate(
    text: str,
    speaker: str,
    model_size: str,
    device: str,
    precision: str,
    language: str,
    instruct: str = "",
    seed: int = -1,
    max_new_tokens: int = 2048,
    do_sample: bool = True,
    top_p: float = 0.9,
    top_k: int = 50,
    temperature: float = 0.9,
    repetition_penalty: float = 1.0,
    attention: str = "auto",
    unload_models: bool = False,
):
    if not text or not text.strip():
        raise ValueError("Text is required")
    _set_seed(seed)
    model = _load_model("CustomVoice", model_size, device, precision, attention)
    mapped_lang = LANGUAGE_MAP.get(language, "auto")
    with _maybe_autocast(_resolve_device(device), precision):
        wavs, sr = model.generate_custom_voice(
            text=text,
            language=mapped_lang,
            speaker=str(speaker).lower().replace(" ", "_"),
            instruct=instruct if instruct and instruct.strip() else None,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            top_p=top_p,
            top_k=top_k,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
        )
    audio = _to_comfy_audio(wavs, sr)
    if unload_models:
        _MODEL_CACHE.clear()
        model_management.soft_empty_cache()
        try:
            import gc
            gc.collect()
            gc.collect()
        except Exception:
            pass
    return (audio,)


def _voice_design_generate(
    text: str,
    instruct: str,
    model_size: str,
    device: str,
    precision: str,
    language: str,
    seed: int = -1,
    max_new_tokens: int = 2048,
    do_sample: bool = True,
    top_p: float = 0.9,
    top_k: int = 50,
    temperature: float = 0.9,
    repetition_penalty: float = 1.0,
    attention: str = "auto",
    unload_models: bool = False,
):
    if not text or not text.strip() or not instruct or not instruct.strip():
        raise ValueError("Text and instruct are required")
    _set_seed(seed)
    model = _load_model("VoiceDesign", model_size, device, precision, attention)
    mapped_lang = LANGUAGE_MAP.get(language, "auto")
    with _maybe_autocast(_resolve_device(device), precision):
        wavs, sr = model.generate_voice_design(
            text=text,
            language=mapped_lang,
            instruct=instruct,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            top_p=top_p,
            top_k=top_k,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
        )
    audio = _to_comfy_audio(wavs, sr)
    if unload_models:
        _MODEL_CACHE.clear()
        model_management.soft_empty_cache()
        try:
            import gc
            gc.collect()
            gc.collect()
        except Exception:
            pass
    return (audio,)


def _voice_clone_generate(
    reference_audio: Any,
    target_text: str,
    model_size: str,
    device: str,
    precision: str,
    language: str,
    reference_text: str = "",
    x_vector_only: bool = False,
    allow_empty_ref_text: bool = False,
    voice: Any = None,
    seed: int = -1,
    max_new_tokens: int = 2048,
    do_sample: bool = True,
    top_p: float = 0.9,
    top_k: int = 50,
    temperature: float = 0.9,
    repetition_penalty: float = 1.0,
    attention: str = "auto",
    unload_models: bool = False,
):
    if not target_text or not target_text.strip():
        raise ValueError("Target text is required")
    prompt = None
    if voice is not None:
        if isinstance(voice, str) and voice.strip():
            prompt = _load_voice_from_file(voice.strip())
        elif isinstance(voice, list):
            prompt = voice
    if prompt is None:
        if (not reference_text or not reference_text.strip()) and not x_vector_only:
            if allow_empty_ref_text:
                x_vector_only = True
            else:
                raise ValueError("reference_text is required unless x_vector_only is enabled")
        if reference_audio is None:
            raise ValueError("reference_audio is required when voice is not provided")
    _set_seed(seed)
    model = _load_model("Base", model_size, device, precision, attention)
    mapped_lang = LANGUAGE_MAP.get(language, "auto")
    with _maybe_autocast(_resolve_device(device), precision):
        if prompt is not None:
            wavs, sr = model.generate_voice_clone(
                text=target_text,
                language=mapped_lang,
                voice_clone_prompt=prompt,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                top_p=top_p,
                top_k=top_k,
                temperature=temperature,
                repetition_penalty=repetition_penalty,
            )
        else:
            audio_tuple = _audio_to_tuple(reference_audio)
            wavs, sr = model.generate_voice_clone(
                text=target_text,
                language=mapped_lang,
                ref_audio=audio_tuple,
                ref_text=reference_text.strip() if reference_text and reference_text.strip() else None,
                x_vector_only_mode=bool(x_vector_only),
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                top_p=top_p,
                top_k=top_k,
                temperature=temperature,
                repetition_penalty=repetition_penalty,
            )
    audio = _to_comfy_audio(wavs, sr)
    if unload_models:
        _MODEL_CACHE.clear()
        model_management.soft_empty_cache()
        try:
            import gc
            gc.collect()
            gc.collect()
        except Exception:
            pass
    return (audio,)


class Qwen3TTSCustomVoiceBasic:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": OrderedDict([
                ("text", ("STRING", {"multiline": True, "default": "Hello from Qwen3-TTS.", "tooltip": "Text to synthesize"})),
                ("speaker", (SPEAKER_CHOICES, {"default": "Ryan"})),
                ("model_size", (["0.6B", "1.7B"], {"default": "1.7B"})),
                ("language", (LANGUAGE_CHOICES, {"default": "Auto"})),
            ]),
            "optional": OrderedDict([
                ("instruct", ("STRING", {"multiline": True, "default": "", "tooltip": "Style instruction"})),
                ("unload_models", ("BOOLEAN", {"default": True, "tooltip": "Unload cached models after generation"})),
                ("seed", ("INT", {"default": -1, "min": -1, "max": 0xffffffffffffffff})),
            ]),
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "generate"
    CATEGORY = "🧪AILab/🎙️QwenTTS"

    def generate(self, text, speaker, model_size, language, instruct="", unload_models=True, seed=-1):
        return _custom_voice_generate(
            text=text,
            speaker=speaker,
            model_size=model_size,
            device="auto",
            precision="bf16",
            language=language,
            instruct=instruct,
            seed=seed,
            max_new_tokens=2048,
            do_sample=False,
            top_p=0.9,
            top_k=50,
            temperature=0.9,
            repetition_penalty=1.0,
            unload_models=unload_models,
        )


class Qwen3TTSCustomVoiceAdvanced:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": OrderedDict([
                ("text", ("STRING", {"multiline": True, "default": "Hello from Qwen3-TTS.", "tooltip": "Text to synthesize"})),
                ("speaker", (SPEAKER_CHOICES, {"default": "Ryan"})),
                ("model_size", (["0.6B", "1.7B"], {"default": "1.7B"})),
                ("device", (_available_devices(), {"default": "auto"})),
                ("precision", (["bf16", "fp16", "fp32"], {"default": "bf16"})),
                ("language", (LANGUAGE_CHOICES, {"default": "Auto"})),
            ]),
            "optional": OrderedDict([
                ("instruct", ("STRING", {"multiline": True, "default": "", "tooltip": "Style instruction"})),
                ("max_new_tokens", ("INT", {"default": 2048, "min": 256, "max": 4096, "step": 256})),
                ("do_sample", ("BOOLEAN", {"default": False})),
                ("top_p", ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0, "step": 0.05})),
                ("top_k", ("INT", {"default": 50, "min": 0, "max": 200, "step": 1})),
                ("temperature", ("FLOAT", {"default": 0.9, "min": 0.1, "max": 2.0, "step": 0.05})),
                ("repetition_penalty", ("FLOAT", {"default": 1.0, "min": 0.5, "max": 2.0, "step": 0.05})),
                ("attention", (ATTENTION_OPTIONS, {"default": "auto"})),
                ("unload_models", ("BOOLEAN", {"default": True, "tooltip": "Unload cached models after generation"})),
                ("seed", ("INT", {"default": -1, "min": -1, "max": 0xffffffffffffffff})),
            ]),
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "generate"
    CATEGORY = "🧪AILab/🎙️QwenTTS"

    def generate(self, text, speaker, model_size, device, precision, language, instruct="", max_new_tokens=2048, do_sample=True, top_p=0.9, top_k=50, temperature=0.9, repetition_penalty=1.0, attention="auto", unload_models=True, seed=-1):
        return _custom_voice_generate(
            text=text,
            speaker=speaker,
            model_size=model_size,
            device=device,
            precision=precision,
            language=language,
            instruct=instruct,
            seed=seed,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            top_p=top_p,
            top_k=top_k,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            attention=attention,
            unload_models=unload_models,
        )


class Qwen3TTSVoiceDesignBasic:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": OrderedDict([
                ("text", ("STRING", {"multiline": True, "default": "Hello from Qwen3-TTS VoiceDesign.", "tooltip": "Text to synthesize"})),
                ("instruct", ("STRING", {"multiline": True, "default": "A warm, gentle female voice.", "tooltip": "Voice description"})),
                ("model_size", (["1.7B"], {"default": "1.7B"})),
                ("language", (LANGUAGE_CHOICES, {"default": "Auto"})),
            ]),
            "optional": OrderedDict([
                ("unload_models", ("BOOLEAN", {"default": True, "tooltip": "Unload cached models after generation"})),
                ("seed", ("INT", {"default": -1, "min": -1, "max": 0xffffffffffffffff})),
            ]),
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "generate"
    CATEGORY = "🧪AILab/🎙️QwenTTS"

    def generate(self, text, instruct, model_size, language, unload_models=True, seed=-1):
        return _voice_design_generate(
            text=text,
            instruct=instruct,
            model_size=model_size,
            device="auto",
            precision="bf16",
            language=language,
            seed=seed,
            max_new_tokens=2048,
            do_sample=False,
            top_p=0.9,
            top_k=50,
            temperature=0.9,
            repetition_penalty=1.0,
            unload_models=unload_models,
        )


class Qwen3TTSVoiceDesignAdvanced:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": OrderedDict([
                ("text", ("STRING", {"multiline": True, "default": "Hello from Qwen3-TTS VoiceDesign.", "tooltip": "Text to synthesize"})),
                ("instruct", ("STRING", {"multiline": True, "default": "A warm, gentle female voice.", "tooltip": "Voice description"})),
                ("model_size", (["1.7B"], {"default": "1.7B"})),
                ("device", (_available_devices(), {"default": "auto"})),
                ("precision", (["bf16", "fp16", "fp32"], {"default": "bf16"})),
                ("language", (LANGUAGE_CHOICES, {"default": "Auto"})),
            ]),
            "optional": OrderedDict([
                ("max_new_tokens", ("INT", {"default": 2048, "min": 256, "max": 4096, "step": 256})),
                ("do_sample", ("BOOLEAN", {"default": False})),
                ("top_p", ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0, "step": 0.05})),
                ("top_k", ("INT", {"default": 50, "min": 0, "max": 200, "step": 1})),
                ("temperature", ("FLOAT", {"default": 0.9, "min": 0.1, "max": 2.0, "step": 0.05})),
                ("repetition_penalty", ("FLOAT", {"default": 1.0, "min": 0.5, "max": 2.0, "step": 0.05})),
                ("attention", (ATTENTION_OPTIONS, {"default": "auto"})),
                ("unload_models", ("BOOLEAN", {"default": True})),
                ("seed", ("INT", {"default": -1, "min": -1, "max": 0xffffffffffffffff})),
            ]),
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "generate"
    CATEGORY = "🧪AILab/🎙️QwenTTS"

    def generate(self, text, instruct, model_size, device, precision, language, max_new_tokens=2048, do_sample=True, top_p=0.9, top_k=50, temperature=0.9, repetition_penalty=1.0, attention="auto", unload_models=True, seed=-1):
        return _voice_design_generate(
            text=text,
            instruct=instruct,
            model_size=model_size,
            device=device,
            precision=precision,
            language=language,
            seed=seed,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            top_p=top_p,
            top_k=top_k,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            attention=attention,
            unload_models=unload_models,
        )


class Qwen3TTSVoiceCloneBasic:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": OrderedDict([
                ("target_text", ("STRING", {"multiline": True, "default": "Hello, this is a cloned voice.", "tooltip": "Text to speak"})),
                ("model_size", (["0.6B", "1.7B"], {"default": "1.7B"})),
                ("language", (LANGUAGE_CHOICES, {"default": "Auto"})),
            ]),
            "optional": OrderedDict([
                ("reference_audio", ("AUDIO", {"tooltip": "Reference audio for cloning (not needed if voice is provided)"})),
                ("reference_text", ("STRING", {"multiline": True, "default": "", "tooltip": "Transcript of reference audio"})),
                ("x_vector_only", ("BOOLEAN", {"default": False, "tooltip": "Skip ref_text by using speaker embedding only"})),
                ("voice", ("VOICE",)),
                ("unload_models", ("BOOLEAN", {"default": True, "tooltip": "Unload cached models after generation"})),
                ("seed", ("INT", {"default": -1, "min": -1, "max": 0xffffffffffffffff})),
            ]),
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "generate"
    CATEGORY = "🧪AILab/🎙️QwenTTS"

    def generate(self, target_text, model_size, language, reference_audio=None, reference_text="", x_vector_only=False, voice=None, unload_models=True, seed=-1):
        return _voice_clone_generate(
            reference_audio=reference_audio,
            target_text=target_text,
            model_size=model_size,
            device="auto",
            precision="bf16",
            language=language,
            reference_text=reference_text,
            x_vector_only=x_vector_only,
            allow_empty_ref_text=True,
            voice=voice,
            seed=seed,
            max_new_tokens=2048,
            do_sample=False,
            top_p=0.9,
            top_k=50,
            temperature=0.9,
            repetition_penalty=1.0,
            unload_models=unload_models,
        )


class Qwen3TTSVoiceCloneAdvanced:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": OrderedDict([
                ("target_text", ("STRING", {"multiline": True, "default": "Hello, this is a cloned voice.", "tooltip": "Text to speak"})),
                ("model_size", (["0.6B", "1.7B"], {"default": "1.7B"})),
                ("device", (_available_devices(), {"default": "auto"})),
                ("precision", (["bf16", "fp16", "fp32"], {"default": "bf16"})),
                ("language", (LANGUAGE_CHOICES, {"default": "Auto"})),
            ]),
            "optional": OrderedDict([
                ("reference_audio", ("AUDIO", {"tooltip": "Reference audio for cloning (not needed if voice is provided)"})),
                ("reference_text", ("STRING", {"multiline": True, "default": "", "tooltip": "Transcript of reference audio"})),
                ("x_vector_only", ("BOOLEAN", {"default": False, "tooltip": "Skip ref_text by using speaker embedding only"})),
                ("voice", ("VOICE",)),
                ("max_new_tokens", ("INT", {"default": 2048, "min": 256, "max": 4096, "step": 256})),
                ("do_sample", ("BOOLEAN", {"default": False})),
                ("top_p", ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0, "step": 0.05})),
                ("top_k", ("INT", {"default": 50, "min": 0, "max": 200, "step": 1})),
                ("temperature", ("FLOAT", {"default": 0.9, "min": 0.1, "max": 2.0, "step": 0.05})),
                ("repetition_penalty", ("FLOAT", {"default": 1.0, "min": 0.5, "max": 2.0, "step": 0.05})),
                ("attention", (ATTENTION_OPTIONS, {"default": "auto"})),
                ("unload_models", ("BOOLEAN", {"default": True})),
                ("seed", ("INT", {"default": -1, "min": -1, "max": 0xffffffffffffffff})),
            ]),
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "generate"
    CATEGORY = "🧪AILab/🎙️QwenTTS"

    def generate(self, target_text, model_size, device, precision, language, reference_audio=None, reference_text="", x_vector_only=False, voice=None, max_new_tokens=2048, do_sample=True, top_p=0.9, top_k=50, temperature=0.9, repetition_penalty=1.0, attention="auto", unload_models=True, seed=-1):
        return _voice_clone_generate(
            reference_audio=reference_audio,
            target_text=target_text,
            model_size=model_size,
            device=device,
            precision=precision,
            language=language,
            reference_text=reference_text,
            x_vector_only=x_vector_only,
            allow_empty_ref_text=False,
            voice=voice,
            seed=seed,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            top_p=top_p,
            top_k=top_k,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            attention=attention,
            unload_models=unload_models,
        )


NODE_CLASS_MAPPINGS = {
    "AILab_Qwen3TTSCustomVoice": Qwen3TTSCustomVoiceBasic,
    "AILab_Qwen3TTSCustomVoice_Advanced": Qwen3TTSCustomVoiceAdvanced,
    "AILab_Qwen3TTSVoiceDesign": Qwen3TTSVoiceDesignBasic,
    "AILab_Qwen3TTSVoiceDesign_Advanced": Qwen3TTSVoiceDesignAdvanced,
    "AILab_Qwen3TTSVoiceClone": Qwen3TTSVoiceCloneBasic,
    "AILab_Qwen3TTSVoiceClone_Advanced": Qwen3TTSVoiceCloneAdvanced,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AILab_Qwen3TTSCustomVoice": "Custom Voice (QwenTTS)",
    "AILab_Qwen3TTSCustomVoice_Advanced": "Custom Voice (QwenTTS) Advanced",
    "AILab_Qwen3TTSVoiceDesign": "Voice Design (QwenTTS)",
    "AILab_Qwen3TTSVoiceDesign_Advanced": "Voice Design (QwenTTS) Advanced",
    "AILab_Qwen3TTSVoiceClone": "Voice Clone (QwenTTS)",
    "AILab_Qwen3TTSVoiceClone_Advanced": "Voice Clone (QwenTTS) Advanced",
}
