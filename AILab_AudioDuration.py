import os
import subprocess
from typing import Any, Optional, Tuple

import numpy as np
import torch


class AILab_AudioDuration:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "audio_path": ("STRING", {"multiline": False, "default": ""}),
                "audio": ("AUDIO",),
                "fps": ("FLOAT",),
            },
        }

    RETURN_TYPES = ("INT", "FLOAT", "INT", "STRING")
    RETURN_NAMES = ("duration_int", "duration_float", "frames", "audio_path")
    FUNCTION = "run"
    CATEGORY = "🧪AILab/🔊Audio"

    def run(self, audio_path: str = "", audio: Any = None, fps: Optional[float] = None) -> Tuple[int, float, int, float, str]:
        seconds = None
        frames = 0
        out_path = audio_path or ""

        if isinstance(audio_path, str) and audio_path.strip():
            path = audio_path.strip()
            if not os.path.isfile(path):
                raise RuntimeError(f"audio_path not found: {path}")
            seconds = self._probe_duration(path)
            out_path = path
        elif audio is not None:
            seconds = self._duration_from_audio(audio)
            extracted = self._extract_path(audio)
            if extracted:
                out_path = extracted
        else:
            raise RuntimeError("Provide audio_path or connect an AUDIO input.")

        if seconds is None or seconds <= 0:
            raise RuntimeError("Could not determine a valid duration.")

        ms = float(seconds) * 1000.0
        if isinstance(fps, (int, float)) and fps > 0:
            round_sec = round(ms / 1000.0)
            frames = int(fps * (round_sec / 8 * 8) + 1)
        return (int(seconds), float(seconds), int(frames), str(out_path))

    @staticmethod
    def _probe_duration(path: str) -> float:
        cmd = [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            path,
        ]
        out = subprocess.check_output(cmd, text=True).strip()
        try:
            return float(out)
        except ValueError:
            raise RuntimeError(f"ffprobe returned non-numeric duration: {out!r}")

    @staticmethod
    def _duration_from_audio(audio: Any) -> Optional[float]:
        if isinstance(audio, dict):
            d = audio.get("duration")
            if isinstance(d, (int, float)) and d > 0:
                return float(d)

        waveform, sr = AILab_AudioDuration._extract_waveform(audio)
        if waveform is None or not sr:
            return None

        frames = AILab_AudioDuration._count_frames(waveform)
        if not frames:
            return None
        return float(frames) / float(sr)

    @staticmethod
    def _extract_waveform(audio: Any) -> Tuple[Optional[Any], Optional[float]]:
        if isinstance(audio, dict):
            waveform = audio.get("waveform")
            if waveform is None:
                waveform = audio.get("samples")
            if waveform is None:
                waveform = audio.get("data")
            sr = audio.get("sample_rate")
            if sr is None:
                sr = audio.get("sr")
            return waveform, AILab_AudioDuration._to_float(sr)
        return None, None

    @staticmethod
    def _extract_path(audio: Any) -> Optional[str]:
        if isinstance(audio, dict):
            for key in ("path", "filepath", "file", "audio_path"):
                value = audio.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
        return None

    @staticmethod
    def _count_frames(waveform: Any) -> Optional[int]:
        if isinstance(waveform, torch.Tensor):
            waveform = waveform.detach().cpu().numpy()
        if isinstance(waveform, np.ndarray):
            if waveform.ndim == 1:
                return waveform.shape[0]
            if waveform.ndim >= 2:
                return waveform.shape[-1]
        try:
            return len(waveform)
        except Exception:
            return None

    @staticmethod
    def _to_float(value: Any) -> Optional[float]:
        if value is None:
            return None
        try:
            if hasattr(value, "item"):
                return float(value.item())
            return float(value)
        except Exception:
            return None


NODE_CLASS_MAPPINGS = {
    "Audio Duration": AILab_AudioDuration,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Audio Duration": "Audio Duration & Frames",
}
