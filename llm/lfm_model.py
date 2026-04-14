import json
import os
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Prefer local LFM2.5-1.2B-Instruct (no HuggingFace API); fallback to HuggingFace
MODEL_ID_HF = "liquidai/LFM2.5-1.2B-Thinking"
_BASE = Path(__file__).resolve().parent.parent
DEFAULT_LOCAL_PATH = _BASE / "LFM2.5-1.2B-Instruct"
MODEL_UNDER_MODELS = _BASE / "models" / "LFM2.5-1.2B-Instruct"

_tokenizer = None
_model = None
_loaded_path = None


def _model_path():
    """Local path if set or exists; else None (use HuggingFace)."""
    path_env = os.environ.get("LOCAL_LLM_PATH", "").strip()
    if path_env:
        p = Path(path_env).expanduser().resolve()
        if p.exists():
            return p
    for candidate in (DEFAULT_LOCAL_PATH, MODEL_UNDER_MODELS):
        if candidate.exists():
            return candidate
    return None


def is_available():
    """True if local model path exists and can be used."""
    return _model_path() is not None


def _verify_local_hf_snapshot(model_dir: Path) -> None:
    cfg = model_dir / "config.json"
    if not cfg.is_file():
        raise FileNotFoundError(f"Missing config.json under {model_dir}.")
    raw = cfg.read_text(encoding="utf-8", errors="replace").strip()
    if raw.startswith("version https://git-lfs"):
        raise ValueError(f"{cfg} is a Git LFS pointer; download the real model files.")
    obj = json.loads(raw)
    if not obj.get("model_type"):
        raise ValueError(f"{cfg} has no model_type.")


def load_model():
    global _tokenizer, _model, _loaded_path

    local_path = _model_path()
    use_local = local_path is not None
    path_str = str(local_path) if use_local else MODEL_ID_HF

    if _model is not None and _loaded_path == path_str:
        return _tokenizer, _model

    if use_local:
        _verify_local_hf_snapshot(Path(path_str))
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        _tokenizer = AutoTokenizer.from_pretrained(path_str, local_files_only=True, trust_remote_code=True)
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True  # faster convs when input sizes fixed
        _model = AutoModelForCausalLM.from_pretrained(
            path_str,
            local_files_only=True,
            device_map="auto" if torch.cuda.is_available() else None,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
        )
        if not torch.cuda.is_available():
            _model = _model.to("cpu")
    else:
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_ID_HF)
        _model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID_HF,
            device_map="auto",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        )
    _model.eval()
    _loaded_path = path_str
    return _tokenizer, _model
