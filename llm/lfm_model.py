import os
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Prefer local LFM2.5-1.2B-Instruct (no HuggingFace API); fallback to HuggingFace
MODEL_ID_HF = "liquidai/LFM2.5-1.2B-Thinking"
_BASE = Path(__file__).resolve().parent.parent
DEFAULT_LOCAL_PATH = _BASE / "LFM2.5-1.2B-Instruct"

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
    if DEFAULT_LOCAL_PATH.exists():
        return DEFAULT_LOCAL_PATH
    return None


def is_available():
    """True if local model path exists and can be used."""
    return _model_path() is not None


def load_model():
    global _tokenizer, _model, _loaded_path

    local_path = _model_path()
    use_local = local_path is not None
    path_str = str(local_path) if use_local else MODEL_ID_HF

    if _model is not None and _loaded_path == path_str:
        return _tokenizer, _model

    if use_local:
        _tokenizer = AutoTokenizer.from_pretrained(path_str, local_files_only=True)
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True  # faster convs when input sizes fixed
        _model = AutoModelForCausalLM.from_pretrained(
            path_str,
            local_files_only=True,
            device_map="auto" if torch.cuda.is_available() else None,
            dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            trust_remote_code=True,
        )
        if not torch.cuda.is_available():
            _model = _model.to("cpu")
    else:
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_ID_HF)
        _model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID_HF,
            device_map="auto",
            dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        )
    _model.eval()
    _loaded_path = path_str
    return _tokenizer, _model
