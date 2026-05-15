# -*- coding: utf-8 -*-
"""
Machine translation (Arabic: Google, then local LFM, then Argos; other targets: local LFM only):

  For target Arabic (ar):
  1) Google Cloud Translation v2 when the client is available (API key or ADC) and not disabled
  2) Local LFM when Google is unavailable or returns no usable translation
  3) Argos Translate (offline) as last resort for Arabic

Set EXTERNAL_MT_DISABLED=1 to block Google Translation API calls; Argos still works unless DISABLE_ARGOS_MT=1.
Optional DISABLE_GOOGLE_MT=1 to skip Google (LFM then Argos for ar).
ENABLE_LOCAL_LLM_TRANSLATION=0 disables LFM MT.

Optional: LEGALAI_MT_CACHE_DIR for on-disk MT chunk cache. Argos: pip install argostranslate.
"""

from __future__ import annotations

import hashlib
import html
import json
import logging
import os
import re
import threading
import time
import urllib.error
import urllib.request
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlencode

_LOGGER = logging.getLogger(__name__)

_CHUNK_CHARS = 4500
_ARGOS_CHUNK_CHARS = 1200
_MAX_ATTEMPTS = 3
_TRANSLATION_CACHE_MAX = 512
_SUPPORTED_TARGET_LANGS = {"ar", "en", "fr", "de"}

_v2_client = None
_translation_cache: "OrderedDict[str, Tuple[str, str]]" = OrderedDict()
_argos_install_lock = threading.Lock()
_argos_pairs_installed: set[str] = set()

_ISO639_3_TO_2 = {
    "eng": "en",
    "deu": "de",
    "fra": "fr",
    "spa": "es",
    "ita": "it",
    "por": "pt",
    "nld": "nl",
    "pol": "pl",
    "rus": "ru",
    "zho": "zh",
    "jpn": "ja",
    "kor": "ko",
    "ara": "ar",
}

def external_mt_disabled() -> bool:
    return os.getenv("EXTERNAL_MT_DISABLED", "").strip().lower() in ("1", "true", "yes")


def argos_mt_disabled() -> bool:
    return os.getenv("DISABLE_ARGOS_MT", "").strip().lower() in ("1", "true", "yes")


def mt_auto_per_chunk_for_mixed() -> bool:
    return os.getenv("MT_AUTO_PER_CHUNK_MIXED", "1").strip().lower() not in ("0", "false", "no")


def google_mt_disabled() -> bool:
    return os.getenv("DISABLE_GOOGLE_MT", "").strip().lower() in ("1", "true", "yes")


def local_lfm_mt_enabled() -> bool:
    return os.getenv("ENABLE_LOCAL_LLM_TRANSLATION", "1").strip().lower() not in ("0", "false", "no")


def _normalize_lang_code(value: str, *, fallback: str = "en") -> str:
    code = (value or "").strip().lower().split("-")[0]
    if len(code) == 3:
        code = _ISO639_3_TO_2.get(code, code[:2])
    if not code:
        return fallback
    return code


def _normalize_target_lang(value: Optional[str]) -> str:
    code = _normalize_lang_code(value or "ar", fallback="ar")
    if code not in _SUPPORTED_TARGET_LANGS:
        return "ar"
    return code


class _TranslateV2RestApiKeyClient:
    """
    Cloud Translation v2 via REST ?key=…

    The ``google.cloud.translate_v2.Client`` constructor accepts ``ClientOptions(api_key=…)`` but
    does not pass the key into requests; it still loads Application Default Credentials. For API-key
    auth we call the public REST endpoint instead.
    """

    _URL = "https://translation.googleapis.com/language/translate/v2"

    def __init__(self, api_key: str):
        self._api_key = api_key

    def translate(
        self,
        values: Any,
        target_language: Optional[str] = None,
        format_: Optional[str] = None,
        source_language: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        if not isinstance(values, str):
            raise TypeError("REST API key client expects a single string per call")
        url = f"{self._URL}?{urlencode({'key': self._api_key})}"
        body: Dict[str, Any] = {
            "q": values,
            "target": target_language or "ar",
            "format": (format_ or "text") or "text",
        }
        if source_language and str(source_language).strip().lower() not in ("und", "unknown", ""):
            body["source"] = source_language
        req = urllib.request.Request(
            url,
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json; charset=utf-8"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                parsed = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", errors="replace")[:800]
            raise RuntimeError(f"Cloud Translation HTTP {e.code}: {detail}") from e
        translations = (parsed.get("data") or {}).get("translations") or ()
        if not translations or not isinstance(translations[0], dict):
            raise ValueError("Cloud Translation returned no translations")
        return translations[0]


def _get_translate_v2_client():
    global _v2_client
    if _v2_client is not None:
        return _v2_client
    if external_mt_disabled() or google_mt_disabled():
        return None
    try:
        from google.cloud import translate_v2 as translate_v2

        api_key = os.getenv("GOOGLE_TRANSLATION_API_KEY", "").strip()
        if api_key:
            creds_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "").strip()
            if creds_path and not Path(creds_path).expanduser().is_file():
                _LOGGER.warning(
                    "GOOGLE_APPLICATION_CREDENTIALS points to a missing file; "
                    "ignoring it and using GOOGLE_TRANSLATION_API_KEY (REST) for Translation v2."
                )
            _v2_client = _TranslateV2RestApiKeyClient(api_key)
        else:
            _v2_client = translate_v2.Client()
    except Exception as e:
        _LOGGER.warning("Google Cloud Translation v2 unavailable: %s", e)
        _v2_client = None
    return _v2_client


def _glossary_path() -> Path:
    return Path(__file__).resolve().parent.parent / "data" / "legal_mt_glossary_v1.json"


def _parse_glossary_pairs(raw: Any) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    for p in raw or []:
        if isinstance(p, (list, tuple)) and len(p) >= 2:
            a, b = str(p[0]).strip(), str(p[1]).strip()
            if a and b:
                out.append((a.lower(), b))
    return out


def load_glossary_pairs() -> List[Tuple[str, str]]:
    """Global `pairs` from glossary JSON (no per-source rows). Kept for backwards compatibility."""
    return load_glossary_pairs_for_source("__")


def load_glossary_pairs_for_source(source_lang: str) -> List[Tuple[str, str]]:
    """
    Merges global `pairs` plus `by_source[iso2]` from glossary JSON (deduped, order preserved).
    """
    path = _glossary_path()
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        _LOGGER.debug("Glossary load skipped: %s", e)
        return []

    merged: List[Tuple[str, str]] = []
    seen: set[str] = set()
    src = (source_lang or "").strip().lower().split("-")[0]

    def _add(pairs: List[Tuple[str, str]]) -> None:
        for a, b in pairs:
            k = f"{a}\x00{b}"
            if k in seen:
                continue
            seen.add(k)
            merged.append((a, b))

    _add(_parse_glossary_pairs(raw.get("pairs")))
    by_src = raw.get("by_source") or {}
    if isinstance(by_src, dict):
        extra = by_src.get(src) or by_src.get(src.upper())
        _add(_parse_glossary_pairs(extra))
    return merged


def glossary_version_label() -> Optional[str]:
    try:
        raw = json.loads(_glossary_path().read_text(encoding="utf-8"))
        v = raw.get("version")
        return str(v) if v else "1"
    except Exception:
        return None


def _disk_cache_base() -> Optional[Path]:
    d = (os.getenv("LEGALAI_MT_CACHE_DIR") or "").strip()
    if not d:
        return None
    p = Path(d).expanduser()
    try:
        p.mkdir(parents=True, exist_ok=True)
    except Exception:
        return None
    return p


def _disk_cache_path(backend: str, source_lang: str, text: str) -> Optional[Path]:
    base = _disk_cache_base()
    if base is None:
        return None
    h = _translation_cache_key(backend, source_lang, text)
    return base / f"{backend}_{h[:2]}" / f"{h}.json"


def _disk_cache_get(backend: str, source_lang: str, text: str) -> Optional[Tuple[str, str]]:
    path = _disk_cache_path(backend, source_lang, text)
    if path is None or not path.is_file():
        return None
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
        t = obj.get("t")
        st = obj.get("st")
        if isinstance(t, str) and st in ("ok", "partial"):
            return t, st
    except Exception:
        return None
    return None


def _disk_cache_put(backend: str, source_lang: str, text: str, value: Tuple[str, str]) -> None:
    path = _disk_cache_path(backend, source_lang, text)
    if path is None:
        return
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({"t": value[0], "st": value[1]}, ensure_ascii=False),
            encoding="utf-8",
        )
    except Exception:
        pass


def _translation_cache_key(backend: str, source_lang: str, text: str) -> str:
    h = hashlib.sha256(f"{backend}|{source_lang}|{text}".encode("utf-8", errors="replace")).hexdigest()
    return h


def _cache_get(backend: str, source_lang: str, text: str) -> Optional[Tuple[str, str]]:
    k = _translation_cache_key(backend, source_lang, text)
    if k in _translation_cache:
        _translation_cache.move_to_end(k)
        return _translation_cache[k]
    d = _disk_cache_get(backend, source_lang, text)
    if d is not None:
        _translation_cache[k] = d
        _translation_cache.move_to_end(k)
        while len(_translation_cache) > _TRANSLATION_CACHE_MAX:
            _translation_cache.popitem(last=False)
    return d


def _cache_put(backend: str, source_lang: str, text: str, value: Tuple[str, str]) -> None:
    k = _translation_cache_key(backend, source_lang, text)
    _translation_cache[k] = value
    _translation_cache.move_to_end(k)
    while len(_translation_cache) > _TRANSLATION_CACHE_MAX:
        _translation_cache.popitem(last=False)
    _disk_cache_put(backend, source_lang, text, value)


def _apply_glossary_ar(text: str, pairs: List[Tuple[str, str]]) -> str:
    if not text or not pairs:
        return text
    t = text
    for src_lower, ar in pairs:
        if len(src_lower) < 4:
            continue
        t = re.compile(re.escape(src_lower), re.IGNORECASE).sub(ar, t)
    return t


def _argos_effective_source(src: str) -> str:
    s = (src or "").strip().lower().split("-")[0]
    if len(s) == 3:
        s = _ISO639_3_TO_2.get(s, s[:2])
    if s in ("und", "", "unknown"):
        return "en"
    return s


def _ensure_argos_pair_for_source(from_code: str) -> bool:
    """Download & install Argos package ``from_code`` → ``ar`` if missing (first run may need network)."""
    a = _argos_effective_source(from_code)
    if a == "ar":
        return False
    key = f"{a}->ar"
    if key in _argos_pairs_installed:
        return True
    try:
        import argostranslate.package as ap
    except ImportError:
        _LOGGER.warning("argostranslate not installed; run: pip install argostranslate")
        return False

    with _argos_install_lock:
        if key in _argos_pairs_installed:
            return True
        try:
            installed = ap.get_installed_packages()
            if any(p.from_code == a and p.to_code == "ar" for p in installed):
                _argos_pairs_installed.add(key)
                return True
            ap.update_package_index()
            available = ap.get_available_packages()
            pkg = next((p for p in available if p.from_code == a and p.to_code == "ar"), None)
            if pkg is None:
                return False
            _LOGGER.info("Installing Argos model %s → ar (first run may take a minute)…", a)
            ap.install_from_path(pkg.download())
            _argos_pairs_installed.add(key)
            return True
        except Exception as e:
            _LOGGER.warning("Argos package install for %s→ar failed: %s", a, e)
            return False


def _translate_piece_argos(piece: str, src: str) -> Optional[str]:
    if argos_mt_disabled():
        return None
    try:
        import argostranslate.translate as at
    except ImportError:
        return None

    fc = _argos_effective_source(src)
    for try_fc in (fc, "en"):
        if try_fc == "ar":
            continue
        if not _ensure_argos_pair_for_source(try_fc):
            continue
        try:
            return at.translate(piece, try_fc, "ar")
        except Exception as e:
            _LOGGER.debug("Argos translate %s→ar failed: %s", try_fc, e)
    return None


def _translate_via_google(text: str, src: str, pairs: List[Tuple[str, str]]) -> Tuple[Optional[str], str]:
    """Returns (translated_full_text or None, status ok|partial)."""
    client = _get_translate_v2_client()
    if client is None:
        return None, "skipped"

    chunks_out: List[str] = []
    overall = "ok"
    t = text
    i = 0
    backend = "google"
    while i < len(t):
        piece = t[i : i + _CHUNK_CHARS]
        i += _CHUNK_CHARS
        cached = _cache_get(backend, src, piece)
        if cached is not None:
            chunks_out.append(cached[0])
            if cached[1] == "partial":
                overall = "partial"
            continue
        translated_piece: Optional[str] = None
        st_piece = "ok"
        for attempt in range(_MAX_ATTEMPTS):
            try:
                kwargs: Dict[str, Any] = {"target_language": "ar", "format_": "text"}
                if src not in ("und", "unknown", ""):
                    kwargs["source_language"] = src
                res = client.translate(piece, **kwargs)
                if isinstance(res, dict):
                    translated_piece = res.get("translatedText") or piece
                else:
                    translated_piece = piece
                break
            except Exception as e:
                _LOGGER.warning("Google translate attempt %s failed: %s", attempt + 1, e)
                if attempt == _MAX_ATTEMPTS - 1:
                    translated_piece = piece
                    overall = "partial"
                    st_piece = "partial"
                else:
                    time.sleep(0.25 * (2**attempt))
        if translated_piece is None:
            translated_piece = piece
            overall = "partial"
            st_piece = "partial"
        out_piece = _apply_glossary_ar(translated_piece, pairs)
        _cache_put(backend, src, piece, (out_piece, st_piece))
        chunks_out.append(out_piece)
    return "".join(chunks_out), overall


def _translate_via_argos(text: str, src: str, pairs: List[Tuple[str, str]]) -> Tuple[Optional[str], str]:
    """Offline Argos NMT. Returns (text or None, ok|partial)."""
    chunks_out: List[str] = []
    overall = "ok"
    backend = "argos"
    t = text
    i = 0
    while i < len(t):
        piece = t[i : i + _ARGOS_CHUNK_CHARS]
        i += _ARGOS_CHUNK_CHARS
        cached = _cache_get(backend, src, piece)
        if cached is not None:
            chunks_out.append(cached[0])
            if cached[1] == "partial":
                overall = "partial"
            continue
        out = _translate_piece_argos(piece, src)
        if out is None:
            overall = "partial"
            st_piece = "partial"
        else:
            st_piece = "ok"
        out_piece = _apply_glossary_ar(out if out is not None else piece, pairs)
        _cache_put(backend, src, piece, (out_piece, st_piece if out is not None else "partial"))
        chunks_out.append(out_piece)
    return "".join(chunks_out), overall


def _translate_piece_local_lfm(piece: str, src: str, target: str) -> Optional[str]:
    if not local_lfm_mt_enabled():
        return None
    try:
        from . import local_llm as local_llm_mod
    except Exception:
        return None
    if not getattr(local_llm_mod, "is_available", lambda: False)():
        return None

    src_code = _normalize_lang_code(src, fallback="en")
    target_code = _normalize_target_lang(target)
    piece_clean = html.unescape(piece or "")
    piece_clean = re.sub(r"[\u200B-\u200F\u202A-\u202E\u2066-\u2069]", "", piece_clean)
    prompt = (
        "You are a strict legal translation assistant.\n"
        f"Translate the input from {src_code} to {target_code}.\n"
        "Rules:\n"
        "- Keep placeholders like [Name], [Date], [Montant], and article numbers exactly as-is.\n"
        "- Keep line breaks and clause order.\n"
        "- Do not add explanations, labels, or quotes.\n"
        "- Output only the translation text.\n\n"
        "Input:\n"
        f"{piece_clean}"
    )
    out = local_llm_mod.generate(prompt, max_new_tokens=700, do_sample=False)
    if not isinstance(out, str):
        return None
    out = out.strip()
    if not out or out.startswith("[LLM load error:"):
        return None
    out = html.unescape(out)
    out = out.replace("&quot;", '"').replace("quot;", '"').replace("&amp;", "&")
    out = re.sub(r"[\u200B-\u200F\u202A-\u202E\u2066-\u2069]", "", out)
    return out.strip()


def _translate_via_local_lfm(
    text: str,
    src: str,
    target_lang: str,
    pairs: List[Tuple[str, str]],
) -> Tuple[Optional[str], str]:
    chunks_out: List[str] = []
    overall = "ok"
    backend = f"local_lfm_{target_lang}"
    t = text
    i = 0
    while i < len(t):
        piece = t[i : i + _ARGOS_CHUNK_CHARS]
        i += _ARGOS_CHUNK_CHARS
        cached = _cache_get(backend, src, piece)
        if cached is not None:
            chunks_out.append(cached[0])
            if cached[1] == "partial":
                overall = "partial"
            continue
        out = _translate_piece_local_lfm(piece, src, target_lang)
        if out is None:
            overall = "partial"
            out_piece = piece
            st_piece = "partial"
        else:
            out_piece = out
            st_piece = "ok"
        if target_lang == "ar":
            out_piece = _apply_glossary_ar(out_piece, pairs)
        _cache_put(backend, src, piece, (out_piece, st_piece))
        chunks_out.append(out_piece)
    return "".join(chunks_out), overall


def _google_ar_result_usable(original: str, g_out: Optional[str], g_st: str) -> bool:
    """True when Google returned text we should accept instead of falling back to LFM or Argos."""
    if g_out is None:
        return False
    o = (original or "").strip()
    g = (g_out or "").strip()
    if not g:
        return False
    if g_st == "partial" and g == o:
        return False
    return True


def translate_plain_to_target(
    text: str,
    source_lang: str,
    *,
    target_lang: str = "ar",
    apply_glossary: bool = True,
) -> Tuple[str, str, str]:
    """
    Returns (output_text, status, provider) where:
      status: ok | partial | skipped
      provider: google_cloud_translate_v2 | local_lfm_translate | argos_translate | none

    For Arabic: Google first, then local LFM, then Argos. For other targets (en/fr/de): local LFM only.
    """
    src = _normalize_lang_code(source_lang, fallback="und")
    target = _normalize_target_lang(target_lang)
    if not isinstance(text, str) or not text.strip():
        return "", "skipped", "none"
    if src == target or src == "und":
        return text.strip(), "skipped", "none"

    pairs = (load_glossary_pairs_for_source(src) if apply_glossary and target == "ar" else [])

    if target == "ar":
        g_out, g_st = _translate_via_google(text, src, pairs)
        if _google_ar_result_usable(text, g_out, g_st):
            return g_out, g_st, "google_cloud_translate_v2"

        llm_out, llm_st = _translate_via_local_lfm(text, src, target, pairs)
        if llm_out is not None:
            if not (llm_st == "partial" and llm_out.strip() == text.strip()):
                return llm_out, llm_st, "local_lfm_translate"

        a_out, a_st = _translate_via_argos(text, src, pairs)
        if a_out is None:
            return text, "skipped", "none"
        if a_st == "partial" and a_out.strip() == text.strip():
            return text, "skipped", "none"
        return a_out, a_st, "argos_translate"

    llm_out, llm_st = _translate_via_local_lfm(text, src, target, pairs)
    if llm_out is not None:
        if not (llm_st == "partial" and llm_out.strip() == text.strip()):
            return llm_out, llm_st, "local_lfm_translate"
    return text, "skipped", "none"


def translate_plain_to_arabic(
    text: str,
    source_lang: str,
    *,
    apply_glossary: bool = True,
) -> Tuple[str, str, str]:
    return translate_plain_to_target(
        text,
        source_lang,
        target_lang="ar",
        apply_glossary=apply_glossary,
    )


def enrich_ocr_chunks_with_arabic(
    ocr_chunks: List[Dict[str, Any]],
    source_lang: str,
    *,
    requested: bool,
    per_chunk: bool = False,
    document_is_mixed: bool = False,
    target_lang: str = "ar",
) -> Dict[str, Any]:
    """
    Adds `translated_ar_text` to each chunk when MT succeeds. Always returns a metadata dict.

    When ``per_chunk`` is True (or implied by caller), uses each chunk's ``detected_lang`` when set,
    falling back to ``source_lang`` for that chunk.
    """
    gv = glossary_version_label()
    meta: Dict[str, Any] = {
        "translation_version": "3",
        "glossary_version": gv,
        "translation_status": "disabled",
        "translation_provider": None,
        "translation_target_lang": _normalize_target_lang(target_lang),
        "per_chunk": bool(per_chunk),
        "skip_reason": None,
        "unsupported_languages": [],
    }
    if not requested:
        return meta
    if external_mt_disabled() and argos_mt_disabled() and not local_lfm_mt_enabled():
        meta["translation_status"] = "skipped"
        meta["skip_reason"] = "EXTERNAL_MT_DISABLED"
        return meta

    doc_src = (source_lang or "").strip().lower().split("-")[0]
    target = _normalize_target_lang(target_lang)
    if not per_chunk and doc_src in (target, "und"):
        meta["translation_status"] = "skipped"
        meta["skip_reason"] = "source_already_target_or_undetermined"
        return meta

    google_ok = _get_translate_v2_client() is not None
    if not google_ok and argos_mt_disabled() and not local_lfm_mt_enabled():
        meta["translation_status"] = "skipped"
        meta["skip_reason"] = "google_translate_client_unavailable_and_argos_disabled"
        return meta

    provider_used: Optional[str] = None
    any_partial = False
    unsupported: List[str] = []

    for c in ocr_chunks:
        raw = (c.get("normalized_text") or c.get("text") or "").strip()
        if not raw:
            c["translated_ar_text"] = ""
            continue

        eff = doc_src
        if per_chunk:
            dl = c.get("detected_lang")
            if isinstance(dl, str) and dl.strip() and dl.strip().lower().split("-")[0] not in ("", "und"):
                eff = dl.strip().lower().split("-")[0]
            elif doc_src not in ("ar", "und"):
                eff = doc_src

        if eff in (target,):
            c["translated_text"] = raw
            if target == "ar":
                c["translated_ar_text"] = raw
            continue

        if eff == "und":
            eff = doc_src if doc_src not in (target, "und") else "en"

        translated, st, prov = translate_plain_to_target(raw, eff, target_lang=target)
        if prov == "none" and st == "skipped" and raw.strip() and eff not in (target, "und"):
            unsupported.append(eff)
        c["translated_text"] = translated
        if target == "ar":
            c["translated_ar_text"] = translated
        if prov != "none":
            provider_used = prov
        if st == "partial":
            any_partial = True

    meta["unsupported_languages"] = sorted(set(unsupported))

    if provider_used is None:
        meta["translation_status"] = "skipped"
        if meta["unsupported_languages"]:
            meta["skip_reason"] = "unsupported_source_language"
        else:
            meta["skip_reason"] = "no_translation_backend_available"
        return meta

    meta["translation_provider"] = provider_used
    meta["translation_status"] = "partial" if any_partial else "ok"
    if any_partial and meta["unsupported_languages"]:
        meta["skip_reason"] = "partial_success_with_unsupported_or_failed_chunks"
    return meta
