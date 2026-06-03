from __future__ import annotations

from typing import Optional

from sqlalchemy import text
from app.db.session import engine
from app.db.session import Base  # noqa
from app.db import models  # noqa


def init_db() -> None:
    """
    Creates tables if missing, and applies ultra-light schema tweaks safely.
    We do NOT drop anything.
    """
    # Create tables if not exist (won't override existing)
    Base.metadata.create_all(bind=engine)

    # Safe schema tweak: add role column if missing (SQLite supports ADD COLUMN)
    # If it already exists, ignore.
    try:
        with engine.begin() as conn:
            # Check if role column exists
            cols = conn.execute(text("PRAGMA table_info(users);")).fetchall()
            col_names = {c[1] for c in cols}  # (cid, name, type,...)
            if "role" not in col_names:
                conn.execute(text("ALTER TABLE users ADD COLUMN role VARCHAR DEFAULT 'user';"))
            if "email_verified" not in col_names:
                conn.execute(text("ALTER TABLE users ADD COLUMN email_verified INTEGER DEFAULT 1;"))
            # Seed admin account: always treat as verified so it can log in without email verification
            conn.execute(
                text("UPDATE users SET email_verified = 1 WHERE email = 'admin@test.com';")
            )
    except Exception:
        pass

    # Safe schema tweaks: add optional analyses metadata columns if missing
    # (SQLite supports ADD COLUMN; ignore if already exists)
    try:
        with engine.begin() as conn:
            cols = conn.execute(text("PRAGMA table_info(analyses);"))
            col_names = {c[1] for c in cols.fetchall()}

            if "mime_type" not in col_names:
                conn.execute(text("ALTER TABLE analyses ADD COLUMN mime_type VARCHAR;"))
            if "sha256" not in col_names:
                conn.execute(text("ALTER TABLE analyses ADD COLUMN sha256 VARCHAR;"))
            if "page_count" not in col_names:
                conn.execute(text("ALTER TABLE analyses ADD COLUMN page_count INTEGER;"))
            if "ocr_used" not in col_names:
                conn.execute(text("ALTER TABLE analyses ADD COLUMN ocr_used INTEGER;"))
            if "detected_lang" not in col_names:
                conn.execute(text("ALTER TABLE analyses ADD COLUMN detected_lang VARCHAR;"))
    except Exception:
        pass

    # Safe schema tweaks: add upload storage columns for profile_user_documents
    try:
        with engine.begin() as conn:
            cols = conn.execute(text("PRAGMA table_info(profile_user_documents);")).fetchall()
            col_names = {c[1] for c in cols}
            if "mime_type" not in col_names:
                conn.execute(text("ALTER TABLE profile_user_documents ADD COLUMN mime_type VARCHAR;"))
            if "file_bytes" not in col_names:
                conn.execute(text("ALTER TABLE profile_user_documents ADD COLUMN file_bytes BLOB;"))
    except Exception:
        pass

    # Safe schema tweak: add author_id to legato_deal_messages for multi-user threads
    try:
        with engine.begin() as conn:
            cols = conn.execute(text("PRAGMA table_info(legato_deal_messages);")).fetchall()
            col_names = {c[1] for c in cols}
            if "author_id" not in col_names:
                conn.execute(text("ALTER TABLE legato_deal_messages ADD COLUMN author_id INTEGER;"))
    except Exception:
        pass

    # Persist social post / profile avatar bytes in SQLite (survives container redeploys)
    try:
        with engine.begin() as conn:
            cols = conn.execute(text("PRAGMA table_info(social_posts);")).fetchall()
            col_names = {c[1] for c in cols}
            if "image_mime_type" not in col_names:
                conn.execute(text("ALTER TABLE social_posts ADD COLUMN image_mime_type VARCHAR;"))
            if "image_bytes" not in col_names:
                conn.execute(text("ALTER TABLE social_posts ADD COLUMN image_bytes BLOB;"))

            cols = conn.execute(text("PRAGMA table_info(legato_profiles);")).fetchall()
            col_names = {c[1] for c in cols}
            if "avatar_mime_type" not in col_names:
                conn.execute(text("ALTER TABLE legato_profiles ADD COLUMN avatar_mime_type VARCHAR;"))
            if "avatar_bytes" not in col_names:
                conn.execute(text("ALTER TABLE legato_profiles ADD COLUMN avatar_bytes BLOB;"))
    except Exception:
        pass

    _backfill_social_images_from_disk()


def _disk_path_from_static_url(url: Optional[str]) -> Optional[str]:
    import os

    if not url or "/static/" not in url:
        return None
    rel = url[url.index("/static/") + 1 :]
    return os.path.join(*rel.split("/"))


def _guess_image_mime(ext: str, content_type: Optional[str] = None) -> str:
    ct = (content_type or "").split(";")[0].strip().lower()
    if ct.startswith("image/"):
        return ct[:255]
    return {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".gif": "image/gif",
        ".webp": "image/webp",
        ".jfif": "image/jpeg",
        ".heic": "image/heic",
        ".heif": "image/heif",
    }.get((ext or "").lower(), "image/jpeg")


def _normalize_uploaded_image(data: bytes, ext: str, content_type: Optional[str] = None) -> tuple[bytes, str, str]:
    """Convert HEIC and other non-web-safe formats to JPEG for avatar/post display."""
    from io import BytesIO

    from PIL import Image

    try:
        from pillow_heif import register_heif_opener

        register_heif_opener()
    except ImportError:
        pass

    ext_l = (ext or "").lower()
    if ext_l == ".jpeg":
        ext_l = ".jpg"
    ct = (content_type or "").split(";")[0].strip().lower()
    browser_safe_ext = {".jpg", ".png", ".gif", ".webp"}
    needs_convert = ext_l in {".heic", ".heif"} or "heic" in ct or "heif" in ct

    if not needs_convert and ext_l in browser_safe_ext:
        return data, ext_l, _guess_image_mime(ext_l, content_type)

    try:
        img = Image.open(BytesIO(data))
        img.load()
        if img.mode in ("RGBA", "LA"):
            background = Image.new("RGB", img.size, (255, 255, 255))
            background.paste(img, mask=img.split()[-1])
            img = background
        elif img.mode == "P":
            img = img.convert("RGB")
        elif img.mode != "RGB":
            img = img.convert("RGB")
        out = BytesIO()
        img.save(out, format="JPEG", quality=90, optimize=True)
        return out.getvalue(), ".jpg", "image/jpeg"
    except Exception:
        if ext_l in browser_safe_ext:
            return data, ext_l, _guess_image_mime(ext_l, content_type)
        raise


def _set_json_avatar_url(payload_json: str, avatar_url: str) -> str:
    import json

    try:
        prof = json.loads(payload_json or "{}")
    except Exception:
        prof = {}
    if not isinstance(prof, dict):
        prof = {}
    prof["avatar_url"] = avatar_url
    prof["avatarUrl"] = avatar_url
    return json.dumps(prof, ensure_ascii=False)


def _backfill_social_images_from_disk() -> None:
    """Best effort: load static files into DB blobs if files still exist on disk."""
    import json
    import os

    base = (os.getenv("IMAGE_BASE_URL") or os.getenv("PUBLIC_API_URL") or os.getenv("API_PUBLIC_URL") or "").strip().rstrip("/")

    try:
        with engine.begin() as conn:
            posts = conn.execute(
                text(
                    "SELECT id, image_url FROM social_posts "
                    "WHERE image_bytes IS NULL AND image_url IS NOT NULL AND trim(image_url) != ''"
                )
            ).fetchall()
            for pid, url in posts:
                if url and "/api/posts/" in url:
                    continue
                path = _disk_path_from_static_url(url)
                if not path or not os.path.isfile(path):
                    continue
                with open(path, "rb") as f:
                    data = f.read()
                if not data:
                    continue
                mime = _guess_image_mime(os.path.splitext(path)[-1])
                conn.execute(
                    text("UPDATE social_posts SET image_bytes = :b, image_mime_type = :m WHERE id = :id"),
                    {"b": data, "m": mime, "id": pid},
                )
                if base:
                    conn.execute(
                        text("UPDATE social_posts SET image_url = :u WHERE id = :id"),
                        {"u": f"{base}/static/post_images/post_{pid}{_ext_from_mime(mime)}", "id": pid},
                    )

            profiles = conn.execute(text("SELECT user_id, payload_json, avatar_bytes FROM legato_profiles")).fetchall()
            for user_id, payload_json, avatar_bytes in profiles:
                if avatar_bytes:
                    if base:
                        mime_row = conn.execute(
                            text("SELECT avatar_mime_type FROM legato_profiles WHERE user_id = :uid"),
                            {"uid": user_id},
                        ).fetchone()
                        mime = mime_row[0] if mime_row else "image/jpeg"
                        conn.execute(
                            text("UPDATE legato_profiles SET payload_json = :j WHERE user_id = :uid"),
                            {
                                "j": _set_json_avatar_url(
                                    payload_json,
                                    f"{base}/static/profile_avatars/user_{user_id}{_ext_from_mime(mime)}",
                                ),
                                "uid": user_id,
                            },
                        )
                    continue
                try:
                    prof = json.loads(payload_json or "{}")
                except Exception:
                    prof = {}
                raw_url = (prof.get("avatarUrl") or prof.get("avatar_url") or "").strip()
                if not raw_url or "/api/profile/" in raw_url:
                    continue
                path = _disk_path_from_static_url(raw_url)
                if not path or not os.path.isfile(path):
                    continue
                with open(path, "rb") as f:
                    data = f.read()
                if not data:
                    continue
                mime = _guess_image_mime(os.path.splitext(path)[-1])
                new_url = (
                    f"{base}/static/profile_avatars/user_{user_id}{_ext_from_mime(mime)}"
                    if base
                    else raw_url
                )
                conn.execute(
                    text(
                        "UPDATE legato_profiles SET avatar_bytes = :b, avatar_mime_type = :m, payload_json = :j "
                        "WHERE user_id = :uid"
                    ),
                    {
                        "b": data,
                        "m": mime,
                        "j": _set_json_avatar_url(payload_json, new_url),
                        "uid": user_id,
                    },
                )
    except Exception:
        pass

    _sync_social_images_to_static()


def _ext_from_mime(mime: Optional[str]) -> str:
    m = (mime or "").lower()
    if "png" in m:
        return ".png"
    if "gif" in m:
        return ".gif"
    if "webp" in m:
        return ".webp"
    return ".jpg"


def _write_post_image_file(post_id: int, data: bytes, ext: str) -> None:
    import os

    os.makedirs("static/post_images", exist_ok=True)
    path = os.path.join("static", "post_images", f"post_{post_id}{ext}")
    with open(path, "wb") as f:
        f.write(data)


def _write_avatar_image_file(user_id: int, data: bytes, ext: str) -> None:
    import os

    os.makedirs("static/profile_avatars", exist_ok=True)
    path = os.path.join("static", "profile_avatars", f"user_{user_id}{ext}")
    with open(path, "wb") as f:
        f.write(data)


def _sync_social_images_to_static() -> None:
    """Write DB image blobs to static/ so nginx/FastAPI can serve them after redeploy."""
    import os

    base = (os.getenv("IMAGE_BASE_URL") or os.getenv("PUBLIC_API_URL") or os.getenv("API_PUBLIC_URL") or "").strip().rstrip("/")
    try:
        with engine.begin() as conn:
            posts = conn.execute(
                text("SELECT id, image_bytes, image_mime_type, image_url FROM social_posts WHERE image_bytes IS NOT NULL")
            ).fetchall()
            for pid, data, mime, url in posts:
                if not data:
                    continue
                ext = _ext_from_mime(mime)
                _write_post_image_file(int(pid), data, ext)
                if base:
                    conn.execute(
                        text("UPDATE social_posts SET image_url = :u WHERE id = :id"),
                        {"u": f"{base}/static/post_images/post_{pid}{ext}", "id": pid},
                    )

            profiles = conn.execute(
                text("SELECT user_id, avatar_bytes, avatar_mime_type, payload_json FROM legato_profiles WHERE avatar_bytes IS NOT NULL")
            ).fetchall()
            for user_id, data, mime, payload_json in profiles:
                if not data:
                    continue
                ext = _ext_from_mime(mime)
                _write_avatar_image_file(int(user_id), data, ext)
                if base:
                    conn.execute(
                        text(
                            "UPDATE legato_profiles SET payload_json = :j WHERE user_id = :uid"
                        ),
                        {
                            "j": _set_json_avatar_url(
                                payload_json,
                                f"{base}/static/profile_avatars/user_{user_id}{ext}",
                            ),
                            "uid": user_id,
                        },
                    )
    except Exception:
        pass
