"""Auto-reseed OmniRogue credentials from a captured browser request.

Copy either of these from browser DevTools (Network tab, right-click the
request, Copy as cURL / Copy as Python requests) and paste it into a file:

  1. The Clerk token refresh  ->  clerk.omnirogue.com/v1/client/sessions/<sid>/tokens
     Contains the __client rotating-token cookie, so the proxy can auto-refresh
     the short-lived JWT. THIS IS THE ONE YOU WANT for a long-running server.

  2. The chat request         ->  omnirogue.com/api/llm/chat
     Contains the live __session JWT but no __client cookie, so the proxy can
     serve requests until that JWT expires but cannot rotate it.

Everything is derived from the JWTs themselves, so field order and extra
analytics cookies do not matter.

Usage:
    python reseed.py capture.txt
    python reseed.py capture.txt --write-env .env
    python reseed.py capture.txt --push https://host --key sk-...
"""
from __future__ import annotations

import argparse
import base64
import json
import re
import sys
from typing import Any, Dict, Optional, Tuple

_JWT_RE = re.compile(r"eyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+")
_SESSION_URL_RE = re.compile(r"sessions/(sess_[A-Za-z0-9]+)/tokens")
_ACTIVE_CTX_RE = re.compile(r"clerk_active_context['\"]?\s*[:=]\s*['\"]?(sess_[A-Za-z0-9]+)")

# Required to identify the session at all. Without these there is nothing usable.
_REQUIRED = ("session_id", "user_id", "instance_id", "jwt")
# Needed only for automatic JWT rotation / Cloudflare friendliness.
_OPTIONAL = ("client_cookie", "client_uat", "cf_bm", "cfuvid")

_ENV_KEYS = [
    ("OMNIROGUE_SESSION_ID", "session_id"),
    ("OMNIROGUE_USER_ID", "user_id"),
    ("OMNIROGUE_INSTANCE_ID", "instance_id"),
    ("OMNIROGUE_JWT", "jwt"),
    ("OMNIROGUE_CLIENT_COOKIE", "client_cookie"),
    ("OMNIROGUE_CLIENT_UAT", "client_uat"),
    ("OMNIROGUE_CF_BM", "cf_bm"),
    ("OMNIROGUE_CFUVID", "cfuvid"),
]


def _find_value(name: str, text: str) -> Optional[str]:
    """Find a cookie value in python-dict or curl-cookie-header style.

    The name must match exactly, so looking up `__client` never accidentally
    returns the value of `__client_uat`.
    """
    patterns = [
        r"['\"]" + re.escape(name) + r"['\"]\s*[:=]\s*['\"]([^'\"]+)['\"]",
        r"(?:^|[;\s&'\"(])" + re.escape(name) + r"=([^;'\"\s&]+)",
    ]
    for pat in patterns:
        match = re.search(pat, text)
        if match:
            return match.group(1)
    return None


def _b64json(segment: str) -> Dict[str, Any]:
    segment += "=" * (-len(segment) % 4)
    return json.loads(base64.urlsafe_b64decode(segment))


def _decode_jwt(token: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    parts = token.split(".")
    if len(parts) < 2:
        return {}, {}
    try:
        return _b64json(parts[0]), _b64json(parts[1])
    except Exception:
        return {}, {}


def _classify_jwts(text: str) -> Tuple[Optional[str], Optional[str], Dict[str, Any]]:
    """Split the JWTs in a capture into (session token, __client cookie, claims).

    A session token carries `sid`/`sub` claims; the __client cookie carries a
    `rotating_token`. Classifying by content means the surrounding format of the
    capture is irrelevant.
    """
    session_token: Optional[str] = None
    client_cookie: Optional[str] = None
    claims: Dict[str, Any] = {}
    header: Dict[str, Any] = {}

    for token in _JWT_RE.findall(text):
        payload_hdr, payload = _decode_jwt(token)
        if "sid" in payload or "sub" in payload:
            # Prefer the freshest session token when several are present.
            if session_token is None or payload.get("iat", 0) >= claims.get("iat", 0):
                session_token, claims, header = token, payload, payload_hdr
        elif "rotating_token" in payload or str(payload.get("id", "")).startswith("client_"):
            if client_cookie is None or len(token) > len(client_cookie):
                client_cookie = token

    if header.get("kid"):
        claims = {**claims, "_kid": header["kid"]}
    return session_token, client_cookie, claims


def parse_captured_request(text: str) -> Dict[str, Any]:
    """Extract every proxy credential found in a captured request."""
    session_token, client_cookie, claims = _classify_jwts(text)

    url_match = _SESSION_URL_RE.search(text)
    ctx_match = _ACTIVE_CTX_RE.search(text)
    session_id = (
        (url_match.group(1) if url_match else None)
        or claims.get("sid")
        or (ctx_match.group(1) if ctx_match else None)
    )

    parsed: Dict[str, Any] = {
        "session_id": session_id,
        "user_id": claims.get("sub"),
        "instance_id": claims.get("_kid"),
        "jwt": session_token or _find_value("__session", text),
        "client_cookie": client_cookie or _find_value("__client", text),
        "client_uat": _find_value("__client_uat", text),
        "cf_bm": _find_value("__cf_bm", text),
        "cfuvid": _find_value("_cfuvid", text),
    }
    parsed["_missing"] = [name for name in _REQUIRED if not parsed.get(name)]
    parsed["_missing_optional"] = [name for name in _OPTIONAL if not parsed.get(name)]
    parsed["_can_refresh"] = bool(parsed.get("client_cookie"))
    return parsed


def to_env_block(parsed: Dict[str, Any]) -> str:
    return "\n".join(
        f"{env}={parsed[key]}" for env, key in _ENV_KEYS if parsed.get(key)
    )


def _write_env(path: str, parsed: Dict[str, Any]) -> None:
    from pathlib import Path

    updates = dict(line.split("=", 1) for line in to_env_block(parsed).splitlines())
    target = Path(path)
    existing = target.read_text("utf-8").splitlines() if target.is_file() else []
    out, seen = [], set()
    for line in existing:
        key = line.split("=", 1)[0] if "=" in line else None
        if key in updates:
            out.append(f"{key}={updates[key]}")
            seen.add(key)
        else:
            out.append(line)
    for key, value in updates.items():
        if key not in seen:
            out.append(f"{key}={value}")
    target.write_text("\n".join(out) + "\n", "utf-8")


def _push(base: str, key: str, text: str) -> None:
    import requests

    resp = requests.post(
        f"{base.rstrip('/')}/auth/reseed",
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        json={"raw": text},
        timeout=30,
    )
    print(f"POST /auth/reseed -> {resp.status_code}")
    print(resp.text[:1000])


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Reseed OmniRogue credentials from a captured browser request."
    )
    parser.add_argument("file", help="File containing the cURL or Python-requests capture")
    parser.add_argument("--push", metavar="BASE_URL", help="POST the capture to a live /auth/reseed")
    parser.add_argument("--key", help="API key, required with --push")
    parser.add_argument("--write-env", metavar="PATH", help="Update the OMNIROGUE_* keys in this .env")
    args = parser.parse_args(argv)

    with open(args.file, encoding="utf-8") as handle:
        text = handle.read()
    parsed = parse_captured_request(text)

    print("Parsed credentials:")
    for env, key in _ENV_KEYS:
        value = parsed.get(key)
        shown = (value[:40] + "...") if value and len(value) > 40 else (value or "")
        flag = "OK     " if value else ("MISSING" if key in _REQUIRED else "absent ")
        print(f"  {env:24} {flag}  {shown}")

    if parsed["_missing"]:
        print("\nERROR: required fields missing:", ", ".join(parsed["_missing"]))
        print("This capture does not identify a session. Re-copy the request.")
        return 1

    if not parsed["_can_refresh"]:
        print(
            "\nWARNING: no __client cookie in this capture, so the JWT cannot be\n"
            "auto-rotated. The proxy will work only until the current JWT expires.\n"
            "For durable operation, capture the Clerk request instead:\n"
            "  clerk.omnirogue.com/v1/client/sessions/<sid>/tokens"
        )

    print("\n--- env block (paste into Railway / .env) ---")
    print(to_env_block(parsed))

    if args.write_env:
        _write_env(args.write_env, parsed)
        print(f"\nWrote {args.write_env}")
    if args.push:
        if not args.key:
            parser.error("--push requires --key")
        _push(args.push, args.key, text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
