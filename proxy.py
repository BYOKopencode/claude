"""Core OmniRogue proxy: Clerk JWT rotation and chat/completion proxy."""
import json
import time
import base64
import uuid
import threading
from datetime import datetime, timezone
from typing import Optional, Dict, Any, Generator

import requests

from config import UserConfig, settings


class OmniRogueProxy:
    """
    Manages Clerk session state and proxies chat requests to OmniRogue.
    Thread-safe. Auto-refreshes JWT before every outbound call.
    """

    def __init__(self, user: UserConfig):
        self._lock = threading.Lock()

        # Identity
        self.name = user.name
        self.session_id = user.session_id
        self.user_id = user.user_id
        self.user_email = user.user_email
        self.instance_id = user.instance_id

        # JWT (auto-rotated)
        self._jwt = user.jwt

        # Device cookies
        self._client_cookie = user.client_cookie
        self._client_uat = user.client_uat
        self._cf_bm = user.cf_bm
        self._cfuvid = user.cfuvid

        # Endpoints
        self.clerk_base = settings.omnirogue_clerk_base
        self.frontend_base = settings.omnirogue_frontend_base
        self.api_base = settings.omnirogue_api_base
        self.clerk_version = settings.clerk_version
        self.clerk_js_version = settings.clerk_js_version

        self._http = requests.Session()

    # ── Cookie Builders ───────────────────────────────────────────────

    def _clerk_cookies(self) -> str:
        parts = []
        if self._client_cookie:
            parts.append(f"__client={self._client_cookie}")
        if self._client_uat:
            parts.append(f"__client_uat={self._client_uat}")
        if self._cf_bm:
            parts.append(f"__cf_bm={self._cf_bm}")
        if self._cfuvid:
            parts.append(f"_cfuvid={self._cfuvid}")
        return "; ".join(parts)

    def _frontend_cookies(self) -> str:
        clerk = self._clerk_cookies()
        session = f"__session={self._jwt}"
        return f"{clerk}; {session}" if clerk else session

    # ── JWT Helpers ───────────────────────────────────────────────────

    def _jwt_claims(self) -> Dict[str, Any]:
        try:
            payload_b64 = self._jwt.split(".")[1]
            pad = 4 - len(payload_b64) % 4
            if pad != 4:
                payload_b64 += "=" * pad
            return json.loads(base64.urlsafe_b64decode(payload_b64))
        except Exception:
            return {}

    def _jwt_expiry(self) -> Optional[int]:
        return self._jwt_claims().get("exp")

    def _needs_refresh(self, buffer: int = 30) -> bool:
        """Whether the JWT should be rotated before the next outbound call.

        Clerk session tokens are short-lived (often only 60s), so a fixed 60s
        buffer would mark every token stale the moment it is issued and force a
        refresh on every single request. Scale the buffer to token lifetime.
        """
        claims = self._jwt_claims()
        exp = claims.get("exp")
        if not exp:
            return True
        issued_at = claims.get("iat")
        if issued_at and exp > issued_at:
            lifetime = exp - issued_at
            buffer = min(buffer, max(2, lifetime // 6))
        return (exp - buffer) < time.time()

    # ── Token Rotation ────────────────────────────────────────────────

    def _refresh_token(self):
        url = (
            f"{self.clerk_base}/v1/client/sessions/{self.session_id}/tokens"
            f"?__clerk_api_version={self.clerk_version}"
            f"&_clerk_js_version={self.clerk_js_version}"
        )

        headers = {
            "accept": "*/*",
            "accept-language": "en-GB,en;q=0.5",
            "content-type": "application/x-www-form-urlencoded",
            "cookie": self._clerk_cookies(),
            "origin": self.frontend_base,
            "referer": f"{self.frontend_base}/",
            "sec-fetch-dest": "empty",
            "sec-fetch-mode": "cors",
            "sec-fetch-site": "same-site",
            "sec-gpc": "1",
            "user-agent": (
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/142.0.0.0 Safari/537.36"
            ),
        }

        data = {"organization_id": "", "token": self._jwt}
        resp = self._http.post(url, headers=headers, data=data, timeout=15)
        resp.raise_for_status()

        payload = resp.json()
        if "jwt" in payload and isinstance(payload["jwt"], str):
            self._jwt = payload["jwt"]
            return

        token_obj = payload.get("last_active_token") or {}
        if token_obj.get("jwt"):
            self._jwt = token_obj["jwt"]
            return

        raise RuntimeError("Token refresh response missing jwt field")

    def ensure_fresh(self, force: bool = False):
        """Refresh this user's JWT when needed, or unconditionally when forced."""
        with self._lock:
            if not self._client_cookie:
                # No __client rotating-token cookie -> cannot mint a new JWT.
                # Serve with the seed JWT; surface a clear error only if forced.
                if force:
                    raise RuntimeError(
                        "Cannot refresh JWT: this capture has no __client cookie. "
                        "Capture the Clerk token request "
                        "(clerk.omnirogue.com/v1/client/sessions/<sid>/tokens) instead."
                    )
                return
            if force or self._needs_refresh():
                self._refresh_token()

    # ── Public helpers ────────────────────────────────────────────────

    def update_cookies(self, jwt: Optional[str] = None, client_cookie: Optional[str] = None,
                       client_uat: Optional[str] = None, cf_bm: Optional[str] = None,
                       cfuvid: Optional[str] = None):
        with self._lock:
            if jwt is not None:
                self._jwt = jwt
            if client_cookie is not None:
                self._client_cookie = client_cookie
            if client_uat is not None:
                self._client_uat = client_uat
            if cf_bm is not None:
                self._cf_bm = cf_bm
            if cfuvid is not None:
                self._cfuvid = cfuvid

    # ── Model parsing ─────────────────────────────────────────────────

    def _parse_model(self, model: str) -> tuple[str, str]:
        if "/" in model:
            return model.split("/", 1)
        return "anthropic", model

    # ── Chat ──────────────────────────────────────────────────────────

    def _chat_request(self, messages, model: str, stream: bool) -> requests.Response:
        self.ensure_fresh()
        provider, model_id = self._parse_model(model)

        url = f"{self.frontend_base}/api/llm/chat"
        headers = {
            "accept": "*/*",
            "accept-language": "en-GB,en;q=0.5",
            "content-type": "application/json",
            "cookie": self._frontend_cookies(),
            "origin": self.frontend_base,
            "referer": f"{self.frontend_base}/create/ai-chat",
            "sec-fetch-dest": "empty",
            "sec-fetch-mode": "cors",
            "sec-fetch-site": "same-origin",
            "sec-gpc": "1",
            "user-agent": (
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/142.0.0.0 Safari/537.36"
            ),
        }
        payload = {
            "provider": provider,
            "model": model_id,
            "messages": messages,
            "stream": stream,
        }
        response = self._http.post(url, headers=headers, json=payload, stream=stream, timeout=60)
        # Upstream sends UTF-8 but omits charset on text/event-stream, and
        # requests then defaults to ISO-8859-1, which mangles emoji and
        # non-ASCII text in both resp.text and iter_lines(decode_unicode=True).
        response.encoding = "utf-8"
        return response

    def chat_sync(self, messages, model: str) -> Dict[str, Any]:
        resp = self._chat_request(messages, model, stream=False)
        resp.raise_for_status()

        text = resp.text.strip()
        content = ""
        usage = {}

        if text.startswith("data:"):
            for line in text.splitlines():
                line = line.strip()
                if not line.startswith("data: "):
                    continue
                data = line[6:]
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                    if chunk.get("type") == "token":
                        content += chunk.get("token", "")
                    elif chunk.get("type") == "done":
                        content = chunk.get("content", content)
                        usage = chunk.get("usage", {})
                except json.JSONDecodeError:
                    continue
        else:
            try:
                data = json.loads(text)
                content = data.get("content", "")
                usage = data.get("usage", {})
            except json.JSONDecodeError:
                content = text

        now = int(time.time())
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        total_tokens = usage.get("total_tokens", prompt_tokens + completion_tokens)

        return {
            "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
            "object": "chat.completion",
            "created": now,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content, "refusal": None},
                    "logprobs": None,
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
                "prompt_tokens_details": {"cached_tokens": 0},
                "completion_tokens_details": {"reasoning_tokens": 0},
            },
            "system_fingerprint": None,
        }

    def chat_stream(self, messages, model: str, include_usage: bool = False) -> Generator[Dict[str, Any], None, None]:
        with self._chat_request(messages, model, stream=True) as resp:
            resp.raise_for_status()
            cmpl_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
            now = int(time.time())
            usage = {}

            for line in resp.iter_lines(decode_unicode=True):
                if not line or not line.startswith("data: "):
                    continue
                data = line[6:]
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue

                if chunk.get("type") == "token":
                    yield {
                        "id": cmpl_id,
                        "object": "chat.completion.chunk",
                        "created": now,
                        "model": model,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"content": chunk.get("token", "")},
                                "logprobs": None,
                                "finish_reason": None,
                            }
                        ],
                    }
                elif chunk.get("type") == "done":
                    usage = chunk.get("usage", {})
                    final_chunk: Dict[str, Any] = {
                        "id": cmpl_id,
                        "object": "chat.completion.chunk",
                        "created": now,
                        "model": model,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {},
                                "logprobs": None,
                                "finish_reason": "stop",
                            }
                        ],
                    }
                    if include_usage and usage:
                        final_chunk["usage"] = {
                            "prompt_tokens": usage.get("prompt_tokens", 0),
                            "completion_tokens": usage.get("completion_tokens", 0),
                            "total_tokens": usage.get("total_tokens", 0),
                        }
                    yield final_chunk

    # ── Backend API ───────────────────────────────────────────────────

    def save_conversation(self, title: str, conversation_id: str, model: str = "claude-sonnet-4.6") -> Dict[str, Any]:
        self.ensure_fresh()

        url = f"{self.api_base}/api/ai-assistant/conversations"
        headers = {
            "accept": "application/json, text/plain, */*",
            "authorization": f"Bearer {self._jwt}",
            "content-type": "application/json",
            "origin": self.frontend_base,
            "referer": f"{self.frontend_base}/",
            "sec-fetch-dest": "empty",
            "sec-fetch-mode": "cors",
            "sec-fetch-site": "same-site",
            "sec-gpc": "1",
            "user-agent": (
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/142.0.0.0 Safari/537.36"
            ),
            "x-clerk-user-email": self.user_email,
            "x-clerk-user-id": self.user_id,
        }
        payload = {
            "title": title,
            "personality": {"mode": "create-chat", "modelId": model},
            "agentType": "create-chat",
            "conversationId": conversation_id,
        }
        resp = self._http.post(url, headers=headers, json=payload, timeout=15)
        resp.raise_for_status()
        return resp.json()

    # ── Status ────────────────────────────────────────────────────────

    def status(self) -> Dict[str, Any]:
        claims = self._jwt_claims()
        exp = claims.get("exp")
        return {
            "name": self.name,
            "session_id": self.session_id,
            "user_id": self.user_id,
            "user_email": self.user_email,
            "jwt_expires_at": datetime.fromtimestamp(exp, tz=timezone.utc).isoformat() if exp else None,
            "jwt_needs_refresh": self._needs_refresh(),
            "jwt_issuer": claims.get("iss"),
            "jwt_audience": claims.get("azp"),
        }

    def reseed(self, session_id=None, user_id=None, instance_id=None,
               user_email=None, jwt=None, client_cookie=None, client_uat=None,
               cf_bm=None, cfuvid=None, refresh=True):
        """Replace identity + cookies from a fresh capture, then optionally refresh."""
        with self._lock:
            if session_id is not None:
                self.session_id = session_id
            if user_id is not None:
                self.user_id = user_id
            if instance_id is not None:
                self.instance_id = instance_id
            if user_email is not None:
                self.user_email = user_email
            if jwt is not None:
                self._jwt = jwt
            if client_cookie is not None:
                self._client_cookie = client_cookie
            if client_uat is not None:
                self._client_uat = client_uat
            if cf_bm is not None:
                self._cf_bm = cf_bm
            if cfuvid is not None:
                self._cfuvid = cfuvid
        if refresh:
            self.ensure_fresh(force=True)

    def export_env(self) -> Dict[str, str]:
        """Current credentials as OMNIROGUE_* env vars (only non-empty ones)."""
        candidates = {
            "OMNIROGUE_SESSION_ID": self.session_id,
            "OMNIROGUE_USER_ID": self.user_id,
            "OMNIROGUE_USER_EMAIL": self.user_email,
            "OMNIROGUE_INSTANCE_ID": self.instance_id,
            "OMNIROGUE_JWT": self._jwt,
            "OMNIROGUE_CLIENT_COOKIE": self._client_cookie,
            "OMNIROGUE_CLIENT_UAT": self._client_uat,
            "OMNIROGUE_CF_BM": self._cf_bm,
            "OMNIROGUE_CFUVID": self._cfuvid,
        }
        return {k: v for k, v in candidates.items() if v}
