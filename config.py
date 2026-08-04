"""Environment-driven configuration and multi-user credential loading."""
from __future__ import annotations

import json
import secrets
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class UserConfig(BaseModel):
    """One API consumer and its independent OmniRogue/Clerk identity."""

    model_config = ConfigDict(extra="ignore")

    api_key: str = Field(min_length=1)
    name: str | None = None
    session_id: str = Field(min_length=1)
    user_id: str = Field(min_length=1)
    user_email: str = Field(min_length=1)
    instance_id: str = Field(min_length=1)
    jwt: str = Field(min_length=1)
    # Optional: needed only for automatic JWT rotation / Cloudflare.
    client_cookie: str = ""
    client_uat: str = ""
    cf_bm: str = ""
    cfuvid: str = ""

    @field_validator("*", mode="after")
    @classmethod
    def reject_placeholder_values(cls, value: str | None, info) -> str | None:
        """Catch unfilled placeholders early.

        Credentials are sent as HTTP header values, which must be latin-1
        encodable. Copy-pasted placeholders such as `<jwt — see .env>` would
        otherwise fail deep inside the request with an opaque codec error.
        """
        if not isinstance(value, str):
            return value
        try:
            value.encode("latin-1")
        except UnicodeEncodeError as exc:
            raise ValueError(
                f"{info.field_name} contains a non-latin-1 character "
                f"({exc.object[exc.start]!r}); it looks like an unreplaced placeholder "
                "rather than a real credential"
            ) from exc
        if value.startswith("<") and value.endswith(">"):
            raise ValueError(
                f"{info.field_name} is still the placeholder {value!r}; "
                "replace it with the real value"
            )
        return value


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # Authentication / multi-user sources
    require_api_key: bool = True
    omnirogue_users_file: str = "users.json"
    omnirogue_users: str | None = None

    # Captured browser request (curl / python-requests dump)
    omnirogue_capture_file: str = "capture.txt"
    omnirogue_capture: str | None = None

    # Legacy single-user configuration
    omnirogue_api_key: str | None = None
    omnirogue_session_id: str | None = None
    omnirogue_user_id: str | None = None
    omnirogue_user_email: str | None = None
    omnirogue_instance_id: str | None = None
    omnirogue_jwt: str | None = None
    omnirogue_client_cookie: str | None = None
    omnirogue_client_uat: str | None = None
    omnirogue_cf_bm: str | None = None
    omnirogue_cfuvid: str | None = None

    # Endpoints
    omnirogue_clerk_base: str = "https://clerk.omnirogue.com"
    omnirogue_frontend_base: str = "https://omnirogue.com"
    omnirogue_api_base: str = "https://api.omnirogue.com"
    clerk_version: str = "2025-11-10"
    clerk_js_version: str = "5.127.1"

    # Server / MCP
    host: str = "0.0.0.0"
    port: int = 5000
    log_level: str = "info"
    mcp_server_name: str = "omnirogue-agent"
    mcp_server_version: str = "1.0.0"
    default_model: str = "claude-sonnet-4.6"

    def _parse_user_array(self, value: Any, source: str) -> list[UserConfig]:
        try:
            raw = json.loads(value) if isinstance(value, str) else value
            if not isinstance(raw, list):
                raise ValueError("must be a JSON array")
            return [UserConfig.model_validate(item) for item in raw]
        except (json.JSONDecodeError, ValidationError, ValueError, TypeError) as exc:
            raise RuntimeError(f"Invalid user configuration in {source}: {exc}") from exc

    def load_users(self) -> list[UserConfig]:
        """Merge file, inline JSON, and legacy sources; later sources override by key."""
        merged: dict[str, UserConfig] = {}

        users_path = Path(self.omnirogue_users_file)
        if users_path.is_file():
            for user in self._parse_user_array(users_path.read_text("utf-8"), str(users_path)):
                merged[user.api_key] = user

        if self.omnirogue_users:
            for user in self._parse_user_array(self.omnirogue_users, "OMNIROGUE_USERS"):
                merged[user.api_key] = user

        # Captured browser request -> credentials (highest-priority single user).
        capture_text = self.omnirogue_capture
        if not capture_text:
            capture_path = Path(self.omnirogue_capture_file)
            if capture_path.is_file():
                capture_text = capture_path.read_text("utf-8")
        if capture_text:
            from reseed import parse_captured_request

            parsed = parse_captured_request(capture_text)
            if parsed["_missing"]:
                raise RuntimeError(
                    "Captured request is missing required fields: "
                    + ", ".join(parsed["_missing"])
                    + ". Re-copy the request from DevTools."
                )
            api_key = self.omnirogue_api_key
            if not api_key:
                if self.require_api_key:
                    raise RuntimeError(
                        "Captured request found but OMNIROGUE_API_KEY is not set. "
                        "Add OMNIROGUE_API_KEY=<your-key>, or set "
                        "REQUIRE_API_KEY=false for local single-user development."
                    )
                api_key = f"local-{secrets.token_urlsafe(16)}"
            captured = UserConfig.model_validate({
                "api_key": api_key,
                "name": "captured",
                "session_id": parsed["session_id"],
                "user_id": parsed["user_id"],
                "user_email": self.omnirogue_user_email or "captured@omnirogue.local",
                "instance_id": parsed["instance_id"],
                "jwt": parsed["jwt"],
                "client_cookie": parsed.get("client_cookie") or "",
                "client_uat": parsed.get("client_uat") or "",
                "cf_bm": parsed.get("cf_bm") or "",
                "cfuvid": parsed.get("cfuvid") or "",
            })
            merged[captured.api_key] = captured

        # `api_key` is handled separately: the credential fields below define
        # whether legacy mode is in use at all.
        legacy_values = {
            "session_id": self.omnirogue_session_id,
            "user_id": self.omnirogue_user_id,
            "user_email": self.omnirogue_user_email,
            "instance_id": self.omnirogue_instance_id,
            "jwt": self.omnirogue_jwt,
            "client_cookie": self.omnirogue_client_cookie,
            "client_uat": self.omnirogue_client_uat,
            "cf_bm": self.omnirogue_cf_bm,
            "cfuvid": self.omnirogue_cfuvid,
        }
        if any(value is not None for value in legacy_values.values()) and not capture_text:
            missing = [key for key, value in legacy_values.items() if value is None]
            if missing:
                raise RuntimeError(
                    "Incomplete legacy user configuration; missing: " + ", ".join(missing)
                )

            api_key = self.omnirogue_api_key
            if not api_key:
                if self.require_api_key:
                    raise RuntimeError(
                        "Legacy configuration found but OMNIROGUE_API_KEY is not set. "
                        "Add OMNIROGUE_API_KEY=<your-key> to authenticate requests, "
                        "or set REQUIRE_API_KEY=false for local single-user development."
                    )
                # Auth disabled: any key is accepted, so a random internal key is fine.
                api_key = f"local-{secrets.token_urlsafe(16)}"

            legacy = UserConfig.model_validate({**legacy_values, "api_key": api_key, "name": "legacy"})
            merged[legacy.api_key] = legacy

        if not merged:
            raise RuntimeError(
                "No OmniRogue users configured. Add capture.txt, users.json, OMNIROGUE_USERS, "
                "or the legacy OMNIROGUE_* variables."
            )
        return list(merged.values())


settings = Settings()
users = settings.load_users()
