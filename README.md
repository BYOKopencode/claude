# OmniRogue Agent API

A deployable, agent-friendly bridge for **OmniRogue**.

- **OpenAI-compatible** `/v1/chat/completions` endpoint (streaming + non-streaming)
- **API-key authentication** with **multi-user** support — each key maps to its own OmniRogue identity + independent JWT rotation
- **MCP server** for tools like **Zed**, **Claude Code**, **Cursor**, etc.
- **Railway-ready** with `Dockerfile`, `railway.json`, and env-var config

## Quick start (local)

```bash
cp .env.example .env
cp users.json.example users.json   # fill in one entry per user (each with its own api_key)
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 5000
```

Test:

```bash
# /health is public (no key required)
curl http://localhost:5000/health

# authenticated request — supply your API key
curl http://localhost:5000/v1/chat/completions \
  -H "Authorization: Bearer sk-alice-CHANGE-ME" \
  -H "Content-Type: application/json" \
  -d '{"model":"claude-sonnet-4.6","messages":[{"role":"user","content":"hi"}]}'
```

## Authentication & multi-user

Every request (except `/health`) requires an API key when `REQUIRE_API_KEY=true`
(the default). Present it via **either** header:

```
Authorization: Bearer <api_key>      # OpenAI-style
X-API-Key: <api_key>
```

Each API key resolves to a distinct user with its own Clerk session and JWT
rotation state, so multiple people can share one deployment safely.

**Configure users** in any of these ways (all sources are merged):

1. **`users.json`** file (see `users.json.example`) — one object per user.
2. **`OMNIROGUE_USERS`** env var — the same JSON array, inline on one line.
3. **Legacy single-user** flat `OMNIROGUE_*` env vars + `OMNIROGUE_API_KEY`.

Set `REQUIRE_API_KEY=false` for local single-user dev: requests with no/any
key fall back to the first configured user.

## Environment variables

| Variable | Required | Description |
|----------|----------|-------------|
| `REQUIRE_API_KEY` | no | Require a valid API key on every request (default `true`) |
| `OMNIROGUE_USERS_FILE` | no | Path to a JSON array of users (default `users.json`) |
| `OMNIROGUE_USERS` | no | Inline JSON array of users (alternative to the file) |
| `OMNIROGUE_API_KEY` | no | API key for the **legacy** single-user mode |
| `OMNIROGUE_SESSION_ID` | legacy | Clerk session ID (single-user mode) |
| `OMNIROGUE_USER_ID` | legacy | Clerk user ID (single-user mode) |
| `OMNIROGUE_USER_EMAIL` | legacy | Clerk user email (single-user mode) |
| `OMNIROGUE_INSTANCE_ID` | legacy | Clerk instance ID (single-user mode) |
| `OMNIROGUE_JWT` | legacy | Current `__session` JWT (single-user mode) |
| `OMNIROGUE_CLIENT_COOKIE` | legacy | `__client` cookie value (single-user mode) |
| `OMNIROGUE_CLIENT_UAT` | legacy | `__client_uat` value (single-user mode) |
| `OMNIROGUE_CF_BM` | legacy | `__cf_bm` cookie (single-user mode) |
| `OMNIROGUE_CFUVID` | legacy | `_cfuvid` cookie (single-user mode) |
| `PORT` | no | Server port (default `5000`) |
| `HOST` | no | Server host (default `0.0.0.0`) |
| `MCP_SERVER_NAME` | no | MCP server name (default `omnirogue-agent`) |

Each user object needs: `api_key`, `session_id`, `user_id`, `user_email`,
`instance_id`, `jwt`, `client_cookie`, `client_uat`, `cf_bm`, `cfuvid` (and
optional `name`). See `.env.example` and `users.json.example`.

## Deploy to Railway

1. Create a new Railway project.
2. Connect this GitHub repo (or push the folder to Railway).
3. Add all environment variables from `.env.example` in Railway Variables.
4. Railway will build the `Dockerfile` and expose port `5000`.

Health check is configured at `/health`.

## Using with agents

### OpenAI-compatible endpoint

```json
{
  "baseURL": "https://your-service.up.railway.app/v1",
  "apiKey": "sk-alice-CHANGE-ME",
  "model": "claude-sonnet-4.6"
}
```

### MCP (Zed, Claude Code, Cursor, etc.)

#### Local stdio mode

Run `mcp_stdio.py`; it uses the first configured user. Configure users through
`users.json`, `OMNIROGUE_USERS`, or the legacy environment variables.

#### Remote SSE mode (Railway)

Point your MCP client at `https://your-service.up.railway.app/mcp/sse` and send
the same API-key header. Remote SSE connections are isolated by authenticated user.

## API endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/v1/chat/completions` | OpenAI-compatible chat |
| POST | `/v1/conversations` | Save conversation to OmniRogue backend |
| GET | `/health` | Public health + configured-user count |
| POST | `/auth/refresh` | Force current user's Clerk JWT refresh |
| POST | `/auth/cookies` | Hot-swap current user's cookies/JWT |
| GET | `/mcp/sse` | Authenticated MCP SSE transport |

## Notes

- Each user's proxy auto-refreshes its own Clerk JWT before authenticated requests.
- Sensitive values come from configuration and are never returned by the API.
- Keep API keys and Clerk credentials secret and rotate them regularly.
