import os
import sys
import asyncio
import json
import httpx
from pathlib import Path
from dotenv import load_dotenv

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
load_dotenv()

from cryptography.hazmat.primitives.ciphers.aead import AESGCM
import binascii

from app.agents.tools.mcp_client import (
    execute_mcp_tool_call_session,
)

def decrypt_field_sync(data: dict) -> str:
    key_secret = "wekraft_default_sec_key_32bytes!"
    raw_key = key_secret.ljust(32, "0")[:32].encode("utf-8")
    aesgcm = AESGCM(raw_key)

    iv = binascii.unhexlify(data["iv"])
    ct = binascii.unhexlify(data["ciphertext"])
    tag = binascii.unhexlify(data["tag"])
    return aesgcm.decrypt(iv, ct + tag, None).decode("utf-8")

sentry_cred = {
    "ciphertext": "98ce2ac40b03251410b5296282794ff999cf285eee1d531162d5a954a8e29611d796a8ea3a7780c5d2396032b752a93cc9fc5d7d6c7a8d4f87",
    "iv": "a59133840ee152a2b2faff93",
    "tag": "0782997ca84388c8cb0196fd856f5001",
}

notion_cred = {
    "ciphertext": "348a716392e87aee43e86eda555be2510865135f90db20c7e65b5f8276ac5b5462b8a8f667ccd026c71d01cb1aa640edff9aaafec42806dfa26f084b7fdd9e46feb07bcb488281b315ae0472d99ace9edcdcbf7016be",
    "iv": "f152db16fd218ed4d714b078",
    "tag": "ad1a310e704c7511de5490956d3f28da",
}

github_cred = {
    "ciphertext": "068224c3efe908c8b7e2c06ea4a3a649d124d2ce439f48a631c466da3c2529fe1cfc39f1a45d8712",
    "iv": "679f6f3f9d324d3d4a72c40a",
    "tag": "078280225ceb33ef351be08e7803f99b",
}

linear_cred = {
    "ciphertext": "ff981bcfc724aecd2125570c257d8f3119851a94f0e7cf154e52fee5921daed905dc99964f19cf205521f003928d31726060f74baf1b3c6f8e3af0da9c7ef5b705512832ec715d2015dc49241c3c172ea854afd5b4a7",
    "iv": "50353b24b0f4da0591b9d8fe",
    "tag": "e00847418fe95868eb9f61848f7d1cf1",
}

async def fetch_linear(token: str):
    url = "https://mcp.linear.app/mcp"
    res = await execute_mcp_tool_call_session(url, token, "list_issues", {})
    raw = res.get("raw", res)
    issues = []
    if isinstance(raw, dict):
        content = raw.get("content", [])
        for item in content:
            if item.get("type") == "text":
                try:
                    parsed = json.loads(item.get("text", "{}"))
                    if "issues" in parsed:
                        issues = parsed["issues"]
                except Exception:
                    pass
    return issues

async def fetch_sentry(token: str):
    url = "https://mcp.sentry.dev/mcp"
    # Find org first
    res_orgs = await execute_mcp_tool_call_session(url, token, "find_organizations", {})
    org_slug = "vrsa-solution-6q"
    if not res_orgs.get("isError"):
        raw_org = res_orgs.get("structuredContent") or res_orgs.get("raw")
        if isinstance(raw_org, dict) and "organizations" in raw_org:
            org_slug = raw_org["organizations"][0].get("slug", org_slug)

    res_issues = await execute_mcp_tool_call_session(
        url, token, "search_issues",
        {"organizationSlug": org_slug, "query": "is:unresolved", "sort": "date"}
    )
    raw = res_issues.get("raw", res_issues)
    text_content = ""
    if isinstance(raw, dict):
        content = raw.get("content", [])
        for item in content:
            if item.get("type") == "text":
                text_content += item.get("text", "")
    return text_content, org_slug

async def fetch_github(token: str):
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github.v3+json",
        "User-Agent": "Wekraft-Agent/1.0",
    }
    async with httpx.AsyncClient(headers=headers, timeout=20.0) as client:
        r = await client.get("https://api.github.com/user/issues?filter=all&state=all&per_page=30")
        if r.status_code == 200:
            return r.json()
    return []

async def main():
    linear_token = decrypt_field_sync(linear_cred)
    sentry_token = decrypt_field_sync(sentry_cred)
    github_token = decrypt_field_sync(github_cred)
    notion_token = decrypt_field_sync(notion_cred)

    print("Fetching Linear issues...")
    linear_issues = await fetch_linear(linear_token)
    print(f"Found {len(linear_issues)} Linear issues.")

    print("Fetching Sentry issues...")
    sentry_text, sentry_org = await fetch_sentry(sentry_token)
    print("Sentry fetched.")

    print("Fetching GitHub issues...")
    gh_issues = await fetch_github(github_token)
    print(f"Found {len(gh_issues)} GitHub issues.")

    # Format Notion page content
    page_title = "Unified Issue Triage & Status Report (Linear, Sentry & GitHub)"

    lines = []
    lines.append("# Unified Issue Triage & Tracking Dashboard")
    lines.append("\n> **Automated Cross-Platform Sync:** Live issues retrieved across **Linear**, **Sentry**, and **GitHub** for centralized triage and execution.")
    lines.append("\n---\n")

    # Linear Section
    lines.append("## 1. 🎯 Linear Tickets & Tasks")
    if linear_issues:
        lines.append("| ID | Title | Priority | Status | URL |")
        lines.append("| :--- | :--- | :--- | :--- | :--- |")
        for iss in linear_issues:
            iid = iss.get("id") or iss.get("key", "")
            title = iss.get("title", "").replace("|", "-")
            priority = iss.get("priority", {})
            p_name = priority.get("name", "None") if isinstance(priority, dict) else str(priority)
            status = iss.get("state", {}).get("name") or iss.get("status", "Active")
            url = iss.get("url", "")
            lines.append(f"| **[{iid}]({url})** | {title} | `{p_name}` | {status} | [Open in Linear]({url}) |")
    else:
        lines.append("No active Linear tickets found.")

    lines.append("\n---\n")

    # Sentry Section
    lines.append("## 2. 🚨 Sentry Unresolved Production Issues")
    lines.append(f"**Organization:** `{sentry_org}`\n")
    if sentry_text:
        lines.append(sentry_text)
    else:
        lines.append("No unresolved Sentry issues found.")

    lines.append("\n---\n")

    # GitHub Section
    lines.append("## 3. 🐙 GitHub Issues & Pull Requests")
    if gh_issues:
        lines.append("| # | Title | Repository | State | Author | Link |")
        lines.append("| :--- | :--- | :--- | :--- | :--- | :--- |")
        for item in gh_issues:
            num = item.get("number")
            title = item.get("title", "").replace("|", "-")
            repo = item.get("repository", {}).get("full_name", "")
            state = item.get("state", "open")
            author = item.get("user", {}).get("login", "")
            html_url = item.get("html_url", "")
            state_badge = "🟢 `open`" if state == "open" else "🟣 `closed`"
            lines.append(f"| **#{num}** | {title} | `{repo}` | {state_badge} | @{author} | [GitHub Issue]({html_url}) |")
    else:
        lines.append("No GitHub issues found.")

    full_markdown_content = "\n".join(lines)
    print("\nPrepared Markdown Content length:", len(full_markdown_content))

    print("\nCreating Notion page via MCP...")
    notion_url = "https://mcp.notion.com/mcp"
    create_args = {
        "creation_mode": "draft",
        "allow_async": False,
        "pages": [
            {
                "properties": {
                    "title": page_title
                },
                "content": full_markdown_content,
                "icon": "📊"
            }
        ]
    }

    res_create = await execute_mcp_tool_call_session(notion_url, notion_token, "notion-create-pages", create_args)
    print("Notion Create Result:")
    print(json.dumps(res_create, indent=2))

if __name__ == "__main__":
    asyncio.run(main())
