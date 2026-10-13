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
    fetch_mcp_tools_list_session,
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

async def main():
    sentry_token = decrypt_field_sync(sentry_cred)
    notion_token = decrypt_field_sync(notion_cred)
    github_token = decrypt_field_sync(github_cred)
    linear_token = decrypt_field_sync(linear_cred)

    linear_url = "https://mcp.linear.app/mcp"
    sentry_url = "https://mcp.sentry.dev/mcp"
    notion_url = "https://mcp.notion.com/mcp"

    print("=== 1. FETCHING LINEAR ISSUES ===")
    res_linear_issues = await execute_mcp_tool_call_session(linear_url, linear_token, "list_issues", {})
    linear_data = res_linear_issues.get("raw", res_linear_issues)
    print("Linear raw keys / content:", type(linear_data))
    print("Linear result snippet:", str(linear_data)[:500])

    print("\n=== 2. FETCHING SENTRY ISSUES ===")
    res_sentry_orgs = await execute_mcp_tool_call_session(sentry_url, sentry_token, "find_organizations", {})
    print("Sentry orgs:", res_sentry_orgs)
    org_slug = "vrsa-solution-6q"
    if not res_sentry_orgs.get("isError") and isinstance(res_sentry_orgs.get("raw"), list) and len(res_sentry_orgs["raw"]) > 0:
        org_slug = res_sentry_orgs["raw"][0].get("slug", org_slug)
    
    res_sentry_issues = await execute_mcp_tool_call_session(
        sentry_url, sentry_token, "search_issues",
        {"organizationSlug": org_slug, "query": "is:unresolved", "sort": "date"}
    )
    sentry_data = res_sentry_issues.get("raw", res_sentry_issues)
    print("Sentry result snippet:", str(sentry_data)[:500])

    print("\n=== 3. FETCHING GITHUB ISSUES ===")
    headers = {
        "Authorization": f"Bearer {github_token}",
        "Accept": "application/vnd.github.v3+json",
        "User-Agent": "Wekraft-Agent/1.0",
    }
    github_issues = []
    async with httpx.AsyncClient(headers=headers, timeout=20.0) as client:
        # User repos or issues across user repos
        r_user = await client.get("https://api.github.com/user")
        print("GitHub User:", r_user.status_code, r_user.json().get("login") if r_user.status_code == 200 else r_user.text)
        
        # Get authenticated user's issues
        r_gh_issues = await client.get("https://api.github.com/user/issues?filter=all&state=all&per_page=20")
        print("GitHub user/issues status:", r_gh_issues.status_code)
        if r_gh_issues.status_code == 200:
            github_issues = r_gh_issues.json()
            print(f"Fetched {len(github_issues)} GitHub issues.")
            for item in github_issues[:5]:
                print(f" - #{item.get('number')} {item.get('title')} ({item.get('state')}) in {item.get('repository', {}).get('full_name')}")
        else:
            print("Failed user/issues, trying repo issues...")
            r_repos = await client.get("https://api.github.com/user/repos?sort=updated&per_page=5")
            if r_repos.status_code == 200:
                repos = r_repos.json()
                for repo in repos:
                    repo_full = repo.get("full_name")
                    r_repo_issues = await client.get(f"https://api.github.com/repos/{repo_full}/issues?state=all&per_page=10")
                    if r_repo_issues.status_code == 200:
                        iss = r_repo_issues.json()
                        github_issues.extend(iss)
                        print(f"Repo {repo_full}: {len(iss)} issues")

    print("\n=== 4. TESTING NOTION MCP / API ===")
    try:
        notion_tools = await fetch_mcp_tools_list_session(notion_url, notion_token)
        print(f"Discovered {len(notion_tools)} Notion MCP tools: {[t['name'] for t in notion_tools]}")
    except Exception as e:
        print("Notion MCP tools error:", e)

    # Also test Notion REST API directly
    notion_headers = {
        "Authorization": f"Bearer {notion_token}",
        "Notion-Version": "2022-06-28",
        "Content-Type": "application/json",
    }
    async with httpx.AsyncClient(headers=notion_headers, timeout=20.0) as client:
        r_notion_search = await client.post("https://api.notion.com/v1/search", json={})
        print("Notion REST search status:", r_notion_search.status_code)
        if r_notion_search.status_code == 200:
            search_data = r_notion_search.json()
            print(f"Notion Search Results: {len(search_data.get('results', []))} items found")
            for item in search_data.get("results", []):
                obj_type = item.get("object")
                title_prop = item.get("properties", {}).get("title") or item.get("title")
                title_str = ""
                if isinstance(title_prop, list) and len(title_prop) > 0:
                    title_str = title_prop[0].get("plain_text", "")
                elif isinstance(title_prop, dict) and "title" in title_prop:
                    t_list = title_prop.get("title", [])
                    if t_list:
                        title_str = t_list[0].get("plain_text", "")
                print(f" - [{obj_type}] ID: {item.get('id')}, Title: {title_str or 'Untitled'}")
        else:
            print("Notion search response:", r_notion_search.text)

if __name__ == "__main__":
    asyncio.run(main())
