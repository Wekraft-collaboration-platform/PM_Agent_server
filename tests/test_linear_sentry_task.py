import os
import sys
import asyncio
import json
from pathlib import Path
from dotenv import load_dotenv

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
load_dotenv()

from app.agents.tools.mcp_client import (
    execute_mcp_tool_call_session,
    fetch_mcp_tools_list_session,
    _call_tool_with_retry,
)

def decrypt_field_sync(data: dict) -> str:
    """Decrypts AES-256-GCM token from Convex credentials object."""
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    import binascii

    key_secret = "wekraft_default_sec_key_32bytes!"
    raw_key = key_secret.ljust(32, "0")[:32].encode("utf-8")
    aesgcm = AESGCM(raw_key)

    iv = binascii.unhexlify(data["iv"])
    ct = binascii.unhexlify(data["ciphertext"])
    tag = binascii.unhexlify(data["tag"])
    return aesgcm.decrypt(iv, ct + tag, None).decode("utf-8")

LINEAR_CRED = {
    "ciphertext": "9779a5b4b10d15c62313ce204b5f0dec038417a202815386b7df9a2ecb5085a28b071c6a183ae191e425f219b2b0d632f1f0691827cfa43534ee7746a19ef084cfbf61e78c02867ddab1730ba7e50d4936c9fd8ead17",
    "iv": "0029c0f9089cec0206b0c153",
    "tag": "6e320b1929d285f6cafeaf966bdc38e0",
}

SENTRY_CRED = {
    "ciphertext": "c0bc0524db17c64c76c04ffaf09168fd8a985d1806cb89f9d9d8458c53c65073a446a826ba665975388ff5bc64b0a5aa0addf95cfc0e76bd92",
    "iv": "8881fc68355cdfa896643d73",
    "tag": "5c4af560a156f563f986606770239eba",
}

async def main():
    linear_token = decrypt_field_sync(LINEAR_CRED)
    sentry_token = decrypt_field_sync(SENTRY_CRED)

    print("Decrypted Linear Token:", linear_token[:12], "... length:", len(linear_token))
    print("Decrypted Sentry Token:", sentry_token[:12], "... length:", len(sentry_token))

    linear_url = "https://mcp.linear.app/mcp"
    sentry_url = "https://mcp.sentry.dev/mcp"

    # Test Linear tools
    print("\n--- Fetching Linear Tools ---")
    linear_tools = await fetch_mcp_tools_list_session(linear_url, linear_token)
    print(f"Discovered {len(linear_tools)} Linear tools.")
    print("Sample tools:", [t["name"] for t in linear_tools[:10]])

    # Test Sentry tools
    print("\n--- Fetching Sentry Tools ---")
    sentry_tools = await fetch_mcp_tools_list_session(sentry_url, sentry_token)
    print(f"Discovered {len(sentry_tools)} Sentry tools.")
    print("Sample tools:", [t["name"] for t in sentry_tools[:10]])

    # 1. Query Linear issues
    print("\n--- Linear: list_issues ---")
    res_linear_issues = await execute_mcp_tool_call_session(linear_url, linear_token, "list_issues", {})
    print("Linear Issues raw result:")
    print(json.dumps(res_linear_issues, indent=2)[:2000])

    # 2. Query Linear teams & projects
    print("\n--- Linear: list_teams ---")
    res_linear_teams = await execute_mcp_tool_call_session(linear_url, linear_token, "list_teams", {})
    print(json.dumps(res_linear_teams, indent=2)[:1000])

    # 3. Query Sentry org & issues
    print("\n--- Sentry: find_organizations ---")
    res_sentry_orgs = await execute_mcp_tool_call_session(sentry_url, sentry_token, "find_organizations", {})
    print(json.dumps(res_sentry_orgs, indent=2)[:1000])

    org_slug = "vrsa-solution-6q"
    if not res_sentry_orgs.get("isError"):
        try:
            # Let's see what's returned
            raw_org = res_sentry_orgs.get("raw")
            print("Sentry org raw:", raw_org)
        except Exception:
            pass

    print("\n--- Sentry: search_issues (unresolved) ---")
    res_sentry_issues = await execute_mcp_tool_call_session(
        sentry_url, sentry_token, "search_issues",
        {"organizationSlug": org_slug, "query": "is:unresolved", "sort": "date"}
    )
    print("Sentry Issues result:")
    print(json.dumps(res_sentry_issues, indent=2)[:2500])

if __name__ == "__main__":
    asyncio.run(main())
