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

GITHUB_CRED = {
    "ciphertext": "068224c3efe908c8b7e2c06ea4a3a649d124d2ce439f48a631c466da3c2529fe1cfc39f1a45d8712",
    "iv": "679f6f3f9d324d3d4a72c40a",
    "tag": "078280225ceb33ef351be08e7803f99b",
}

OWNER = "ronitrai27"
REPO = "customer_agent_punjabi"

async def main():
    token = decrypt_field_sync(GITHUB_CRED)
    print(f"Decrypted GitHub Token: {token[:8]}... (len: {len(token)})")

    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github.v3+json",
        "User-Agent": "Wekraft-Agent/1.0",
    }

    async with httpx.AsyncClient(headers=headers, timeout=20.0) as client:
        # 1. Fetch Repository Details
        print(f"\n================ 1. REPOSITORY METADATA ({OWNER}/{REPO}) ================")
        r_repo = await client.get(f"https://api.github.com/repos/{OWNER}/{REPO}")
        print("Status:", r_repo.status_code)
        if r_repo.status_code == 200:
            data = r_repo.json()
            print("Name:", data.get("name"))
            print("Full Name:", data.get("full_name"))
            print("Description:", data.get("description"))
            print("Default Branch:", data.get("default_branch"))
            print("Visibility:", data.get("visibility"))
            print("Open Issues Count:", data.get("open_issues_count"))
            print("Language:", data.get("language"))
            print("Updated At:", data.get("updated_at"))
            print("HTML URL:", data.get("html_url"))
        else:
            print("Repo response:", r_repo.text)

        # 2. Fetch Open & All Issues
        print(f"\n================ 2. ISSUES ({OWNER}/{REPO}) ================")
        r_issues = await client.get(f"https://api.github.com/repos/{OWNER}/{REPO}/issues?state=all&per_page=10")
        print("Status:", r_issues.status_code)
        if r_issues.status_code == 200:
            issues = r_issues.json()
            print(f"Total Issues & PRs Returned: {len(issues)}")
            for idx, item in enumerate(issues, 1):
                is_pr = "pull_request" in item
                kind = "Pull Request" if is_pr else "Issue"
                print(f"{idx}. [{kind} #{item.get('number')}] {item.get('title')}")
                print(f"   - State: {item.get('state')}")
                print(f"   - Author: @{item.get('user', {}).get('login')}")
                print(f"   - Created: {item.get('created_at')}")
                print(f"   - Comments: {item.get('comments')}")
                print(f"   - Link: {item.get('html_url')}")
        else:
            print("Issues response:", r_issues.text)

        # 3. Fetch Pull Requests
        print(f"\n================ 3. PULL REQUESTS ({OWNER}/{REPO}) ================")
        r_prs = await client.get(f"https://api.github.com/repos/{OWNER}/{REPO}/pulls?state=all&per_page=10")
        print("Status:", r_prs.status_code)
        if r_prs.status_code == 200:
            prs = r_prs.json()
            print(f"Total PRs: {len(prs)}")
            for idx, pr in enumerate(prs, 1):
                print(f"{idx}. [PR #{pr.get('number')}] {pr.get('title')}")
                print(f"   - State: {pr.get('state')}")
                print(f"   - Author: @{pr.get('user', {}).get('login')}")
                print(f"   - Branch: {pr.get('head', {}).get('ref')} -> {pr.get('base', {}).get('ref')}")
                print(f"   - Merged: {pr.get('merged_at') is not None}")
                print(f"   - Link: {pr.get('html_url')}")
        else:
            print("PR response:", r_prs.text)

        # 4. Fetch Recent Commits
        print(f"\n================ 4. RECENT COMMITS ({OWNER}/{REPO}) ================")
        r_commits = await client.get(f"https://api.github.com/repos/{OWNER}/{REPO}/commits?per_page=5")
        print("Status:", r_commits.status_code)
        if r_commits.status_code == 200:
            commits = r_commits.json()
            print(f"Total Recent Commits: {len(commits)}")
            for idx, c in enumerate(commits, 1):
                sha = c.get("sha", "")[:7]
                author = c.get("commit", {}).get("author", {}).get("name")
                msg = c.get("commit", {}).get("message", "").split("\n")[0]
                date = c.get("commit", {}).get("author", {}).get("date")
                print(f"{idx}. [{sha}] {msg} (by {author} on {date})")
                print(f"   - Link: {c.get('html_url')}")
        else:
            print("Commits response:", r_commits.text)

        # 5. Fetch Branches
        print(f"\n================ 5. BRANCHES ({OWNER}/{REPO}) ================")
        r_branches = await client.get(f"https://api.github.com/repos/{OWNER}/{REPO}/branches")
        print("Status:", r_branches.status_code)
        if r_branches.status_code == 200:
            branches = r_branches.json()
            print(f"Branches ({len(branches)}):")
            for b in branches:
                print(f" - {b.get('name')} (commit: {b.get('commit', {}).get('sha', '')[:7]})")
        else:
            print("Branches response:", r_branches.text)

if __name__ == "__main__":
    asyncio.run(main())
